import asyncio
import json
import logging
import time
from typing import AsyncIterator

import httpx

from backend.core.account_pool import AccountPool
from backend.core.config import settings
from backend.core.request_trace import trace_context_fields
from backend.services.auth_resolver import BASE_URL, AuthResolver
from backend.upstream.payload_builder import build_chat_payload
from backend.upstream.qwen_executor import QwenExecutor
from backend.upstream.sse_consumer import parse_sse_chunk

log = logging.getLogger("qwen2api.client")


class QwenClient:
    def __init__(self, account_pool: AccountPool):
        self.account_pool = account_pool
        self.auth_resolver = AuthResolver(account_pool) if account_pool is not None else None
        self.executor = QwenExecutor(self, account_pool)
        self._deleted_chat_ids: set[str] = set()
        self._deleting_chat_ids: dict[str, asyncio.Future[bool]] = {}
        self._delete_lock = asyncio.Lock()

        # HTTP连接池配置（对齐 ds2api 的高性能设置）
        limits = httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=30.0,
        )
        # 增加 read timeout 以支持长任务（工具调用可能需要更长时间）
        timeout = httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)
        self._http_client = httpx.AsyncClient(
            limits=limits,
            timeout=timeout,
            http2=True,
            follow_redirects=True,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._http_client.aclose()
        return False

    @staticmethod
    def _build_headers(token: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {token}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": f"{BASE_URL}/",
            "Origin": BASE_URL,
            "Connection": "keep-alive",
            "Content-Type": "application/json",
        }

    async def _request_json(self, method: str, path: str, token: str, body: dict | None = None, timeout: float = 30.0) -> dict:
        resp = await self._http_client.request(
            method,
            f"{BASE_URL}{path}",
            headers=self._build_headers(token),
            json=body,
            timeout=timeout,
        )
        return {"status": resp.status_code, "body": resp.text}

    async def create_chat(self, token: str, model: str, chat_type: str = "t2t", *, use_prewarmed: bool = True) -> str:
        return await self.executor.create_chat(token, model, chat_type=chat_type, use_prewarmed=use_prewarmed)

    async def delete_chat(self, token: str, chat_id: str):
        if not token or not chat_id:
            return True
        res = await self._request_json("DELETE", f"/api/v2/chats/{chat_id}", token, timeout=20.0)
        status = int(res.get("status") or 0)
        if status in (200, 204, 404):
            return True
        body = (res.get("body") or "")[:200]
        raise RuntimeError(f"delete_chat HTTP {status}: {body}")

    async def delete_chat_reliable(
        self,
        token: str,
        chat_id: str,
        *,
        source: str = "cleanup",
        attempts: int | None = None,
        base_delay: float | None = None,
    ) -> bool:
        """Best-effort chat deletion with bounded retries."""
        if not token or not chat_id:
            return True
        trace = trace_context_fields()
        delete_future: asyncio.Future[bool] | None = None
        owns_delete = False
        async with self._delete_lock:
            if chat_id in self._deleted_chat_ids:
                log.info(
                    "[DeleteChat] skip_duplicate chat_id=%s source=%s surface=%s req_id=%s marker=%s context_chat_id=%s",
                    chat_id,
                    source,
                    trace["surface"],
                    trace["req_id"],
                    trace["marker"],
                    trace["chat_id"],
                )
                return True
            delete_future = self._deleting_chat_ids.get(chat_id)
            if delete_future is None:
                delete_future = asyncio.get_running_loop().create_future()
                self._deleting_chat_ids[chat_id] = delete_future
                owns_delete = True
            else:
                log.info(
                    "[DeleteChat] skip_inflight chat_id=%s source=%s surface=%s req_id=%s marker=%s context_chat_id=%s",
                    chat_id,
                    source,
                    trace["surface"],
                    trace["req_id"],
                    trace["marker"],
                    trace["chat_id"],
                )
        if not owns_delete:
            return await asyncio.shield(delete_future)

        max_attempts = max(1, int(attempts or settings.CHAT_DELETE_RETRY_ATTEMPTS))
        delay = max(0.0, float(base_delay if base_delay is not None else settings.CHAT_DELETE_RETRY_DELAY_SECONDS))
        last_error: Exception | None = None
        deleted = False
        try:
            for attempt in range(1, max_attempts + 1):
                try:
                    await self.delete_chat(token, chat_id)
                    deleted = True
                    async with self._delete_lock:
                        self._deleted_chat_ids.add(chat_id)
                        if len(self._deleted_chat_ids) > 5000:
                            self._deleted_chat_ids.clear()
                            self._deleted_chat_ids.add(chat_id)
                    log.info(
                        "[DeleteChat] deleted chat_id=%s source=%s attempt=%s surface=%s req_id=%s marker=%s context_chat_id=%s",
                        chat_id,
                        source,
                        attempt,
                        trace["surface"],
                        trace["req_id"],
                        trace["marker"],
                        trace["chat_id"],
                    )
                    return True
                except Exception as exc:
                    last_error = exc
                    if attempt >= max_attempts:
                        break
                    log.warning(
                        "[DeleteChat] retry chat_id=%s source=%s attempt=%s/%s surface=%s req_id=%s marker=%s error=%s",
                        chat_id,
                        source,
                        attempt,
                        max_attempts,
                        trace["surface"],
                        trace["req_id"],
                        trace["marker"],
                        exc,
                    )
                    await asyncio.sleep(delay * attempt)
            log.warning(
                "[DeleteChat] failed chat_id=%s source=%s surface=%s req_id=%s marker=%s error=%s",
                chat_id,
                source,
                trace["surface"],
                trace["req_id"],
                trace["marker"],
                last_error,
            )
            return False
        finally:
            async with self._delete_lock:
                current_future = self._deleting_chat_ids.get(chat_id)
                if current_future is delete_future:
                    if not delete_future.done():
                        delete_future.set_result(deleted)
                    self._deleting_chat_ids.pop(chat_id, None)

    def delete_chat_background(
        self,
        token: str,
        chat_id: str,
        *,
        source: str = "cleanup",
        attempts: int | None = None,
        base_delay: float | None = None,
    ) -> asyncio.Task[None] | None:
        """Schedule best-effort chat deletion without blocking the caller."""
        if not token or not chat_id:
            return None

        async def runner() -> None:
            try:
                await self.delete_chat_reliable(
                    token,
                    chat_id,
                    source=source,
                    attempts=attempts,
                    base_delay=base_delay,
                )
            except Exception as exc:
                log.warning("[DeleteChat] background_failed chat_id=%s source=%s error=%s", chat_id, source, exc)

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(runner())
            task.set_name(f"delete-chat-{chat_id[:8]}")
            return task
        except RuntimeError as exc:
            log.warning("[DeleteChat] schedule_failed chat_id=%s source=%s error=%s", chat_id, source, exc)
            return None

    async def list_chats(self, token: str, limit: int = 50) -> list[dict]:
        res = await self._request_json("GET", f"/api/v2/chats?limit={limit}", token, timeout=20.0)
        if res["status"] != 200:
            return []
        try:
            data = json.loads(res.get("body", "{}"))
        except Exception:
            return []
        chats = data.get("data", [])
        return chats if isinstance(chats, list) else []

    @staticmethod
    def _classify_auth_failure(status_code: int | None, text: str) -> tuple[str, str]:
        lower = (text or "").lower()
        if any(keyword in lower for keyword in (
            "banned",
            "suspended",
            "disabled",
            "deactivated",
            "blocked",
            "violation",
            "violated",
            "封禁",
            "封号",
            "禁用",
            "停用",
            "违规",
        )):
            return "banned", "账号疑似被封禁或禁用"
        if "aliyun_waf" in lower or "<!doctype" in lower or "captcha" in lower:
            return "unknown", "官网返回风控/WAF 页面，需浏览器登录刷新后复验"
        if status_code in (401, 403):
            return "auth_error", "Token 已失效或认证失败"
        if status_code == 429:
            return "rate_limited", "官网验证接口限流"
        if status_code is None:
            return "unknown", "官网验证请求失败"
        return "unknown", f"官网验证接口返回 HTTP {status_code}"

    async def verify_token_detail(self, token: str) -> dict:
        """Probe chat.qwen.ai auth endpoint and preserve the reason.

        This is intentionally a direct official-site probe. Callers that manage accounts
        should use verify_account(), which refreshes expired tokens and then probes again.
        """
        if not token:
            return {
                "valid": False,
                "status_code": "auth_error",
                "status_text": "认证失效",
                "error": "Token 为空",
                "upstream_status": None,
            }

        try:
            resp = await self._http_client.get(
                f"{BASE_URL}/api/v1/auths/",
                headers=self._build_headers(token),
                timeout=15.0,
            )
            body_preview = resp.text[:1000]
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except Exception as e:
                    log.warning(
                        f"[verify_token] JSON 解析失败（可能被拦截或代理异常）: "
                        f"{e}, status={resp.status_code}, text={resp.text[:100]}"
                    )
                    status_code, error = self._classify_auth_failure(resp.status_code, resp.text)
                    # 保持旧行为：WAF/HTML 页面不立即判死，交给账号验证流程继续处理。
                    return {
                        "valid": status_code == "unknown",
                        "status_code": status_code,
                        "status_text": "未知" if status_code == "unknown" else "失效",
                        "error": error,
                        "upstream_status": resp.status_code,
                    }

                if data.get("role") == "user":
                    return {
                        "valid": True,
                        "status_code": "valid",
                        "status_text": "正常",
                        "error": "",
                        "upstream_status": resp.status_code,
                    }
                status_code, error = self._classify_auth_failure(resp.status_code, json.dumps(data, ensure_ascii=False))
                return {
                    "valid": False,
                    "status_code": status_code,
                    "status_text": "失效",
                    "error": error or "官网认证响应不是有效用户",
                    "upstream_status": resp.status_code,
                }

            status_code, error = self._classify_auth_failure(resp.status_code, body_preview)
            return {
                "valid": False,
                "status_code": status_code,
                "status_text": "封禁" if status_code == "banned" else "认证失效",
                "error": error,
                "upstream_status": resp.status_code,
            }
        except Exception as e:
            log.warning(f"[verify_token] HTTP 请求异常: {e}")
            return {
                "valid": False,
                "status_code": "unknown",
                "status_text": "未知",
                "error": f"官网验证请求异常: {e}",
                "upstream_status": None,
            }

    async def verify_token(self, token: str) -> bool:
        """Backward-compatible boolean token check."""
        detail = await self.verify_token_detail(token)
        # 兼容旧逻辑：WAF/HTML 这类非明确 token 失败不在此处判死。
        return bool(detail.get("valid"))

    def _mark_account_valid(self, acc, message: str = "官网验证通过"):
        acc.valid = True
        acc.activation_pending = False
        acc.status_code = "valid"
        acc.last_error = message
        acc.consecutive_failures = 0
        acc.rate_limit_strikes = 0

    def _mark_account_failed(self, acc, status_code: str, error: str):
        acc.valid = False
        acc.status_code = status_code or "auth_error"
        acc.last_error = error or "官网验证失败"
        if status_code == "pending_activation":
            acc.activation_pending = True
        if status_code == "banned":
            acc.activation_pending = False
        acc.consecutive_failures += 1

    async def verify_account(self, acc) -> dict:
        """Verify an account against chat.qwen.ai and refresh expired tokens.

        Flow: probe current token on official site -> if expired and password exists,
        open browser login to refresh token -> probe the refreshed token again. Only
        explicit banned/disabled responses are reported as banned.
        """
        before_token = acc.token or ""
        result = {
            "email": acc.email,
            "valid": False,
            "refreshed": False,
            "refresh_attempted": False,
            "refresh_ok": False,
            "status_code": "auth_error",
            "status_text": "认证失效",
            "error": "",
            "upstream_status": None,
        }

        first = await self.verify_token_detail(before_token)
        result.update({k: first.get(k) for k in ("valid", "status_code", "status_text", "error", "upstream_status")})
        if first.get("valid") and first.get("status_code") == "valid":
            self._mark_account_valid(acc, "官网 token 验证通过")
            await self.account_pool.save()
            self.account_pool._reset_concurrency_limits()
            return result

        if first.get("status_code") == "banned":
            self._mark_account_failed(acc, "banned", first.get("error", "账号疑似被封禁"))
            await self.account_pool.save()
            self.account_pool._reset_concurrency_limits()
            result.update({"valid": False, "status_code": "banned", "status_text": "封禁"})
            return result

        if acc.password and self.auth_resolver is not None:
            log.info(f"[校验] {acc.email} token 不可用/过期，正在打开官网登录刷新 token...")
            result["refresh_attempted"] = True
            refresh_ok = await self.auth_resolver.refresh_token(acc)
            result["refresh_ok"] = refresh_ok
            result["refreshed"] = bool(refresh_ok and acc.token and acc.token != before_token)

            if refresh_ok:
                second = await self.verify_token_detail(acc.token)
                result.update({k: second.get(k) for k in ("valid", "status_code", "status_text", "error", "upstream_status")})
                if second.get("valid") and second.get("status_code") == "valid":
                    self._mark_account_valid(
                        acc,
                        "官网验证通过，Token 已刷新" if result["refreshed"] else "官网验证通过，Token 仍有效",
                    )
                    await self.account_pool.save()
                    self.account_pool._reset_concurrency_limits()
                    result.update({"valid": True, "status_code": "valid", "status_text": "正常", "error": acc.last_error})
                    return result
                if second.get("status_code") == "banned":
                    self._mark_account_failed(acc, "banned", second.get("error", "账号疑似被封禁"))
                    await self.account_pool.save()
                    self.account_pool._reset_concurrency_limits()
                    result.update({"valid": False, "status_code": "banned", "status_text": "封禁"})
                    return result

        final_status = "pending_activation" if getattr(acc, "activation_pending", False) else (result.get("status_code") or "auth_error")
        if final_status in ("unknown", "rate_limited"):
            # 未知/临时失败不要误报封禁，但也不能作为可用账号调度。
            final_status = "auth_error"
        final_error = result.get("error") or "Token 失效，且自动刷新未成功"
        if acc.password and result.get("refresh_attempted") and not result.get("refresh_ok"):
            final_error = f"{final_error}；已尝试打开官网登录刷新但未获取到有效 Token"
        elif not acc.password:
            final_error = f"{final_error}；未保存密码，无法自动刷新 Token"

        self._mark_account_failed(acc, final_status, final_error)
        await self.account_pool.save()
        self.account_pool._reset_concurrency_limits()
        result.update({
            "valid": False,
            "status_code": acc.get_status_code(),
            "status_text": acc.get_status_text(),
            "error": acc.last_error,
        })
        return result

    async def list_models(self, token: str) -> list:
        try:
            resp = await self._http_client.get(
                f"{BASE_URL}/api/models",
                headers=self._build_headers(token),
                timeout=10.0,
            )
            if resp.status_code != 200:
                return []
            try:
                return resp.json().get("data", [])
            except Exception as e:
                log.warning(f"[list_models] JSON 解析失败: {e}, status={resp.status_code}, text={resp.text[:100]}")
                return []
        except Exception:
            return []

    # ---- cached upstream-pool model list ----
    # Fetches chat.qwen.ai /api/models using a valid token from the account pool
    # (not the caller's API KEY). Cached for _UPSTREAM_MODELS_TTL seconds.
    _UPSTREAM_MODELS_TTL = 300
    _upstream_models_cache: list[dict] = []
    _upstream_models_fetched_at: float = 0.0

    async def list_models_from_pool(self) -> list[dict]:
        now = time.time()
        if self._upstream_models_cache and (now - self._upstream_models_fetched_at) < self._UPSTREAM_MODELS_TTL:
            return self._upstream_models_cache
        if self.account_pool is None:
            return []
        acc = None
        try:
            acc = await self.account_pool.acquire_wait(timeout=5)
            if not acc:
                return []
            models = await self.list_models(acc.token)
            if models:
                QwenClient._upstream_models_cache = models
                QwenClient._upstream_models_fetched_at = now
            return models
        except Exception as e:
            log.warning(f"[list_models_from_pool] failed: {e}")
            return []
        finally:
            if acc is not None:
                self.account_pool.release(acc)

    def _build_payload(
        self,
        chat_id: str,
        model: str,
        content: str,
        has_custom_tools: bool = False,
        files: list[dict] | None = None,
        chat_type: str = "t2t",
        image_options: dict | None = None,
        thinking_enabled: bool | None = None,
        enable_search: bool = False,
    ) -> dict:
        return build_chat_payload(
            chat_id,
            model,
            content,
            has_custom_tools,
            files=files,
            chat_type=chat_type,
            image_options=image_options,
            thinking_enabled=thinking_enabled,
            enable_search=enable_search,
        )

    def parse_sse_chunk(self, chunk: str) -> list[dict]:
        return parse_sse_chunk(chunk)

    async def stream(
        self,
        token: str,
        chat_id: str,
        model: str,
        content: str,
        has_custom_tools: bool = False,
        files: list[dict] | None = None,
        chat_type: str = "t2t",
        image_options: dict | None = None,
        thinking_enabled: bool | None = None,
        enable_search: bool = False,
        system: str = "",
    ):
        async for event in self.executor.stream(
            token,
            chat_id,
            model,
            content,
            has_custom_tools,
            files=files,
            chat_type=chat_type,
            image_options=image_options,
            thinking_enabled=thinking_enabled,
            enable_search=enable_search,
            system=system,
        ):
            yield event

    async def stream_chat_once(self, token: str, chat_id: str, payload: dict) -> AsyncIterator[dict]:
        try:
            payload_dump = json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception:
            payload_dump = str(payload)
        log.info("[Qwen请求] POST %s/api/v2/chat/completions?chat_id=%s\n%s", BASE_URL, chat_id, payload_dump)

        # 使用全局连接池，复用连接（对齐 ds2api）
        async with self._http_client.stream(
            "POST",
            f"{BASE_URL}/api/v2/chat/completions?chat_id={chat_id}",
            headers={**self._build_headers(token), "Accept": "text/event-stream"},
            json=payload,
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                try:
                    body_text = body.decode("utf-8", errors="replace")
                except Exception:
                    body_text = str(body)
                log.warning("[Qwen响应] HTTP %s chat_id=%s body=%r", resp.status_code, chat_id, body_text)
                yield {"status": resp.status_code, "body": body}
                return
            # 使用 aiter_text() 保证 UTF-8 正确处理和 SSE 格式完整
            async for chunk in resp.aiter_text():
                if chunk:
                    log.info("[Qwen流式] chat_id=%s chunk=%r", chat_id, chunk)
                    yield {"chunk": chunk}
            yield {"status": "streamed"}

    async def post_chat_completion_once(self, token: str, chat_id: str, payload: dict, timeout: float = 60.0) -> dict:
        resp = await self._http_client.post(
            f"{BASE_URL}/api/v2/chat/completions?chat_id={chat_id}",
            headers={**self._build_headers(token), "X-Accel-Buffering": "no"},
            json=payload,
            timeout=timeout,
        )
        return {"status": resp.status_code, "body": resp.text}

    async def get_vision_task_status(self, token: str, task_id: str, timeout: float = 30.0) -> dict:
        resp = await self._http_client.get(
            f"{BASE_URL}/api/v1/tasks/status/{task_id}",
            headers=self._build_headers(token),
            timeout=timeout,
        )
        return {"status": resp.status_code, "body": resp.text}

    async def get_chat_detail(self, token: str, chat_id: str, timeout: float = 30.0) -> dict:
        resp = await self._http_client.get(
            f"{BASE_URL}/api/v2/chats/{chat_id}",
            headers=self._build_headers(token),
            timeout=timeout,
        )
        return {"status": resp.status_code, "body": resp.text}

    async def chat_stream_events_with_retry(
        self,
        model: str,
        content: str,
        has_custom_tools: bool = False,
        files: list[dict] | None = None,
        fixed_account=None,
        existing_chat_id: str | None = None,
        delete_on_close: bool = False,
        use_prewarmed: bool = True,
        chat_type: str = "t2t",
        image_options: dict | None = None,
        thinking_enabled: bool | None = None,
        enable_search: bool = False,
        system: str = "",
    ):
        async for item in self.executor.chat_stream_events_with_retry(
            model,
            content,
            has_custom_tools,
            files=files,
            fixed_account=fixed_account,
            existing_chat_id=existing_chat_id,
            delete_on_close=delete_on_close,
            use_prewarmed=use_prewarmed,
            chat_type=chat_type,
            image_options=image_options,
            thinking_enabled=thinking_enabled,
            enable_search=enable_search,
            system=system,
        ):
            yield item
