from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio
import json
import logging
from typing import Any

from backend.adapter.standard_request import StandardRequest
from backend.adapter.cli_proxy import CLIProxy
from backend.core.config import resolve_model
from backend.core.request_logging import new_request_id, request_context, update_request_context
from backend.runtime import stream_presenter
from backend.runtime.execution import collect_completion_run, cleanup_runtime_resources
from backend.runtime.visible_text import VisibleTextSanitizer, sanitize_visible_text
from backend.services.auth_quota import resolve_auth_context
from backend.services.completion_bridge import force_fresh_chat_after_empty_response, is_empty_upstream_response
from backend.services.token_calc import calculate_usage

log = logging.getLogger("qwen2api.gemini")
router = APIRouter()

GEMINI_STREAM_MEDIA_TYPE = "application/json"


def _build_standard_request(model: str, body: dict, *, stream: bool | None = None) -> StandardRequest:
    """使用 CLIProxy 进行协议转换"""
    standard_request = CLIProxy.from_gemini(model, body, stream=stream)
    CLIProxy.log_conversion("gemini", standard_request.response_model, len(standard_request.prompt), len(standard_request.tools))
    return standard_request


def _gemini_chunk_payload(text: str) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": text}],
                    "role": "model",
                }
            }
        ]
    }


async def _load_and_validate_request(request: Request, model: str, *, force_stream: bool | None = None):
    app = request.app
    users_db = app.state.users_db
    client = app.state.qwen_client

    auth = await resolve_auth_context(request, users_db)
    token = auth.token

    body = await request.json()
    try:
        log.info("[GEMINI] Client Request (Raw JSON):\n%s", json.dumps(body, ensure_ascii=False, indent=2))
    except Exception:
        log.info("[GEMINI] Client Request (Raw):\n%s", body)
    standard_request = _build_standard_request(model, body, stream=force_stream)
    update_request_context(resolved_model=standard_request.resolved_model)
    return users_db, client, token, standard_request


@router.post("/v1beta/models/{model}:generateContent")
@router.post("/v1/models/{model}:generateContent")
@router.post("/models/{model}:generateContent")
async def gemini_generate_content(model: str, request: Request):
    with request_context(req_id=new_request_id(), surface="gemini", requested_model=model):
        users_db, client, token, standard_request = await _load_and_validate_request(request, model, force_stream=False)
        content = standard_request.prompt
        log.info(f"[Gemini] route=generateContent model={standard_request.resolved_model}, stream={standard_request.stream}, prompt_len={len(content)}")

        try:
            execution = await collect_completion_run(client, standard_request, content)
            if is_empty_upstream_response(execution):
                force_fresh_chat_after_empty_response(standard_request)
                await cleanup_runtime_resources(client, execution.acc, execution.chat_id, preserve_chat=False)
                raise RuntimeError("empty upstream response")
        except Exception as e:
            log.error(f"Gemini proxy failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        visible_text = sanitize_visible_text(execution.state.answer_text)
        usage = calculate_usage(content, visible_text)
        users = await users_db.get()
        for u in users:
            if u["id"] == token:
                u["used_tokens"] += usage["total_tokens"]
                break
        await users_db.save(users)
        await cleanup_runtime_resources(client, execution.acc, execution.chat_id)

        log.info(f"[Gemini] Request complete. Generated {len(visible_text)} visible characters.")
        return JSONResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": visible_text}],
                            "role": "model",
                        }
                    }
                ]
            }
        )


@router.post("/v1beta/models/{model}:streamGenerateContent")
@router.post("/v1/models/{model}:streamGenerateContent")
@router.post("/models/{model}:streamGenerateContent")
async def gemini_stream_generate_content(model: str, request: Request):
    with request_context(req_id=new_request_id(), surface="gemini", requested_model=model):
        users_db, client, token, standard_request = await _load_and_validate_request(request, model, force_stream=True)
        content = standard_request.prompt
        log.info(f"[Gemini] route=streamGenerateContent model={standard_request.resolved_model}, stream={standard_request.stream}, prompt_len={len(content)}")

        async def generate():
            queue: asyncio.Queue[str | None] = asyncio.Queue()
            answer_sanitizer = VisibleTextSanitizer()

            async def on_delta(evt, text_chunk, _):
                if text_chunk and evt.get("phase") == "answer":
                    visible_chunk = answer_sanitizer.feed(text_chunk)
                    if visible_chunk:
                        await queue.put(stream_presenter.gemini_text_chunk(visible_chunk))

            async def runner():
                execution = None
                try:
                    execution = await collect_completion_run(
                        client,
                        standard_request,
                        content,
                        capture_events=False,
                        on_delta=on_delta,
                    )
                    if is_empty_upstream_response(execution):
                        force_fresh_chat_after_empty_response(standard_request)
                        await cleanup_runtime_resources(client, execution.acc, execution.chat_id, preserve_chat=False)
                        raise RuntimeError("empty upstream response")

                    visible_text = sanitize_visible_text(execution.state.answer_text)
                    usage = calculate_usage(content, visible_text)
                    users = await users_db.get()
                    for u in users:
                        if u["id"] == token:
                            u["used_tokens"] += usage["total_tokens"]
                            break
                    await users_db.save(users)
                    await cleanup_runtime_resources(client, execution.acc, execution.chat_id)
                    log.info(f"[Gemini] Request complete. Generated {len(visible_text)} visible characters.")
                except Exception as e:
                    await queue.put(json.dumps({"error": str(e)}) + "\n")
                finally:
                    visible_tail = answer_sanitizer.flush()
                    if visible_tail:
                        await queue.put(stream_presenter.gemini_text_chunk(visible_tail))
                    await queue.put(None)

            task = asyncio.create_task(runner())
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
            await task

        return StreamingResponse(generate(), media_type=GEMINI_STREAM_MEDIA_TYPE)
