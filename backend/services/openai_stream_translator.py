from __future__ import annotations

import json
from typing import Any, Callable

from backend.adapter.standard_request import CLAUDE_CODE_OPENAI_PROFILE, OPENCLAW_OPENAI_PROFILE
from backend.runtime.execution import RuntimeToolDirective
from backend.toolcall.parser import parse_tool_calls_detailed


STRICT_TOOL_TEXT_PREFIXES = ("{", "[", "`", "<")
BUFFERED_TOOL_CALLS_ONLY = "buffered_tool_calls_only"
DIRECTIVE_DRIVEN_TOOL_CALLS = "directive_driven_tool_calls"
TOOL_CALL_PREFIX_PROBE = "##TOOL_CALL"
MIN_TOOL_PREFIX_CACHE_CHARS = 12


class OpenAIStreamTranslator:
    def __init__(
        self,
        *,
        completion_id: str,
        created: int,
        model_name: str,
        client_profile: str,
        build_final_directive: Callable[[str], RuntimeToolDirective] | None = None,
        allowed_tool_names: list[str] | None = None,
    ):
        self.completion_id = completion_id
        self.created = created
        self.model_name = model_name
        self.client_profile = client_profile
        self.build_final_directive = build_final_directive
        self.allowed_tool_names = {name for name in (allowed_tool_names or []) if isinstance(name, str) and name}
        self.pending_chunks: list[str] = []
        self.role_chunk_sent = False
        self.emitted_tool_index = 0
        self.answer_fragments: list[str] = []
        self.buffered_toolish_fragments: list[str] = []
        self.pending_content_chunks: list[str] = []
        self.tool_calls_emitted = False
        self.reasoning_started = False
        self.reasoning_closed = False
        self.tool_text_detection_mode = self._resolve_tool_text_detection_mode(client_profile)
        self.tool_call_finalize_mode = self._resolve_tool_call_finalize_mode(client_profile)
        self.enable_prefix_probe = bool(self.allowed_tool_names)
        self.prefix_probe_buffer = ""
        self.prefix_probe_decided = False
        self._suspicion_suffix = ""  # 暂存可能是 ##TOOL_CALL## 前缀的尾部片段

    @staticmethod
    def _resolve_tool_text_detection_mode(client_profile: str) -> str:
        if client_profile == OPENCLAW_OPENAI_PROFILE:
            return "strict_prefix"
        return "accept_any_tool_syntax"

    @staticmethod
    def _resolve_tool_call_finalize_mode(client_profile: str) -> str:
        if client_profile == CLAUDE_CODE_OPENAI_PROFILE:
            return BUFFERED_TOOL_CALLS_ONLY
        return DIRECTIVE_DRIVEN_TOOL_CALLS

    def _split_suspicion_suffix(self, text: str) -> tuple[str, str]:
        """把文本末尾可能是 ##TOOL_CALL## 前缀的部分切出来暂存，返回 (safe, suspicion)。"""
        full_marker = TOOL_CALL_PREFIX_PROBE + "##"  # "##TOOL_CALL##"
        for i in range(min(len(full_marker), len(text)), 0, -1):
            suffix = text[-i:]
            if suffix[0] == "#" and full_marker.startswith(suffix):
                return text[:-i], suffix
        return text, ""

    def _looks_like_tool_output(self, text_chunk: str) -> bool:
        if not text_chunk:
            return False
        lowered = text_chunk.lower()
        common_markers = (
            "tool does not exists",
            "function.name:",
            "##tool_call##",
            "##tool_call",
            "##end_call##",
            '"tool_calls"',
            '"function":',
        )
        if any(marker in lowered for marker in common_markers):
            return True
        if self.allowed_tool_names:
            detailed = parse_tool_calls_detailed(text_chunk, self.allowed_tool_names)
            if detailed.get("saw_tool_syntax"):
                if self.tool_text_detection_mode == "strict_prefix":
                    stripped = text_chunk.lstrip()
                    return stripped.startswith(STRICT_TOOL_TEXT_PREFIXES)
                return True
        return False

    def _should_finalize_tool_calls(self, directive: RuntimeToolDirective) -> bool:
        if directive.stop_reason != "tool_use":
            return False
        if self.tool_call_finalize_mode == BUFFERED_TOOL_CALLS_ONLY:
            return bool(self.buffered_toolish_fragments)
        return True

    def _ensure_role_chunk(self) -> None:
        if self.role_chunk_sent:
            return
        yield_payload = {
            "id": self.completion_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        self.pending_chunks.append(f"data: {json.dumps(yield_payload, ensure_ascii=False)}\n\n")
        self.role_chunk_sent = True

    def _emit_content_chunk(self, text_chunk: str) -> None:
        chunk = (
            f"data: {json.dumps({'id': self.completion_id, 'object': 'chat.completion.chunk', 'created': self.created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': {'content': text_chunk}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
        )
        self.pending_chunks.append(chunk)
        self.pending_content_chunks.append(chunk)

    def _emit_reasoning_chunk(self, text_chunk: str) -> None:
        """把 Qwen 的思考内容以 DeepSeek R1 风格 reasoning_content 发出去，
        让网页端/客户端能显示推理过程。"""
        self.reasoning_started = True
        chunk = (
            f"data: {json.dumps({'id': self.completion_id, 'object': 'chat.completion.chunk', 'created': self.created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': {'reasoning_content': text_chunk}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
        )
        self.pending_chunks.append(chunk)

    def _close_reasoning_if_needed(self) -> None:
        if not self.reasoning_started or self.reasoning_closed:
            return
        chunk = (
            f"data: {json.dumps({'id': self.completion_id, 'object': 'chat.completion.chunk', 'created': self.created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': {'reasoning_content': ''}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
        )
        self.pending_chunks.append(chunk)
        chunk_content = (
            f"data: {json.dumps({'id': self.completion_id, 'object': 'chat.completion.chunk', 'created': self.created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': {'content': ''}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
        )
        self.pending_chunks.append(chunk_content)
        self.reasoning_closed = True

    def _discard_pending_content_chunks(self) -> None:
        if not self.pending_content_chunks:
            return
        pending_content_ids = {id(chunk) for chunk in self.pending_content_chunks}
        self.pending_chunks = [chunk for chunk in self.pending_chunks if id(chunk) not in pending_content_ids]
        self.pending_content_chunks = []

    def on_delta(self, evt: dict[str, Any], text_chunk: str | None, tool_calls: list[dict[str, Any]] | None) -> None:
        self._ensure_role_chunk()

        phase = evt.get("phase")
        status = evt.get("status")

        if phase in ("think", "thinking_summary") and status == "finished":
            self._close_reasoning_if_needed()
            return

        if text_chunk and phase in ("think", "thinking_summary"):
            # 把思考内容作为 reasoning_content 发给客户端（DeepSeek R1 风格）
            # 网页端 TestPage 会单独显示这段"推理过程"
            self._emit_reasoning_chunk(text_chunk)
            return

        if text_chunk and phase == "answer":
            self._close_reasoning_if_needed()
            self.answer_fragments.append(text_chunk)
            if self.enable_prefix_probe and not self.prefix_probe_decided:
                self.prefix_probe_buffer += text_chunk
                if len(self.prefix_probe_buffer) < MIN_TOOL_PREFIX_CACHE_CHARS:
                    return
                buffered_probe = self.prefix_probe_buffer
                self.prefix_probe_buffer = ""
                self.prefix_probe_decided = True
                if buffered_probe.lstrip().startswith(TOOL_CALL_PREFIX_PROBE):
                    self.buffered_toolish_fragments.append(buffered_probe)
                    return
                text_chunk = buffered_probe
            if self._looks_like_tool_output(text_chunk):
                self.buffered_toolish_fragments.append(text_chunk)
            elif self.buffered_toolish_fragments:
                self.buffered_toolish_fragments.append(text_chunk)
            else:
                # 发给客户端前，检查尾部是否是 ##TOOL_CALL## 的开始，避免跨 chunk 被漏发
                if self._suspicion_suffix:
                    text_chunk = self._suspicion_suffix + text_chunk
                    self._suspicion_suffix = ""
                if self._looks_like_tool_output(text_chunk):
                    self.buffered_toolish_fragments.append(text_chunk)
                else:
                    safe_part, suspicion = self._split_suspicion_suffix(text_chunk)
                    if suspicion:
                        self._suspicion_suffix = suspicion
                        if safe_part:
                            self._emit_content_chunk(safe_part)
                    else:
                        self._emit_content_chunk(text_chunk)
            return

        if tool_calls:
            self.emit_tool_calls(tool_calls)

    def emit_tool_calls(self, tool_calls: list[dict[str, Any]]) -> None:
        self._ensure_role_chunk()
        self._close_reasoning_if_needed()
        for tool_call in tool_calls:
            idx = self.emitted_tool_index
            self.emitted_tool_index += 1
            self.pending_chunks.append(
                f"data: {json.dumps({'id': self.completion_id, 'object': 'chat.completion.chunk', 'created': self.created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': idx, 'id': tool_call['id'], 'type': 'function', 'function': {'name': tool_call['name'], 'arguments': json.dumps(tool_call['input'], ensure_ascii=False)}}]}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
            )
        if tool_calls:
            self.tool_calls_emitted = True

    def drain_chunks(self) -> list[str]:
        chunks = list(self.pending_chunks)
        self.pending_chunks.clear()
        return chunks

    def finalize(self, finish_reason: str) -> list[str]:
        final_finish_reason = finish_reason
        self._close_reasoning_if_needed()
        if self.enable_prefix_probe and self.prefix_probe_buffer:
            if self.prefix_probe_buffer.startswith(TOOL_CALL_PREFIX_PROBE):
                self.buffered_toolish_fragments.append(self.prefix_probe_buffer)
            elif not self.tool_calls_emitted:
                self._emit_content_chunk(self.prefix_probe_buffer)
            self.prefix_probe_buffer = ""
            self.prefix_probe_decided = True
        # 把尾部可疑暂存也加入缓冲，避免遗漏
        if self._suspicion_suffix:
            self.buffered_toolish_fragments.append(self._suspicion_suffix)
            self._suspicion_suffix = ""
        buffered_text = "".join(self.buffered_toolish_fragments)
        has_tool_marker = "##tool_call" in buffered_text.lower() or "<tool_call>" in buffered_text.lower()
        if self.build_final_directive is not None and not self.tool_calls_emitted:
            directive = self.build_final_directive("".join(self.answer_fragments))
            if self._should_finalize_tool_calls(directive):
                self._discard_pending_content_chunks()
                tool_calls = [
                    {
                        "id": block["id"],
                        "name": block["name"],
                        "input": block.get("input", {}),
                    }
                    for block in directive.tool_blocks
                    if block.get("type") == "tool_use"
                ]
                if tool_calls:
                    self.emit_tool_calls(tool_calls)
                    final_finish_reason = "tool_calls"
            elif buffered_text and not has_tool_marker and finish_reason != "tool_calls":
                # 只有确认不含工具调用标记时才以普通文本发出
                self._emit_content_chunk(buffered_text)
        elif buffered_text and not self.tool_calls_emitted and not has_tool_marker and finish_reason != "tool_calls":
            self._emit_content_chunk(buffered_text)

        chunks = self.drain_chunks()
        chunks.append(
            f"data: {json.dumps({'id': self.completion_id, 'object': 'chat.completion.chunk', 'created': self.created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': final_finish_reason}]}, ensure_ascii=False)}\n\n"
        )
        chunks.append("data: [DONE]\n\n")
        return chunks
