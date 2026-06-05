from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from backend.adapter.standard_request import CLAUDE_CODE_OPENAI_PROFILE, StandardRequest
from backend.core.config import settings
from backend.core.request_logging import update_request_context
from backend.core.request_trace import find_test_markers, log_test_prompt
from backend.runtime.stream_metrics import StreamMetrics
from backend.runtime.visible_text import sanitize_visible_text
from backend.services import tool_parser
from backend.services.workspace_context import build_workspace_final_reminder
from backend.toolcall.formats_qnml import render_qnml_tool_calls
from backend.toolcall.normalize import normalize_tool_name
from backend.toolcall.stream_state import StreamingToolCallState


# Qwen 鍋跺皵鐢熸垚鐨勬瘨鎬?宸ュ叿涓嶅瓨鍦?鎴?鏃犳硶缁х画"骞昏銆?
# 鍦ㄦ祦寮忔敹鍒板墠 20 瀛楁椂璇嗗埆锛岃Е鍙戞棭鏈熸嫤鎴?+ retry 鑰屼笉鏄祦缁欏鎴风銆?
_TOXIC_REFUSAL_RE = re.compile(
    r"Tool\s+\S+\s+(?:does\s+not\s+exists?|is\s+not\s+(?:available|registered))"
    r"|I\s+cannot\s+execute\s+this\s+tool"
    r"|I['\u2019]?\s*m\s+sorry[,. ]"
    r"|I\s+cannot\s+(?:help|assist|proceed|continue|support|perform)"
    r"|I['\u2019]?m\s+not\s+(?:able|designed)\s+to"
    r"|unable\s+to\s+(?:proceed|continue|perform|complete)"
    r"|(?:\u8be5)?\u5de5\u5177.{0,12}?\u4e0d\u5b58\u5728"
    r"|\u65e0\u6cd5(?:\u7ee7\u7eed|\u8fdb\u884c|\u652f\u6301|\u5b8c\u6210|\u6267\u884c)"
    r"|\u4e0d\u80fd(?:\u7ee7\u7eed|\u8fdb\u884c|\u652f\u6301|\u5b8c\u6210|\u6267\u884c)"
    r"|\u62b1\u6b49.{0,20}?(?:\u65e0\u6cd5|\u4e0d\u80fd|\u4e0d\u652f\u6301)",
    re.IGNORECASE,
)


log = logging.getLogger("qwen2api.runtime")

QNML_TOOL_MARKERS = (
    "<|qnml|tool_calls",
    "</|qnml|tool_calls",
    "<|qnml|invoke",
    "</|qnml|invoke",
    "<|qnml|parameter",
    "</|qnml|parameter",
)
LEGACY_XML_TOOL_MARKERS = (
    "<tool_calls",
    "</tool_calls",
    "<invoke",
    "</invoke",
    "<parameter",
    "</parameter",
    "<tool_call",
    "</tool_call",
)
LEGACY_HASH_TOOL_MARKERS = ("##tool_call##", "##end_call##")
TEXTUAL_TOOL_MARKERS = QNML_TOOL_MARKERS + LEGACY_XML_TOOL_MARKERS + LEGACY_HASH_TOOL_MARKERS
_CONTROL_TOOL_NAMES = {
    "Agent",
    "AskUserQuestion",
    "CronCreate",
    "CronDelete",
    "CronList",
    "EnterPlanMode",
    "ExitPlanMode",
    "EnterWorktree",
    "ExitWorktree",
    "Monitor",
    "PushNotification",
    "ScheduleWakeup",
    "TaskCreate",
    "TaskDelete",
    "TaskGet",
    "TaskList",
    "TaskOutput",
    "TaskStop",
    "TaskUpdate",
}


def has_textual_tool_marker(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in TEXTUAL_TOOL_MARKERS)


def tool_directive_visible_text(directive: "RuntimeToolDirective", fallback_text: str = "") -> str:
    """Return safe visible text for non-tool responses, preferring filtered text blocks."""
    if directive.stop_reason == "tool_use":
        return ""
    text_parts: list[str] = []
    saw_text_block = False
    for block in directive.tool_blocks:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        saw_text_block = True
        text = block.get("text")
        if isinstance(text, str) and text:
            text_parts.append(text)
    if saw_text_block:
        return sanitize_visible_text("".join(text_parts))
    return sanitize_visible_text(fallback_text or "")


@dataclass(slots=True)
class RuntimeAttemptState:
    answer_text: str = ""
    reasoning_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    blocked_tool_names: list[str] = field(default_factory=list)
    finish_reason: str = "stop"
    empty_upstream_response: bool = False
    raw_events: list[dict[str, Any]] = field(default_factory=list)
    emitted_visible_output: bool = False
    stage_metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeExecutionResult:
    state: RuntimeAttemptState
    chat_id: str | None
    acc: Any | None


@dataclass(slots=True)
class RuntimeToolDirective:
    tool_blocks: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "end_turn"


@dataclass(slots=True)
class RuntimeRetryDirective:
    retry: bool
    next_prompt: str
    reason: str | None = None


@dataclass(slots=True)
class FinalDeliveryGap:
    required_markers: list[str] = field(default_factory=list)
    missing_artifact_markers: list[str] = field(default_factory=list)
    final_phrase: str = ""
    final_phrase_missing: bool = False
    target_paths: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RuntimeRetryContinuation:
    should_continue: bool
    next_prompt: str


@dataclass(slots=True)
class RuntimeRetryLoop:
    prompt: str
    max_attempts: int


@dataclass(slots=True)
class RuntimeAttemptPlan:
    loop: RuntimeRetryLoop
    prompt: str


@dataclass(slots=True)
class AnthropicStreamCompletionResult:
    chunks: list[str]


@dataclass(slots=True)
class AnthropicStreamSuccessResult:
    chunks: list[str]
    usage_delta: int


@dataclass(slots=True)
class RuntimeAttemptOutcome:
    execution: RuntimeExecutionResult
    continuation: RuntimeRetryContinuation


@dataclass(slots=True)
class RuntimeAttemptCursor:
    index: int
    number: int


TRAILING_IDLE_AFTER_TOOL_SECONDS = 2.0


__all__ = [
    "RuntimeAttemptState",
    "RuntimeExecutionResult",
    "RuntimeToolDirective",
    "RuntimeRetryDirective",
    "RuntimeRetryContinuation",
    "RuntimeRetryLoop",
    "RuntimeAttemptPlan",
    "AnthropicStreamCompletionResult",
    "AnthropicStreamSuccessResult",
    "RuntimeAttemptOutcome",
    "RuntimeAttemptCursor",
    "anthropic_stream_stop_reason",
    "anthropic_stream_usage_delta",
    "build_retry_loop",
    "build_tool_directive",
    "build_usage_delta_factory",
    "begin_runtime_attempt",
    "cleanup_runtime_resources",
    "collect_completion_run",
    "collect_completion_run_with_recovery",
    "continue_after_retry_directive",
    "evaluate_retry_directive",
    "extract_blocked_tool_names",
    "finalize_anthropic_stream_success",
    "complete_anthropic_stream_success",
    "has_recent_search_no_results",
    "has_recent_unchanged_read_result",
    "inject_assistant_message",
    "native_tool_calls_to_markup",
    "parse_tool_directive_once",
    "plan_runtime_attempts",
    "recent_same_tool_identity_count",
    "request_max_attempts",
    "retryable_usage_delta",
    "sanitize_visible_text",
    "should_force_finish_after_tool_use",
    "tool_identity",
    "tool_directive_visible_text",
]


def begin_runtime_attempt(attempt_index: int) -> RuntimeAttemptCursor:
    cursor = RuntimeAttemptCursor(index=attempt_index, number=attempt_index + 1)
    update_request_context(stream_attempt=cursor.number)
    return cursor


def should_force_finish_after_tool_use(stop_reason: str, trailing_idle_seconds: float, visible_output_after_tool: bool) -> bool:
    return stop_reason == "tool_use" and trailing_idle_seconds >= TRAILING_IDLE_AFTER_TOOL_SECONDS and not visible_output_after_tool


def extract_blocked_tool_names(text: str, allowed_tool_names: list[str] | None = None) -> list[str]:
    if not text:
        return []
    if "does not exist" not in text.lower():
        return []
    blocked = re.findall(r"Tool\s+([A-Za-z0-9_.:-]+)\s+does not exists?\.?", text)
    if not blocked:
        return []
    if not allowed_tool_names:
        return blocked
    return [normalize_tool_name(name, allowed_tool_names) for name in blocked]


def _recent_message_texts(messages: list[dict[str, Any]] | None, *, limit: int = 10) -> list[str]:
    texts: list[str] = []
    checked = 0
    for msg in reversed(messages or []):
        checked += 1
        content = msg.get("content", "")
        parts: list[str] = []
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        parts.append(part.get("text", ""))
                    elif part.get("type") == "tool_result":
                        inner = part.get("content", "")
                        if isinstance(inner, str):
                            parts.append(inner)
                        elif isinstance(inner, list):
                            for inner_part in inner:
                                if isinstance(inner_part, dict) and inner_part.get("type") == "text":
                                    parts.append(inner_part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
        merged = "\n".join(text for text in parts if text)
        if merged:
            texts.append(merged)
        if checked >= limit:
            break
    return texts


def has_recent_unchanged_read_result(messages: list[dict[str, Any]] | None, read_path: str | None = None) -> bool:
    if not messages:
        return False
    target = (read_path or "").strip()
    read_by_id: dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        parts = content if isinstance(content, list) else []
        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "tool_use" and part.get("name") == "Read":
                tool_id = str(part.get("id", ""))
                tool_input = part.get("input", {})
                if isinstance(tool_input, dict) and tool_id:
                    path = str(tool_input.get("file_path") or tool_input.get("path") or "").strip()
                    if path:
                        read_by_id[tool_id] = path
            elif part.get("type") == "tool_result":
                inner = part.get("content", "")
                if isinstance(inner, list):
                    text = "\n".join(str(p.get("text", "")) for p in inner if isinstance(p, dict))
                else:
                    text = str(inner or "")
                if "Unchanged since last read" not in text:
                    continue
                tool_use_id = str(part.get("tool_use_id", ""))
                previous_path = read_by_id.get(tool_use_id, "")
                if target:
                    if previous_path and previous_path == target:
                        return True
                    continue
                return True
    return False


def has_recent_search_no_results(messages: list[dict[str, Any]] | None) -> bool:
    for text in _recent_message_texts(messages):
        lowered = text.lower()
        if "websearch" not in lowered:
            continue
        if "did 0 searches" in lowered or '"results": []' in lowered or '"matches": []' in lowered:
            return True
    return False


def tool_identity(tool_name: str, tool_input: Any = None) -> str:
    try:
        if tool_name == "Read" and isinstance(tool_input, dict):
            return f"Read::{tool_input.get('file_path', '').strip()}"
        if tool_name == "read" and isinstance(tool_input, dict):
            return f"read::{tool_input.get('path', '').strip()}"
        return f"{tool_name}::{json.dumps(tool_input or {}, ensure_ascii=False, sort_keys=True)}"
    except Exception:
        return tool_name or ""


def _tool_input_path(tool_input: Any) -> str:
    if not isinstance(tool_input, dict):
        return ""
    value = tool_input.get("file_path") or tool_input.get("path") or ""
    return str(value).strip() if value is not None else ""


_WRITE_TOOL_NAMES = {"Write", "Edit", "NotebookEdit", "fs_put_file", "fs_patch_file", "notebook_patch", "WriteX", "EditX"}
_WRITE_PATH_KEYS = ("file_path", "path", "target_file", "filename", "file", "notebook_path")
_WRITE_SUCCESS_RE = re.compile(
    r"file\s+(?:created|updated|written|saved|modified)\s+successfully|"
    r"(?:created|updated|written|saved|modified)\s+successfully|"
    r"successfully\s+(?:created|updated|wrote|written|saved|modified|applied)|"
    r"(?:created|updated|wrote|written|saved|modified)\s+(?:the\s+)?file|"
    r"file\s+has\s+been\s+(?:created|updated|written|saved|modified)|"
    r"has\s+been\s+(?:created|updated|written|saved|modified)|"
    r"changes?\s+(?:applied|saved)\s+successfully|"
    r"wrote\s+\d+\s+bytes?|"
    r"\u5199\u5165\u6210\u529f|\u66f4\u65b0\u6210\u529f|\u4fdd\u5b58\u6210\u529f|\u521b\u5efa\u6210\u529f|\u4fee\u6539\u6210\u529f|"
    r"\u5df2\u5199\u5165|\u5df2\u66f4\u65b0|\u5df2\u4fdd\u5b58|\u5df2\u521b\u5efa|\u5df2\u4fee\u6539",
    re.IGNORECASE,
)
_WRITE_ERROR_RE = re.compile(
    r"\berror\b|\bfailed\b|\bfailure\b|exception|traceback|invalid|not found|"
    r"incomplete|partial|truncated|timeout|timed out|cancel(?:led)?|aborted|interrupted|"
    r"no changes? (?:made|applied)|unchanged|"
    r"\u5931\u8d25|\u9519\u8bef|\u5f02\u5e38|\u672a\u5b8c\u6210|\u4e0d\u5b8c\u6574|\u90e8\u5206|\u622a\u65ad|\u8d85\u65f6|\u53d6\u6d88|\u4e2d\u65ad|\u672a\u66f4\u6539|\u65e0\u66f4\u6539",
    re.IGNORECASE,
)


def _normalized_guard_path(path: str) -> str:
    return re.sub(r"/+", "/", (path or "").replace("\\", "/").strip()).lower()


def _write_target_path(tool_input: Any) -> str:
    if not isinstance(tool_input, dict):
        return ""
    for key in _WRITE_PATH_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _stable_write_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _stable_write_payload(value[key])
            for key in sorted(value)
            if key not in _WRITE_PATH_KEYS
        }
    if isinstance(value, list):
        return [_stable_write_payload(item) for item in value]
    return value


def _write_payload_signature(tool_input: Any) -> str:
    if not isinstance(tool_input, dict):
        return ""
    payload = _stable_write_payload(tool_input)
    if payload in ({}, [], "", None):
        return ""
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return repr(payload)


def _is_write_like_tool(tool_name: str) -> bool:
    return tool_name in _WRITE_TOOL_NAMES


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                nested = part.get("content") if part.get("type") in {"tool_result", "function_call_output"} else part.get("text")
                if nested is not None:
                    parts.append(_content_text(nested))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(text for text in parts if text)
    if content is None:
        return ""
    return str(content)


def _tool_result_success(part_or_message: dict[str, Any]) -> bool:
    if part_or_message.get("is_error") is True:
        return False
    text = _content_text(part_or_message.get("content", ""))
    if _WRITE_ERROR_RE.search(text):
        return False
    return bool(_WRITE_SUCCESS_RE.search(text))


def _iter_assistant_tool_uses(message: dict[str, Any]) -> list[dict[str, Any]]:
    uses: list[dict[str, Any]] = []
    content = message.get("content", [])
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "tool_use":
                uses.append(part)
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            fn = tool_call.get("function", {}) if isinstance(tool_call.get("function"), dict) else {}
            raw_args = fn.get("arguments", "{}")
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args else raw_args
            except (json.JSONDecodeError, ValueError):
                parsed_args = {}
            uses.append({
                "type": "tool_use",
                "id": tool_call.get("id"),
                "name": fn.get("name", ""),
                "input": parsed_args if isinstance(parsed_args, dict) else {},
            })
    return uses


def _latest_successful_write_result_identity(messages: list[dict[str, Any]] | None) -> tuple[str, str]:
    """Return path and write payload when the latest client input is a successful write result."""
    if not messages:
        return "", ""
    write_identity_by_tool_id: dict[str, tuple[str, str]] = {}
    latest_successful_path = ""
    latest_successful_signature = ""

    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            tool_uses = _iter_assistant_tool_uses(message)
            if not tool_uses:
                latest_successful_path = ""
                latest_successful_signature = ""
                continue
            for tool_use in tool_uses:
                tool_id = str(tool_use.get("id") or "")
                tool_name = normalize_tool_name(str(tool_use.get("name", "")), _WRITE_TOOL_NAMES)
                tool_input = tool_use.get("input", {})
                target_path = _write_target_path(tool_input)
                signature = _write_payload_signature(tool_input)
                if tool_id and _is_write_like_tool(tool_name) and target_path and signature:
                    write_identity_by_tool_id[tool_id] = (target_path, signature)
            continue

        if message.get("role") == "tool":
            tool_call_id = str(message.get("tool_call_id") or "")
            if tool_call_id in write_identity_by_tool_id:
                target_path, signature = write_identity_by_tool_id[tool_call_id]
                if _tool_result_success(message):
                    latest_successful_path, latest_successful_signature = target_path, signature
            continue

        if message.get("role") != "user":
            continue
        content = message.get("content", [])
        if isinstance(content, list):
            saw_tool_result = False
            for part in content:
                if not isinstance(part, dict) or part.get("type") not in {"tool_result", "function_call_output"}:
                    continue
                saw_tool_result = True
                tool_use_id = str(part.get("tool_use_id") or part.get("call_id") or part.get("id") or "")
                if tool_use_id in write_identity_by_tool_id:
                    target_path, signature = write_identity_by_tool_id[tool_use_id]
                    if _tool_result_success(part):
                        latest_successful_path, latest_successful_signature = target_path, signature
            if not saw_tool_result:
                latest_successful_path = ""
                latest_successful_signature = ""
        elif isinstance(content, str) and content.strip():
            latest_successful_path = ""
            latest_successful_signature = ""

    if latest_successful_path and latest_successful_signature:
        return latest_successful_path, latest_successful_signature
    return "", ""


def _completed_write_visible_text(tool_call: dict[str, Any]) -> str:
    tool_input = tool_call.get("input", {})
    if isinstance(tool_input, dict):
        for key in ("content", "new_string", "text"):
            value = tool_input.get(key)
            if isinstance(value, str) and value.strip():
                return value
    target_path = _write_target_path(tool_input)
    return f"已完成写入：{target_path}" if target_path else "已完成。"


def _add_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _blocked_tools_notice(request: StandardRequest) -> str:
    blocked = [name for name in getattr(request, "retry_blocked_tools", []) if name]
    if not blocked:
        return ""
    return (
        "[RETRY TOOL BLOCKLIST - MUST OBEY]\n"
        f"Do NOT call these tools in the next output: {', '.join(blocked)}.\n"
        "Choose a non-control project tool such as Read/Grep/Glob/Write/Edit/Bash only when it directly advances the current task.\n"
        "[/RETRY TOOL BLOCKLIST]"
    )


def _read_blocklist_notice(request: StandardRequest) -> str:
    paths = [path for path in getattr(request, "retry_read_blocklist", []) if path]
    if not paths:
        return ""
    return (
        "[READ BLOCKLIST - MUST OBEY]\n"
        "These files were already read or returned 'Unchanged since last read'. Do NOT call Read for them again:\n"
        + "\n".join(f"- {path}" for path in paths[-8:])
        + "\nUse the existing tool result, read a different relevant file, write/edit the requested output, or finish.\n"
        "[/READ BLOCKLIST]"
    )


def _retry_guard_notice(request: StandardRequest) -> str:
    notices = [
        _blocked_tools_notice(request),
        _read_blocklist_notice(request),
        build_workspace_final_reminder(getattr(request, "workspace_root", None)),
    ]
    return "\n\n".join(notice for notice in notices if notice)


def _inject_retry_guard(prompt: str, request: StandardRequest, message: str) -> str:
    guard = _retry_guard_notice(request)
    combined = "\n\n".join(part for part in (guard, message) if part)
    return inject_assistant_message(prompt, combined)


def recent_same_tool_identity_count(messages: list[dict[str, Any]] | None, tool_name: str, tool_input: Any = None) -> int:
    target = tool_identity(tool_name, tool_input)
    count = 0
    started = False
    for msg in reversed(messages or []):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            if started:
                break
            continue
        tools = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name")]
        if not tools:
            if started:
                break
            continue
        started = True
        if len(tools) == 1 and tool_identity(tools[0].get("name", ""), tools[0].get("input", {})) == target:
            count += 1
            continue
        break
    return count


def has_recent_openai_same_tool_call(history_messages: list[dict[str, Any]] | None, tool_name: str, tool_input: Any = None) -> bool:
    target = tool_identity(tool_name, tool_input)
    for msg in reversed(history_messages or []):
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            continue
        if len(tool_calls) != 1:
            return False
        fn = tool_calls[0].get("function", {}) if isinstance(tool_calls[0], dict) else {}
        name = fn.get("name", "")
        raw_args = fn.get("arguments", "{}")
        try:
            parsed_args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args else raw_args
        except (json.JSONDecodeError, ValueError):
            parsed_args = {"raw": raw_args}
        return tool_identity(name, parsed_args) == target
    return False
def _extract_reasoning_text(evt: dict[str, Any]) -> str:
    if not isinstance(evt, dict):
        return ""

    content = evt.get("content", "")
    if isinstance(content, str) and content:
        return content

    extra = evt.get("extra", {})
    if not isinstance(extra, dict):
        return ""

    summary_thought = extra.get("summary_thought", {})
    if not isinstance(summary_thought, dict):
        return ""

    summary_content = summary_thought.get("content", [])
    if isinstance(summary_content, list):
        return "".join(part for part in summary_content if isinstance(part, str))
    if isinstance(summary_content, str):
        return summary_content
    return ""


def has_invalid_textual_tool_contract(answer_text: str) -> bool:
    """Detect malformed legacy textual tool contracts that should be retried.

    QNML/legacy ``<tool_calls><invoke ...>`` parse failures are handled by the
    normal parser path below. This function only preserves the older JSON-marker
    guard where malformed JSON or stringified ``input`` was a common failure mode.
    """
    if not answer_text:
        return False
    lowered = answer_text.lower()
    if "##tool_call##" not in lowered and "<tool_call" not in lowered:
        return False
    compact = answer_text.strip()
    tc_m = re.search(r'##TOOL_CALL##\s*(.*?)\s*##END_CALL##', compact, re.DOTALL | re.IGNORECASE)
    if tc_m:
        try:
            obj = json.loads(tc_m.group(1))
        except (json.JSONDecodeError, ValueError):
            return True
        tool_input = obj.get("input", obj.get("args", obj.get("arguments", obj.get("parameters", {}))))
        return isinstance(tool_input, str)
    xml_m = re.search(r'<tool_call\b[^>]*>\s*(.*?)\s*</tool_call>', compact, re.DOTALL | re.IGNORECASE)
    if xml_m:
        try:
            obj = json.loads(xml_m.group(1))
        except (json.JSONDecodeError, ValueError):
            return True
        tool_input = obj.get("input", obj.get("args", obj.get("arguments", obj.get("parameters", {}))))
        return isinstance(tool_input, str)
    return False


def _attempted_tool_for_repair(answer_text: str, request: StandardRequest) -> str:
    return (
        tool_parser.extract_attempted_tool_name(answer_text, request.tool_names)
        or (request.tool_names[0] if request.tool_names else "tool")
    )


def _latest_user_text(history_messages: list[dict[str, Any]] | None) -> str:
    if not history_messages:
        return ""
    for message in reversed(history_messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") in ("text", "input_text")
            )
        else:
            text = str(content or "")
        if text.strip():
            return text
    return ""


_FINAL_CHECKPOINT_RE = re.compile(r"\[[A-Z0-9_-]*CHECKPOINT[A-Z0-9_-]*\]")
_FINAL_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9_:-]{5,}\b")
_FINAL_PHRASE_TOKEN_PATTERN = r"([A-Z0-9][A-Z0-9_:-]{5,})"
_DELIVERY_FILE_RE = re.compile(
    r"[A-Za-z]:[\\/][^\s`\"'<>|]+?\.(?:md|txt|json|csv|html|log)"
    r"|(?<![\w.-])[\w./\\-]+?\.(?:md|txt|json|csv|html|log)",
    re.IGNORECASE,
)
_FINAL_PHRASE_PATTERNS = (
    re.compile(
        r"(?i:(?:last|final)\s+(?:line|sentence|reply|response).{0,80}?"
        r"(?:must|should)\s+(?:be|end\s+(?:with|exactly\s+with)))\s*[:\uff1a]?\s*"
        r"[`\"']?" + _FINAL_PHRASE_TOKEN_PATTERN + r"[`\"']?",
        re.DOTALL,
    ),
    re.compile(
        r"(?:\u6700\u540e\u4e00(?:\u53e5|\u884c)|\u7ed3\u5c3e|\u6700\u7ec8\u56de\u7b54).{0,60}?"
        r"(?:\u5fc5\u987b|\u8981|\u5e94\u8be5).{0,30}?"
        r"(?:(?:\u662f|\u4e3a|\u4ee5|\u7528)|(?i:be|with))?\s*[:\uff1a]?\s*"
        r"[`\"']?" + _FINAL_PHRASE_TOKEN_PATTERN + r"[`\"']?",
        re.DOTALL,
    ),
)


def _delivery_requirement_text(history_messages: list[dict[str, Any]] | None, current_prompt: str) -> str:
    return _latest_user_text(history_messages) or current_prompt or ""


def _normalize_final_phrase_candidate(candidate: str) -> str:
    value = (candidate or "").strip().strip("`\"'")
    value = value.rstrip(".,;:)\uff0c\u3002\uff1b\uff1a")
    if not value:
        return ""
    if _DELIVERY_FILE_RE.fullmatch(value):
        return ""
    if re.search(r"\.(?:md|txt|json|csv|html|log)\Z", value, re.IGNORECASE):
        return ""
    if not _FINAL_TOKEN_RE.fullmatch(value):
        return ""
    return value


def _extract_final_phrase(requirement_text: str) -> str:
    if not requirement_text:
        return ""
    for pattern in _FINAL_PHRASE_PATTERNS:
        match = pattern.search(requirement_text)
        if match:
            candidate = _normalize_final_phrase_candidate(match.group(1))
            if candidate:
                return candidate
    final_context = re.search(
        r"(?:last|final|\u6700\u540e|\u7ed3\u5c3e|\u6700\u7ec8).{0,120}",
        requirement_text,
        re.IGNORECASE | re.DOTALL,
    )
    if final_context:
        tokens = _FINAL_TOKEN_RE.findall(final_context.group(0))
        for token in reversed(tokens):
            candidate = _normalize_final_phrase_candidate(token)
            if candidate:
                return candidate
    return ""


def _extract_required_markers(requirement_text: str) -> list[str]:
    markers: list[str] = []
    for marker in _FINAL_CHECKPOINT_RE.findall(requirement_text or ""):
        _add_unique(markers, marker)
    return markers


def _extract_delivery_target_paths(requirement_text: str) -> list[str]:
    if not requirement_text:
        return []
    all_paths: list[str] = []
    report_paths: list[str] = []
    for match in _DELIVERY_FILE_RE.finditer(requirement_text):
        value = match.group(0).strip().rstrip(".,;:)\uff0c\u3002\uff1b\uff1a")
        lowered = value.lower()
        _add_unique(all_paths, value)
        context = requirement_text[max(0, match.start() - 80):match.start()]
        if (
            "report" in lowered
            or "\u62a5\u544a" in context
            or "\u8f93\u51fa\u7684\u6587\u4ef6" in context
            or "\u62a5\u544a\u6587\u4ef6" in context
        ):
            _add_unique(report_paths, value)
    return report_paths or all_paths


def _path_basename(path: str) -> str:
    return re.split(r"[\\/]+", path.strip())[-1].lower()


def _path_matches_targets(path: str, target_paths: list[str]) -> bool:
    if not target_paths:
        return True
    normalized_path = _normalized_guard_path(path)
    basename = _path_basename(path)
    for target in target_paths:
        normalized_target = _normalized_guard_path(target)
        if (
            normalized_path == normalized_target
            or normalized_path.endswith("/" + normalized_target)
            or normalized_target.endswith("/" + normalized_path)
            or (basename and basename == _path_basename(target))
        ):
            return True
    return False


def _write_body_text(tool_input: Any) -> str:
    if not isinstance(tool_input, dict):
        return ""
    parts: list[str] = []
    for key in ("content", "new_string", "text"):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            parts.append(value)
    return "\n".join(parts)


def _delivered_artifact_text(history_messages: list[dict[str, Any]] | None, target_paths: list[str]) -> str:
    if not history_messages:
        return ""
    tool_path_by_id: dict[str, str] = {}
    chunks: list[str] = []
    for message in history_messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            for tool_use in _iter_assistant_tool_uses(message):
                tool_id = str(tool_use.get("id") or "")
                tool_name = normalize_tool_name(str(tool_use.get("name", "")), _WRITE_TOOL_NAMES)
                tool_input = tool_use.get("input", {})
                target_path = _write_target_path(tool_input) or _tool_input_path(tool_input)
                if tool_id and target_path:
                    tool_path_by_id[tool_id] = target_path
                if _is_write_like_tool(tool_name) and _path_matches_targets(target_path, target_paths):
                    body = _write_body_text(tool_input)
                    if body:
                        chunks.append(body)
            continue

        if message.get("role") == "tool":
            tool_call_id = str(message.get("tool_call_id") or "")
            if _path_matches_targets(tool_path_by_id.get(tool_call_id, ""), target_paths):
                text = _content_text(message.get("content", ""))
                if text:
                    chunks.append(text)
            continue

        if message.get("role") != "user":
            continue
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in {"tool_result", "function_call_output"}:
                continue
            tool_id = str(part.get("tool_use_id") or part.get("call_id") or part.get("id") or "")
            if _path_matches_targets(tool_path_by_id.get(tool_id, ""), target_paths):
                text = _content_text(part.get("content", ""))
                if text:
                    chunks.append(text)
    return "\n".join(chunks)


def _final_delivery_gap(
    *,
    current_prompt: str,
    history_messages: list[dict[str, Any]] | None,
    state: RuntimeAttemptState,
    has_tools: bool,
) -> FinalDeliveryGap | None:
    requirement_text = _delivery_requirement_text(history_messages, current_prompt)
    required_markers = _extract_required_markers(requirement_text)
    final_phrase = _extract_final_phrase(requirement_text)
    if not required_markers and not final_phrase:
        return None

    target_paths = _extract_delivery_target_paths(requirement_text)
    artifact_text = _delivered_artifact_text(history_messages, target_paths)
    visible_answer = state.answer_text or ""
    marker_source = artifact_text if target_paths and has_tools else "\n".join(part for part in (artifact_text, visible_answer) if part)
    missing_markers = [
        marker for marker in required_markers
        if marker not in marker_source
    ]
    final_phrase_missing = bool(final_phrase and not visible_answer.rstrip().endswith(final_phrase))
    if not missing_markers and not final_phrase_missing:
        return None
    return FinalDeliveryGap(
        required_markers=required_markers,
        missing_artifact_markers=missing_markers,
        final_phrase=final_phrase,
        final_phrase_missing=final_phrase_missing,
        target_paths=target_paths,
    )


def _final_delivery_retry_message(gap: FinalDeliveryGap) -> str:
    lines = [
        "[MANDATORY FINAL DELIVERY CHECK FAILED]",
        "The previous response tried to finish, but explicit user delivery requirements are still missing.",
    ]
    if gap.target_paths:
        lines.append("Target report/file path(s): " + ", ".join(gap.target_paths[-4:]))
    if gap.required_markers:
        lines.append("Required marker(s): " + ", ".join(gap.required_markers))
    if gap.missing_artifact_markers:
        lines.append("Missing from delivered artifact/output: " + ", ".join(gap.missing_artifact_markers))
        lines.append("Use Write/Edit only if the report/file needs these missing markers. Do not repeat an identical Write/Edit payload.")
    if gap.final_phrase_missing and gap.final_phrase:
        lines.append(f"The final visible response must end exactly with: {gap.final_phrase}")
    lines.append("Do not restart the task. Complete only the missing delivery items, then finish.")
    lines.append("[/MANDATORY FINAL DELIVERY CHECK FAILED]")
    return "\n".join(lines)


def _delivery_guard_prompt(request: StandardRequest, current_prompt: str | None = None) -> str:
    return current_prompt or getattr(request, "prompt", "") or getattr(request, "full_prompt", "") or ""


def _complete_final_delivery_if_safe(
    *,
    request: StandardRequest,
    state: RuntimeAttemptState,
    history_messages: list[dict[str, Any]] | None,
    current_prompt: str | None = None,
    source: str,
) -> FinalDeliveryGap | None:
    gap = _final_delivery_gap(
        current_prompt=_delivery_guard_prompt(request, current_prompt),
        history_messages=history_messages,
        state=state,
        has_tools=bool(request.tools),
    )
    if gap is None:
        return None
    if gap.missing_artifact_markers:
        log.warning(
            "[FinalDeliveryGuard] source=%s incomplete missing_markers=%s final_phrase_missing=%s targets=%s",
            source,
            gap.missing_artifact_markers,
            gap.final_phrase_missing,
            gap.target_paths[-4:],
        )
        return gap
    if gap.final_phrase_missing and gap.final_phrase:
        # Never mutate user-visible text here. Long-running Claude Code sessions can keep
        # old test prompts in history, and appending their final sentinel leaks into later replies.
        log.info(
            "[FinalDeliveryGuard] source=%s ignored missing final phrase phrase=%s targets=%s",
            source,
            gap.final_phrase,
            gap.target_paths[-4:],
        )
        return None
    return gap



def _user_negated_control_tool(tool_name: str, latest: str) -> bool:
    if not latest:
        return False
    latest_l = latest.lower()
    name = re.escape(tool_name.lower())
    compact_tool = re.sub(r"[^a-z0-9]+", "", tool_name.lower())

    english_patterns = [
        rf"(?:do\s+not|don['\u2019]?t|dont|never|without|no)\s+(?:(?:use|call|create|start|run|invoke|trigger)\s+)?{name}\b",
        rf"\b{name}\b\s+(?:is\s+)?(?:forbidden|not\s+allowed|disabled|disallowed)",
    ]
    if compact_tool and compact_tool != tool_name.lower():
        english_patterns.extend([
            rf"(?:do\s+not|don['\u2019]?t|dont|never|without|no)\s+(?:(?:use|call|create|start|run|invoke|trigger)\s+)?{re.escape(compact_tool)}\b",
            rf"\b{re.escape(compact_tool)}\b\s+(?:is\s+)?(?:forbidden|not\s+allowed|disabled|disallowed)",
        ])
    if any(re.search(pattern, latest_l, re.IGNORECASE) for pattern in english_patterns):
        return True

    # Keep Chinese terms as Unicode escapes so the source survives non-UTF8 rewrites.
    chinese_negatives = (
        "\u4e0d\u8981",      # 不要
        "\u522b",            # 别
        "\u7981\u6b62",      # 禁止
        "\u4e0d\u5f97",      # 不得
        "\u4e0d\u51c6",      # 不准
        "\u4e0d\u7528",      # 不用
        "\u4e0d\u4f7f\u7528",  # 不使用
        "\u65e0\u9700",      # 无需
        "\u522b\u7528",      # 别用
        "\u4e0d\u8981\u7528",  # 不要用
        "\u4e0d\u8981\u4f7f\u7528",  # 不要使用
        "\u4e0d\u8981\u8c03\u7528",  # 不要调用
        "\u4e0d\u8981\u521b\u5efa",  # 不要创建
        "\u4e0d\u8981\u6267\u884c",  # 不要执行
        "\u4e0d\u8981\u89e6\u53d1",  # 不要触发
    )
    chinese_suffixes = (
        "\u7981\u7528",      # 禁用
        "\u7981\u6b62",      # 禁止
        "\u4e0d\u5141\u8bb8",  # 不允许
        "\u4e0d\u8981\u7528",  # 不要用
        "\u4e0d\u8981\u8c03\u7528",  # 不要调用
    )
    negative_prefix = "|".join(re.escape(term) for term in chinese_negatives)
    negative_suffix = "|".join(re.escape(term) for term in chinese_suffixes)
    target_names = [name]
    if compact_tool and compact_tool != tool_name.lower():
        target_names.append(re.escape(compact_tool))
    for target in target_names:
        if re.search(rf"(?:{negative_prefix}).{{0,32}}{target}\b", latest_l, re.IGNORECASE):
            return True
        if re.search(rf"\b{target}\b.{{0,24}}(?:{negative_suffix})", latest_l, re.IGNORECASE):
            return True
    return False


def _user_requested_control_tool(tool_name: str, history_messages: list[dict[str, Any]] | None) -> bool:
    latest = _latest_user_text(history_messages).lower()
    if not latest:
        return False
    if _user_negated_control_tool(tool_name, latest):
        return False
    compact_tool = re.sub(r"[^a-z0-9]+", "", tool_name.lower())
    compact_latest = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", latest)
    if compact_tool and compact_tool in compact_latest:
        return True
    control_keywords = {
        "Agent": ("agent", "subtask", "delegate", "delegation", "background agent", "\u4ee3\u7406", "\u5b50\u4efb\u52a1"),
        "AskUserQuestion": ("askuserquestion", "ask user", "ask me", "question tool", "\u8be2\u95ee", "\u63d0\u95ee"),
        "CronCreate": ("cron", "schedule", "scheduled task", "automation", "\u5b9a\u65f6", "\u8ba1\u5212\u4efb\u52a1"),
        "CronDelete": ("cron", "schedule", "scheduled task", "automation", "\u5b9a\u65f6", "\u8ba1\u5212\u4efb\u52a1"),
        "CronList": ("cron", "schedule", "scheduled task", "automation", "\u5b9a\u65f6", "\u8ba1\u5212\u4efb\u52a1"),
        "TaskCreate": ("taskcreate", "task create", "todo tool", "task management", "\u4efb\u52a1\u7ba1\u7406", "\u5f85\u529e"),
        "TaskUpdate": ("taskupdate", "task update", "todo tool", "task management", "\u4efb\u52a1\u7ba1\u7406", "\u5f85\u529e"),
        "TaskList": ("tasklist", "task list tool", "todo tool", "task management", "\u4efb\u52a1\u7ba1\u7406", "\u5f85\u529e"),
        "TaskGet": ("taskget", "task get", "todo tool", "task management", "\u4efb\u52a1\u7ba1\u7406", "\u5f85\u529e"),
        "TaskOutput": ("taskoutput", "task output", "agent output", "subtask output", "\u5b50\u4efb\u52a1\u8f93\u51fa"),
        "TaskStop": ("taskstop", "task stop", "stop task", "task management", "\u4efb\u52a1\u7ba1\u7406"),
    }
    return any(keyword in latest for keyword in control_keywords.get(tool_name, ()))


def should_retry_textual_tool_contract(answer_text: str) -> bool:
    return has_textual_tool_marker(answer_text)


def _fresh_textual_tool_retry_instruction(tool_name: str) -> str:
    canonical = normalize_tool_name(tool_name or "tool", {"Write", "Read", "Bash", "PowerShell", tool_name or "tool"})
    base = (
        "[MANDATORY]: Do not continue, quote, or repair partial QNML/tool markup from the previous response. "
        "Start over with one fresh, complete tool call only, or continue with the next direct verification step if no more tool input is needed."
    )
    if canonical == "Write":
        return base + " For Write, include both a non-empty file_path and the full content in the same tool call."
    if canonical == "Read":
        return base + " For Read, include one non-empty file_path and do not reread a file that already returned unchanged."
    if canonical in {"Bash", "PowerShell"}:
        return base + " For shell tools, include a complete command and keep it scoped to the active workspace."
    return base

def native_tool_calls_to_markup(tool_calls: list[dict[str, Any]]) -> str:
    """Render native upstream tool calls back into the prompt-visible QNML form."""
    return render_qnml_tool_calls([
        {"name": tool_call.get("name", ""), "input": tool_call.get("input", {})}
        for tool_call in tool_calls
        if isinstance(tool_call, dict)
    ])



def _is_retry_blocked_tool(tool_name: str, request: StandardRequest) -> bool:
    blocked = {name for name in getattr(request, "retry_blocked_tools", []) if name}
    return tool_name in blocked


def _is_disallowed_control_tool(
    tool_name: str,
    request: StandardRequest,
    history_messages: list[dict[str, Any]] | None = None,
) -> bool:
    if not tool_name:
        return False
    if _is_retry_blocked_tool(tool_name, request):
        return True
    if tool_name not in _CONTROL_TOOL_NAMES:
        return False
    if history_messages is None:
        return False
    return not _user_requested_control_tool(tool_name, history_messages)


def _is_blocked_read_call(tool_name: str, tool_input: Any, request: StandardRequest) -> bool:
    if tool_name != "Read":
        return False
    read_path = _tool_input_path(tool_input)
    if not read_path:
        return False
    return read_path in {path for path in getattr(request, "retry_read_blocklist", []) if path}


def _tool_call_block_reason(
    tool_call: dict[str, Any],
    request: StandardRequest,
    history_messages: list[dict[str, Any]] | None = None,
    *,
    include_completed_write_guard: bool = False,
) -> str | None:
    raw_name = str(tool_call.get("name", ""))
    tool_name = normalize_tool_name(raw_name, request.tool_names)
    tool_input = tool_call.get("input", {})
    if _is_disallowed_control_tool(tool_name, request, history_messages):
        return f"disallowed_control_tool:{tool_name}"
    if _is_blocked_read_call(tool_name, tool_input, request):
        return f"blocked_read:{_tool_input_path(tool_input)}"
    if include_completed_write_guard and _is_write_like_tool(tool_name):
        latest_completed_path, latest_completed_signature = _latest_successful_write_result_identity(history_messages)
        target_path = _write_target_path(tool_input)
        current_signature = _write_payload_signature(tool_input)
        if (
            latest_completed_path
            and latest_completed_signature
            and target_path
            and current_signature
            and _normalized_guard_path(latest_completed_path) == _normalized_guard_path(target_path)
            and latest_completed_signature == current_signature
        ):
            return f"completed_write:{target_path}"
    return None

def _filter_invalid_native_tool_calls(
    tool_calls: list[dict[str, Any]],
    request: StandardRequest,
    history_messages: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not tool_calls:
        return [], []
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            invalid.append(tool_call)
            continue
        blocks, stop_reason = tool_parser.parse_tool_calls_silent(
            native_tool_calls_to_markup([tool_call]),
            request.tools,
        )
        parsed = next((block for block in blocks if isinstance(block, dict) and block.get("type") == "tool_use"), None)
        if stop_reason == "tool_use" and parsed:
            fixed = dict(tool_call)
            fixed["name"] = parsed.get("name", fixed.get("name"))
            fixed["input"] = parsed.get("input", fixed.get("input", {}))
            reason = _tool_call_block_reason(fixed, request, history_messages)
            if reason:
                log.warning("[ToolGuard] rejected collected tool call reason=%s name=%s", reason, fixed.get("name", "-"))
                invalid.append(fixed)
                continue
            valid.append(fixed)
        else:
            invalid.append(tool_call)
    return valid, invalid


def _tool_input_preview(input_data: Any, *, limit: int = 260) -> str:
    try:
        raw = json.dumps(input_data if input_data is not None else {}, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        raw = repr(input_data)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw[:limit] + ("...[truncated]" if len(raw) > limit else "")


def _trace_text_fingerprint(label: str, text: str, *, stage: str, chat_id: str | None = None) -> None:
    if not settings.TRACE_RESPONSE_FINGERPRINTS:
        return
    text = text or ""
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    tail_chars = max(0, int(getattr(settings, "TRACE_RESPONSE_TAIL_CHARS", 160) or 0))
    tail = text[-tail_chars:] if tail_chars else ""
    log.info(
        "[Trace] %s stage=%s chat_id=%s chars=%s sha256=%s tail=%r",
        label,
        stage,
        chat_id or "-",
        len(text),
        digest,
        tail,
    )


def _trace_tool_call_inputs(stage: str, tool_calls: list[dict[str, Any]]) -> None:
    if not settings.TRACE_RESPONSE_FINGERPRINTS:
        return
    for idx, tool_call in enumerate(tool_calls, start=1):
        if not isinstance(tool_call, dict):
            continue
        try:
            raw_input = json.dumps(tool_call.get("input", {}), ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            raw_input = repr(tool_call.get("input", {}))
        _trace_text_fingerprint(
            f"tool_input index={idx} id={tool_call.get('id', '-')} name={tool_call.get('name', '-')}",
            raw_input,
            stage=stage,
        )
        input_data = tool_call.get("input", {})
        if isinstance(input_data, dict):
            for field_name in ("content", "new_string", "old_string", "command"):
                field_value = input_data.get(field_name)
                if isinstance(field_value, str):
                    _trace_text_fingerprint(
                        f"tool_input_field index={idx} id={tool_call.get('id', '-')} name={tool_call.get('name', '-')} field={field_name}",
                        field_value,
                        stage=stage,
                    )


def _log_tool_calls(stage: str, tool_calls: list[dict[str, Any]]) -> None:
    for idx, tool_call in enumerate(tool_calls, start=1):
        if not isinstance(tool_call, dict):
            log.info("[ToolCall] stage=%s index=%s invalid_block=%r", stage, idx, tool_call)
            continue
        log.info(
            "[ToolCall] stage=%s index=%s id=%s name=%s input=%s",
            stage,
            idx,
            tool_call.get("id", "-"),
            tool_call.get("name", "-"),
            _tool_input_preview(tool_call.get("input", {})),
        )
    _trace_tool_call_inputs(stage, tool_calls)


async def run_runtime_attempt(
    *,
    client,
    request: StandardRequest,
    current_prompt: str,
    history_messages: list[dict[str, Any]] | None,
    attempt_index: int,
    max_attempts: int,
    allow_after_visible_output: bool = False,
    capture_events: bool = True,
    on_delta: Callable[[dict[str, Any], str | None, list[dict[str, Any]] | None], Awaitable[None]] | None = None,
) -> RuntimeAttemptOutcome:
    attempt_cursor = begin_runtime_attempt(attempt_index)
    execution = await collect_completion_run(
        client,
        request,
        current_prompt,
        capture_events=capture_events,
        on_delta=on_delta,
        history_messages=history_messages,
    )
    retry = evaluate_retry_directive(
        request=request,
        current_prompt=current_prompt,
        history_messages=history_messages,
        attempt_index=attempt_cursor.index,
        max_attempts=max_attempts,
        state=execution.state,
        allow_after_visible_output=allow_after_visible_output,
    )
    if execution.state.empty_upstream_response:
        request.skip_prewarmed_chat_ids = True
        if getattr(request, "persistent_session", False) and getattr(request, "upstream_chat_id", None):
            request.session_chat_invalidated = True
            request.upstream_chat_id = None
            request.prompt = request.full_prompt or request.prompt
    preserve_chat = bool(getattr(request, 'persistent_session', False)) and not execution.state.empty_upstream_response
    continuation = await continue_after_retry_directive(
        client=client,
        execution=execution,
        retry=retry,
        preserve_chat=preserve_chat,
    )
    return RuntimeAttemptOutcome(execution=execution, continuation=continuation)


async def collect_completion_run(
    client,
    request: StandardRequest,
    prompt: str,
    *,
    capture_events: bool = True,
    on_delta: Callable[[dict[str, Any], str | None, list[dict[str, Any]] | None], Awaitable[None]] | None = None,
    history_messages: list[dict[str, Any]] | None = None,
) -> RuntimeExecutionResult:
    chat_id = None
    acc = None
    answer_fragments: list[str] = []
    reasoning_fragments: list[str] = []
    native_tool_calls: list[dict[str, Any]] = []
    rejected_tool_calls: list[dict[str, Any]] = []
    tool_state = StreamingToolCallState()
    emitted_visible_output = False
    first_event_marked = False
    raw_events: list[dict[str, Any]] = []
    metrics = StreamMetrics()
    last_thinking_summary_text = ""

    # 鍒濆鍖?Tool Sieve 鐢ㄤ簬瀹炴椂妫€娴?
    tool_sieve = None
    if request.tools:
        tool_sieve = tool_parser.ToolSieve(request.tool_names)
        log.info("[Collect] tool filter enabled: tools=%s", request.tool_names)

    async def cleanup_empty_upstream_state() -> None:
        if acc is None:
            return
        token = getattr(acc, "token", None)
        pool = getattr(client, "executor", None) and getattr(client.executor, "chat_id_pool", None)
        if chat_id and token:
            delete_fn = getattr(client, "delete_chat_reliable", None)
            if delete_fn is not None:
                await delete_fn(token, chat_id, source="empty_upstream_response")
            else:
                try:
                    await client.delete_chat(token, chat_id)
                except Exception as exc:
                    log.warning("[Collect] delete empty chat failed chat_id=%s error=%s", chat_id, exc)
        if pool is not None:
            try:
                flushed = await pool.flush_account(acc.email)
                log.warning(
                    "[Collect] flushed prewarmed chats after empty upstream response account=%s count=%s",
                    acc.email,
                    flushed,
                )
            except Exception as exc:
                log.warning("[Collect] flush prewarmed chats failed account=%s error=%s", acc.email, exc)

    def _finalize_result(*, reason: str | None = None) -> RuntimeExecutionResult:
        answer_text = "".join(answer_fragments)
        reasoning_text = "".join(reasoning_fragments)
        if native_tool_calls and not answer_text:
            answer_text = native_tool_calls_to_markup(native_tool_calls)

        # 鍏抽敭淇锛氬己鍒惰В鏋愭渶缁堟枃鏈腑鐨勫伐鍏疯皟鐢?
        detected_tool_calls = native_tool_calls or (rejected_tool_calls if reason == "invalid_tool_args" else [])
        final_finish_reason = "invalid_tool_args" if reason == "invalid_tool_args" else ("tool_calls" if native_tool_calls else "stop")

        # 绗竴閲嶏細鍒锋柊 Tool Sieve
        if tool_sieve and not native_tool_calls:
            flush_events = tool_sieve.flush()
            for evt in flush_events:
                if evt.get("type") == "tool_calls":
                    calls = evt.get("calls", [])
                    if calls:
                        # 杞崲涓烘爣鍑嗘牸寮?
                        import uuid
                        detected_tool_calls = [{
                            "type": "tool_use",
                            "id": f"toolu_{uuid.uuid4().hex[:8]}",
                            "name": call["name"],
                            "input": call["input"]
                        } for call in calls]
                        detected_tool_calls, invalid_calls = _filter_invalid_native_tool_calls(detected_tool_calls, request, history_messages)
                        if invalid_calls:
                            rejected_tool_calls.extend(invalid_calls)
                            log.warning(
                                "[Collect] Tool Sieve flush rejected invalid tool calls: tools=%s",
                                [c.get("name") for c in invalid_calls if isinstance(c, dict)],
                            )
                            continue
                        final_finish_reason = "tool_calls"
                        _log_tool_calls("tool_sieve_flush", detected_tool_calls)
                        log.info(
                            "[Collect] Tool Sieve flush detected tool calls: tools=%s",
                            [t.get("name") for t in detected_tool_calls],
                        )
                        break
                elif evt.get("type") == "content":
                    # 鍓╀綑鏂囨湰鍐呭
                    pass
            if rejected_tool_calls and not detected_tool_calls:
                detected_tool_calls = rejected_tool_calls
                final_finish_reason = "invalid_tool_args"

        # 绗簩閲嶏細瑙ｆ瀽鏈€缁堟枃鏈?
        if not detected_tool_calls and request.tools and answer_text:
            # 灏濊瘯浠庢渶缁堟枃鏈腑瑙ｆ瀽宸ュ叿璋冪敤
            tool_blocks, stop_reason = tool_parser.parse_tool_calls_silent(answer_text, request.tools)
            tool_use_blocks = [b for b in tool_blocks if b.get("type") == "tool_use"]

            if tool_use_blocks and stop_reason == "tool_use":
                # 鎵惧埌宸ュ叿璋冪敤锛?
                detected_tool_calls = tool_use_blocks
                detected_tool_calls, invalid_calls = _filter_invalid_native_tool_calls(detected_tool_calls, request, history_messages)
                if invalid_calls:
                    rejected_tool_calls.extend(invalid_calls)
                    log.warning(
                        "[Collect] final text parser rejected invalid tool calls: tools=%s",
                        [c.get("name") for c in invalid_calls if isinstance(c, dict)],
                    )
                    detected_tool_calls = []
                if not detected_tool_calls:
                    finish_reason = "invalid_tool_args" if rejected_tool_calls else "stop"
                    state = RuntimeAttemptState(
                        answer_text=answer_text,
                        reasoning_text=reasoning_text,
                        tool_calls=rejected_tool_calls if rejected_tool_calls else [],
                        blocked_tool_names=[],
                        finish_reason=finish_reason,
                        raw_events=raw_events,
                        emitted_visible_output=emitted_visible_output,
                        stage_metrics=metrics.as_dict(),
                    )
                    return RuntimeExecutionResult(state=state, chat_id=chat_id, acc=acc)
                final_finish_reason = "tool_calls"
                _log_tool_calls("final_text_parse", detected_tool_calls)

                # 浠庢枃鏈腑绉婚櫎宸ュ叿璋冪敤閮ㄥ垎
                text_blocks = [b for b in tool_blocks if b.get("type") == "text"]
                if text_blocks:
                    answer_text = text_blocks[0].get("text", "")
                else:
                    answer_text = ""

                log.info(
                    "[Collect] final text parser detected tool calls: tools=%s, cleaned_text_len=%s",
                    [t.get("name") for t in detected_tool_calls],
                    len(answer_text),
                )

        # 妫€鏌ョ┖杈撳嚭
        if not detected_tool_calls and request.tools and answer_text and has_textual_tool_marker(answer_text):
            attempted = _attempted_tool_for_repair(answer_text, request)
            log.warning(
                "[Collect] blocked unparsed textual tool markup before client output: attempted=%s len=%s reason=%s",
                attempted,
                len(answer_text),
                reason,
            )
            answer_text = ""
            final_finish_reason = "invalid_tool_args"
        empty_upstream_response = not detected_tool_calls and not answer_text.strip() and not reasoning_text.strip()
        if empty_upstream_response:
            log.warning(
                "[Collect] upstream returned empty output: reason=%s chat_id=%s",
                reason,
                chat_id,
            )
            # 濡傛灉鏈?reasoning 浣嗘病鏈?visible output锛岃鏄庢ā鍨嬪彧杈撳嚭浜嗘€濊€冭繃绋?
            if reasoning_text.strip():
                log.warning("[Collect] upstream returned reasoning only without visible output")

        if reason:
            log.info(
                "[Collect] finalize reason=%s chat_id=%s tool_calls=%s answer_chars=%s reasoning_chars=%s finish_reason=%s",
                reason,
                chat_id,
                len(detected_tool_calls),
                len(answer_text),
                len(reasoning_text),
                final_finish_reason,
            )
            _trace_text_fingerprint("upstream_answer", answer_text, stage=reason, chat_id=chat_id)
            if reasoning_text:
                _trace_text_fingerprint("upstream_reasoning", reasoning_text, stage=reason, chat_id=chat_id)
        if detected_tool_calls:
            _log_tool_calls(f"final:{reason or 'stream_end'}", detected_tool_calls)
        metrics.mark("stream_finish", float(len(raw_events)))
        state = RuntimeAttemptState(
            answer_text=answer_text,
            reasoning_text=reasoning_text,
            tool_calls=detected_tool_calls,
            blocked_tool_names=extract_blocked_tool_names(answer_text.strip(), request.tool_names),
            finish_reason=final_finish_reason,
            empty_upstream_response=empty_upstream_response,
            raw_events=raw_events,
            emitted_visible_output=emitted_visible_output,
            stage_metrics=metrics.summary(),
        )
        return RuntimeExecutionResult(state=state, chat_id=chat_id, acc=acc)

    request_chat_type = getattr(request, "chat_type", "t2t") or "t2t"
    use_prewarmed_chat = request_chat_type == "t2t" and not bool(getattr(request, "skip_prewarmed_chat_ids", False))
    existing_chat_id = getattr(request, "upstream_chat_id", None) if request_chat_type == "t2t" else None
    update_request_context(
        surface=getattr(request, "surface", "-"),
        requested_model=getattr(request, "requested_model", None) or getattr(request, "response_model", "-"),
        resolved_model=getattr(request, "resolved_model", "-"),
    )
    test_markers = log_test_prompt(
        log,
        stage="runtime_prompt",
        surface=getattr(request, "surface", "-"),
        model=getattr(request, "resolved_model", "-"),
        stream=bool(getattr(request, "stream", False)),
        tools=list(getattr(request, "tool_names", []) or []),
        prompt=prompt,
    )
    delete_on_close = (request_chat_type != "t2t") or not bool(getattr(request, "persistent_session", False))
    if test_markers or settings.TRACE_RESPONSE_FINGERPRINTS:
        log.info(
            "[RuntimePrompt] marker=%s surface=%s model=%s prompt_len=%s delete_on_close=%s persistent_session=%s existing_chat_id=%s",
            ",".join(test_markers) if test_markers else "-",
            getattr(request, "surface", "-"),
            getattr(request, "resolved_model", "-"),
            len(prompt or ""),
            delete_on_close,
            bool(getattr(request, "persistent_session", False)),
            existing_chat_id or "-",
        )

    async for item in client.chat_stream_events_with_retry(
        request.resolved_model,
        prompt,
        has_custom_tools=bool(request.tools),
        files=getattr(request, "upstream_files", None),
        fixed_account=getattr(request, "bound_account", None),
        existing_chat_id=existing_chat_id,
        delete_on_close=delete_on_close,
        use_prewarmed=use_prewarmed_chat,
        chat_type=request_chat_type,
        thinking_enabled=getattr(request, "thinking_enabled", None),
        enable_search=bool(getattr(request, "enable_search", False)),
        system=getattr(request, "system_prompt", ""),
    ):
        if item.get("type") == "meta":
            chat_id = item.get("chat_id")
            acc = item.get("acc")
            update_request_context(chat_id=chat_id)
            test_markers = find_test_markers(prompt)
            if test_markers:
                log.info(
                    "[TestTrace] stage=upstream_chat marker=%s upstream_chat_id=%s account=%s delete_on_close=%s surface=%s",
                    ",".join(test_markers),
                    chat_id,
                    getattr(acc, "email", "-"),
                    delete_on_close,
                    getattr(request, "surface", "-"),
                )
            metrics.mark("chat_created", float(len(raw_events)))
            continue
        if item.get("type") != "event":
            continue

        evt = item.get("event", {})
        if capture_events:
            raw_events.append(evt)
        if evt.get("type") != "delta":
            continue

        phase = evt.get("phase", "")
        content = evt.get("content", "")

        if phase in ("think", "thinking_summary"):
            reasoning_content = _extract_reasoning_text(evt)
            if phase == "thinking_summary" and reasoning_content:
                if reasoning_content.startswith(last_thinking_summary_text):
                    incremental = reasoning_content[len(last_thinking_summary_text):]
                else:
                    incremental = reasoning_content
                last_thinking_summary_text = reasoning_content
                reasoning_content = incremental
            if not reasoning_content and evt.get("status") != "finished":
                continue
            if reasoning_content:
                reasoning_fragments.append(reasoning_content)
                emitted_visible_output = True
                if not first_event_marked:
                    metrics.mark("first_event", float(len(raw_events)))
                    first_event_marked = True
            if on_delta is not None:
                await on_delta(evt, reasoning_content, None)
            continue

        if phase == "answer" and content:
            answer_fragments.append(content)

            # 姣掓€ф嫆缁濇棭鏈熸嫤鎴細Qwen 鍋跺皵骞昏鍑?"Tool X does not exists." 涔嬬被鏂囨湰銆?
            # 鍦ㄦ爣璁?emitted_visible_output 涔嬪墠璇嗗埆骞舵彁鍓?finalize锛岃 evaluate_retry_directive
            # 鐨?blocked_tool_name 鍒嗘敮鑳芥甯歌Е鍙戦噸璇曪紙鍚﹀垯 emitted=True 鍚庡氨涓?retry 浜嗭級銆?
            if (
                request.tools
                and not emitted_visible_output
                and len("".join(answer_fragments)) >= 20
            ):
                early_answer = "".join(answer_fragments).strip()
                # Valid textual tool calls start as answer text before ToolSieve
                # finishes parsing them. Do not classify those markers as a
                # refusal/toxic answer; let the tool parser consume or block them.
                if not has_textual_tool_marker(early_answer) and _TOXIC_REFUSAL_RE.search(early_answer):
                    toxic_blocked = extract_blocked_tool_names(early_answer, request.tool_names)
                    blocked_name = toxic_blocked[0] if toxic_blocked else "unknown"
                    log.warning(
                        "[Collect] blocked contaminated output before client stream: preview=%r",
                        early_answer[:80],
                    )
                    return _finalize_result(reason=f"blocked_tool_name:{blocked_name}")

            if not first_event_marked:
                metrics.mark("first_event", float(len(raw_events)))
                first_event_marked = True

            # Tool Sieve mirrors ds2api's stream path: only release safe text,
            # buffer possible tool markup, and emit tool calls as soon as a complete
            # QNML/legacy block is parsed instead of waiting for upstream EOF.
            if tool_sieve:
                sieve_events = tool_sieve.process_chunk(content)
                for sieve_evt in sieve_events:
                    if sieve_evt.get("type") == "content":
                        safe_text = sieve_evt.get("text", "")
                        if safe_text:
                            emitted_visible_output = True
                            if on_delta is not None:
                                await on_delta(evt, safe_text, None)
                        continue

                    if sieve_evt.get("type") == "tool_calls":
                        calls = sieve_evt.get("calls", [])
                        if calls:
                            import uuid
                            detected_calls = [{
                                "type": "tool_use",
                                "id": f"toolu_{uuid.uuid4().hex[:8]}",
                                "name": call["name"],
                                "input": call["input"]
                            } for call in calls]
                            detected_calls, invalid_calls = _filter_invalid_native_tool_calls(detected_calls, request, history_messages)
                            if invalid_calls:
                                rejected_tool_calls.extend(invalid_calls)
                                log.warning(
                                    "[Collect] Tool Sieve rejected invalid tool calls: tools=%s",
                                    [c.get("name") for c in invalid_calls if isinstance(c, dict)],
                                )
                                return _finalize_result(reason="invalid_tool_args")
                            native_tool_calls.extend(detected_calls)
                            emitted_visible_output = True
                            _log_tool_calls("tool_sieve_stream", detected_calls)
                            log.info(
                                "[Collect] Tool Sieve detected tool calls in stream: tools=%s",
                                [c.get("name") for c in detected_calls],
                            )
                            if on_delta is not None:
                                await on_delta({**evt, "phase": "tool_call"}, None, detected_calls)
                            return _finalize_result(reason="tool_sieve_detected")

                if request.tools:
                    answer_text = "".join(answer_fragments)
                    if len(answer_fragments) % 3 == 0 or "does not exist" in content.lower():
                        blocked_tool_names = extract_blocked_tool_names(answer_text.strip(), request.tool_names)
                        if blocked_tool_names:
                            return _finalize_result(reason=f"blocked_tool_name:{blocked_tool_names[0]}")
                    if has_textual_tool_marker(answer_text):
                        directive = parse_tool_directive_once(
                            request,
                            RuntimeAttemptState(answer_text=answer_text, reasoning_text="".join(reasoning_fragments)),
                        )
                        if directive.stop_reason == "tool_use":
                            return _finalize_result(reason="textual_tool_use")
                continue

            emitted_visible_output = True
            if on_delta is not None:
                await on_delta(evt, content, None)
            if request.tools:
                answer_text = "".join(answer_fragments)
                if len(answer_fragments) % 3 == 0 or "does not exist" in content.lower():
                    blocked_tool_names = extract_blocked_tool_names(answer_text.strip(), request.tool_names)
                    if blocked_tool_names:
                        return _finalize_result(reason=f"blocked_tool_name:{blocked_tool_names[0]}")
                if has_textual_tool_marker(answer_text):
                    directive = parse_tool_directive_once(
                        request,
                        RuntimeAttemptState(answer_text=answer_text, reasoning_text="".join(reasoning_fragments)),
                    )
                    if directive.stop_reason == "tool_use":
                        return _finalize_result(reason="textual_tool_use")
            continue

        if phase == "tool_call":
            emitted_visible_output = True
            if not first_event_marked:
                metrics.mark("first_event", float(len(raw_events)))
                first_event_marked = True
            completed_calls = tool_state.process_event(evt)
            if completed_calls:
                completed_calls, invalid_calls = _filter_invalid_native_tool_calls(completed_calls, request, history_messages)
                if invalid_calls:
                    rejected_tool_calls.extend(invalid_calls)
                    log.warning(
                        "[Collect] native tool_call rejected invalid tool calls: tools=%s",
                        [c.get("name") for c in invalid_calls if isinstance(c, dict)],
                    )
                    return _finalize_result(reason="invalid_tool_args")
                native_tool_calls.extend(completed_calls)
                _log_tool_calls("native_stream", completed_calls)
                if on_delta is not None:
                    await on_delta(evt, None, completed_calls)
                return _finalize_result(reason="native_tool_use")

    execution = _finalize_result(reason="stream_end")
    if execution.state.empty_upstream_response:
        await cleanup_empty_upstream_state()
    return execution


def parse_tool_directive_once(request: StandardRequest, state: RuntimeAttemptState) -> RuntimeToolDirective:
    if state.tool_calls:
        return RuntimeToolDirective(
            tool_blocks=[
                {
                    "type": "tool_use",
                    "id": tool_call["id"],
                    "name": normalize_tool_name(tool_call["name"], request.tool_names),
                    "input": tool_call.get("input", {}),
                }
                for tool_call in state.tool_calls
            ],
            stop_reason="tool_use",
        )

    if request.tools and state.answer_text:
        tool_blocks, stop_reason = tool_parser.parse_tool_calls_silent(state.answer_text, request.tools)
        if stop_reason != "tool_use" and has_textual_tool_marker(state.answer_text):
            attempted = _attempted_tool_for_repair(state.answer_text, request)
            return RuntimeToolDirective(
                tool_blocks=[{
                    "type": "text",
                    "text": f"Invalid tool-call format was blocked for {attempted}. Retry the current request with a valid tool call.",
                }],
                stop_reason="end_turn",
            )
        return RuntimeToolDirective(tool_blocks=tool_blocks, stop_reason=stop_reason)

    return RuntimeToolDirective(tool_blocks=[{"type": "text", "text": state.answer_text}], stop_reason="end_turn")


# ==================== 鎴柇缁啓 + 娴佸紡 warmup锛圥2-6 & P2-10 鎺ュ叆锛?===================

async def collect_completion_run_with_recovery(
    client,
    request: StandardRequest,
    prompt: str,
    *,
    capture_events: bool = True,
    on_delta: Callable[[dict[str, Any], str | None, list[dict[str, Any]] | None], Awaitable[None]] | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    max_continuation: int = 2,
    warmup_chars: int = 0,
    guard_chars: int = 0,
) -> RuntimeExecutionResult:
    """collect_completion_run 鐨勫寮虹増锛屽彔鍔犱袱涓兘鍔涳細

    1. **Truncation recovery** (P2-6): if answer_text ends inside an unclosed QNML tool block,
       鏈€澶氳拷鍔?max_continuation 娆＄画鍐欒姹傦紝姣忔涓㈠純宸ュ叿瀹氫箟鐪佷笂涓嬫枃棰勭畻锛?
       鐢?deduplicate_continuation 鍘婚櫎澶村熬閲嶅彔鍚庢嫾鎺ャ€?

    2. **娴佸紡 warmup/guard**锛圥2-10锛夛細褰?warmup_chars>0 鏃讹紝鐢?IncrementalTextStreamer
       鍖呰 on_delta 鐨?text 璺緞鈥斺€旇捣姝ョ疮绉?warmup_chars 瀛楃鍚庢墠鏀捐锛?
       浠讳綍鏃跺埢淇濈暀鏈熬 guard_chars 瀛楃鏆備笉鍙戝嚭锛岀粰璺?chunk 娓呮礂棰勭暀绌洪棿銆?

    淇濈暀鍚戝悗鍏煎锛氫笉浼犲彲閫夊弬鏁?鈫?琛屼负涓?collect_completion_run 瀹屽叏涓€鑷淬€?
    """
    from backend.services.truncation_recovery import (
        build_continuation_prompt,
        deduplicate_continuation,
        is_truncated,
    )
    from backend.services.incremental_text_streamer import IncrementalTextStreamer

    wrapped_on_delta = on_delta
    streamer: IncrementalTextStreamer | None = None
    if warmup_chars > 0 and on_delta is not None:
        streamer = IncrementalTextStreamer(
            warmup_chars=warmup_chars,
            guard_chars=max(guard_chars, 64),
        )

        async def _wrapped(evt, text_chunk, tool_calls):
            # 浠呭绾枃鏈?delta 鍋?warmup锛泃ool_calls / thinking / native 鐩存帴閫忎紶
            if text_chunk is None or tool_calls is not None or evt.get("phase") not in ("answer", "text"):
                await on_delta(evt, text_chunk, tool_calls)
                return
            released = streamer.push(text_chunk)
            if released:
                await on_delta(evt, released, None)

        wrapped_on_delta = _wrapped

    result = await collect_completion_run(
        client, request, prompt,
        capture_events=capture_events,
        on_delta=wrapped_on_delta,
        history_messages=history_messages,
    )

    # 鑻?warmup 杩樹繚鐣欑潃灏鹃儴锛宖lush 鍑哄幓
    if streamer is not None and on_delta is not None:
        tail = streamer.finish()
        if tail:
            await on_delta({"phase": "answer"}, tail, None)

    # 鎴柇缁啓
    continues = 0
    while continues < max_continuation:
        state = result.state
        # 鏈夊凡妫€鍑虹殑宸ュ叿璋冪敤灏变笉缁啓锛堣瀹㈡埛绔幓鎵ц閭ｄ釜 tool锛?
        if state.tool_calls:
            break
        # No tools: skip recovery; without QNML/legacy marker context plain text can false-positive.
        if not request.tools:
            break
        if not is_truncated(state.answer_text):
            break

        continues += 1
        log.info(
            "[TruncRecover] detected unclosed tool call, continuation attempt=%d chat_id=%s len=%d",
            continues, result.chat_id, len(state.answer_text),
        )

        assistant_ctx, followup = build_continuation_prompt(state.answer_text, anchor_chars=2000)
        # 缁啓 prompt = 鍘?prompt + assistant 宸茶緭鍑虹殑閿氱偣 + user 缁啓鎸囦护
        cont_prompt = (
            f"{prompt.rstrip()}\n\nAssistant: {assistant_ctx}\n\nHuman: {followup}\n\nAssistant:"
        )

        cont_result = await collect_completion_run(
            client, request, cont_prompt,
            capture_events=False,
            on_delta=on_delta,  # 不经过 streamer，续写内容直接透传
            history_messages=history_messages,
        )
        try:
            cont_text = cont_result.state.answer_text
            if not cont_text or not cont_text.strip():
                log.info("[TruncRecover] empty continuation, stopping")
                break

            deduped = deduplicate_continuation(state.answer_text, cont_text)
            if not deduped.strip():
                log.info("[TruncRecover] continuation fully overlapped existing, stopping")
                break

            merged_answer = state.answer_text + deduped
            merged_state = RuntimeAttemptState(
                answer_text=merged_answer,
                reasoning_text=state.reasoning_text,
                tool_calls=cont_result.state.tool_calls or state.tool_calls,
                blocked_tool_names=cont_result.state.blocked_tool_names or state.blocked_tool_names,
                finish_reason=cont_result.state.finish_reason or state.finish_reason,
                raw_events=state.raw_events,
                emitted_visible_output=state.emitted_visible_output or cont_result.state.emitted_visible_output,
                stage_metrics=state.stage_metrics,
            )
            result = RuntimeExecutionResult(state=merged_state, chat_id=result.chat_id, acc=result.acc)
            log.info(
                "[TruncRecover] continuation=%d produced %d new chars; total=%d",
                continues, len(deduped), len(merged_answer),
            )
            # 鑻ョ画鍐欏畬鎴愬悗宸查棴鍚堝垯鏀跺伐
            if not is_truncated(merged_answer):
                break
        finally:
            bound_account = getattr(request, "bound_account", None)
            preserve_continuation_chat = bool(getattr(request, "persistent_session", False))
            if cont_result.acc is not None and bound_account is not None and cont_result.acc is bound_account:
                if not preserve_continuation_chat and cont_result.chat_id and getattr(cont_result.acc, "token", None):
                    await client.delete_chat_reliable(
                        cont_result.acc.token,
                        cont_result.chat_id,
                        source="truncation_recovery",
                    )
            else:
                await cleanup_runtime_resources(
                    client,
                    cont_result.acc,
                    cont_result.chat_id,
                    preserve_chat=preserve_continuation_chat,
                )

    return result



def _filter_tool_directive(
    directive: RuntimeToolDirective,
    request: StandardRequest,
    history_messages: list[dict[str, Any]] | None = None,
) -> RuntimeToolDirective:
    if directive.stop_reason != "tool_use":
        return directive
    filtered_blocks: list[dict[str, Any]] = []
    blocked_reasons: list[str] = []
    for block in directive.tool_blocks:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            filtered_blocks.append(block)
            continue
        reason = _tool_call_block_reason(
            block,
            request,
            history_messages,
            include_completed_write_guard=True,
        )
        if reason:
            blocked_reasons.append(reason)
            log.warning("[ToolGuard] blocked final directive reason=%s name=%s", reason, block.get("name", "-"))
            if reason.startswith("completed_write:"):
                filtered_blocks.append({"type": "text", "text": _completed_write_visible_text(block)})
            continue
        filtered_blocks.append(block)
    if any(isinstance(block, dict) and block.get("type") == "tool_use" for block in filtered_blocks):
        return RuntimeToolDirective(tool_blocks=filtered_blocks, stop_reason="tool_use")
    if blocked_reasons:
        if filtered_blocks:
            return RuntimeToolDirective(tool_blocks=filtered_blocks, stop_reason="end_turn")
        return RuntimeToolDirective(
            tool_blocks=[{
                "type": "text",
                "text": "",
            }],
            stop_reason="end_turn",
        )
    return RuntimeToolDirective(tool_blocks=filtered_blocks, stop_reason="end_turn")

def build_tool_directive(
    request: StandardRequest,
    state: RuntimeAttemptState,
    history_messages: list[dict[str, Any]] | None = None,
) -> RuntimeToolDirective:
    if not state.tool_calls and state.finish_reason == "stop":
        _complete_final_delivery_if_safe(
            request=request,
            state=state,
            history_messages=history_messages,
            source="directive",
        )
    raw_directive = parse_tool_directive_once(request, state)
    directive = _filter_tool_directive(raw_directive, request, history_messages)
    log.info(
        f"[ToolDirective] tool_blocks={len(directive.tool_blocks)} raw_tool_blocks={len(raw_directive.tool_blocks)} "
        f"stop_reason={directive.stop_reason} has_tool_use={any(b.get('type') == 'tool_use' for b in directive.tool_blocks)}"
    )
    return directive


def anthropic_stream_usage_delta(prompt: str, answer_text: str) -> int:
    return len(answer_text) + len(prompt)


def anthropic_stream_stop_reason(request: StandardRequest, state: RuntimeAttemptState, pending_chunks: list[str]) -> str:
    if state.tool_calls or any('"type": "tool_use"' in chunk for chunk in pending_chunks):
        return "tool_use"
    return build_tool_directive(request, state).stop_reason


def finalize_anthropic_stream_success(*, request: StandardRequest, prompt: str, execution: RuntimeExecutionResult, translator) -> AnthropicStreamSuccessResult:
    stop_reason = anthropic_stream_stop_reason(request, execution.state, translator.pending_chunks)
    chunks = translator.finalize(answer_text=execution.state.answer_text, stop_reason=stop_reason)
    return AnthropicStreamSuccessResult(
        chunks=chunks,
        usage_delta=anthropic_stream_usage_delta(prompt, execution.state.answer_text),
    )


async def complete_anthropic_stream_success(*, users_db, token: str, client, prompt: str, request: StandardRequest, execution: RuntimeExecutionResult, translator) -> AnthropicStreamCompletionResult:
    from backend.services.auth_quota import add_used_tokens

    stream_success = finalize_anthropic_stream_success(
        request=request,
        prompt=prompt,
        execution=execution,
        translator=translator,
    )
    await add_used_tokens(users_db, token, stream_success.usage_delta)
    await cleanup_runtime_resources(client, execution.acc, execution.chat_id)
    return AnthropicStreamCompletionResult(chunks=stream_success.chunks)


def inject_assistant_message(prompt: str, message: str) -> str:
    next_prompt = prompt.rstrip()
    if next_prompt.endswith("Assistant:"):
        return next_prompt[:-len("Assistant:")] + message + "\nAssistant:"
    return next_prompt + "\n\n" + message + "\nAssistant:"


def retryable_usage_delta(prompt: str):
    return lambda execution, current_prompt=None: len(execution.state.answer_text) + len(current_prompt or prompt)


def build_usage_delta_factory(prompt: str) -> Callable[[RuntimeExecutionResult, Any | None], int]:
    return lambda execution, current_prompt=None: len(execution.state.answer_text) + len(current_prompt or prompt)


def request_max_attempts(request: StandardRequest) -> int:
    # 宸ュ叿妯″紡涓嬬粰妯″瀷鏇村閲嶈瘯鏈轰細锛堟瘨鎬у够瑙?閲嶅璋冪敤鍦烘櫙甯歌锛夛紝
    # 鍘熷€?2 鍦ㄥ杞?retry 閲屽お瀹规槗鐢ㄥ畬锛屽崌鍒?4
    return 4 if request.tools else settings.MAX_RETRIES


def plan_runtime_attempts(request: StandardRequest, *, initial_prompt: str) -> RuntimeAttemptPlan:
    loop = build_retry_loop(request, initial_prompt=initial_prompt)
    return RuntimeAttemptPlan(loop=loop, prompt=loop.prompt)


def build_retry_loop(request: StandardRequest, *, initial_prompt: str) -> RuntimeRetryLoop:
    return RuntimeRetryLoop(
        prompt=initial_prompt,
        max_attempts=request_max_attempts(request),
    )


def evaluate_retry_directive(
    *,
    request: StandardRequest,
    current_prompt: str,
    history_messages: list[dict[str, Any]] | None,
    attempt_index: int,
    max_attempts: int,
    state: RuntimeAttemptState,
    allow_after_visible_output: bool = False,
) -> RuntimeRetryDirective:
    can_retry_after_output = allow_after_visible_output or not state.emitted_visible_output

    def _retry(reason: str, next_prompt: str) -> RuntimeRetryDirective:
        log.info(
            "[Retry] reason=%s attempt=%s/%s client=%s blocked=%s read_blocked=%s finish_reason=%s emitted=%s",
            reason,
            attempt_index + 1,
            max_attempts,
            getattr(request, "client_profile", "-"),
            getattr(request, "retry_blocked_tools", [])[:8],
            getattr(request, "retry_read_blocklist", [])[-8:],
            state.finish_reason,
            state.emitted_visible_output,
        )
        return RuntimeRetryDirective(retry=True, next_prompt=next_prompt, reason=reason)

    if not state.tool_calls and state.finish_reason == "stop" and can_retry_after_output:
        delivery_gap = _complete_final_delivery_if_safe(
            request=request,
            state=state,
            history_messages=history_messages,
            current_prompt=current_prompt,
            source="retry",
        )
        if delivery_gap is not None:
            log.warning(
                "[FinalDeliveryGuard] retry missing_markers=%s final_phrase_missing=%s targets=%s",
                delivery_gap.missing_artifact_markers,
                delivery_gap.final_phrase_missing,
                delivery_gap.target_paths[-4:],
            )
            if attempt_index >= max_attempts - 1:
                return RuntimeRetryDirective(retry=False, next_prompt=current_prompt, reason="final_delivery_incomplete_exhausted")
            return _retry(
                "final_delivery_incomplete",
                _inject_retry_guard(current_prompt, request, _final_delivery_retry_message(delivery_gap)),
            )

    if attempt_index >= max_attempts - 1:
        return RuntimeRetryDirective(retry=False, next_prompt=current_prompt, reason=None)

    if state.finish_reason == "invalid_tool_args" and request.tools and can_retry_after_output:
        attempted_tools = [
            normalize_tool_name(str(call.get("name", "")), request.tool_names)
            for call in state.tool_calls
            if isinstance(call, dict) and call.get("name")
        ]
        disallowed = [name for name in attempted_tools if _is_disallowed_control_tool(name, request, history_messages)]
        for name in disallowed:
            _add_unique(request.retry_blocked_tools, name)
        blocked_read_paths: list[str] = []
        previously_blocked_read_paths = {
            path for path in getattr(request, "retry_read_blocklist", []) if path
        }
        repeated_blocked_read_paths: list[str] = []
        for call in state.tool_calls:
            if not isinstance(call, dict):
                continue
            tool_name = normalize_tool_name(str(call.get("name", "")), request.tool_names)
            tool_input = call.get("input", {})
            if tool_name != "Read":
                continue
            read_path = _tool_input_path(tool_input)
            if not read_path:
                continue
            if read_path in previously_blocked_read_paths:
                _add_unique(repeated_blocked_read_paths, read_path)
            _add_unique(request.retry_read_blocklist, read_path)
            _add_unique(blocked_read_paths, read_path)
        if blocked_read_paths:
            if len(repeated_blocked_read_paths) == len(blocked_read_paths):
                log.warning(
                    "[Retry] stop_repeated_blocked_read attempt=%s/%s paths=%s finish_reason=%s",
                    attempt_index + 1,
                    max_attempts,
                    repeated_blocked_read_paths[-4:],
                    state.finish_reason,
                )
                return RuntimeRetryDirective(retry=False, next_prompt=current_prompt, reason="repeated_blocked_read")
            joined_paths = "; ".join(blocked_read_paths[-4:])
            force_text = (
                "[MANDATORY]: The previous Read call was blocked because that file was already returned unchanged or is on the retry read blocklist. "
                f"Do NOT call Read again for: {joined_paths}. "
                "Use the existing result, choose another relevant tool, write/edit the requested output, or finish."
            )
            return _retry(
                f"blocked_read:{blocked_read_paths[-1]}",
                _inject_retry_guard(current_prompt, request, force_text),
            )
        force_text = (
            "[MANDATORY]: The previous tool call was invalid, disallowed, or incomplete. "
            "Do NOT reuse or continue any partial QNML/tool markup from the previous response. "
            "Start a fresh complete tool call with all required arguments, or continue with direct project tools such as Read/Grep/Glob/Write/Edit/Bash when they advance the task. "
            "Use control/task/scheduling/delegation tools only when they are clearly necessary for the current task context or explicitly requested."
        )
        reason = f"unexpected_control_tool:{disallowed[0]}" if disallowed else "invalid_tool_args"
        return _retry(reason, _inject_retry_guard(current_prompt, request, force_text))
    if state.blocked_tool_names and request.tools:
        if not can_retry_after_output:
            return RuntimeRetryDirective(retry=False, next_prompt=current_prompt, reason=None)
        blocked_name = normalize_tool_name(state.blocked_tool_names[0], request.tool_names)
        reminder_prompt = tool_parser.inject_format_reminder(
            current_prompt,
            blocked_name,
            client_profile=getattr(request, "client_profile", CLAUDE_CODE_OPENAI_PROFILE),
        )
        return _retry(
            f"blocked_tool_name:{blocked_name}",
            _inject_retry_guard(reminder_prompt, request, ""),
        )

    if request.tools:
        directive: RuntimeToolDirective | None = None
        if state.answer_text:
            saw_contract_markup = should_retry_textual_tool_contract(state.answer_text)
            if saw_contract_markup and can_retry_after_output:
                if has_invalid_textual_tool_contract(state.answer_text):
                    fallback_tool_name = _attempted_tool_for_repair(state.answer_text, request)
                    reminder_prompt = tool_parser.inject_format_reminder(
                        current_prompt,
                        fallback_tool_name,
                        client_profile=getattr(request, "client_profile", CLAUDE_CODE_OPENAI_PROFILE),
                    )
                    return _retry(
                        f"invalid_textual_tool_contract:{fallback_tool_name}",
                        _inject_retry_guard(reminder_prompt, request, _fresh_textual_tool_retry_instruction(fallback_tool_name)),
                    )
                directive = parse_tool_directive_once(request, state)
                if directive.stop_reason != "tool_use":
                    fallback_tool_name = _attempted_tool_for_repair(state.answer_text, request)
                    reminder_prompt = tool_parser.inject_format_reminder(
                        current_prompt,
                        fallback_tool_name,
                        client_profile=getattr(request, "client_profile", CLAUDE_CODE_OPENAI_PROFILE),
                    )
                    return _retry(
                        f"unparsed_textual_tool_contract:{fallback_tool_name}",
                        _inject_retry_guard(reminder_prompt, request, _fresh_textual_tool_retry_instruction(fallback_tool_name)),
                    )
        if directive is None:
            directive = parse_tool_directive_once(request, state)

        if directive.stop_reason == "tool_use":
            first_tool = next((b for b in directive.tool_blocks if b.get("type") == "tool_use"), None)
            if first_tool:
                tool_name = str(first_tool.get("name", ""))
                tool_input = first_tool.get("input", {})
                if _is_disallowed_control_tool(tool_name, request, history_messages) and can_retry_after_output:
                    _add_unique(request.retry_blocked_tools, tool_name)
                    force_text = (
                        f"[MANDATORY]: Do NOT call {tool_name} again in this retry. That control/task/scheduling/delegation call was not clearly necessary for the current task context or was previously blocked. "
                        "Continue with direct project tools such as Read/Grep/Glob/Write/Edit/Bash unless a control tool is clearly required."
                    )
                    return _retry(
                        f"unexpected_control_tool:{tool_name}",
                        _inject_retry_guard(current_prompt, request, force_text),
                    )

                if getattr(request, "client_profile", CLAUDE_CODE_OPENAI_PROFILE) == "openclaw_openai":
                    repeated_same_tool = has_recent_openai_same_tool_call(history_messages, tool_name, tool_input)
                else:
                    repeated_same_tool = recent_same_tool_identity_count(history_messages, tool_name, tool_input) >= 1
                if repeated_same_tool and can_retry_after_output:
                    force_text = (
                        f"[MANDATORY]: You already called {tool_name} with the same input. "
                        "Do NOT repeat the same tool call. Use the previous tool result, choose the next relevant tool, write/edit the requested output, or finish."
                    )
                    return _retry(
                        f"repeated_same_tool:{tool_name}",
                        _inject_retry_guard(current_prompt, request, force_text),
                    )

                if tool_name == "Read" and has_recent_unchanged_read_result(history_messages, _tool_input_path(tool_input)):
                    read_path = _tool_input_path(tool_input)
                    log.info(
                        "[Retry] pass_through_unchanged_read path=%s attempt=%s/%s client=%s",
                        read_path or "-",
                        attempt_index + 1,
                        max_attempts,
                        getattr(request, "client_profile", "-"),
                    )

                if tool_name == "WebSearch" and has_recent_search_no_results(history_messages) and can_retry_after_output:
                    force_text = (
                        "[MANDATORY]: The last WebSearch returned no results. "
                        "Do NOT call WebSearch again with similar wording. Use another tool or finish with the best available answer."
                    )
                    return _retry(
                        "search_no_results",
                        _inject_retry_guard(current_prompt, request, force_text),
                    )

    if (
        (state.empty_upstream_response or not state.answer_text)
        and not state.tool_calls
        and state.finish_reason == "stop"
        and not state.emitted_visible_output
    ):
        return _retry("empty_upstream_response", current_prompt)

    return RuntimeRetryDirective(retry=False, next_prompt=current_prompt, reason=None)

async def continue_after_retry_directive(*, client, execution, retry: RuntimeRetryDirective, preserve_chat: bool = False) -> RuntimeRetryContinuation:
    if not retry.retry:
        return RuntimeRetryContinuation(should_continue=False, next_prompt=retry.next_prompt)
    await cleanup_runtime_resources(client, execution.acc, execution.chat_id, preserve_chat=preserve_chat)
    if not preserve_chat:
        await asyncio.sleep(0.15)
    return RuntimeRetryContinuation(should_continue=True, next_prompt=retry.next_prompt)


async def cleanup_runtime_resources(client, acc, chat_id: str | None, *, preserve_chat: bool = False) -> None:
    if acc is None:
        return
    token = getattr(acc, "token", None)
    client.account_pool.release(acc)
    if preserve_chat:
        return
    if chat_id and token:
        background_delete = getattr(client, "delete_chat_background", None)
        if background_delete is not None:
            background_delete(token, chat_id, source="runtime_cleanup")
            return
        delete_fn = getattr(client, "delete_chat_reliable", None)
        if delete_fn is not None:
            async def delete_runner() -> None:
                try:
                    await delete_fn(token, chat_id, source="runtime_cleanup")
                except Exception as exc:
                    log.warning("[Cleanup] background delete_chat failed chat_id=%s error=%s", chat_id, exc)

            try:
                asyncio.create_task(delete_runner())
            except RuntimeError:
                await delete_fn(token, chat_id, source="runtime_cleanup")
            return
        try:
            async def raw_delete_runner() -> None:
                try:
                    await client.delete_chat(token, chat_id)
                except Exception as exc:
                    log.warning("[Cleanup] background delete_chat failed chat_id=%s error=%s", chat_id, exc)

            asyncio.create_task(raw_delete_runner())
        except Exception as exc:
            log.warning("[Cleanup] delete_chat failed chat_id=%s error=%s", chat_id, exc)
