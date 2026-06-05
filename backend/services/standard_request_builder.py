from __future__ import annotations

from backend.adapter.standard_request import StandardRequest
from backend.core.config import resolve_model
from backend.services.model_modes import parse_model_mode
from backend.services.prompt_builder import messages_to_prompt
from backend.toolcall.normalize import build_tool_name_registry


def _coerce_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "enable", "enabled", "auto", "thinking"}:
            return True
        if lowered in {"0", "false", "no", "off", "disable", "disabled", "fast", "none"}:
            return False
    return None


def _extract_thinking_enabled(req_data: dict) -> bool | None:
    if "enable_thinking" in req_data:
        return _coerce_bool(req_data.get("enable_thinking"))
    if "thinking" in req_data:
        thinking = req_data.get("thinking")
        if isinstance(thinking, dict):
            for key in ("enabled", "enable", "enabled_thinking", "enable_thinking"):
                if key in thinking:
                    return _coerce_bool(thinking.get(key))
        else:
            return _coerce_bool(thinking)
    if "thinking_mode" in req_data:
        return _coerce_bool(req_data.get("thinking_mode"))
    return None


def build_chat_standard_request(req_data: dict, *, default_model: str, surface: str, client_profile: str = "openclaw_openai") -> StandardRequest:
    requested_model = req_data.get("model", default_model)
    model_mode = parse_model_mode(requested_model, default_model=default_model)
    explicit_thinking = _extract_thinking_enabled(req_data)
    thinking_enabled = True if model_mode.force_thinking else explicit_thinking
    enable_search = bool(_coerce_bool(req_data.get("enable_search")) or False)
    if model_mode.mode == "search":
        enable_search = True
    if model_mode.chat_type == "deep_research":
        enable_search = True
    prompt_result = messages_to_prompt(req_data, client_profile=client_profile)
    tools = prompt_result.tools
    tool_names = [tool_name for tool_name in (tool.get("name") for tool in tools) if isinstance(tool_name, str) and tool_name]
    return StandardRequest(
        prompt=prompt_result.prompt,
        system_prompt=prompt_result.system_prompt,
        response_model=requested_model,
        resolved_model=resolve_model(model_mode.base_model),
        surface=surface,
        client_profile=client_profile,
        requested_model=requested_model,
        stream=req_data.get("stream", False),
        tools=tools,
        tool_names=tool_names,
        tool_name_registry=build_tool_name_registry(tool_names),
        tool_enabled=prompt_result.tool_enabled,
        chat_type=model_mode.chat_type,
        thinking_enabled=thinking_enabled,
        force_thinking=model_mode.force_thinking,
        enable_search=enable_search,
        model_mode=model_mode.mode,
        skip_prewarmed_chat_ids=model_mode.chat_type != "t2t",
    )
