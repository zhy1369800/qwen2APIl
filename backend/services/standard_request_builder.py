from __future__ import annotations

from backend.adapter.standard_request import StandardRequest
from backend.core.config import resolve_model
from backend.services.prompt_builder import messages_to_prompt
from backend.toolcall.normalize import build_tool_name_registry


def _resolve_thinking_enabled(req_data: dict) -> bool:
    reasoning_effort = str(req_data.get("reasoning_effort", "") or "").strip().lower()
    if reasoning_effort == "none":
        return False

    reasoning = req_data.get("reasoning")
    if isinstance(reasoning, dict):
        enabled = reasoning.get("enabled")
        if enabled is False:
            return False
        effort = str(reasoning.get("effort", "") or "").strip().lower()
        if effort == "none":
            return False

    thinking = req_data.get("thinking")
    if thinking is False:
        return False
    if isinstance(thinking, dict) and thinking.get("enabled") is False:
        return False

    include_reasoning = req_data.get("include_reasoning")
    if include_reasoning is False:
        return False

    return True


def build_chat_standard_request(req_data: dict, *, default_model: str, surface: str, client_profile: str = "openclaw_openai") -> StandardRequest:
    requested_model = req_data.get("model", default_model)
    prompt_result = messages_to_prompt(req_data, client_profile=client_profile)
    tools = prompt_result.tools
    tool_names = [tool_name for tool_name in (tool.get("name") for tool in tools) if isinstance(tool_name, str) and tool_name]
    return StandardRequest(
        prompt=prompt_result.prompt,
        response_model=requested_model,
        resolved_model=resolve_model(requested_model),
        surface=surface,
        client_profile=client_profile,
        stream=req_data.get("stream", False),
        tools=tools,
        tool_names=tool_names,
        tool_name_registry=build_tool_name_registry(tool_names),
        tool_enabled=prompt_result.tool_enabled,
        thinking_enabled=_resolve_thinking_enabled(req_data),
    )
