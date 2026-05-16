from __future__ import annotations

from backend.adapter.standard_request import StandardRequest
from backend.core.config import resolve_model
from backend.services.prompt_builder import messages_to_prompt
from backend.toolcall.normalize import build_tool_name_registry


def _resolve_thinking_enabled(req_data: dict) -> bool:
    feature_config = req_data.get("feature_config")
    if not isinstance(feature_config, dict):
        feature_config = {}

    reasoning_effort = str(req_data.get("reasoning_effort", "") or "").strip().lower()
    if reasoning_effort == "none":
        return False

    reasoning = req_data.get("reasoning")
    if isinstance(reasoning, dict):
        if reasoning.get("enabled") is False:
            return False
        if str(reasoning.get("effort", "") or "").strip().lower() == "none":
            return False

    thinking = req_data.get("thinking")
    if thinking is False:
        return False
    if isinstance(thinking, dict) and thinking.get("enabled") is False:
        return False

    enable_thinking = req_data.get("enable_thinking")
    if enable_thinking is False:
        return False

    if req_data.get("include_reasoning") is False:
        return False
    if feature_config.get("thinking_enabled") is False:
        return False
    if feature_config.get("auto_thinking") is False and feature_config.get("thinking_enabled") is not True:
        return False
    if str(feature_config.get("thinking_mode", "") or "").strip().lower() == "none":
        return False

    if reasoning_effort in ("low", "medium", "high"):
        return True
    if isinstance(reasoning, dict) and (reasoning.get("enabled") is True or reasoning.get("effort")):
        return True
    if thinking is True or (isinstance(thinking, dict) and thinking.get("enabled") is True):
        return True
    if enable_thinking is True:
        return True
    if req_data.get("include_reasoning") is True:
        return True
    if feature_config.get("thinking_enabled") is True:
        return True
    if feature_config.get("auto_thinking") is True:
        return True
    if str(feature_config.get("thinking_mode", "") or "").strip().lower() in ("auto", "on", "enabled"):
        return True

    model_name = str(req_data.get("model", "")).lower()
    if any(k in model_name for k in ("think", "reason", "r1", "o1", "o3")):
        return True

    return False


def _resolve_auto_search_enabled(req_data: dict) -> bool:
    feature_config = req_data.get("feature_config")
    if not isinstance(feature_config, dict):
        feature_config = {}

    auto_search = req_data.get("auto_search")
    if isinstance(auto_search, bool):
        return auto_search

    enable_search = req_data.get("enable_search")
    if isinstance(enable_search, bool):
        return enable_search

    auto_search_fc = feature_config.get("auto_search")
    if isinstance(auto_search_fc, bool):
        return auto_search_fc

    return False


def build_chat_standard_request(
    req_data: dict,
    *,
    default_model: str,
    surface: str,
    client_profile: str = "openclaw_openai",
    native_fc_enabled: bool = False,
) -> StandardRequest:
    requested_model = req_data.get("model", default_model)
    prompt_result = messages_to_prompt(req_data, client_profile=client_profile, native_fc_enabled=native_fc_enabled)
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
        auto_search_enabled=_resolve_auto_search_enabled(req_data),
    )
