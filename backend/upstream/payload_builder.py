import time
import uuid


CUSTOM_TOOL_COMPAT_FEATURE_CONFIG = {
    "thinking_enabled": True,
    "output_schema": "phase",
    "research_mode": "normal",
    "auto_thinking": True,
    "thinking_mode": "Auto",
    "thinking_format": "summary",
    "auto_search": False,
    "code_interpreter": True,  # 开启原生代码解释器
    "plugins_enabled": False,
}

def build_chat_payload(
    chat_id: str,
    model: str,
    content: str,
    has_custom_tools: bool = False,
    files: list[dict] | None = None,
    thinking_enabled: bool = True,
) -> dict:
    ts = int(time.time())
    feature_config = {
        **CUSTOM_TOOL_COMPAT_FEATURE_CONFIG,
        # 虽然开启原生 function_calling 可能会导致上游产生 "Tool X does not exists" 的拦截文本，
        # 但我们已经在 execution.py 中实现了抢救逻辑，可以从拦截响应中提取工具调用。
        # 开启此开关能显著提升 Qwen 输出结构化工具调用的稳定性。
        "function_calling": has_custom_tools,
        "enable_tools": has_custom_tools,
        "enable_function_call": has_custom_tools,
        "tool_choice": "auto" if has_custom_tools else "none",
    }
    if not thinking_enabled:
        feature_config["thinking_enabled"] = False
        feature_config["auto_thinking"] = False
    return {
        "stream": True,
        "version": "2.1",
        "incremental_output": True,
        "chat_id": chat_id,
        "chat_mode": "normal",
        "model": model,
        "parent_id": None,
        "messages": [
            {
                "fid": str(uuid.uuid4()),
                "parentId": None,
                "childrenIds": [str(uuid.uuid4())],
                "role": "user",
                "content": content,
                "user_action": "chat",
                "files": files or [],
                "timestamp": ts,
                "models": [model],
                "chat_type": "t2t",
                "feature_config": feature_config,
                "extra": {"meta": {"subChatType": "t2t"}},
                "sub_chat_type": "t2t",
                "parent_id": None,
            }
        ],
        "timestamp": ts,
    }
