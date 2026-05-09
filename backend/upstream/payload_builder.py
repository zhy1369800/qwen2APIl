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
    "code_interpreter": False,
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
        # Our Anthropic/OpenAI bridge relies on textual JSON/XML tool directives
        # that are parsed locally. Enabling Qwen native function_calling here causes
        # upstream interception such as `Tool Read/Bash does not exists.` for custom
        # local tools that only exist in the bridge layer.
        "function_calling": False,
        # Additional safeguards to prevent tool call interception
        "enable_tools": False,
        "enable_function_call": False,
        "tool_choice": "none",
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
