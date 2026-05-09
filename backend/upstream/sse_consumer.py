import json
import logging

log = logging.getLogger("qwen2api.sse")


def parse_sse_chunk(chunk: str) -> list[dict]:
    events = []
    for line in chunk.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
            events.append(obj)
        except Exception:
            continue

    parsed = []
    for evt in events:
        if evt.get("choices"):
            delta = evt["choices"][0].get("delta", {})
            content = delta.get("content", "")

            # Log if content contains "Tool" and "does not exist"
            if content and "Tool" in content and "does not exist" in content:
                log.warning(f"[SSE] Detected tool interception: content={content!r} phase={delta.get('phase')} status={delta.get('status')} extra={delta.get('extra')}")

            parsed.append(
                {
                    "type": "delta",
                    "phase": delta.get("phase", "answer"),
                    "content": content,
                    "status": delta.get("status", ""),
                    "extra": delta.get("extra", {}),
                    "function_call": delta.get("function_call"),
                }
            )
    return parsed
