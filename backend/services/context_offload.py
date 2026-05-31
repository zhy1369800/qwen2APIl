from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from backend.adapter.standard_request import CLAUDE_CODE_OPENAI_PROFILE

SYSTEM_CONTEXT_FILE_PREFIX = "qwen2api_context"
SYSTEM_CONTEXT_PROMPT_NOTE = (
    "System context files named qwen2api_context*.txt/.md/.json/.log may be attached. "
    "Use them as supporting long-term context, while the inline messages remain authoritative for the latest turn and tool state. "
    "User-uploaded files are separate user inputs and should also be respected."
)


@dataclass(slots=True)
class LocalContextFile:
    filename: str
    ext: str
    content_type: str
    text: str
    sha256: str
    purpose: str = "context"
    local_path: str = ""


@dataclass(slots=True)
class ContextOffloadPlan:
    mode: str
    inline_messages: list[dict[str, Any]]
    generated_files: list[LocalContextFile] = field(default_factory=list)
    summary_text: str = ""
    estimated_prompt_len: int = 0
    note: str = ""


class ContextOffloader:
    def __init__(self, settings):
        self.settings = settings

    def estimate_prompt_len(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, client_profile: str = "") -> int:
        total = 0
        for msg in messages or []:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        total += len(str(part.get("text", "")))
                        total += len(str(part.get("content", "")))
            total += 24
        total += sum(len(str(tool.get("name", ""))) + len(str(tool.get("description", ""))) for tool in (tools or []))
        if client_profile == CLAUDE_CODE_OPENAI_PROFILE:
            total += 512
        return total

    def _extract_text(self, msg: dict[str, Any]) -> str:
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        chunks.append(str(part.get("text", "")))
                    elif part.get("type") == "tool_result":
                        chunks.append(str(part.get("content", "")))
            return "\n".join(chunk for chunk in chunks if chunk)
        return str(content)

    def _make_file(self, base_name: str, ext: str, text: str, content_type: str) -> LocalContextFile:
        data = text.encode("utf-8")
        return LocalContextFile(
            filename=f"{base_name}.{ext}",
            ext=ext,
            content_type=content_type,
            text=text,
            sha256=hashlib.sha256(data).hexdigest(),
        )

    def plan(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        client_profile: str = "",
        keep_recent_messages: int | None = None,
    ) -> ContextOffloadPlan:
        estimated = self.estimate_prompt_len(messages, tools=tools, client_profile=client_profile)
        if estimated <= self.settings.CONTEXT_INLINE_MAX_CHARS:
            return ContextOffloadPlan(mode="inline", inline_messages=messages, estimated_prompt_len=estimated)

        recent_count = self.settings.CONTEXT_INLINE_RECENT_MESSAGES if keep_recent_messages is None else keep_recent_messages
        recent_count = max(1, int(recent_count or 1))
        inline_messages = list(messages[-recent_count:]) if messages else []
        older_messages = list(messages[:-recent_count]) if messages and len(messages) > recent_count else []

        if not inline_messages:
            inline_messages = [{"role": "user", "content": SYSTEM_CONTEXT_PROMPT_NOTE}]
        else:
            last_inline = dict(inline_messages[-1])
            latest_text = self._extract_text(last_inline)
            if latest_text.strip():
                last_inline["content"] = f"{latest_text.strip()}\n\n{SYSTEM_CONTEXT_PROMPT_NOTE}"
            else:
                last_inline["content"] = SYSTEM_CONTEXT_PROMPT_NOTE
            inline_messages[-1] = last_inline

        serialized_parts: list[str] = []
        for idx, msg in enumerate(older_messages, 1):
            role = msg.get("role", "unknown")
            text = self._extract_text(msg)
            if not text.strip():
                continue
            serialized_parts.append(f"## Message {idx} [{role}]\n{text.strip()}\n")
        attachment_text = "\n".join(serialized_parts).strip()
        summary_text = attachment_text[:1200] if attachment_text else ""

        if estimated <= self.settings.CONTEXT_FORCE_FILE_MAX_CHARS:
            mode = "hybrid"
        else:
            mode = "file"

        generated_files: list[LocalContextFile] = []
        if attachment_text:
            generated_files.append(
                self._make_file(
                    f"{SYSTEM_CONTEXT_FILE_PREFIX}_history",
                    "txt",
                    attachment_text,
                    "text/plain",
                )
            )

        return ContextOffloadPlan(
            mode=mode,
            inline_messages=inline_messages,
            generated_files=generated_files,
            summary_text=summary_text,
            estimated_prompt_len=estimated,
            note=SYSTEM_CONTEXT_PROMPT_NOTE,
        )
