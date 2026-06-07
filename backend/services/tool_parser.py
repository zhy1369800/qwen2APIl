import json
import logging
import re
import uuid
from typing import Any, cast

from backend.adapter.standard_request import CLAUDE_CODE_OPENAI_PROFILE, OPENCLAW_OPENAI_PROFILE
from backend.core.request_logging import get_request_context
from backend.services.tool_arg_fixer import fix_tool_call_arguments
from backend.services.tool_name_obfuscation import from_qwen_name
from backend.toolcall.normalize import build_tool_name_registry, normalize_tool_name
from backend.toolcall.parser import parse_tool_calls_detailed

__all__ = ["parse_tool_calls", "parse_tool_calls_detailed", "inject_format_reminder", "parse_tool_calls_silent", "ToolSieve"]

log = logging.getLogger("qwen2api.tool_parser")


CASE_SENSITIVE_TOOL_NAMES = {"Bash", "Edit", "Write", "Read", "Grep", "Glob", "WebFetch", "WebSearch"}


def _normalize_tool_name_case(name: str, tool_names: set[str]) -> str:
    if not isinstance(name, str) or not name:
        return name
    if name in tool_names:
        return name
    lowered = name.lower()
    for candidate in tool_names:
        if candidate.lower() == lowered:
            if candidate in CASE_SENSITIVE_TOOL_NAMES:
                return candidate
            return candidate
    return name


def _find_tool_use_json(text: str, tool_names: set[str]):
    i = 0
    while i < len(text):
        pos = text.find('{', i)
        if pos == -1:
            break
        depth = 0
        for j in range(pos, len(text)):
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[pos:j + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict) and obj.get("type") == "tool_use" and obj.get("name"):
                            normalized_name = normalize_tool_name(obj.get("name", ""), tool_names)
                            if normalized_name in tool_names:
                                obj = dict(obj)
                                obj["name"] = normalized_name
                                return pos, obj

                    except (json.JSONDecodeError, ValueError):
                        pass
                    break
        i = pos + 1

    return None


def _extract_first_xml_tool_call(text: str) -> str | None:
    wrapped_match = re.search(r"<tool_calls>\s*(<tool_call>[\s\S]*?</tool_call>)\s*</tool_calls>", text, re.IGNORECASE)
    if wrapped_match:
        return wrapped_match.group(1)

    tool_call_match = re.search(r"<tool_call>\s*(\{[\s\S]*?\}|[\s\S]*?)\s*</tool_call>", text, re.IGNORECASE)
    if tool_call_match:
        return tool_call_match.group(0)
    return None


def _extract_invoke_xml_tool_call(answer: str) -> tuple[str, dict[str, Any], str] | None:
    m = re.search(r"<invoke\s+name=\"([^\"]+)\"\s*>([\s\S]*?)</invoke>", answer, re.IGNORECASE)
    if not m:
        return None
    raw_name = m.group(1).strip()
    body = m.group(2) or ""
    params: dict[str, Any] = {}
    for pm in re.finditer(r"<parameter\s+name=\"([^\"]+)\"\s*>([\s\S]*?)</parameter>", body, re.IGNORECASE):
        key = pm.group(1).strip()
        val = pm.group(2).strip()
        if key:
            params[key] = val
    prefix = answer[:m.start()].strip()
    return raw_name, params, prefix


def _extract_function_xml_tool_call(answer: str) -> tuple[str, dict[str, Any], str] | None:
    m = re.search(r"<function=([A-Za-z0-9_.:-]+)\s*>([\s\S]*?)</function>", answer, re.IGNORECASE)
    if not m:
        return None
    raw_name = m.group(1).strip()
    body = m.group(2) or ""
    params: dict[str, Any] = {}
    # 兼容 <parameter name="k">v</parameter>
    for pm in re.finditer(r"<parameter\s+name=\"([^\"]+)\"\s*>([\s\S]*?)</parameter>", body, re.IGNORECASE):
        key = pm.group(1).strip()
        val = pm.group(2).strip()
        if key:
            params[key] = val
    # 兼容简写 <path>...</path> / <command>...</command> 等
    for tm in re.finditer(r"<([A-Za-z_][A-Za-z0-9_]*)>\s*([\s\S]*?)\s*</\1>", body, re.IGNORECASE):
        key = tm.group(1).strip()
        if key.lower() == "parameter":
            continue
        val = tm.group(2).strip()
        if key and val and key not in params:
            params[key] = val
    prefix = answer[:m.start()].strip()
    return raw_name, params, prefix


def _extract_first_json_tool_call(text: str) -> str | None:
    normalized = text.strip()

    # 优先查找完整的 JSON 对象
    # markers 按优先级：Qwen 官方 tool_calls 外层包装 > 单对象 > 松散片段
    markers = [
        '<tool_call>{"name"',
        '<tool_calls><tool_call>{"name"',
        '{"tool_calls"',
        '{"name"',
        '"name":',
        '"name="',
        'function.name:',
    ]
    start_positions = [normalized.find(marker) for marker in markers if normalized.find(marker) != -1]
    if not start_positions:
        return None
    start = min(start_positions)
    candidate = normalized[start:]

    wrapped_match = re.search(r"<tool_calls>\s*(<tool_call>[\s\S]*?</tool_call>)\s*</tool_calls>", candidate, re.IGNORECASE)
    if wrapped_match:
        return wrapped_match.group(1)

    tool_call_match = re.search(r"<tool_call>\s*(\{[\s\S]*?\}|[\s\S]*?)\s*</tool_call>", candidate, re.IGNORECASE)
    if tool_call_match:
        return tool_call_match.group(0)

    json_start = candidate.find("{")
    if json_start == -1:
        return None
    depth = 0
    for idx in range(json_start, len(candidate)):
        ch = candidate[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                json_str = candidate[json_start:idx + 1]
                # 验证是否是有效的工具调用 JSON
                try:
                    obj = json.loads(json_str)
                    if isinstance(obj, dict) and "name" in obj:
                        return json_str
                except (json.JSONDecodeError, ValueError):
                    pass
                return json_str
    return candidate[json_start:]


def _normalize_fragmented_tool_call(answer: str) -> str:
    text = answer.strip()
    if "##TOOL_CALL##" in text and "##END_CALL##" in text:
        return text

    extracted_tool_call = _extract_first_xml_tool_call(text) or _extract_first_json_tool_call(text)
    if extracted_tool_call:
        return extracted_tool_call

    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Tool\s+[A-Za-z0-9_.:-]*\s*does not exists?\\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```[\s\S]*?```", "", text)

    extracted_tool_call = _extract_first_xml_tool_call(text) or _extract_first_json_tool_call(text)
    if extracted_tool_call:
        return extracted_tool_call

    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[•●·\-*]+\s*", "", line)
        line = line.replace("END_CALL##", "##END_CALL##")
        if line:
            lines.append(line)

    normalized = "\n".join(lines)
    if "TOOL_CALL##" in normalized and "##TOOL_CALL##" not in normalized:
        normalized = normalized.replace("TOOL_CALL##", "##TOOL_CALL##")
    if "##END_CALL##" in normalized and "##TOOL_CALL##" not in normalized and '"name"' in normalized:
        normalized = f"##TOOL_CALL##\n{normalized}"
    return normalized


def _coerce_tool_input(name: str, input_data: Any, tools: list[dict[str, Any]]) -> Any:
    if not isinstance(input_data, dict):
        return input_data

    # 修正 AskUserQuestion 工具参数
    if name == "AskUserQuestion":
        fixed = dict(input_data)

        # 如果只有 question 字段，转换为 questions 数组
        if "question" in fixed and "questions" not in fixed:
            question_text = fixed.pop("question")
            fixed["questions"] = [{
                "question": question_text,
                "header": "Question",
                "options": [
                    {"label": "Yes", "description": "Confirm"},
                    {"label": "No", "description": "Decline"}
                ],
                "multiSelect": False
            }]
            log.info(f"[ToolCoerce] Fixed AskUserQuestion: converted 'question' to 'questions' array")

        # 确保 questions 是数组
        if "questions" in fixed:
            if not isinstance(fixed["questions"], list):
                fixed["questions"] = [fixed["questions"]]

            # 验证每个问题的格式
            for i, q in enumerate(fixed["questions"]):
                if not isinstance(q, dict):
                    continue

                # 确保有必需字段
                if "question" not in q:
                    q["question"] = "Please provide your input"
                if "header" not in q:
                    q["header"] = "Question"
                if "multiSelect" not in q:
                    q["multiSelect"] = False

                # 确保 options 格式正确
                if "options" not in q:
                    q["options"] = [
                        {"label": "Continue", "description": "Proceed"},
                        {"label": "Cancel", "description": "Stop"}
                    ]
                elif isinstance(q.get("options"), list):
                    for j, opt in enumerate(q["options"]):
                        if isinstance(opt, str):
                            q["options"][j] = {"label": opt, "description": opt}
                        elif isinstance(opt, dict):
                            if "label" not in opt:
                                opt["label"] = opt.get("description", f"Option {j+1}")
                            if "description" not in opt:
                                opt["description"] = opt.get("label", "")

        return fixed

    # 修正 Agent 工具参数
    if name == "Agent":
        fixed = dict(input_data)
        if "description" not in fixed:
            fixed["description"] = "Execute sub-task"
        if "prompt" not in fixed:
            fixed["prompt"] = fixed.get("description", "Execute the task")
        return fixed

    # 修正 Read 工具参数
    if name == "Read":
        fixed = dict(input_data)
        if "file_path" not in fixed:
            if "path" in fixed:
                fixed["file_path"] = fixed.pop("path")
            elif "filename" in fixed:
                fixed["file_path"] = fixed.pop("filename")
        return fixed

    # 修正 Bash 工具参数
    if name == "Bash":
        fixed = dict(input_data)
        if "command" not in fixed:
            if "cmd" in fixed:
                fixed["command"] = fixed.pop("cmd")
            elif "script" in fixed:
                fixed["command"] = fixed.pop("script")
        return fixed

    # 原有的 query/queries 转换逻辑
    query_value = input_data.get("query")
    queries = input_data.get("queries")
    if query_value or "queries" not in input_data:
        return input_data
    if not any(isinstance(tool, dict) and isinstance(tool.get("parameters"), dict) and isinstance(tool["parameters"].get("properties"), dict) and "query" in tool["parameters"]["properties"] for tool in tools):
        return input_data

    if isinstance(queries, list):
        merged = "\n".join(str(item).strip() for item in queries if str(item).strip())
        if merged:
            coerced = dict(input_data)
            coerced.pop("queries", None)
            coerced["query"] = merged
            return coerced
    if isinstance(queries, str) and queries.strip():
        coerced = dict(input_data)
        coerced.pop("queries", None)
        coerced["query"] = queries.strip()
        return coerced

    return input_data


def parse_tool_calls(answer: str, tools: list):
    return _parse_tool_calls(answer, tools, emit_logs=True)


def parse_tool_calls_silent(answer: str, tools: list):
    return _parse_tool_calls(answer, tools, emit_logs=False)


def parse_bracket_tool_call(text: str) -> dict | None:
    m = re.search(r'\[Tool\s*Call:\s*([A-Za-z0-9_.-]+)\s*with\s*(.*?)\]', text, re.IGNORECASE)
    if not m:
        return None
    name = m.group(1)
    args_str = m.group(2)
    
    kv_pattern = r'([A-Za-z0-9_.-]+)\s*=\s*("[^"]*"|\'[^\']*\'|[^\s,]+)'
    pairs = re.findall(kv_pattern, args_str)
    
    inputs = {}
    for k, v in pairs:
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        else:
            if v.isdigit():
                v = int(v)
            else:
                try:
                    v = float(v)
                except ValueError:
                    pass
        inputs[k] = v
    return {"name": name, "input": inputs, "start": m.start(), "end": m.end()}


def parse_new_bracket_tool_call(text: str) -> dict | None:
    m = re.search(r'\[Tool\s*Call\]\s*([A-Za-z0-9_.-]+)\s*\n*(\{.*)', text, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    name = m.group(1).strip()
    json_part = m.group(2).strip()
    
    depth = 0
    json_end = -1
    for idx, ch in enumerate(json_part):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                json_end = idx
                break
    if json_end != -1:
        json_str = json_part[:json_end + 1]
        try:
            inputs = json.loads(json_str)
            if isinstance(inputs, dict):
                return {"name": name, "input": inputs, "start": m.start(), "end": m.start() + len(text) - len(json_part) + json_end + 1}
        except Exception:
            pass
    return None


def parse_xml_tool_code(text: str) -> dict | None:
    m = re.search(r'<tool_code>\s*(.*?)\s*</tool_code>', text, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    content = m.group(1).strip()
    
    # 匹配 function_name(...) 形式
    fn_match = re.match(r'^([A-Za-z0-9_.-]+)\s*\((.*)\)$', content, re.DOTALL)
    if not fn_match:
        return None
    name = fn_match.group(1)
    args_str = fn_match.group(2).strip()
    
    # 匹配 key = value 键值对
    # 支持带引号的字符串参数以及数字、布尔、None 值
    kv_pattern = r'([A-Za-z0-9_.-]+)\s*=\s*("[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\'|[^\s,]+)'
    pairs = re.findall(kv_pattern, args_str)
    
    inputs = {}
    for k, v in pairs:
        v = v.strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v_content = v[1:-1]
            # 简单的反转义
            v_content = v_content.replace('\\"', '"').replace("\\'", "'").replace('\\n', '\n').replace('\\t', '\t')
            inputs[k] = v_content
        else:
            if v.lower() == 'true':
                inputs[k] = True
            elif v.lower() == 'false':
                inputs[k] = False
            elif v.lower() == 'none':
                inputs[k] = None
            elif v.isdigit():
                inputs[k] = int(v)
            else:
                try:
                    inputs[k] = float(v)
                except ValueError:
                    inputs[k] = v
    return {"name": name, "input": inputs, "start": m.start(), "end": m.end()}


def _parse_tool_calls(answer: str, tools: list, *, emit_logs: bool):
    answer = _normalize_fragmented_tool_call(answer)
    ctx = get_request_context()
    req_tag = f"req={ctx.get('req_id', '-')} chat={ctx.get('chat_id', '-')}"
    if not tools:
        return [{"type": "text", "text": answer}], "end_turn"
    tool_names = {t.get("name") for t in tools if t.get("name")}
    tool_registry = build_tool_name_registry(tool_names)

    def _log_debug(message: str) -> None:
        if emit_logs:
            log.debug(message)

    def _log_info(message: str) -> None:
        if emit_logs:
            log.info(message)

    def _log_warning(message: str) -> None:
        if emit_logs:
            log.warning(message)

    # 强制记录原始输入用于调试（但遵守 emit_logs 开关：ToolSieve 流式解析每 chunk 都调一次，
    # 若无条件记录会刷 1000+ 行 [ToolParse]——只在 finalize/诊断场景打印）
    if emit_logs:
        log.info(f"[ToolParse] [{req_tag}] 原始回复({len(answer)}字): {answer[:500]!r}")

    def _make_tool_block(name, input_data, prefix=""):
        # 入站反混淆：Qwen 返回的别名（ReadX）→ 客户端原名（Read）。
        # 未知别名原样返回，不影响 Qwen 直接返回原名的兼容路径。
        name = from_qwen_name(name)
        normalized_name = normalize_tool_name(name, tool_registry.values())
        cased_name = _normalize_tool_name_case(normalized_name, tool_names)
        if tool_names and cased_name not in tool_names:
            _log_warning(f"[ToolParse] 工具名不匹配，返回未注册工具调用: name={name!r}, normalized={normalized_name!r}, cased={cased_name!r}, tools={tool_names}")
            # We do NOT return as text here anymore. We return it as a tool_use block so execution.py can explicitly block and retry it.
        coerced_input = _coerce_tool_input(cased_name, input_data, tools)
        # 智能引号修复 + Edit/StrReplace 的 old_string fuzzy 修复
        coerced_input = fix_tool_call_arguments(cased_name, coerced_input)
        tool_id = f"toolu_{uuid.uuid4().hex[:8]}"
        blocks = []
        if prefix:
            blocks.append({"type": "text", "text": prefix})
        blocks.append({"type": "tool_use", "id": tool_id, "name": cased_name, "input": coerced_input})
        _log_info(f"[ToolParse] 返回工具块: original={name!r}, normalized={normalized_name!r}, final={cased_name!r}, input={json.dumps(coerced_input, ensure_ascii=False)[:200]}")
        return blocks, "tool_use"

    bracket_call = parse_bracket_tool_call(answer)
    if bracket_call:
        name = bracket_call["name"]
        inp = bracket_call["input"]
        prefix = answer[:bracket_call["start"]].strip()
        if emit_logs:
            log.info(f"[ToolParse] [{req_tag}] ✓ Bracket格式 [Tool Call]: name={name!r}, input={str(inp)[:120]}")
        return _make_tool_block(name, inp, prefix)

    new_bracket_call = parse_new_bracket_tool_call(answer)
    if new_bracket_call:
        name = new_bracket_call["name"]
        inp = new_bracket_call["input"]
        prefix = answer[:new_bracket_call["start"]].strip()
        if emit_logs:
            log.info(f"[ToolParse] [{req_tag}] ✓ 新Bracket格式 [Tool Call] name\\n{{...}}: name={name!r}, input={str(inp)[:120]}")
        return _make_tool_block(name, inp, prefix)

    xml_tool_code = parse_xml_tool_code(answer)
    if xml_tool_code:
        name = xml_tool_code["name"]
        inp = xml_tool_code["input"]
        prefix = answer[:xml_tool_code["start"]].strip()
        if emit_logs:
            log.info(f"[ToolParse] [{req_tag}] ✓ XML格式 <tool_code>: name={name!r}, input={str(inp)[:120]}")
        return _make_tool_block(name, inp, prefix)

    detailed = parse_tool_calls_detailed(answer, tool_names)
    detailed_calls = cast(list[dict[str, Any]], detailed["calls"])
    if detailed_calls:
        first_call = detailed_calls[0]
        _log_info(f"[ToolParse] ✓ 详细解析格式: source={detailed['source']}, name={first_call['name']!r}, input={json.dumps(first_call['input'], ensure_ascii=False)[:200]}")
        return _make_tool_block(first_call["name"], first_call["input"])

    tc_m = re.search(r'##TOOL_CALL##\s*(.*?)\s*##END_CALL##', answer, re.DOTALL | re.IGNORECASE)
    if tc_m:
        try:
            obj = json.loads(tc_m.group(1))
            name = obj.get("name", "")
            inp = obj.get("input", obj.get("args", obj.get("arguments", obj.get("parameters", {}))))
            if isinstance(inp, str):
                try:
                    inp = json.loads(inp)
                except Exception:
                    inp = {"value": inp}
            prefix = answer[:tc_m.start()].strip()
            _log_info(f"[ToolParse] ✓ ##TOOL_CALL## 格式: name={name!r}, input={str(inp)[:120]}")
            return _make_tool_block(name, inp, prefix)
        except (json.JSONDecodeError, ValueError) as e:
            _log_warning(f"[ToolParse] ##TOOL_CALL## 格式解析失败: {e}, content={tc_m.group(1)[:100]!r}")

    xml_m = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', answer, re.DOTALL | re.IGNORECASE)
    if xml_m:
        try:
            obj = json.loads(xml_m.group(1))
            name = obj.get("name", "")
            inp = obj.get("input", obj.get("args", obj.get("arguments", obj.get("parameters", {}))))
            if isinstance(inp, str):
                try:
                    inp = json.loads(inp)
                except Exception:
                    inp = {"value": inp}
            prefix = answer[:xml_m.start()].strip()
            _log_info(f"[ToolParse] ✓ XML格式 <tool_call>: name={name!r}, input={str(inp)[:120]}")
            return _make_tool_block(name, inp, prefix)
        except (json.JSONDecodeError, ValueError) as e:
            _log_warning(f"[ToolParse] XML格式解析失败: {e}, content={xml_m.group(1)[:100]!r}")

    invoke_xml = _extract_invoke_xml_tool_call(answer)
    if invoke_xml:
        name, inp, prefix = invoke_xml
        _log_info(f"[ToolParse] ✓ XML格式 <invoke>: name={name!r}, input={str(inp)[:120]}")
        return _make_tool_block(name, inp, prefix)

    function_xml = _extract_function_xml_tool_call(answer)
    if function_xml:
        name, inp, prefix = function_xml
        _log_info(f"[ToolParse] ✓ XML格式 <function=...>: name={name!r}, input={str(inp)[:120]}")
        return _make_tool_block(name, inp, prefix)

    cb_m = re.search(r'```tool_call\s*\n(.*?)\n```', answer, re.DOTALL)
    if cb_m:
        try:
            obj = json.loads(cb_m.group(1).strip())
            name = obj.get("name", "")
            inp = obj.get("input", obj.get("args", {}))
            if isinstance(inp, str):
                try:
                    inp = json.loads(inp)
                except Exception:
                    inp = {"value": inp}
            prefix = answer[:cb_m.start()].strip()
            _log_info(f"[ToolParse] ✓ 代码块格式 tool_call: name={name!r}, input={str(inp)[:120]}")
            return _make_tool_block(name, inp, prefix)
        except (json.JSONDecodeError, ValueError) as e:
            _log_warning(f"[ToolParse] 代码块格式解析失败: {e}")

    stripped = re.sub(r'```json\s*\n?', '', answer)
    stripped = re.sub(r'\n?```', '', stripped)
    result = _find_tool_use_json(stripped, tool_names)
    if result:
        pos, tool_call = result
        prefix = stripped[:pos].strip()
        tool_id = tool_call.get("id") or f"toolu_{uuid.uuid4().hex[:8]}"
        _log_info(f"[ToolParse] ✓ 旧JSON格式 tool_call: name={tool_call['name']!r}")
        blocks = []
        if prefix:
            blocks.append({"type": "text", "text": prefix})
        blocks.append({
            "type": "tool_use",
            "id": tool_id,
            "name": tool_call["name"],
            "input": _coerce_tool_input(tool_call["name"], tool_call.get("input", {}), tools),
        })
        return blocks, "tool_use"

    # 尝试解析纯 JSON 格式: {"name": "...", "input": {...}}
    stripped_clean = stripped.strip()
    try:
        if stripped_clean.startswith('{') and stripped_clean.endswith('}'):
            obj = json.loads(stripped_clean)
            if isinstance(obj, dict) and "name" in obj:
                name = obj.get("name", "")
                inp = obj.get("input", obj.get("args", obj.get("arguments", obj.get("parameters", {}))))
                if isinstance(inp, str):
                    try:
                        inp = json.loads(inp)
                    except Exception:
                        inp = {"value": inp}
                _log_info(f"[ToolParse] ✓ 纯JSON格式: name={name!r}, input={str(inp)[:120]}")
                return _make_tool_block(name, inp)
    except (json.JSONDecodeError, ValueError) as e:
        _log_debug(f"[ToolParse] 纯JSON格式解析失败: {e}, content={stripped_clean[:200]!r}")

    _log_warning(f"[ToolParse] ✗ 未检测到工具调用，作为普通文本返回。工具列表: {tool_names}")
    return [{"type": "text", "text": answer}], "end_turn"


class ToolSieve:
    """工具调用流式检测器 - 实时检测并分离工具调用"""

    def __init__(self, tool_names: list[str]):
        self.tool_names = set(tool_names) if tool_names else set()
        self.pending = ""
        self.capture = ""
        self.capturing = False
        self.pending_tool_calls = []
        self.tool_calls_detected = False
        self.args_is_quoted_string = False
        self.has_skipped_first_quote = False
        self.stream_active = False
        self.args_started = False

    def process_chunk(self, chunk: str) -> list[dict]:
        """
        处理一个chunk，返回事件列表
        事件类型：
        - {"type": "content", "text": "..."}  # 普通文本
        - {"type": "tool_calls", "calls": [...]}  # 工具调用
        """
        if not chunk:
            return []

        self.pending += chunk
        events = []

        # 如果正在捕获工具调用
        if self.capturing:
            self.capture += chunk
            self.pending = ""

            # 1. 识别工具名并激活流式
            if not getattr(self, "stream_active", False):
                m_name = re.search(r'"name"\s*:\s*"([^"]+)"', self.capture)
                xml_invoke_name = None
                xml_function_name = None
                xml_tool_code_name = None
                bracket_tool_name = None
                if not m_name:
                    m_invoke = re.search(r"<invoke\s+name=\"([^\"]+)\"", self.capture, re.IGNORECASE)
                    if m_invoke:
                        xml_invoke_name = m_invoke.group(1)
                    else:
                        m_function = re.search(r"<function=([A-Za-z0-9_.:-]+)\s*>", self.capture, re.IGNORECASE)
                        if m_function:
                            xml_function_name = m_function.group(1)
                        else:
                            m_tool_code = re.search(r"<tool_code>\s*([A-Za-z0-9_.-]+)", self.capture, re.IGNORECASE)
                            if m_tool_code:
                                xml_tool_code_name = m_tool_code.group(1)
                            else:
                                m_bracket = re.search(r'\[Tool\s*Call\]\s*([A-Za-z0-9_.-]+)', self.capture, re.IGNORECASE)
                                if m_bracket:
                                    bracket_tool_name = m_bracket.group(1)
                if m_name:
                    raw_name = m_name.group(1)
                elif xml_invoke_name or xml_function_name or xml_tool_code_name or bracket_tool_name:
                    raw_name = xml_invoke_name or xml_function_name or xml_tool_code_name or bracket_tool_name
                else:
                    raw_name = None
                    
                if raw_name:
                    from backend.services.tool_name_obfuscation import from_qwen_name
                    from backend.toolcall.normalize import normalize_tool_name
                    self.stream_tool_name = normalize_tool_name(from_qwen_name(raw_name), self.tool_names)
                    self.stream_active = True
                    self.stream_brace_depth = 0
                    self.stream_completed = False
                    
                    if xml_tool_code_name or (self.tool_names and self.stream_tool_name not in self.tool_names):
                        self.stream_ignored = True
                    else:
                        self.stream_ignored = False
                        
                    if not getattr(self, "stream_ignored", False):
                        import uuid
                        self.stream_tool_id = f"toolu_{uuid.uuid4().hex[:8]}"
                        events.append({
                            "type": "tool_calls_start",
                            "calls": [{"type": "tool_call_stream_start", "id": self.stream_tool_id, "name": self.stream_tool_name}]
                        })

            # 2. 定位并提取 arguments 的开始
            if getattr(self, "stream_active", False) and not getattr(self, "args_started", False):
                is_bracket_layout = "[tool call]" in self.capture.lower()
                if is_bracket_layout:
                    brace_pos = self.capture.find("{")
                    if brace_pos != -1:
                        self.args_started = True
                        args_start_str = self.capture[brace_pos:]
                    else:
                        args_start_str = ""
                else:
                    m_input = re.search(r'"(?:input|arguments|args)"\s*:\s*(.*)', self.capture, re.DOTALL)
                    if m_input:
                        self.args_started = True
                        args_start_str = m_input.group(1)
                    else:
                        args_start_str = ""

                if getattr(self, "args_started", False) and args_start_str:
                    cleaned_start = args_start_str.lstrip()
                    if cleaned_start.startswith('"'):
                        self.args_is_quoted_string = True
                    
                    clean_args = ""
                    for ch in args_start_str:
                        if self.stream_brace_depth == 0 and ch not in "{[\"":
                            continue
                        
                        if self.args_is_quoted_string and ch == '"' and self.stream_brace_depth == 0 and not self.has_skipped_first_quote:
                            self.has_skipped_first_quote = True
                            continue
                            
                        clean_args += ch
                        if ch in "{[":
                            self.stream_brace_depth += 1
                        elif ch in "}]":
                            self.stream_brace_depth -= 1
                            if self.stream_brace_depth == 0:
                                self.stream_completed = True
                                break
                                
                    if clean_args and not getattr(self, "stream_ignored", False):
                        if self.args_is_quoted_string:
                            clean_args = clean_args.replace('\\"', '"')
                            if self.stream_completed and clean_args.endswith('"'):
                                clean_args = clean_args[:-1]
                        if clean_args:
                            events.append({
                                "type": "tool_calls_chunk",
                                "calls": [{"type": "tool_call_stream_chunk", "arguments": clean_args}]
                            })

            # 3. 提取 arguments 的后续增量
            elif getattr(self, "stream_active", False) and getattr(self, "args_started", False) and not getattr(self, "stream_completed", False):
                clean_args = ""
                for ch in chunk:
                    if self.stream_brace_depth == 0 and ch not in "{[\"":
                        continue
                    
                    if self.args_is_quoted_string and ch == '"' and self.stream_brace_depth == 0 and not self.has_skipped_first_quote:
                        self.has_skipped_first_quote = True
                        continue
                        
                    clean_args += ch
                    if ch in "{[":
                        self.stream_brace_depth += 1
                    elif ch in "}]":
                        self.stream_brace_depth -= 1
                        if self.stream_brace_depth == 0:
                            self.stream_completed = True
                            break
                            
                if clean_args and not getattr(self, "stream_ignored", False):
                    if self.args_is_quoted_string:
                        clean_args = clean_args.replace('\\"', '"')
                        if self.stream_completed and clean_args.endswith('"'):
                            clean_args = clean_args[:-1]
                    if clean_args:
                        events.append({
                            "type": "tool_calls_chunk",
                            "calls": [{"type": "tool_call_stream_chunk", "arguments": clean_args}]
                        })
            # --- END INCREMENTAL STREAMING LOGIC ---

            # 针对 tool_code 的闭合检测
            if getattr(self, "stream_active", False) and not getattr(self, "stream_completed", False):
                if "</tool_code>" in self.capture.lower():
                    self.stream_completed = True

            # 尝试解析
            prefix, calls, suffix, ready = self._consume_tool_capture()

            if ready and calls:
                # 解析成功
                if prefix and not getattr(self, "stream_active", False):
                    events.append({"type": "content", "text": prefix})

                # If we streamed, we should NOT emit full `tool_calls` again to avoid duplicate!
                # Wait, `execution.py` expects `tool_calls` to trigger execution!
                # We can just emit it, but `execution.py` should execute it without translator emitting it again.
                # Actually, `execution.py` will catch `type: tool_calls` and run the tool.
                # Translator handles `type: tool_calls` by doing `self.emit_tool_calls(tool_calls)`, which would emit a duplicate tool block.
                # So we can set a flag `emitted_stream = True` to tell `execution.py` or `translator` to skip emitting the full block.
                # Let's just add `is_full_duplicate: True` to the final `tool_calls` event.
                
                self.pending_tool_calls = calls
                self.tool_calls_detected = True
                self.pending = suffix
                self.capture = ""
                self.capturing = False
                self.stream_active = False

            return events

        # 检测工具调用开始
        start = self._find_tool_start(self.pending)

        if start >= 0:
            # 找到工具调用开始
            prefix = self.pending[:start]
            if prefix:
                events.append({"type": "content", "text": prefix})

            self.capture = self.pending[start:]
            self.pending = ""
            self.capturing = True
            events.append({"type": "tool_detected"})
        else:
            # 没找到，输出安全部分
            safe, hold = self._split_safe_content(self.pending)
            if safe:
                events.append({"type": "content", "text": safe})
            self.pending = hold

        return events

    def _find_tool_start(self, text: str) -> int:
        """查找工具调用开始位置"""
        markers = [
            '{"tool_calls"',
            '{"name":',
            '<tool_call>',
            '<tool_code>',
            '<invoke name=',
            '<function=',
            '##TOOL_CALL##',
            'function.name:',
            '[Tool Call:',
            '[tool call:',
            '[Tool Call]',
            '[tool call]',
        ]

        positions = []
        for marker in markers:
            pos = text.find(marker)
            if pos >= 0:
                positions.append(pos)

        return min(positions) if positions else -1

    def _consume_tool_capture(self) -> tuple[str, list, str, bool]:
        """尝试解析捕获的工具调用"""
        if not self.capture:
            return "", [], "", False

        # 尝试解析工具调用
        try:
            # 使用现有的解析逻辑
            blocks, stop_reason = parse_tool_calls_silent(self.capture,
                [{"name": name} for name in self.tool_names])

            if stop_reason == "tool_use":
                # 找到工具��用
                tool_blocks = [b for b in blocks if b.get("type") == "tool_use"]
                if tool_blocks:
                    # 转换为标准格式
                    calls = [{
                        "name": tb["name"],
                        "input": tb["input"]
                    } for tb in tool_blocks]

                    # 提取前缀文本
                    text_blocks = [b for b in blocks if b.get("type") == "text"]
                    prefix = text_blocks[0]["text"] if text_blocks else ""

                    return prefix, calls, "", True
        except Exception as e:
            log.debug(f"[ToolSieve] 解析失败: {e}")

        # 还不完整或解析失败
        return "", [], "", False

    def _split_safe_content(self, text: str) -> tuple[str, str]:
        """分离安全内容和需要保留的部分"""
        # 保留最后几个字符，防止工具调用标记被截断
        if len(text) < 20:
            return "", text

        return text[:-10], text[-10:]

    def _clean_args_chunk(self, chunk: str) -> str:
        """清理流式参数块中的结尾标记"""
        if not chunk:
            return ""
        if "##END" in chunk:
            chunk = chunk[:chunk.find("##END")]
        if "END_CALL" in chunk:
            chunk = chunk[:chunk.find("END_CALL")]
        if "</tool" in chunk:
            chunk = chunk[:chunk.find("</tool")]
        return chunk

    def flush(self) -> list[dict]:
        """刷新剩余内容"""
        events = []

        if self.pending_tool_calls:
            events.append({"type": "tool_calls", "calls": self.pending_tool_calls})
            self.pending_tool_calls = []

        if self.capturing and self.capture:
            # 尝试最后一次解析
            prefix, calls, suffix, ready = self._consume_tool_capture()
            if ready and calls:
                if prefix:
                    events.append({"type": "content", "text": prefix})
                events.append({"type": "tool_calls", "calls": calls})
                self.tool_calls_detected = True
                if suffix:
                    events.append({"type": "content", "text": suffix})
            else:
                # 解析失败，检查是否看起来像工具调用
                if not self._looks_like_incomplete_tool_call(self.capture):
                    events.append({"type": "content", "text": self.capture})

        if self.pending:
            events.append({"type": "content", "text": self.pending})

        return events

    def _looks_like_incomplete_tool_call(self, text: str) -> bool:
        """检查文本是否看起来像不完整的工具调用"""
        markers = ['{"tool_calls"', '{"name":', '<tool_call>', '<tool_code>', '<tool_code', '<invoke name=', '<function=', '##TOOL_CALL##', 'function.name:']
        return any(marker in text for marker in markers)

    def has_tool_calls(self) -> bool:
        """是否检测到工具调用"""
        return self.tool_calls_detected or bool(self.pending_tool_calls)


def inject_format_reminder(prompt: str, tool_name: str, *, client_profile: str = OPENCLAW_OPENAI_PROFILE, blocked_tool: str | None = None) -> str:
    """Inject a format correction reminder into the prompt before the final 'Assistant:' tag.
    Used when upstream produced the toxic 'Tool X does not exists.' hallucination —
    the reminder teaches the model to emit the text-marker format without that phrase."""
    # 如果有被屏蔽的幻觉工具（如 code_interpreter），在 reminder 里明确禁止
    blocked_line = ""
    if blocked_tool and blocked_tool != tool_name:
        blocked_line = (
            f"\u26a0\ufe0f STRICTLY FORBIDDEN: Do NOT use '{blocked_tool}' \u2014 it is NOT available here.\n"
            f"\u26a0\ufe0f \u4e25\u683c\u7981\u6b62\uff1a\u4e0d\u8981\u4f7f\u7528 '{blocked_tool}'\uff0c\u8be5\u5de5\u5177\u5728\u5f53\u524d\u73af\u5883\u4e0d\u53ef\u7528\u3002\n"
        )
    if client_profile == CLAUDE_CODE_OPENAI_PROFILE:
        reminder = (
            f"{blocked_line}"
            "[CORRECTION / \u7ea0\u6b63]: Your previous output used a forbidden tool.\n"
            f"要调用 {tool_name}，只输出这个精确格式，不要有其他文本：\n"
            f"To invoke {tool_name}, output ONLY this exact format with NO other text:\n"
            "##TOOL_CALL##\n"
            f'{{"name": "{tool_name}", "input": {{"arg1": "value1", "arg2": "value2"}}}}\n'
            "##END_CALL##\n\n"
            "ABSOLUTELY FORBIDDEN in your next output:\n"
            "- Any disclaimer about a tool being unavailable, missing, or unregistered\n"
            "- Any sentence claiming you are unable to run a function\n"
            "- Any apology for failing to invoke something\n"
            "These ##TOOL_CALL## blocks are plain TEXT MARKERS the proxy parses — not native function calls.\n"
        )
    else:
        reminder = (
            "[CORRECTION / 纠正]: 请用正确的 ##TOOL_CALL## 格式重新发起调用。\n"
            "You MUST use ##TOOL_CALL## format and NOTHING ELSE:\n"
            "##TOOL_CALL##\n"
            f'{{"name": {json.dumps(tool_name)}, "input": {{...your args here...}}}}\n'
            "##END_CALL##\n"
            "不要输出任何声称无法执行工具的话。The ##TOOL_CALL## blocks are TEXT MARKERS, not native functions.\n"
        )
    prompt = prompt.rstrip()
    if prompt.endswith("Assistant:"):
        return prompt[: -len("Assistant:")] + reminder + "\nAssistant:"
    return prompt + "\n\n" + reminder + "\nAssistant:"


