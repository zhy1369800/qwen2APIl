# 上下文转文件与工具结果截断配置

本文档说明 qwen2API 中与历史上下文转文件、agent/tools 请求上下文保留，以及 Tool Result inline 截断相关的环境变量。

这些配置主要用于解决长任务场景中的两个问题：

1. 历史消息过长导致 prompt 膨胀、上游截断或模型遗忘早期上下文。
2. 工具结果过短截断导致模型看不到关键结果，从而反复调用同一个工具。

## 一、上下文转文件相关配置

### `CONTEXT_INLINE_MAX_CHARS`

默认值：

```env
CONTEXT_INLINE_MAX_CHARS=4000
```

用途：

用于估算当前请求的上下文长度。若估算长度小于等于该值，则全部上下文保持 inline，不触发历史上下文转文件。

超过该值时，系统会尝试将较早历史消息生成 `qwen2api_context*.txt` 上下文文件，并作为文件附件上传到 Qwen 上游。

注意：

- 当前估算会考虑 `messages` 和 `tools` 的部分长度。
- 但真正转文件的内容只来自较早的 `messages`，不会把 tools schema 或 skill 定义转入文件。

生产建议：

```env
CONTEXT_INLINE_MAX_CHARS=6000
```

如果更保守，可使用默认值：

```env
CONTEXT_INLINE_MAX_CHARS=4000
```

测试上下文转文件能力时，可以临时调低：

```env
CONTEXT_INLINE_MAX_CHARS=100
```

---

### `CONTEXT_FORCE_FILE_MAX_CHARS`

默认值：

```env
CONTEXT_FORCE_FILE_MAX_CHARS=10000
```

用途：

用于区分 `hybrid` 和 `file` 模式：

```text
estimated <= CONTEXT_INLINE_MAX_CHARS
  -> inline

CONTEXT_INLINE_MAX_CHARS < estimated <= CONTEXT_FORCE_FILE_MAX_CHARS
  -> hybrid

estimated > CONTEXT_FORCE_FILE_MAX_CHARS
  -> file
```

当前实现中，`hybrid` 和 `file` 的实际处理差异较小，主要都是：

```text
最近 N 条消息保留 inline
更早历史消息转为上下文文件
```

生产建议：

```env
CONTEXT_FORCE_FILE_MAX_CHARS=16000
```

如果希望更保守，可继续使用默认值：

```env
CONTEXT_FORCE_FILE_MAX_CHARS=10000
```

测试时可以临时调低：

```env
CONTEXT_FORCE_FILE_MAX_CHARS=200
```

---

### `CONTEXT_ATTACH_WITH_TOOLS`

默认值：

```env
CONTEXT_ATTACH_WITH_TOOLS=false
```

用途：

控制带 `tools` 的 agent 请求是否允许自动生成历史上下文文件。

- `false`：默认不开启。带 tools 的请求不自动生成历史上下文文件，行为更接近传统 inline 模式。
- `true`：带 tools 的请求也允许把较早历史消息转为上下文文件，并挂载到 Qwen 上游请求的 `files` 字段中。

推荐默认关闭的原因：

- agent/tools 请求对最近工具状态非常敏感。
- 文件附件读取依赖 Qwen 上游文件解析能力。
- `tools + files` 同时启用在不同上游状态下可能有稳定性差异。

生产建议：

如果尚未充分验证文件附件稳定性：

```env
CONTEXT_ATTACH_WITH_TOOLS=false
```

如果已验证 tools 场景下 `qwen2api_context*.txt` 能被 Qwen 读取，可以灰度开启：

```env
CONTEXT_ATTACH_WITH_TOOLS=true
```

---

### `CONTEXT_INLINE_RECENT_MESSAGES`

默认值：

```env
CONTEXT_INLINE_RECENT_MESSAGES=6
```

用途：

触发上下文转文件时，保留最近 N 条 `messages` 在 inline prompt 中，更早历史消息转为上下文文件。

这样可以保证：

- 当前任务仍在 inline 中。
- 最近的 `tool_use` / `tool_result` 仍在 inline 中。
- 模型不会因为最近工具状态被转入文件而重复调用工具。

示例：

```env
CONTEXT_INLINE_RECENT_MESSAGES=6
```

表示：

```text
最近 6 条消息 inline
更早消息 -> qwen2api_context.txt
```

生产建议：

普通场景：

```env
CONTEXT_INLINE_RECENT_MESSAGES=6
```

长 agent 任务较多、工具调用链较长时：

```env
CONTEXT_INLINE_RECENT_MESSAGES=8
```

仅测试附件读取能力时可临时设置：

```env
CONTEXT_INLINE_RECENT_MESSAGES=1
```

不建议生产使用 `1`，因为它可能导致最近工具状态被移出 inline。

## 二、Tool Result inline 截断相关配置

历史上下文转文件只处理较早的 `messages`。最近 N 条消息仍会保留 inline，因此这些 inline 消息中的 Tool Result 仍需要长度控制，避免 prompt 被单条工具结果撑爆。

### `TOOL_RESULT_INLINE_MAX_CHARS`

默认值：

```env
TOOL_RESULT_INLINE_MAX_CHARS=6000
```

用途：

控制带 `tools` 请求中，标准 `role=tool` 工具结果最多保留多少字符到 inline prompt。

原先该值硬编码为 `6000`，现在可通过环境变量调整。

生产建议：

```env
TOOL_RESULT_INLINE_MAX_CHARS=8000
```

如果上游延迟、token 压力明显，可以使用默认值或降低：

```env
TOOL_RESULT_INLINE_MAX_CHARS=6000
```

如果模型仍因看不到完整工具结果而反复调用工具，可适度提高：

```env
TOOL_RESULT_INLINE_MAX_CHARS=10000
```

---

### `TOOL_RESULT_INLINE_NO_TOOLS_MAX_CHARS`

默认值：

```env
TOOL_RESULT_INLINE_NO_TOOLS_MAX_CHARS=300
```

用途：

控制无 `tools` 请求中，被识别为 Tool Result 的内容最多 inline 多少字符。

生产建议：

如果无 tools 场景很少包含工具结果，可保持默认：

```env
TOOL_RESULT_INLINE_NO_TOOLS_MAX_CHARS=300
```

如果客户端偶尔不传 tools 但 messages 里仍包含工具结果，建议调高：

```env
TOOL_RESULT_INLINE_NO_TOOLS_MAX_CHARS=1000
```

---

### `TOOL_MESSAGE_INLINE_MAX_CHARS`

默认值：

```env
TOOL_MESSAGE_INLINE_MAX_CHARS=6000
```

用途：

控制带 `tools` 请求中，历史 user message 内被识别为 Tool Result 的文本最多 inline 多少字符。

它和 `TOOL_RESULT_INLINE_MAX_CHARS` 的区别：

- `TOOL_RESULT_INLINE_MAX_CHARS`：处理标准 `role=tool` 消息。
- `TOOL_MESSAGE_INLINE_MAX_CHARS`：处理被包装在 `role=user` 消息中的工具结果，例如 `[Tool Result]...[/Tool Result]` 或 JSON results。

生产建议：

建议和 `TOOL_RESULT_INLINE_MAX_CHARS` 保持一致：

```env
TOOL_MESSAGE_INLINE_MAX_CHARS=8000
```

如果 prompt 压力较大，可保持默认：

```env
TOOL_MESSAGE_INLINE_MAX_CHARS=6000
```

## 三、对话历史消息截断长度配置

为避免长任务或携带大参数的工具消息在历史对话中被过早截断损坏，系统提供了以下环境变量来灵活控制各角色消息的历史保留截断上限：

### `ASSISTANT_MESSAGE_CLAUDE_INLINE_MAX_CHARS`

默认值：

```env
ASSISTANT_MESSAGE_CLAUDE_INLINE_MAX_CHARS=500
```

用途：

在 `tools` 启用且使用 Claude 兼容模式（`CLAUDE_CODE_OPENAI_PROFILE`）时，限制非工具调用的普通助手回复最大保留字符数。

### `ASSISTANT_MESSAGE_INLINE_MAX_CHARS`

默认值：

```env
ASSISTANT_MESSAGE_INLINE_MAX_CHARS=1400
```

用途：

在普通或非 Claude 兼容模式下，限制非工具调用的普通助手回复的最大保留字符数。在未启用 `tools` 的普通对话中也生效。

### `USER_MESSAGE_INLINE_MAX_CHARS`

默认值：

```env
USER_MESSAGE_INLINE_MAX_CHARS=1600
```

用途：

在 `tools` 启用时，限制用户普通输入（非工具结果）的最大保留字符数。

### `TOOL_CALL_INLINE_MAX_CHARS`

默认值：

```env
TOOL_CALL_INLINE_MAX_CHARS=8000
```

用途：

专门用于保护助手回复中包含 `##TOOL_CALL##` 的工具调用（例如写入文件等携带大量参数的消息）。设置该上限为较大的值（例如 8000 字符以上）可以有效防止工具调用参数被截断损坏，避免 JSON 结构破碎以及由此引发的重复工具调用。

---

## 四、推荐配置组合

### 保守稳定版

适合刚上线或不希望 tools 请求受到附件机制影响的场景。

```env
CONTEXT_INLINE_MAX_CHARS=4000
CONTEXT_FORCE_FILE_MAX_CHARS=10000
CONTEXT_ATTACH_WITH_TOOLS=false
CONTEXT_INLINE_RECENT_MESSAGES=6

TOOL_RESULT_INLINE_MAX_CHARS=6000
TOOL_RESULT_INLINE_NO_TOOLS_MAX_CHARS=1000
TOOL_MESSAGE_INLINE_MAX_CHARS=6000
```

特点：

- tools 请求不自动转上下文文件。
- 风险最低。
- 工具结果截断仍然可配置。

---

### 推荐灰度版

适合已验证 `tools + qwen2api_context.txt` 能被 Qwen 上游读取的场景。

```env
CONTEXT_INLINE_MAX_CHARS=6000
CONTEXT_FORCE_FILE_MAX_CHARS=16000
CONTEXT_ATTACH_WITH_TOOLS=true
CONTEXT_INLINE_RECENT_MESSAGES=6

TOOL_RESULT_INLINE_MAX_CHARS=8000
TOOL_RESULT_INLINE_NO_TOOLS_MAX_CHARS=1000
TOOL_MESSAGE_INLINE_MAX_CHARS=8000
```

特点：

- tools 请求也可自动转较早历史为文件。
- 最近 6 条消息仍然 inline，保护当前任务和工具状态。
- 工具结果保留更多内容，降低重复调用工具概率。

---

### 激进长任务版

适合长 agent 任务较多，并且确认附件读取、parse、上游稳定性都较好的场景。

```env
CONTEXT_INLINE_MAX_CHARS=8000
CONTEXT_FORCE_FILE_MAX_CHARS=20000
CONTEXT_ATTACH_WITH_TOOLS=true
CONTEXT_INLINE_RECENT_MESSAGES=8

TOOL_RESULT_INLINE_MAX_CHARS=10000
TOOL_RESULT_INLINE_NO_TOOLS_MAX_CHARS=1200
TOOL_MESSAGE_INLINE_MAX_CHARS=10000
```

特点：

- 更适合复杂多轮工具调用。
- inline 保留更多最近状态。
- 工具结果更不容易被过早截断。
- 成本是 prompt 更大、延迟可能更高。

---

### 附件读取测试配置

仅用于验证 Qwen 是否能读取自动生成的上下文文件，不建议生产使用。

```env
CONTEXT_INLINE_MAX_CHARS=100
CONTEXT_FORCE_FILE_MAX_CHARS=200
CONTEXT_ATTACH_WITH_TOOLS=true
CONTEXT_INLINE_RECENT_MESSAGES=1

TOOL_RESULT_INLINE_MAX_CHARS=6000
TOOL_RESULT_INLINE_NO_TOOLS_MAX_CHARS=300
TOOL_MESSAGE_INLINE_MAX_CHARS=6000
```

测试方法：

1. 在较早历史消息中放入唯一暗号。
2. 最新用户消息只问“请根据历史上下文回答暗号是什么”，不要包含暗号本身。
3. 如果模型能回答暗号，说明上下文文件大概率被 Qwen 成功读取。

## 四、实现注意事项

### 1. 转文件内容来源

上下文文件保存的是客户端传入的原始 `payload["messages"]` 中较早的消息。

不会保存：

- `prompt_builder` 截断后的内容。
- tools schema。
- skill 定义。
- 最终拼接后的完整 prompt。
- 工具协议说明。

### 2. tools 和 skill 不会被转文件

`tools` 只参与长度估算，不会被写入 `qwen2api_context.txt`。

skill 如果是通过 tools 注入，也不会被转文件。

只有当客户端自己把 skill 文本放进较早 `messages` 中时，它才会被当作普通历史消息转入文件。

### 3. 文件链接无法访问不代表失败

上传到 Qwen 后返回的 URL 通常是 OSS 临时签名链接，可能因过期、签名、STS token、复制截断、Referer 或权限策略导致浏览器无法打开。

判断文件是否可被 Qwen 使用，应优先看：

- `files/parse` 是否成功。
- `parse_status` 是否为 `success`。
- 上游 chat payload 的 `files` 字段是否包含 `qwen2api_context*.txt`。
- 模型是否能回答只存在于历史文件中的内容。

### 4. 不建议完全取消 Tool Result 截断

历史转文件解决的是“旧上下文太长”的问题；Tool Result 截断解决的是“当前 inline 工具结果太长”的问题。

即使开启上下文转文件，最近 N 条消息仍然 inline，其中可能包含很长的工具结果。因此仍需要保留截断阈值，只是现在可以通过环境变量调节。
