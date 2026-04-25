# Zhiyue-Bot Architecture Mapping (from MumuBot)

## 1) Layer Mapping

| MumuBot (Go) | Zhiyue-Bot (Python asyncio) | Notes |
|---|---|---|
| `main.go` | `main.py`, `core/runtime.py` | 异步启动、停止、组件装配 |
| `internal/agent/react_agent.go` | `core/agent.py` | ReAct 主循环、消息缓冲、调度、工具调用 |
| `internal/agent/concurrency.go` | `core/concurrency.py` | 群级任务并发与队列控制 |
| `internal/tools/*` | `plugins/*` | 业务工具插件化（统一注册器） |
| `internal/onebot/client.go` | `adapters/onebot/client.py` | OneBot 通信适配层 |
| `internal/mcp/manager.go` | `adapters/mcp/manager.py` | MCP 工具加载适配层 |
| `internal/llm/*` | `adapters/llm/*` | Chat/Embedding/Vision 模型适配层 |
| `internal/vector/milvus.go` | `adapters/vector/milvus.py` | 向量库适配层 |
| `internal/web/*` | `adapters/web/app.py` | 管理后台适配骨架 |
| `internal/config/config.go` | `internal/config/schema.py`, `internal/config/loader.py` | 配置模型与加载 |
| `internal/persona/persona.go` | `internal/persona/persona.py` | 人格与提示词 |
| `internal/memory/*` | `internal/memory/*` | 记忆模型与管理 |
| `internal/learning/learner.go` | `internal/learning/learner.py` | 后台学习循环 |
| `internal/jargon/manager.go` | `internal/jargon/manager.py` | 黑话管理器 |
| `internal/utils/ringbuffer.go` | `internal/utils/ring_buffer.py` | 环形缓冲 |

## 2) Tools -> Plugins Mapping

| MumuBot tools file | Zhiyue-Bot plugin file |
|---|---|
| `internal/tools/memory.go` | `plugins/memory.py` |
| `internal/tools/jargon.go` | `plugins/jargon.py` |
| `internal/tools/expression.go` | `plugins/expression.py` |
| `internal/tools/member.go` | `plugins/member.py` |
| `internal/tools/interaction.go` | `plugins/interaction.py` |
| `internal/tools/sticker.go` | `plugins/sticker.py` |
| `internal/tools/mood.go` | `plugins/mood.py` |
| `internal/tools/style_classification.go` | `plugins/style_classification.py` |
| `internal/tools/tools.go` (群信息类工具) | `plugins/group_info.py` |
| `internal/tools/tools.go` (`request_get`) | `plugins/web_request.py` |
| `internal/tools/eino_hooks.go` | `plugins/hooks.py` |
| `internal/tools/tools.go` (ToolContext) | `plugins/context.py` |

## 3) Runtime Structure

- `core/`：Core 引擎层（Agent、并发调度、装配）
- `adapters/`：Adapter 层（OneBot/LLM/MCP/Web/Milvus）
- `internal/`：Internal 领域层（配置、人格、记忆、学习、黑话、工具类）
- `plugins/`：业务功能层（映射 MumuBot 的 tools）

## 4) Compatibility Shims

为减少迁移期改动，保留了兼容入口：

- `internal/agent/agent.py` -> re-export `core/agent.py`
- `internal/llm/client.py` -> re-export `adapters/llm/chat.py`
- `internal/config/config.py` -> re-export 新配置接口
