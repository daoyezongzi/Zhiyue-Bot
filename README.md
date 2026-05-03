# Zhiyue-Bot

Zhiyue-Bot 是一个基于 OneBot + LLM 的异步群聊 Agent。  
Zhiyue-Bot is an async group-chat agent built on OneBot + LLM.

## 功能 / Features

1. 人格与情绪驱动回复  
Persona- and mood-driven replies with dynamic style adjustment.

2. 消息调度队列 + 500ms 去抖  
Async message queue with 500ms debounce to merge burst messages.

3. LLM 重试与降级链路  
LLM retry and fallback chain (Primary -> Fallback) on timeout/429/connection errors.

4. 三位一体记忆架构  
Three-layer memory architecture:
- 短期记忆：会话级近期对话窗口  
Short-term memory: session-local recent turns
- 长期记忆：`user_memories` 向量集合  
Long-term memory: `user_memories` vector collection
- 外部知识：`external_knowledge` 向量集合  
External knowledge: `external_knowledge` vector collection

5. 记忆代谢（异步沉淀）  
Async memory metabolizer:
- 达到阈值或检测到话题切换时触发  
Triggers on threshold or topic shift
- 自动总结并写入长期记忆  
Summarizes and stores into long-term memory
- 保留最近 3 条短期对话保证上下文连续  
Keeps last 3 turns for local continuity

6. RAG 检索注入  
RAG retrieval injection into prompt:
- 历史背景（往事）  
Historical background (past events)
- 相关知识（外部知识）  
Related knowledge (external knowledge)

7. 话题系统完整闭环  
Topic loop (assignment -> summary -> archive -> recall):
- 群消息自动归属到进行中话题，低信号消息可保持 no-topic  
Auto-assign group messages to active topics while keeping low-signal messages as no-topic
- 达到阈值后自动更新结构化摘要（标题/摘要/事实/未完事项/关键词）  
Auto-refresh structured summaries after threshold
- 按活跃度与时效自动归档，并在后续对话中召回历史话题  
Auto-archive by activity/timeliness and recall archived topics for future turns

8. 管理后台（可选）  
Optional admin web dashboard:
- 查看状态、重置运行态  
View runtime status and reset state
- 更新 system prompt  
Update system prompt
- 配置背景图  
Update dashboard background

9. 详细可观测日志  
Detailed observability logs for enqueue/debounce/retrieval/fallback/reply.

## 架构概览 / Architecture

- `core/agent.py`: 核心调度与 think/reply 流程  
Core scheduling and think/reply pipeline
- `adapters/llm/chat.py`: Chat 请求、重试、模型降级  
Chat requests, retries, and model fallback
- `adapters/llm/embedding.py`: Embedding 请求与限流/异常处理  
Embedding requests with rate-limit/error handling
- `internal/memory/memory_manager.py`: 记忆代谢与 RAG 检索编排  
Memory metabolizer and RAG orchestration
- `internal/memory/vector_storage.py`: ChromaDB 多集合向量存储  
Multi-collection ChromaDB vector storage
- `internal/topic/manager.py`: 话题归属、摘要、归档、召回闭环  
Topic assignment, summary, archive and recall loop

## 快速开始 / Quick Start

1. 安装依赖 / Install dependencies

```bash
pip install -r requirements.txt
```

2. 准备配置 / Prepare config

```bash
cp .env.example .env
cp config/config.yaml.example config/config.yaml
```

`config/persona.prompt` 为静态人格模板文件（已从 `config.yaml` 拆分）。  
`config/persona.prompt` is the static persona template (moved out from `config.yaml`).

3. 填写关键项 / Fill required fields
- `LLM_API_KEY`（或 `LLM_PROVIDER + <PROVIDER>_API_KEY`）  
`LLM_API_KEY` (or `LLM_PROVIDER + <PROVIDER>_API_KEY`)
- `ONEBOT_WS_URL` 与 `ONEBOT_WS_MODE`  
`ONEBOT_WS_URL` and `ONEBOT_WS_MODE`
- 至少一个启用群组（`config/config.yaml` 的 `groups`）  
At least one enabled group in `config/config.yaml`.
- 人格正文请直接编辑 `config/persona.prompt`（支持 `{{Name}}`、`{{QQ}}`、`{{Interests}}`、`{{AliasNames}}`、`{{StyleLine}}` 占位符）  
Edit `config/persona.prompt` for persona text (supports placeholders above).

4. 启动 / Run

```bash
python main.py
```

## 关键配置 / Key Config Notes

### `PERSONA_MASTER_ID` 是什么？

`PERSONA_MASTER_ID` 对应 OneBot 上报事件中的 `user_id`。  
For OneBot, `PERSONA_MASTER_ID` is matched against incoming `user_id`.

在 QQ 场景下，它就是目标用户的 QQ 号（数字 ID / UIN）。  
In QQ scenarios, this is the target QQ numeric ID (UIN).

### 记忆相关配置 / Memory-related

- `MEMORY_CHROMA_PATH`: ChromaDB 持久化目录 / ChromaDB persistence path
- `MEMORY_SHORT_TERM_THRESHOLD`: 触发沉淀的短期条数阈值 / short-term threshold
- `MEMORY_SHORT_TERM_KEEP_LAST`: 沉淀后保留条数 / turns kept after metabolism
- `MEMORY_TOPIC_SHIFT_THRESHOLD`: 话题切换灵敏度 / topic-shift sensitivity
- `MEMORY_RAG_TOP_K`: 每次检索条数 / top-k retrieval size
- `MEMORY_TOPIC_ENABLED`: 是否启用话题闭环 / enable topic loop
- `MEMORY_TOPIC_STORE_PATH`: 话题持久化文件路径 / topic store file path
- `MEMORY_TOPIC_MAX_ACTIVE_PER_GROUP`: 每群最大活跃话题数 / max active topics per group
- `MEMORY_TOPIC_SUMMARY_TRIGGER_MESSAGES`: 触发摘要更新的消息阈值 / summary trigger threshold

## 依赖说明 / Dependency Notes

- 若安装了 `chromadb`，将启用持久化向量存储。  
With `chromadb` installed, persistent vector storage is enabled.
- 若 `chromadb` 不可用，系统会回退到内存模式（开发可用，重启不持久）。  
If unavailable, it falls back to in-memory mode (works for dev, non-persistent).

## 许可证 / License

当前仓库未显式声明许可证，请按项目维护者要求使用。  
No explicit license is declared in this repository; follow maintainer policy.


参考：https://github.com/SugarMGP/MumuBot
使用py语言重构了该项目的核心内容。日后会按照我的想法进行更新。
感谢白糖大大的mumubot 做的真的很棒很棒！

## Memory Governance (新增)

新增配置：
- `MEMORY_STORE_PATH`
- `MEMORY_AUTO_INGEST_ENABLED`
- `MEMORY_CONVERGENCE_INTERVAL_MINUTES`
- `MEMORY_CANDIDATE_GRACE_HOURS`
- `MEMORY_CANDIDATE_PROMOTE_EVIDENCE`

新增后台 API：
- `GET /api/memories`
- `GET /api/memories/{memory_id}`
- `POST /api/memories`
- `POST /api/memories/{memory_id}/archive`
- `POST /api/memories/{memory_id}/activate`
- `POST /api/memories/{memory_id}/candidate`
- `DELETE /api/memories/{memory_id}`
- `POST /api/memories/convergence`
