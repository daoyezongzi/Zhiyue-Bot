# 2026-05-04 更新总结：话题闭环 + 记忆治理 + Web 对接

## 1. 本次目标
- 对齐 MumuBot 的核心思路，补齐纸月的“话题系统闭环”和“长期记忆治理闭环”。
- 让治理能力可视化、可操作：在后台 API 与 Dashboard 中直接查看与控制。

## 2. 已落地能力

### 2.1 话题系统闭环（Topic Loop）
- 新增 `internal/topic/manager.py`，支持：
  - 话题归属（复用/新建）
  - 话题摘要刷新
  - 话题归档与历史召回
  - 运行时快照与管理查询
- Agent 消息流已接入：
  - 用户消息入话题
  - 助手回复入话题
  - Prompt 构建阶段注入当前话题与相关历史话题
- Web API 已接入：
  - `GET /api/topics`
  - `GET /api/topics/{topic_id}`
  - `POST /api/topics/{topic_id}/archive`
  - `POST /api/topics/{topic_id}/activate`
  - `POST /api/topics/{topic_id}/summary/refresh`

### 2.2 长期记忆治理闭环（Memory Governance）
- 记忆模型升级（`internal/memory/models.py`）：
  - 增加 `canonical_type / status / evidence_count / source_kind / source_ref / fact_key / updated_at / user_id`
  - 状态规范：`active / candidate / archived / legacy`
- 记忆管理器升级（`internal/memory/memory_manager.py`）：
  - 候选记忆摄取与低信号过滤
  - 同槽位（`fact_key`）去重/增强（`evidence_count` 累计）
  - 候选晋升（candidate -> active）
  - 冲突归档（同 key 的旧 active 自动归档）
  - 周期收敛任务（候选超时归档、满足条件晋升）
  - JSON 持久化（重启后保持治理状态）
  - 召回时仅放行 `active/legacy`（降噪）
- Agent 已接入治理：
  - 用户消息、助手回复都会进入治理入口
  - 管理状态增加 `long_term_memory` 快照

### 2.3 Tool Call 观测增强
- 保留并完善了工具调用日志能力（结构化记录、分页查询、过滤、清理、统计）。
- 后台 API：
  - `GET /api/tool-calls`
  - `GET /api/tool-calls/{tool_call_id}`
  - `DELETE /api/tool-calls/{tool_call_id}`
  - `POST /api/tool-calls/clear`

## 3. 新增/扩展的记忆治理 API
- `GET /api/memories`
- `GET /api/memories/{memory_id}`
- `POST /api/memories`
- `POST /api/memories/{memory_id}/archive`
- `POST /api/memories/{memory_id}/activate`
- `POST /api/memories/{memory_id}/candidate`
- `DELETE /api/memories/{memory_id}`
- `POST /api/memories/convergence`

## 4. 配置项扩展
- `memory_store_path`
- `memory_auto_ingest_enabled`
- `memory_convergence_interval_minutes`
- `memory_candidate_grace_hours`
- `memory_candidate_promote_evidence`
- `tool_call_store_path`
- `tool_call_max_entries`

对应示例与加载链路已同步：
- `internal/config/schema.py`
- `internal/config/loader.py`
- `config/config.yaml.example`
- `.env.example`

## 5. Dashboard 对接
- 在 `web_ui/dashboard.html` 增加“长期记忆治理”面板：
  - 治理快照展示（总量、状态分布、类型分布、规则）
  - 长期记忆列表刷新
  - 手动触发收敛

## 6. 影响文件（核心）
- `internal/topic/__init__.py`
- `internal/topic/manager.py`
- `internal/memory/models.py`
- `internal/memory/memory_manager.py`
- `internal/memory/__init__.py`
- `core/agent.py`
- `adapters/web/admin_service.py`
- `internal/config/schema.py`
- `internal/config/loader.py`
- `config/config.yaml.example`
- `.env.example`
- `web_ui/dashboard.html`

## 7. 验证
- 已执行：`python -m compileall core adapters internal`
- 结果：通过。
