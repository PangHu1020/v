# 不足与 Phase-2 待办

按"已知不足"和"Phase-2 待办"两块整理。每条都标注影响面、代码位置、推荐解法。

---

## 1. Phase-1 已知不足

### 1.1 Bus 消费者无重试 / 失败语义粗糙

[backend/app/bus/consumer.py:_dispatch](../backend/app/bus/consumer.py) 当前的失败处理：handler 抛任何异常 → 推 DLQ + ack。

问题：

- 没有按错误类型区分（瞬时 vs 永久）。
- 没有 retry 上限或指数退避。
- LLM 短暂故障会让消息直接进 DLQ，客户体验不连贯。

**建议**：Phase-2 引入 `(error_type, retry_count)` 元数据；瞬时错误（`asyncio.TimeoutError`、`httpx.NetworkError`）走有限退避重试；永久错误（解析失败、4xx）直接 DLQ。

### 1.2 RedisCheckpointer 的 `alist` 排序按 checkpoint id 字典序

[backend/v/memory/checkpointer.py:alist](../backend/v/memory/checkpointer.py)

LangGraph checkpoint id 是 ULID 风格、单调递增字符串，所以字典序通常等同时间序。但**不保证**——某些 LangGraph 版本可能换 ID 算法。

**建议**：Phase-2 接入 Postgres 持久 checkpointer 时一并加 `created_at` 列做权威排序。

### 1.3 出站没有重试 + 失败可能丢消息

[backend/app/channels/wecom/outbound.py:send_text](../backend/app/channels/wecom/outbound.py)
[backend/app/channels/feishu/outbound.py:send_text](../backend/app/channels/feishu/outbound.py)

抛出后由 worker 的 try/except 捕获 → 推 DLQ。但**回执已经丢失**——客户没收到 AI 回复，且 ack 已经发出去。

**建议**：

- send_text 内部加 tenacity 重试（瞬时 5xx / 429 退避 3 次）。
- 仍失败后写入 `agent.outbound_dlq` 表（Phase-2 新建），由人工或定时任务复投。
- 客户长时间无回复时主动发"系统繁忙，请稍后重试"占位（避免静默失败）。

### 1.4 Token cache 仅进程内

[backend/app/channels/wecom/outbound.py:_access_token](../backend/app/channels/wecom/outbound.py)
[backend/app/channels/feishu/outbound.py:_tenant_access_token](../backend/app/channels/feishu/outbound.py)

多 worker / 多 pod 时每个进程独立刷新 token，存在重复请求和潜在竞态。

**建议**：Phase-2 多实例部署时将 token 缓存到 Redis（带分布式锁），让所有 worker 共享。

### 1.5 Debouncer 状态在重启后会丢

[backend/app/channels/debounce.py:Debouncer](../backend/app/channels/debounce.py) 用 Redis hash 存累积消息，但**定时器是 asyncio task 在内存里**。进程重启后：

- Redis 里残留了 pending 数据（pexpire 设了 4 倍窗口防泄漏）。
- 没有 task 来 flush 它，等到 expire 就自然消失。

**建议**：

- ① 启动时扫描所有 `debounce:*` key 并补发 flush task（小数据量，可接受）。
- ② 或者直接接受丢失这一段——客户重发即可，对话连续性靠长记忆兜底。

Phase-1 选 ②；Phase-2 多实例时考虑 ①。

### 1.6 Health 端点不区分 ready vs live

[backend/app/gateway/routers/health.py:health](../backend/app/gateway/routers/health.py) 只有一个端点。

K8s 习惯：

- liveness：进程活着即可（不查依赖）。
- readiness：所有依赖（PG + Redis + bus consumer task）都健康。

**建议**：Phase-2 拆 `/livez`（永远 200）+ `/readyz`（依赖检查）。

### 1.7 main.py lifespan 异常处理薄

[backend/app/main.py:lifespan](../backend/app/main.py)

PG 池或 Redis 创建失败时，FastAPI 会启动失败但错误日志可能淹没在 uvicorn 输出里。

**建议**：明确 fail-fast 日志 + 退出码；CI 加一个 `make startup-smoke` 验证 lifespan 能正常启停。

### 1.8 没有 metrics

仅有日志。生产需要：

- LLM 调用 latency p50/p95/p99
- Token 使用量（fallback 触发率）
- Bus 队列深度
- 出站失败率
- 会话时长分布

**建议**：Phase-2 加 Prometheus 指标（`prometheus-client`），暴露 `/metrics`。

### 1.9 没有日志聚合规范

`structlog` 输出 JSON 但没有定义"必须字段"清单。不同模块的字段名可能漂移（已经看到 `latency_ms` 在 LLMCaller 用 vs `latency` 在别处可能用）。

**建议**：定义 [doc/logging_schema.md](logging_schema.md)（待写）规范字段命名 + 必填项。

### 1.10 测试有少量 mock 边界粗糙

- [backend/test/unit/store/test_postgres.py](../backend/test/unit/store/test_postgres.py) 用 MagicMock 模拟 asyncpg，覆盖度高但不能保证 SQL 真正能执行——靠 `backend/test/e2e/test_migration.py` 的真实 PG 兜底。
- LLMCaller 测试只 mock 内部 chat model；没有验证 prompt 格式或 token 限制。
- 没有性能/压力测试。

**建议**：Phase-2 加：

- 一个 LLM eval 集（[backend/eval/](../backend/eval/) 目前是空目录）。
- 一个 bus throughput 测试（fakeredis 1k 消息 / 秒级别）。
- 长时间稳定性测试（连续 1 小时无内存泄漏）。

### 1.11 Schema 迁移无版本管理

直接 `psql -f` 重跑。如果 schema 变化，修改文件需要全部重跑（DROP TABLE IF EXISTS CASCADE）。开发期 OK，生产不行。

**建议**：Phase-2 引入 Alembic（[CLAUDE.md 已写](../CLAUDE.md) "起步先不用 Alembic"——schema 稳定后切）。

### 1.12 单租户假设零散在代码里

虽然每张 `agent.*` 表预留了 `tenant_id` 列，但代码层 channel 适配器、Settings、bus shard 都按"全局唯一一份"来构造（main.py lifespan）。

**建议**：Phase-2 多租户时需要：

- per-tenant 配置加载（按 webhook 来源识别租户）。
- per-tenant LLMCaller / outbound（不同租户可能配不同 API key）。
- per-tenant shard prefix（避免跨租户消息撞 key）。

---

## 2. Phase-2 待办（按优先级）

### P0：Slack 人工接管（handoff）

**为什么是 P0**：CLAUDE.md 锁定了 `transfer_to_human` 工具的实现路径（LangGraph `interrupt()`）。这是闭环最关键的容错出口；没有它，AI 答错时无法转人工，客服系统不能上线。

**范围**：

- [backend/app/operator/slack/](../backend/app/operator/) 新目录：
  - `signature.py`：Slack 请求签名（HMAC SHA256, X-Slack-Request-Timestamp + body）
  - `router.py`：`POST /operator/slack/events`（事件订阅 + button 回调）
  - `outbound.py`：`SlackOutbound`，发告警 + 转发人工回复
- [backend/v/tools/](../backend/v/tools/) 新目录：
  - `transfer_to_human.py`：`@tool` 装饰，调用 `interrupt()`
- [backend/v/memory/checkpointer.py](../backend/v/memory/checkpointer.py)：加 `migrate_hot_to_cold(thread_id)` / `migrate_cold_to_hot(thread_id)`
- [backend/v/hooks/](../backend/v/hooks/)：加 `on_interrupt` / `on_resume` 钩子
- [backend/v/agents/nodes.py](../backend/v/agents/nodes.py)：让 agent_node 处理 `transfer_to_human` 工具调用
- 更新 [backend/app/main.py](../backend/app/main.py) 装配 Slack 适配器

**复杂度**：中。LangGraph interrupt 文档清晰；难点在客户与人工消息的双向转发设计。

### P1：MCP Client + 商品状态 / 广告 MCP

**为什么是 P1**：用户明确"商品状态查询、广告"通过 MCP 接入。没有这些工具 agent 只能空谈。

**范围**：

- [backend/v/mcp/](../backend/v/mcp/) 新目录：
  - `client.py`：stdio + HTTP/SSE 双 transport 抽象
  - `auth.py`：OAuth 2.0（device flow）+ API Key
  - `cache.py`：5min in-mem L1 + 24h Redis L2，按 (tool_name, args_hash) 键
  - `registry.py`：从 settings 加载已配置的 MCP server，连接 + 健康检查
- [backend/v/tools/mcp_tool.py](../backend/v/tools/)：把每个 MCP tool 转成 LangChain `@tool`
- 写类工具（如下单 / 退货操作）在 cache 层 opt-out

**注意**：商品状态/广告的 MCP **server 端**不在本仓库范围。本仓库只做 client + 集成。

### P2：ARQ Cron + 三个主动场景

**为什么是 P2**：proactive 是产品差异化点，但客户不发就不触发，AI 答疑链可以独立工作。

**范围**：

- [backend/v/cron/](../backend/v/cron/) 新目录：
  - `worker.py`：ARQ 工作器入口
  - `tasks/logistics.py`：物流送达通知（事件触发，订阅业务侧消息总线 / DB watcher）
  - `tasks/ad_hoc_ad.py`：手动触发广告推送（admin API 入口）
  - `tasks/repurchase.py`：复购提醒（每天扫一次，找符合条件的客户）
  - `tasks/consolidate_session.py`：会话整合延迟任务（Phase-1 debounce 的反面——合并完整 session 写入 session_memory）
- 把 ARQ 共享 Redis（`ARQ_REDIS_URL` env）
- main.py lifespan 启动 ARQ worker（或独立进程，按部署形态选）

### P2：NL2SQL Subagent

**为什么是 P2**：CLAUDE.md 锁定了 `subagent` 工具的具体业务是 NL2SQL（查 dw + meta schema）。schema 已就绪，只缺 subagent 实现。

**范围**：

- [backend/v/tools/subagent.py](../backend/v/tools/)：`@tool` 装饰，调用一个独立的 LangGraph 子图
- 子图节点：
  - `schema_link_node`：查 `meta.table_info` / `column_info`，按用户问题召回相关表/列
  - `sql_gen_node`：LLMCaller 调 main_primary 生成 SQL（structured output 约束）
  - `sql_exec_node`：在 `dw` schema 上 read-only 执行
  - `format_node`：把结果格式化为自然语言
- [scripts/seed_meta.py](../scripts/) 新脚本：把 dw 5 张表的元信息批量写入 meta schema（一次性）

### P3：Memory Extractor + Recall Memory 工具

**为什么是 P3**：补全长期记忆写路径，让 AI 在跨会话时能记住客户偏好。

**范围**：

- [backend/v/memory/memory_extractor.py](../backend/v/memory/)（已占位）：
  - 从 session_memory 总结里抽取 user_profile 字段（结构化输出）
  - 写入 `agent.user_profile` JSONB；处理冲突 / 去重 / 过时
  - 抽取 episodes（向量化），写入 `agent.memory_episodes`
  - 用 DeepSeek flash + Pydantic structured output
- [backend/v/agents/background/summarizer.py](../backend/v/agents/background/) 新文件：
  - 接管 hooks 触发的 session 总结（context 超阈值 / 30min TTL 前 / on_session_end）
  - 写 `agent.session_memory` 然后调 memory_extractor
- [backend/v/tools/recall_memory.py](../backend/v/tools/)：
  - 输入查询文本 → embed → pgvector cosine 召回 top-K
  - 应用时间衰减加权
  - 返回最相关的 episodes 给 agent

### P4：Skill Loader

**为什么是 P4**：Skill 是 SOP 注入机制，能显著提升 AI 在特定业务（如退货流程）的准确度。但对核心闭环不是阻塞。

**范围**：

- [backend/v/skills/](../backend/v/skills/) 新目录：
  - `loader.py`：扫描 `SKILL_INTERNAL_REPO_PATH` + 已 whitelist 的 community 来源
  - `model.py`：Skill schema（markdown SOP / SKILL.md+exec）
  - `executor.py`：可执行脚本沙箱（subprocess + resource limits）
  - `registry.py`：内存索引 + 按 intent 召回
- [backend/v/agents/nodes.py](../backend/v/agents/nodes.py:enter_node)：根据当前消息 intent 注入相关 SOP 到 system prompt
- 信任边界：仓库内 + whitelist → 可执行；其余 → markdown only

### P5：可观测性 + 运维

- Prometheus metrics
- Grafana dashboard
- 日志结构规范文档
- liveness / readiness 拆分
- 启动 smoke 测试
- 多实例部署文档（Redis 共享 token、shard 数调整、ARQ worker 横向扩展）

### P6：性能 + 稳定性

- Bus throughput 基准测试
- LLM eval 集（提示词回归）
- 长时间稳定性（24h 连续 1 RPS）
- pgvector HNSW 调优（ef_construction、M）

### P7：多租户

按 §1.12 描述的范围逐项落实。

---

## 3. 决策记录

为了未来维护方便，以下"看似奇怪"的设计是有意为之，请勿"修复"：

| 决策 | 位置 | 原因 |
| --- | --- | --- |
| 出站不走 bus | [backend/app/CLAUDE.md](../backend/app/CLAUDE.md) | 入站流不被回执污染；出站语义本来就是直连 |
| Subagent 不抽象统一基类 | [backend/v/CLAUDE.md](../backend/v/CLAUDE.md) | "共享上下文 subagent" 与"独立上下文 subagent" 是两种不同事物，强行抽象退化为空接口 |
| 30min 静默后开新 session_id | [doc/data_flow.md §3](data_flow.md#3-session-生命周期) | 旧上下文大概率无关 + 浪费 token；连续性靠长记忆兜底 |
| RedisCheckpointer 是 Phase-1 唯一 checkpointer | [backend/v/memory/checkpointer.py](../backend/v/memory/checkpointer.py) | Slack handoff 才需要冷存储；Phase-1 没 handoff 就用不上 |
| Channel 层做 debounce 而非 bus 层 | [backend/app/CLAUDE.md](../backend/app/CLAUDE.md) | bus 是路由 + 并发控制层；语义合并属于平台适配 |
| 单 LLM provider（DeepSeek）+ 同家族降级 | [backend/v/CLAUDE.md](../backend/v/CLAUDE.md) | 跨家族降级会引入风格 / 协议差异，先单家族；多家族留待 Phase-2 真有需求时再开 |
