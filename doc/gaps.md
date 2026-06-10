# 不足与 Phase-3 待办

按"已知不足"和"Phase-3 待办"两块整理。每条都标注影响面、代码位置、推荐解法。

---

## 1. 当前已知不足

### 1.1 Bus 消费者无重试 / 失败语义粗糙

[backend/app/bus/consumer.py:_dispatch](../backend/app/bus/consumer.py) 当前的失败处理：handler 抛任何异常 → 推 DLQ + ack。

问题：

- 没有按错误类型区分（瞬时 vs 永久）。
- 没有 retry 上限或指数退避。
- LLM 短暂故障会让消息直接进 DLQ，客户体验不连贯。

**建议**：引入 `(error_type, retry_count)` 元数据；瞬时错误（`asyncio.TimeoutError`、`httpx.NetworkError`）走有限退避重试；永久错误（解析失败、4xx）直接 DLQ。

### 1.2 RedisCheckpointer.alist 排序按 checkpoint id 字典序

[backend/v/agents/checkpointer.py:alist](../backend/v/agents/checkpointer.py)

LangGraph checkpoint id 是 ULID 风格、单调递增字符串，所以字典序通常等同时间序。但**不保证**——某些 LangGraph 版本可能换 ID 算法。

**建议**：给 RedisCheckpointer 写入一份 `created_at` 索引（ZSET），`alist` 用 ZSET 分页代替 hash 全扫描。

### 1.3 出站没有重试 + 失败可能丢消息

[backend/app/wecom_aibot/outbound.py:send_text](../backend/app/wecom_aibot/outbound.py)
~~`SlackOutbound.post_handoff_alert`~~ (已删除)

抛出后由 worker 的 try/except 捕获 → 推 DLQ。但**回执已经丢失**——客户没收到 AI 回复，且 ack 已经发出去。

**修改建议**：

- send_text 内部加 tenacity 重试（瞬时 5xx / 429 退避 3 次）。
- 仍失败后写入 `agent.outbound_dlq` 表（新建），由人工或定时任务复投。
- 客户长时间无回复时主动发"系统繁忙"占位（避免静默失败）。

### 1.4 ~~Token cache 仅进程内~~（已不适用）

WeCom HTTP webhook 和 Feishu 已移除；WeCom 智能机器人 WS 不需要 access_token 刷新。此条目已过时，无需处理。

### 1.5 Debouncer 状态在重启后会丢

[backend/app/wecom_aibot/debounce.py:Debouncer](../backend/app/wecom_aibot/debounce.py) 用 Redis hash 存累积消息，但**定时器是 asyncio task 在内存里**。进程重启后：

- Redis 里残留 pending 数据（pexpire 设了 4 倍窗口防泄漏）。
- 没有 task 来 flush，等到 expire 就自然消失。

**建议**：

- ① 启动时扫描所有 `debounce:*` key 并补发 flush task（小数据量可接受）。
- ② 接受丢失，客户重发即可，连续性靠长记忆兜底。

当前选 ②。多实例部署时考虑 ①。

### 1.6 Health 端点不区分 ready vs live ✅（已收敛）

[backend/app/gateway/routers/health.py](../backend/app/gateway/routers/health.py) 现在拆成三个：

- `/livez` — 永远 `200`，依赖故障**不**翻牌（避免 k8s crash-loop）。
- `/readyz` — 依赖全 ok 时 `200`，任一挂掉返回 `503`，body 里报 down。
- `/health` — 老接口，永远 `200`，body 里携带 down，向后兼容。

### 1.7 main.py lifespan 异常处理薄

[backend/app/main.py:lifespan](../backend/app/main.py)

PG 池或 Redis 创建失败时，FastAPI 会启动失败但错误日志可能淹没在 uvicorn 输出里。

**建议**：明确 fail-fast 日志 + 退出码；CI 加一个 `make startup-smoke` 验证 lifespan 能正常启停。

### 1.8 没有 metrics

仅有日志。生产需要：

- LLM 调用 latency p50/p95/p99
- Token 使用量、fallback 触发率
- Bus 队列深度
- 出站失败率
- 会话时长分布
- recall_memory 调用频率 + 命中率

**建议**：Phase-3 加 `prometheus-client`，暴露 `/metrics`。

### 1.9 没有日志聚合规范

`structlog` 输出 JSON 但没有定义"必须字段"清单。模块多了之后字段名容易漂移（`latency_ms` vs `latency`，`session_id` vs `thread_id`）。

**建议**：写 `doc/logging_schema.md` 规范字段命名 + 必填项 + 事件名约定。

### 1.10 测试有少量 mock 边界粗糙

- LLMCaller 测试只 mock 内部 chat model；没有验证 prompt 格式或 token 限制。
- 没有性能/压力测试。
- consolidate_session / extract_session_memory 走 mock LLM，无 prompt 回归。

**建议**：

- LLM eval 集（[backend/eval/](../backend/eval/) 目前是空目录）。
- 一个 bus throughput 测试（fakeredis 1k 消息 / 秒级别）。
- 长时间稳定性测试（连续 1 小时无内存泄漏）。
- prompt regression：consolidate_session 输出的 SessionSummary 应在固定输入上保持稳定结构。

### 1.11 Schema 迁移无版本管理

直接 `psql -f` 重跑。如果 schema 变化，修改文件需要全部重跑（DROP TABLE IF EXISTS CASCADE）。开发期 OK，生产不行。

**建议**：引入 Alembic（[根 CLAUDE.md](../CLAUDE.md) "起步先不用 Alembic"——schema 稳定后切）。LangGraph 的 PG checkpointer 自带 `checkpoint_migrations` 表管理它自己的演进，不与我们的 schema 冲突。

### 1.12 单租户假设零散在代码里

虽然每张 `agent.*` 表预留了 `tenant_id` 列，但代码层 channel 适配器、Settings、bus shard 都按"全局唯一一份"来构造。

**建议**：多租户时需要：

- per-tenant 配置加载（按 webhook 来源识别租户）。
- per-tenant LLMCaller / outbound（不同租户可能配不同 API key）。
- per-tenant shard prefix（避免跨租户消息撞 key）。
- `agent.session.tenant_id` 进 PK 索引。

### ~~1.13 LangGraph ToolNode Command(resume) 边界 bug~~（已不适用，handoff 层已删除）

### 1.14 Proactive 消息 → 父会话注入还没有 drainer

[backend/v/cron/proactive.py:record_proactive](../backend/v/cron/proactive.py) 把每条主动消息记到 `proactive_log:{ch}:{u}` 列表，但 `on_session_start` 还没接 drainer——客户下次回复时，AI 看不到刚才系统主动推过什么。

**当前规避**：客户在 IM 客户端能看到主动消息，对话上下文靠人脑桥接。

**建议**：on_session_start 增加：

```python
proactive = await redis.lrange(f"proactive_log:{ch}:{u}", 0, -1)
if proactive:
    state["messages"].extend(AIMessage(content=...) for each)
    await redis.delete(...)
```

### ~~1.15 MCPServerConfig.api_key 是明文 env 字符串~~（已移除）

MCP 层已整体删除。此条目不适用。

### 1.16 Skill 加载器只支持 markdown，未做 SKILL.md+exec 沙箱

[backend/v/skills/loader.py](../backend/v/skills/loader.py) `SkillKind = Literal["markdown"]`。CLAUDE.md 提到的可执行 SKILL（subprocess + resource limits + 信任边界）尚未实现。

**建议**：Phase-3 P4-补：

- `executor.py`：subprocess + cgroup / `prlimit` 限制
- 信任边界：仓库内 + whitelist 列表 → 可执行；其他默认 markdown-only
- 社区 registry sync（git pull）

### 1.17 Skill 目录无排序/筛选，全量进 cold 层

[backend/v/skills/registry.py](../backend/v/skills/registry.py) 的 `render_catalog` 把某 channel 下**所有** skill 的 name + description 全列进 cold 层 `<available_skills>` 目录，由模型自主选择并 `load_skill` 取正文。这避免了旧关键词子串匹配的召回问题（"我想退掉这个东西"现在靠模型语义判断，不再漏召），但 skill 数量很多时目录会膨胀，挤占缓存前缀。

**建议**：skill 数量超阈值时，用 embedder 给每个 skill 的 description 离线生成向量，按与近期对话的 cosine 相似度选 top-N 进目录（仍冻结到压缩点以保持缓存友好）。和 [recall_memory](../backend/v/tools/recall_memory.py) 复用同一份 Qwen embedder。

---

## 2. Phase-3 待办（按优先级）

Phase-2 已落地：P0 Slack handoff / P2 ARQ Cron / P3 长记忆 / 通用 subagent / P4 Skill 加载器。（P1 MCP 已移除。）

Phase-3 进度：Group A / B / C（三层记忆 + 中段压缩，已重设计为 inline，无 ARQ 依赖）/ D / E / G 已合入 main。Group F（情绪预判）和 MCP 层已移除，ARQ/cron 层已移除（内存固化改为 inline）。剩下的是 RAG 深度优化、可观测性、多租户三块。

### P0：可观测性 + 运维

**为什么是 P0**：Phase-2 的核心闭环都跑通了，但生产环境出问题第一反应是"看 metrics 找哪段慢/错"——目前没有。

**范围**：

- `backend/v/utils/metrics.py`：`prometheus-client` 包装，定义命名约定
- 关键 metric：
  - `llm_call_latency_seconds{role,model,fallback_used}` Histogram
  - `bus_queue_depth{shard}` Gauge
  - `bus_handler_duration_seconds{outcome}` Histogram
  - `outbound_send_total{channel,outcome}` Counter
  - `handoff_active_sessions` Gauge
- `/metrics` endpoint
- `doc/logging_schema.md`：必填字段（request_id / channel / channel_user_id / session_id）+ 推荐字段
- `/livez` + `/readyz` 拆分
- `make startup-smoke` 验 lifespan 启停
- 1.7 / 1.8 / 1.9 / 1.6 / 1.4 一同收敛

### P1：性能 + 稳定性

**为什么是 P1**：上线前的最后一公里。

**范围**：

- Bus throughput 基准（fakeredis & real Redis 各跑一组）
- LLM eval 集（[backend/eval/](../backend/eval/) 目录目前空），在固定 prompt 上跑回归
- 长时稳定性（连续 1 小时 1 RPS，看内存 / 文件描述符 / token 池）
- pgvector HNSW 调参（ef_construction / M / ef_search）
- 1.1（bus 重试退避）+ 1.3（outbound 重试 + DLQ 表）一起做

### P2：多租户

**为什么是 P2**：不上线之前不需要，但要预留接线。

**范围**：

- per-tenant settings 加载（按 webhook 来源识别）
- per-tenant LLMCaller / outbound 池
- per-tenant shard prefix
- `agent.session.tenant_id NOT NULL` 做 PK 一部分
- per-tenant skill 目录

### P3：Skill 增强

- `executable Skill`（1.16）：subprocess + cgroup 沙箱，信任边界
- skill 语义匹配（1.17）：embedder-based cosine 替换关键词子串

### P4：Schema 迁移工具化

1.11 引入 Alembic。

---

## 3. 决策记录

为了未来维护方便，以下"看似奇怪"的设计是有意为之，请勿"修复"：

| 决策 | 位置 | 原因 |
| --- | --- | --- |
| 出站不走 bus | 架构规则 | 入站流不被回执污染；出站语义本来就是直连 |
| RedisCheckpointer 归在 `backend/v/agents/` 而非 `memory/` | [backend/v/agents/checkpoints/redis.py](../backend/v/agents/checkpoints/redis.py) | checkpointer 是 agent 运行时（state 存档）的一部分，不是用户级记忆 |
| 30min 静默后开新 session_id | [doc/data_flow.md §5](data_flow.md) | 旧上下文大概率无关 + 浪费 token；连续性靠长记忆兜底 |
| Debounce 归在 wecom_aibot 适配层而非 bus 层 | 架构规则 | bus 是路由 + 并发控制层；语义合并属于平台适配 |
| 单 LLM provider（DeepSeek）+ 同家族降级 | [backend/v/CLAUDE.md](../backend/v/CLAUDE.md) | 跨家族降级会引入风格 / 协议差异 |
| Subagent 是单轮、无嵌套工具调用 | [backend/v/tools/subagent.py](../backend/v/tools/subagent.py) | 真有多步需求让父 agent 自己调，避免无界嵌套 |
| Skill 匹配是关键词子串而不是向量 | [backend/v/skills/registry.py](../backend/v/skills/registry.py) | MVP 简单可解释；语义召回作为 P3 升级 |
| LangGraph 节点用闭包而非 functools.partial | [backend/v/agents/graph.py](../backend/v/agents/graph.py) | partial 让 LangGraph 签名检测错过 config 参数 |
| LLM 直接产出"合并后 profile"而非写 merge 规则 | [backend/v/memory/memory_extractor.py](../backend/v/memory/memory_extractor.py) | 合并语义是任务级的；写规则会教条，让 LLM 看现有 + 新输入直接出 |
