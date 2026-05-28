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

**Phase-2 P0 部分缓解**：冷路径已经走 langgraph 官方 `AsyncPostgresSaver`，PG 端排序权威。热路径仍是字典序。

**建议**：给 RedisCheckpointer 写入一份 `created_at` 索引（ZSET），`alist` 用 ZSET 分页代替 hash 全扫描。

### 1.3 出站没有重试 + 失败可能丢消息

[backend/app/channels/wecom/outbound.py:send_text](../backend/app/channels/wecom/outbound.py)
[backend/app/channels/feishu/outbound.py:send_text](../backend/app/channels/feishu/outbound.py)
[backend/app/operator/slack/outbound.py:post_handoff_alert](../backend/app/operator/slack/outbound.py)

抛出后由 worker 的 try/except 捕获 → 推 DLQ。但**回执已经丢失**——客户没收到 AI 回复，且 ack 已经发出去。

**���议**：

- send_text 内部加 tenacity 重试（瞬时 5xx / 429 退避 3 次）。
- 仍失败后写入 `agent.outbound_dlq` 表（新建），由人工或定时任务复投。
- 客户长时间无回复时主动发"系统繁忙"占位（避免静默失败）。

### 1.4 Token cache 仅进程内

[backend/app/channels/wecom/outbound.py:_access_token](../backend/app/channels/wecom/outbound.py)
[backend/app/channels/feishu/outbound.py:_tenant_access_token](../backend/app/channels/feishu/outbound.py)

多 worker / 多 pod 时每个进程独立刷新 token，存在重复请求和潜在竞态。FastAPI 主进程和 ARQ worker 进程会**双倍**刷新 token（同一份 corp 配置下）。

**建议**：多实例部署时把 token 缓存搬到 Redis（带分布式锁），让所有 worker 共享。

### 1.5 Debouncer 状态在重启后会丢

[backend/app/channels/debounce.py:Debouncer](../backend/app/channels/debounce.py) 用 Redis hash 存累积消息，但**定时器是 asyncio task 在内存里**。进程重启后：

- Redis 里残留 pending 数据（pexpire 设了 4 倍窗口防泄漏）。
- 没有 task 来 flush，等到 expire 就自然消失。

**建议**：

- ① 启动时扫描所有 `debounce:*` key 并补发 flush task（小数据量可接受）。
- ② 接受丢失，客户重发即可，连续性靠长记忆兜底。

当前选 ②。多实例部署时考虑 ①。

### 1.6 Health 端点不区分 ready vs live

[backend/app/gateway/routers/health.py:health](../backend/app/gateway/routers/health.py) 只有一个端点。

K8s 习惯：

- liveness：进程活着即可（不查依赖）。
- readiness：所有依赖（PG + Redis + bus consumer task）都健康。

**建议**：拆 `/livez`（永远 200）+ `/readyz`（依赖检查）。

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
- 接管次数 / 平均接管时长
- recall_memory / MCP 调用频率 + 命中率

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

### 1.13 LangGraph 1.x ToolNode 在 Command(resume) + 完整生产图组合下的边界 bug

[backend/test/e2e/test_slack_handoff.py](../backend/test/e2e/test_slack_handoff.py) 注释里记录：用真实生产图（enter → agent → tools → ...）走完整 transfer_to_human → on_resume 流程时，ToolNode 在 resume 后会抛 `No message found in input`。

简化图（只 agent → tools → END）的 resume 在单测 [test_handoff.py:TestOnResume](../backend/test/unit/hooks/test_handoff.py) 和 [test_transfer_to_human.py](../backend/test/unit/tools/test_transfer_to_human.py) 里通过。

**当前规避**：e2e 验收测试只断言"挂起 + 迁移 + 转发"部分；resume 后 AI 回复在单测覆盖。

**建议**：在更新 langgraph 版本时复跑完整 e2e；如果上游修复了，把 e2e 的 resume 断言加回来。

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

### 1.15 MCPServerConfig.api_key 是明文 env 字符串

[backend/v/mcp/config.py](../backend/v/mcp/config.py) 把 api_key 整段以 JSON 形式塞进 `MCP_SERVERS_JSON` env var。生产里这意味着 token 出现在进程环境、容器配置、CI 日志里。

**建议**：

- ① 支持 `${SECRET_NAME}` 占位符，运行时从 vault / Secret Manager 解析。
- ② 或者改成单独的 `MCP_{ID}_API_KEY` env 一对一映射。

### 1.16 Skill 加载器只支持 markdown，未做 SKILL.md+exec 沙箱

[backend/v/skills/loader.py](../backend/v/skills/loader.py) `SkillKind = Literal["markdown"]`。CLAUDE.md 提到的可执行 SKILL（subprocess + resource limits + 信任边界）尚未实现。

**建议**：Phase-3 P4-补：

- `executor.py`：subprocess + cgroup / `prlimit` 限制
- 信任边界：仓库内 + whitelist 列表 → 可执行；其他默认 markdown-only
- 社区 registry sync（git pull）

### 1.17 Skill 匹配是关键词子串，召回粗糙

[backend/v/skills/registry.py:_score](../backend/v/skills/registry.py) 用大小写不敏感子串匹配 + 命中数排序。"我想退掉这个东西"这种没有"退款"二字的句子匹配不到 refund_sop。

**建议**：用 embedder 给每个 skill 的 description / intents 离线生成向量，启动时加载到内存；查询时 embed customer query + cosine 重排。和 [recall_memory](../backend/v/tools/recall_memory.py) 复用同一份 Qwen embedder。

---

## 2. Phase-3 待办（按优先级）

Phase-2 已全部落地（P0 Slack handoff / P1 MCP / P2 ARQ Cron / P3 长记忆 / 通用 subagent / P4 Skill 加载器）。

Phase-3 进度：Group A（启动健康自检）/ B（token 计数器）/ C（三层记忆 + 中段压缩）/ D（工具死循环 + 熔断）/ E（intent / reflection 节点）/ F（情绪兜底接管）/ G（WeCom 智能机器人 WS 渠道，含独立 worker + Redis 出站通道）已合入 main。剩下的是运维 / 性能 / 多租户三块。

### P0：可观测性 + 运维

**为什么是 P0**：Phase-2 的核心闭环都跑通了，但生产环境出问题第一反应是"看 metrics 找哪段慢/错"——目前没有。

**范围**：

- `backend/v/utils/metrics.py`：`prometheus-client` 包装，定义命名约定
- 关键 metric：
  - `llm_call_latency_seconds{role,model,fallback_used}` Histogram
  - `bus_queue_depth{shard}` Gauge
  - `bus_handler_duration_seconds{outcome}` Histogram
  - `outbound_send_total{channel,outcome}` Counter
  - `mcp_tool_call_total{server,tool,cache}` Counter
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
- 1.13 跟踪 langgraph 上游修复
- 1.14 接 proactive drainer

### P2：多租户

**为什么是 P2**：不上线之前不需要，但要预留接线。

**范围**：

- per-tenant settings 加载（按 webhook 来源识别）
- per-tenant LLMCaller / outbound 池
- per-tenant shard prefix
- `agent.session.tenant_id NOT NULL` 做 PK 一部分
- per-tenant skill 目录

### P3：MCP 增强

- 1.15 token 占位符 / 二级 env 解析
- OAuth 2.0 device flow 真实接入
- `executable Skill`（1.16）
- skill 语义匹配（1.17）

### P4：Schema 迁移工具化

1.11 引入 Alembic。

---

## 3. 决策记录

为了未来维护方便，以下"看似奇怪"的设计是有意为之，请勿"修复"：

| 决策 | 位置 | 原因 |
| --- | --- | --- |
| 出站不走 bus | [backend/app/CLAUDE.md](../backend/app/CLAUDE.md) | 入站流不被回执污染；出站语义本来就是直连 |
| RedisCheckpointer 归在 `backend/v/agents/` 而非 `memory/` | [backend/v/agents/checkpointer.py](../backend/v/agents/checkpointer.py) | checkpointer 是 agent 运行时（state 存档）的一部分，不是用户级记忆 |
| 30min 静默后开新 session_id | [doc/data_flow.md §5](data_flow.md) | 旧上下文大概率无关 + 浪费 token；连续性靠长记忆兜底 |
| Slack 接管期间客���消息直接转发，不入 graph | [backend/app/bus/worker.py](../backend/app/bus/worker.py) | 操作员在 Slack 已经看到，graph suspended 状态下也无法处理 |
| Channel 层做 debounce 而非 bus 层 | [backend/app/CLAUDE.md](../backend/app/CLAUDE.md) | bus 是路由 + 并发控制层；语义合并属于平台适配 |
| 单 LLM provider（DeepSeek）+ 同家族降级 | [backend/v/CLAUDE.md](../backend/v/CLAUDE.md) | 跨家族降级会引入风格 / 协议差异 |
| MCP 工具命名带 server_id 前缀 | [backend/v/mcp/registry.py](../backend/v/mcp/registry.py) | 不同 server 可能有同名工具（如 search），前缀消除歧义 |
| Subagent 是单轮、无嵌套工具调用 | [backend/v/tools/subagent.py](../backend/v/tools/subagent.py) | 真有多步需求让父 agent 自己调，避免无界嵌套 |
| Skill 匹配是关键词子串而不是向量 | [backend/v/skills/registry.py](../backend/v/skills/registry.py) | MVP 简单可解释；语义召回作为 P3 升级 |
| LangGraph 节点用闭包而非 functools.partial | [backend/v/agents/graph.py](../backend/v/agents/graph.py) | partial 让 LangGraph 签名检测错过 config 参数 |
| 主进程 + ARQ worker 两个进程 | [backend/app/cron_worker.py](../backend/app/cron_worker.py) | 主动任务可独立扩缩 + 故障隔离 |
| LLM 直接产出"合并后 profile"而非写 merge 规则 | [backend/v/memory/memory_extractor.py](../backend/v/memory/memory_extractor.py) | 合并语义是任务级的；写规则会教条，让 LLM 看现有 + 新输入直接出 |
