# 架构与实现方法

## 1. 整体定位

外部客户服务 Agent 平台，通过 WeCom 智能机器人 WebSocket 接入 C 端客户。**已实现的能力**：

- 被动答疑主链路（客户消息 → AI 回复，含工具调用循环）
- Slack 人工接管（`transfer_to_human` 工具 + LangGraph `interrupt()` + 热冷 checkpointer 迁移）
- ARQ 主动触达（物流送达通知 / 广告推送 / 复购提醒 / 会话总结延迟任务）
- 长期记忆（session 总结 → user_profile + event_memory，向量召回 + 时间衰减）
- RAG 检索层（`v/rag/retriever.py` → `agent.knowledge_chunk`，pgvector cosine）
- 通用 subagent 工具（`context_mode` 参数控制共享 / 独立上下文）
- Skill 加载器（markdown SOP，按客户意图匹配，自动注入 system prompt）
- Phase-3 Group B：token 计数器 + metadata-first 降级阶梯
- Phase-3 Group C：三层记忆架构 + 会话中段压缩
- Phase-3 Group D：工具死循环检测 + 熔断（dead-loop guard + circuit breaker）
- Phase-3 Group E：intent / reflection 节点 + 工具安全护栏
- Phase-3 Group G：WeCom 智能机器人（WebSocket）适配器 + 独立 worker 进程

未实现：可观测性（Prometheus metrics）、性能基准、多租户。详见 [gaps.md](gaps.md)。

## 2. 分层

```
┌──────────────────────────────────────────────────────────────┐
│  /backend/app/  ── 外壳层（Shell & Gateway）                  │
│  纯 I/O：WS 帧归一化、签名校验、入站排队、出站投递、           │
│           Slack operator adapter、ARQ worker 入口             │
│  ─────────────────────────────────────────────────────────   │
│  /backend/v/   ── 智能层（Agent Engine）                      │
│  纯逻辑：LangGraph 编排、LLM 推理、RAG 检索、记忆管理、        │
│           工具、cron 任务定义、skill 加载器                    │
└──────────────────────────────────────────────────────────────┘

依赖单向：app → v；v 不得引入 app。
```

为了让 `/v/` 不依赖 `/app/`，对需要跨层的协作使用 **Protocol**（如 `HandoffNotifier`）—— `SlackOutbound` 通过 duck typing 满足，不必在 `/v/` 里 import 它。

## 3. 目录树（已实现部分）

```
backend/
├── app/                                    外壳 & 网关
│   ├── main.py                             FastAPI 入口 + lifespan
│   ├── cron_worker.py                      ARQ WorkerSettings 入口（独立进程）
│   ├── wecom_aibot_worker.py               WeCom 智能机器人 WS 独立 worker 入口（Phase-3 G）
│   ├── gateway/
│   │   ├── middleware.py                   RequestIdMiddleware
│   │   └── routers/health.py               GET /health
│   ├── store/
│   │   ├── postgres.py                     asyncpg 池 + jsonb/vector codecs
│   │   └── redis.py                        redis.asyncio 客户端工厂
│   ├── bus/
│   │   ├── messages.py                     SystemMessage（frozen）
│   │   ├── shard.py                        RedisStreamShard：mmh3 路由
│   │   ├── producer.py / consumer.py       sharded enqueue / 严格串行消费
│   │   └── worker.py                       make_bus_handler：粘合 graph + outbound + handoff
│   ├── wecom_aibot/                        WeCom 智能机器人 WS 适配器（Phase-3 G）
│   │   ├── debounce.py                     500ms 合并窗口
│   │   ├── client.py                       WecomAibotClient（持久 WS + 指数退避重连）
│   │   └── outbound.py                     WecomAibotOutbound（Redis pub/sub 跨进程出站）
│   └── operator/                           Phase-2 P0：人工接管
│       └── slack/
│           ├── signature.py                Slack v0 HMAC-SHA256
│           ├── outbound.py                 SlackOutbound：告警 + 转发
│           └── router.py                   /events + /interactivity 两个路由
└── v/                                      Agent Engine
    ├── configs/base.py                     10 个 BaseSettings + get_settings()
    ├── utils/logging.py                    structlog + bind_request
    ├── agents/
    │   ├── state.py                        CustomerServiceState（add_messages reducer）
    │   ├── nodes.py                        enter（注入 profile + skills）/ agent / exit
    │   ├── edges.py                        路由常量 + route_after_agent
    │   ├── graph.py                        build_graph(checkpointer)
    │   ├── checkpointer.py                 RedisCheckpointer（热路径）
    │   ├── pg_checkpointer.py              open_pg_checkpointer（冷路径，langgraph 官方）
    │   └── checkpointer_migration.py       migrate_hot_to_cold / migrate_cold_to_hot
    ├── rag/                                RAG 检索层
    │   └── retriever.py                    KnowledgeRetriever（pgvector cosine SQL + 格式化）
    ├── memory/                             用户级记忆（不含 checkpointer）
    │   ├── working.py                      user_profile 工作记忆缓存
    │   ├── long_term.py                    read_user_profile（PG）
    │   └── memory_extractor.py             session_memory → user_profile + event_memory（向量化）
    ├── models/
    │   ├── factory.py                      get_chat_model / get_embedding（OpenAI-compat）
    │   └── llm_caller.py                   LLMCaller（30s 超时 + 同家族降级）
    ├── hooks/
    │   ├── session.py                      on_session_start：profile 注入
    │   └── handoff.py                      extract_interrupt / on_interrupt / on_resume / is_suspended
    ├── tools/
    │   ├── calculator.py                   安全 AST 求值（白名单算子）
    │   ├── transfer_to_human.py            LangGraph interrupt 触发
    │   ├── recall_memory.py                pgvector cosine + 时间衰减（per-customer）
    │   ├── search.py                       薄包装层 → rag/retriever.py
    │   └── subagent.py                     通用子 agent，shared / independent 上下文
    ├── cron/                                Phase-2 P2
    │   ├── proactive.py                    deliver_proactive 共享投递助手
    │   └── tasks/
    │       ├── logistics.py                物流签收通知
    │       ├── ad_hoc_ad.py                广告手动推送
    │       ├── repurchase.py               复购提醒（DW 扫描 + 批量）
    │       └── consolidate_session.py      会话总结（结构化输出 → session_memory）
    └── skills/                              Phase-2 P4
        ├── model.py                        Skill schema（YAML frontmatter + body）
        ├── loader.py                       递归加载 markdown SOP
        └── registry.py                     关键词匹配 + 渲染 prompt 段
```

## 4. 关键复用基元

### 4.1 `LLMCaller` ([backend/v/models/llm_caller.py](../backend/v/models/llm_caller.py))

所有 LLM 调用的唯一入口。

- **接口**：`async def chat(role, messages, *, structured=None, tools=None, config=None) -> LLMResult`
- **路由**：role ∈ {`main_primary`, `main_fallback`, `summary`, `memory_extract`}
- **超时**：`asyncio.wait_for(timeout=settings.llm.timeout_seconds)`（默认 30s）
- **降级触发**：`asyncio.TimeoutError` / `openai.APITimeoutError` / `RateLimitError` / `InternalServerError`
- **降级策略**：同家族单次重试（pro → flash）
- **LangSmith**：每次调用透传 `RunnableConfig`，run_name + tags 自动嵌套到父 trace 下

### 4.2 `RedisStreamShard` ([backend/app/bus/shard.py](../backend/app/bus/shard.py))

64 个 shard（可配 `BUS_SHARD_COUNT`）。`mmh3.hash(f"{channel}:{channel_user_id}") % shard_count` 决定归属。共享 DLQ `{prefix}:dlq`。

### 4.3 `Debouncer` ([backend/app/wecom_aibot/debounce.py](../backend/app/wecom_aibot/debounce.py))

合并 500ms 内同一身份的连续消息，asyncio task 定时器 + Redis hash 双层。

### 4.4 `RedisCheckpointer` ([backend/v/agents/checkpointer.py](../backend/v/agents/checkpointer.py))

热路径 LangGraph checkpointer。每个 thread 一个 hash，整体 TTL 1800s。

### 4.5 `open_pg_checkpointer` ([backend/v/agents/pg_checkpointer.py](../backend/v/agents/pg_checkpointer.py))

冷路径，基于 langgraph 官方 `AsyncPostgresSaver`。`search_path=agent,public`，让 checkpoints 表落到 `agent` schema。

### 4.6 `KnowledgeRetriever` ([backend/v/rag/retriever.py](../backend/v/rag/retriever.py))

共享知识语料库（`agent.knowledge_chunk`）的唯一查询入口。pgvector cosine，可选 `source_type` 过滤，top_k 限制 [1, 20]。`tools/search.py` 是它的薄 `@tool` 包装层；future admin endpoints 可直接 import `KnowledgeRetriever`。

### 4.7 ARQ worker ([backend/app/cron_worker.py](../backend/app/cron_worker.py))

独立进程入口（`make cron`）。`on_startup` 构造 PG 池 + Redis + WeCom AIBot outbound + LLMCaller + embedder 装入 ARQ ctx。

### 4.8 `SkillRegistry` ([backend/v/skills/registry.py](../backend/v/skills/registry.py))

启动时从配置目录加载所有 markdown SOP。关键词子串匹配，top-K 渲染进 system prompt。

### 4.9 `AppSettings` ([backend/v/configs/base.py](../backend/v/configs/base.py))

10 个独立 BaseSettings 类（按 env_prefix 分隔），由 `get_settings()` 组合：

| 子类 | env_prefix |
| --- | --- |
| `RuntimeSettings` | `APP_` |
| `LLMSettings` | `LLM_` |
| `EmbeddingSettings` | `EMBEDDING_` |
| `DBSettings` | `POSTGRES_` |
| `RedisSettings` | `REDIS_` |
| `MemorySettings` | `MEMORY_` |
| `BusSettings` | `BUS_` |
| `WecomSettings` | `WECOM_` |
| `WecomAibotSettings` | `WECOM_AIBOT_` |
| `SlackSettings` | `SLACK_` |
| `ARQSettings` | `ARQ_` |
| `SkillSettings` | `SKILL_` |
| `LangSmithSettings` | `LANGSMITH_` |

### 4.10 结构化日志 ([backend/v/utils/logging.py](../backend/v/utils/logging.py))

`structlog` + dev / json 双模式。`bind_request(request_id, channel, channel_user_id, ...)` 上下文管理器。

### 4.11 WeCom 智能机器人 WebSocket 适配器（Phase-3 G）

[backend/app/wecom_aibot/client.py](../backend/app/wecom_aibot/client.py)
[backend/app/wecom_aibot/outbound.py](../backend/app/wecom_aibot/outbound.py)
[backend/app/wecom_aibot_worker.py](../backend/app/wecom_aibot_worker.py)

- `WecomAibotClient`：持久 WS（指数退避重连 1s→60s + 30s 心跳）。入站 frame 归一化为 `SystemMessage(channel="wecom_aibot")` 后送 `Debouncer`，经过同一条 bus。
- `WecomAibotOutbound`：`send_text` 底层是 `redis.publish("wecom_aibot:outbound", ...)` — 任何进程可调用，worker 进程统一通过 WS 转发。
- `wecom_aibot_worker.py`：进程入口。同时跑 `WecomAibotClient.run()` 和 pub/sub 订阅 task。

## 5. 数据库 schema

单 Postgres 实例，多 schema：

| schema | 用途 | 状态 |
| --- | --- | --- |
| `agent` | 记忆 + 会话 + profile + event_memory + knowledge_chunk（pgvector(1024) HNSW cosine） + LangGraph checkpoints | 完整 |
| `dw` | 业务数据仓库（dim_region/customer/product/date + fact_order） | 完整迁移自 legacy MySQL |
| `meta` | NL2SQL 语义元数据（保留 schema，数据空） | 表已建 |

## 6. 工具体系

`AGENT_TOOLS = [calculator, search, recall_memory, subagent, transfer_to_human]`

所有工具使用 `@tool(parse_docstring=True)` — docstring 是 LLM 可见的工具描述（唯一真相源）。

| 工具 | 文件 | 作用 |
| --- | --- | --- |
| `calculator` | [tools/calculator.py](../backend/v/tools/calculator.py) | 安全 AST 求值 |
| `search` | [tools/search.py](../backend/v/tools/search.py) → [rag/retriever.py](../backend/v/rag/retriever.py) | 共享知识库语义召回 |
| `recall_memory` | [tools/recall_memory.py](../backend/v/tools/recall_memory.py) | per-customer 事件记忆 + 时间衰减 |
| `subagent` | [tools/subagent.py](../backend/v/tools/subagent.py) | 单轮 LLM 子任务调用 |
| `transfer_to_human` | [tools/transfer_to_human.py](../backend/v/tools/transfer_to_human.py) | LangGraph interrupt()，suspend 当前会话 |

## 7. 部署形态

三个进程：

```
1. FastAPI 主进程            uv run fastapi dev backend/app/main.py
   ├── Bus consumer task（背景 asyncio）
   └── Slack 路由

2. ARQ Worker                uv run arq backend.app.cron_worker.WorkerSettings
   ├── 物流 / 广告 / 复购定时任务
   ├── 会话总结延迟任务
   └── 长期记忆抽取任务

3. WeCom 智能机器人 Worker    uv run python -m backend.app.wecom_aibot_worker
   ├── 持久 WebSocket（重连 + 心跳）
   ├── 入站 frame → SystemMessage → bus
   └── 订阅 Redis pub/sub `wecom_aibot:outbound` 转发出站
```

## 8. CI/CD

[.github/workflows/ci.yml](../.github/workflows/ci.yml)：push + PR 触发 `ruff check` + `ruff format --check` + `pytest --cov=backend --cov-fail-under=80`。

测试体系：

- `backend/test/unit/`：模块级（fakeredis + mock asyncpg + respx）
- `backend/test/e2e/`：跨模块（真实 PG 容器 + LangGraph + 完整 lifespan）
- 当前覆盖率 84.11%，440 个测试。


## 2. 分层

```
┌──────────────────────────────────────────────────────────────┐
│  /backend/app/  ── 外壳层（Shell & Gateway）                  │
│  纯 I/O：WS 帧归一化、签名校验、入站排队、出站投递、           │
│           Slack operator adapter、ARQ worker 入口             │
│  ─────────────────────────────────────────────────────────   │
│  /backend/v/   ── 智能层（Agent Engine）                      │
│  纯逻辑：LangGraph 编排、LLM 推理、RAG 检索、记忆管理、        │
│           工具、cron 任务定义、skill 加载器                    │
└──────────────────────────────────────────────────────────────┘

依赖单向：app → v；v 不得引入 app。
```

为了让 `/v/` 不依赖 `/app/`，对需要跨层的协作使用 **Protocol**（如 `HandoffNotifier`）—— `SlackOutbound` 通过 duck typing 满足，不必在 `/v/` 里 import 它。

## 3. 目录树（已实现部分）

```
backend/
├── app/                                    外壳 & 网关
│   ├── main.py                             FastAPI 入口 + lifespan
│   ├── cron_worker.py                      ARQ WorkerSettings 入口（独立进程）
│   ├── wecom_aibot_worker.py               WeCom 智能机器人 WS 独立 worker 入口（Phase-3 G）
│   ├── gateway/
│   │   ├── middleware.py                   RequestIdMiddleware
│   │   └── routers/health.py               GET /health
│   ├── store/
│   │   ├── postgres.py                     asyncpg 池 + jsonb/vector codecs
│   │   └── redis.py                        redis.asyncio 客户端工厂
│   ├── bus/
│   │   ├── messages.py                     SystemMessage（frozen）
│   │   ├── shard.py                        RedisStreamShard：mmh3 路由
│   │   ├── producer.py / consumer.py       sharded enqueue / 严格串行消费
│   │   └── worker.py                       make_bus_handler：粘合 graph + outbound + handoff
│   ├── channels/                           客户侧适配器
│   │   ├── base.py                         ChannelAdapter ABC
│   │   ├── debounce.py                     500ms 合并窗口
│   │   ├── wecom/                          WeCom：signature/crypto/router/outbound（HTTP webhook）
│   │   ├── wecom_aibot/                    WeCom 智能机器人：WS client + Redis pub/sub outbound（Phase-3 G）
│   │   └── feishu/                         Feishu：HTTP webhook
│   └── operator/                           Phase-2 P0：人工接管
│       └── slack/
│           ├── signature.py                Slack v0 HMAC-SHA256
│           ├── outbound.py                 SlackOutbound：告警 + 转发
│           └── router.py                   /events + /interactivity 两个路由
└── v/                                      Agent Engine
    ├── configs/base.py                     12 个 BaseSettings + get_settings()
    ├── utils/logging.py                    structlog + bind_request
    ├── agents/
    │   ├── state.py                        CustomerServiceState（add_messages reducer）
    │   ├── nodes.py                        enter（注入 profile + skills）/ agent / exit
    │   ├── edges.py                        路由常量 + route_after_agent
    │   ├── graph.py                        build_graph(checkpointer, extra_tools=)
    │   ├── checkpointer.py                 RedisCheckpointer（热路径）
    │   ├── pg_checkpointer.py              open_pg_checkpointer（冷路径，langgraph 官方）
    │   └── checkpointer_migration.py       migrate_hot_to_cold / migrate_cold_to_hot
    ├── memory/                             用户级记忆（不含 checkpointer）
    │   ├── working.py                      user_profile 工作记忆缓存
    │   ├── long_term.py                    read_user_profile（PG）
    │   └── memory_extractor.py             session_memory → user_profile + episodes（向量化）
    ├── models/
    │   ├── factory.py                      get_chat_model / get_embedding（OpenAI-compat）
    │   └── llm_caller.py                   LLMCaller（30s 超时 + 同家族降级）
    ├── hooks/
    │   ├── session.py                      on_session_start：profile 注入
    │   └── handoff.py                      extract_interrupt / on_interrupt / on_resume / append_operator_log / is_suspended
    ├── tools/
    │   ├── transfer_to_human.py            LangGraph interrupt 触发
    │   ├── recall_memory.py                pgvector cosine + 时间衰减
    │   └── subagent.py                     通用子 agent，shared / independent 上下文
    ├── mcp/                                Phase-2 P1
    │   ├── config.py                       MCPServerConfig 解析
    │   ├── client.py                       ClientSession 长连接（stdio/HTTP/SSE）
    │   ├── cache.py                        L1 in-mem + L2 Redis
    │   └── registry.py                     LangChain StructuredTool 包装
    ├── cron/                                Phase-2 P2
    │   ├── proactive.py                    deliver_proactive 共享投递助手
    │   └── tasks/
    │       ├── logistics.py                物流签收通知
    │       ├── ad_hoc_ad.py                广告手动推送
    │       ├── repurchase.py               复购提醒（DW 扫描 + 批量）
    │       └── consolidate_session.py      会话总结（结构化输出 → session_memory）
    └── skills/                              Phase-2 P4
        ├── model.py                        Skill schema（YAML frontmatter + body）
        ├── loader.py                       递归加载 markdown SOP
        └── registry.py                     关键词匹配 + 渲染 prompt 段
```

## 4. 关键复用基元

### 4.1 `LLMCaller` ([backend/v/models/llm_caller.py](../backend/v/models/llm_caller.py))

所有 LLM 调用的唯一入口。

- **接口**：`async def chat(role, messages, *, structured=None, tools=None) -> LLMResult`
- **路由**：role ∈ {`main_primary`, `main_fallback`, `summary`, `memory_extract`}，每个映射到不同 model id（实际 Phase-1/2 主推理走 `main_primary`，总结/抽取/subagent 走 `main_fallback` flash）。
- **超时**：`asyncio.wait_for(timeout=settings.llm.timeout_seconds)`（默认 30s）。
- **降级触发**：`asyncio.TimeoutError` / `openai.APITimeoutError` / `RateLimitError` / `InternalServerError`。
- **降级策略**：同家族单次重试（pro → flash）。跨家族暂未启用。
- **可观测性**：每次调用 emit `llm.invoked`（model / role / latency_ms / fallback_used）。

### 4.2 `RedisStreamShard` ([backend/app/bus/shard.py](../backend/app/bus/shard.py))

64 个 shard（可配 `BUS_SHARD_COUNT`）。`mmh3.hash(f"{channel}:{channel_user_id}") % shard_count` 决定归属。共享 DLQ `{prefix}:dlq`。

### 4.3 `Debouncer` ([backend/app/channels/debounce.py](../backend/app/channels/debounce.py))

合并 500ms 内同一身份的连续消息，asyncio task 定时器 + Redis hash 双层。

### 4.4 `RedisCheckpointer` ([backend/v/agents/checkpointer.py](../backend/v/agents/checkpointer.py))

热路径 LangGraph checkpointer。每个 thread 一个 hash，整体 TTL 1800s。**Phase-2 P0 后：归属在 `agents/`，因为 checkpointer 是 agent 运行时的一部分而非用户记忆。**

### 4.5 `open_pg_checkpointer` ([backend/v/agents/pg_checkpointer.py](../backend/v/agents/pg_checkpointer.py))

冷路径，基于 langgraph 官方 `AsyncPostgresSaver`。在 DSN 上注入 `search_path=agent,public`，让 LangGraph 的 checkpoints/checkpoint_blobs/checkpoint_writes/checkpoint_migrations 表落到 `agent` schema 而非 `public`。

### 4.6 `migrate_hot_to_cold` / `migrate_cold_to_hot` ([backend/v/agents/checkpointer_migration.py](../backend/v/agents/checkpointer_migration.py))

`transfer_to_human` 触发 `on_interrupt` → 把整个 thread 的 checkpoint 链从 Redis 拷贝到 PG（按父引用顺序重放）→ 删源端。`on_resume` 反向。幂等。

### 4.7 `MCPRegistry` ([backend/v/mcp/registry.py](../backend/v/mcp/registry.py))

启动时连接所有配置的 MCP server，调 `list_tools` 自动把每个工具转成 `langchain_core.tools.StructuredTool`，工具名前缀 `{server_id}__{tool_name}`。读类工具结果走 `MCPToolCache`（L1 内存 5min + L2 Redis 24h），写类按 `write_tools` 列表 opt-out。

### 4.8 `MCPToolCache` ([backend/v/mcp/cache.py](../backend/v/mcp/cache.py))

L1 进程内 dict + L2 Redis。键 = `mcp:cache:{server_id}:{tool_name}:{sha256(canonical_args)[:16]}`。L2 hit 自动回填 L1。

### 4.9 ARQ worker ([backend/app/cron_worker.py](../backend/app/cron_worker.py))

独立进程入口（`make cron` / `uv run arq backend.app.cron_worker.WorkerSettings`）。`on_startup` 构造 PG 池 + Redis + channel outbound + LLMCaller + embedder 装入 ARQ ctx。函数列表：`notify_logistics_delivered` / `push_ad` / `consolidate_session` / `extract_session_memory` / `_scheduled_repurchase_run`。每天 09:30 跑一次复购扫描（避开整点 API 拥堵）。

### 4.10 `SkillRegistry` ([backend/v/skills/registry.py](../backend/v/skills/registry.py))

启动时从配置目录加载所有 markdown SOP。每个 skill 的 YAML frontmatter 里有 `intents`（关键词列表），`enter_node` 用这些关键词在客户最新消息里做大小写不敏感子串匹配，按命中数 desc + priority desc 排序，取 top-K（默认 3）渲染进 system prompt。

### 4.11 `AppSettings` ([backend/v/configs/base.py](../backend/v/configs/base.py))

12 个独立 BaseSettings 类（按 env_prefix 分隔），由 `get_settings()` 组合，`@lru_cache` 缓存进程一次：

| 子类 | env_prefix | Phase |
| --- | --- | --- |
| `RuntimeSettings` | `APP_` | P1 |
| `LLMSettings` | `LLM_` | P1 |
| `EmbeddingSettings` | `EMBEDDING_` | P1 |
| `DBSettings` | `POSTGRES_` | P1 |
| `RedisSettings` | `REDIS_` | P1 |
| `MemorySettings` | `MEMORY_` | P1 |
| `BusSettings` | `BUS_` | P1 |
| `WecomSettings` | `WECOM_` | P1 |
| `WecomAibotSettings` | `WECOM_AIBOT_` | P3 G |
| `FeishuSettings` | `FEISHU_` | P1 |
| `SlackSettings` | `SLACK_` | P2 P0 |
| `MCPSettings` | `MCP_` | P2 P1 |
| `ARQSettings` | `ARQ_` | P2 P2 |
| `SkillSettings` | `SKILL_` | P2 P4 |

### 4.12 结构化日志 ([backend/v/utils/logging.py](../backend/v/utils/logging.py))

`structlog` + dev / json 双模式。`bind_request(request_id, channel, channel_user_id, session_id, **extra)` 上下文管理器，进入块期间所有日志自动带这些字段。

### 4.13 WeCom 智能机器人 WebSocket 适配器（Phase-3 G）

[backend/app/channels/wecom_aibot/client.py](../backend/app/channels/wecom_aibot/client.py)
[backend/app/channels/wecom_aibot/outbound.py](../backend/app/channels/wecom_aibot/outbound.py)
[backend/app/wecom_aibot_worker.py](../backend/app/wecom_aibot_worker.py)

与 WeCom HTTP webhook 不同，智能机器人协议要求客户端持有一条长连 WS。所以本渠道由独立 worker 进程承载：

- `WecomAibotClient`：持久 WS（指数退避重连 1s→60s + 30s 心跳）。入站 frame 归一化为 `SystemMessage(channel="wecom_aibot")` 后送 `Debouncer`，与 HTTP 渠道共用同一条 bus。
- `WecomAibotOutbound`：实现与其他渠道 outbound 相同的 `send_text(channel_user_id, text)` 接口，但底层是把 JSON 发布到 Redis pub/sub channel（默认 `wecom_aibot:outbound`）；FastAPI 主进程 / ARQ worker / Slack handoff 任何一处都能往这里发，由 worker 进程统一通过 WS 转发。
- `wecom_aibot_worker.py`：进程入口。同时跑 `WecomAibotClient.run()` 和一个 pub/sub 订阅 task，用 SIGINT/SIGTERM 优雅关停。

配置（`WECOM_AIBOT_*`）：`WS_URL` / `TOKEN` / `HEARTBEAT_SECONDS=30` / `OUTBOUND_PUBSUB_CHANNEL=wecom_aibot:outbound`。`WS_URL` 为空时 worker 启动后立即退出，便于在不启用该渠道的部署里直接跳过。

## 5. 数据库 schema

单 Postgres 实例，多 schema：

| schema | 用途 | 状态 |
| --- | --- | --- |
| `agent` | Agent 记忆 + 会话 + profile + episodes（pgvector(1024) HNSW cosine） + LangGraph checkpoints/blobs/writes/migrations 表（Phase-2 P0 加） | 完整，写入路径全部就绪 |
| `dw` | 业务数据仓库（dim_region/customer/product/date + fact_order） | 完整迁移自 legacy MySQL，含 4 个 FK |
| `meta` | NL2SQL 语义元数据（4 张表） | 表已建，数据空（NL2SQL 业务工具放弃后保留 schema） |

迁移文件 [scripts/sql/](../scripts/sql/)：

- `000_extensions.sql`：`vector` + `pgcrypto`
- `010_schema_agent.sql`：agent 自管表（user_alias / session / session_memory / user_profile / memory_episodes）
- `020_schema_dw.sql`：业务数仓
- `030_schema_meta.sql`：语义元数据

LangGraph 的 PG checkpointer 表由 `AsyncPostgresSaver.setup()` 在首次连接时自动创建到 `agent` schema（通过 `search_path` 设置）。

每张 `agent.*` 表都预留 `tenant_id UUID NULL` 列。

## 6. 工具体系

绑定到主 agent 的工具列表（见 [backend/v/agents/nodes.py:23](../backend/v/agents/nodes.py#L23)）：

| 工具 | 文件 | 作用 |
| --- | --- | --- |
| `transfer_to_human` | [tools/transfer_to_human.py](../backend/v/tools/transfer_to_human.py) | 触发 LangGraph `interrupt()`，suspend 当前会话，等人工 |
| `recall_memory` | [tools/recall_memory.py](../backend/v/tools/recall_memory.py) | pgvector cosine + 时间衰减召回 episodes |
| `subagent` | [tools/subagent.py](../backend/v/tools/subagent.py) | 单轮 LLM 子任务调用，`context_mode` 控制是否带父消息 |

外加 MCP 注册中心动态发现的工具（命名为 `{server_id}__{tool_name}`），编译 graph 时通过 `extra_tools=` 拼接。LLM 看到的工具列表 = `[transfer_to_human, recall_memory, subagent, *mcp_tools]`，`ToolNode` 用同一个列表分发。

## 7. 部署形态

三个进程：

```
1. FastAPI 主进程            uv run fastapi dev backend/app/main.py
   ├── 入站 webhook 处理（WeCom HTTP / Feishu HTTP）
   ├── Bus consumer task（背景 asyncio）
   └── Slack 路由 + 客户路由

2. ARQ Worker                uv run arq backend.app.cron_worker.WorkerSettings
   ├── 物流 / 广告 / 复购定时任务
   ├── 会话总结延迟任务
   └── 长期记忆抽取任务

3. WeCom 智能机器人 Worker    uv run python -m backend.app.wecom_aibot_worker
   ├── 持久 WebSocket（重连 + 心跳）
   ├── 入站 frame → SystemMessage → bus（与 HTTP 渠道共用）
   └── 订阅 Redis pub/sub `wecom_aibot:outbound` 转发出站
```

`docker/docker-compose.yml` 提供 pgvector pg16 + redis 7-alpine。生产部署文档是 Phase-3 范围。

`make` 入口：

| 命令 | 作用 |
| --- | --- |
| `make install` | `uv sync` |
| `make lint` / `make fmt` | ruff check / format |
| `make test` / `make cov` | pytest，覆盖率门槛 80% |
| `make dev` | FastAPI 开发服务器 |
| `make cron` | ARQ worker |
| `make db-up` / `make db-down` | docker-compose 起停 PG + Redis |
| `make db-migrate` | 应用 SQL 迁移 |

## 8. CI/CD

[.github/workflows/ci.yml](../.github/workflows/ci.yml)：push + PR 触发 `ruff check` + `ruff format --check` + `pytest --cov=backend --cov-fail-under=80`，附带 service container（pgvector + redis）。

测试体系：

- `backend/test/unit/`：模块级（fakeredis + mock asyncpg + respx + FastMCP in-memory transport）
- `backend/test/e2e/`：跨模块（真实 PG 容器 + LangGraph + 完整 lifespan）
- 当前覆盖率 84.11%，368 个测试。

测试集组成（粗略）：

- 配置 / 工具 / 渠道：~60
- 总线 / 网关 / 路由：~25
- 智能层（state / nodes / graph / hooks）：~35
- 记忆（checkpointer / migration / extractor / working / long_term）：~30
- 工具（transfer_to_human / recall_memory / subagent）：~30
- MCP：~37
- Cron：~22
- Slack 接管：~28
- Skill：~39
- e2e（迁移 / 入站到回复 / pg checkpointer / handoff）：~12
