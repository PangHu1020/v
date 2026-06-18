# 架构与实现方法

## 1. 整体定位

外部客户服务 Agent 平台，通过 WeCom 智能机器人 WebSocket 接入 C 端客户。**已实现的能力**：

- 被动答疑主链路（客户消息 → AI 回复，含工具调用循环）
- 长期记忆（session 总结 → user_profile + event_memory，向量召回 + 时间衰减）
- RAG 检索层（`v/rag/retriever.py` → `agent.knowledge_chunk`，pgvector cosine）
- 通用 subagent 工具（`context_mode` 参数控制共享 / 独立上下文）
- Skill 加载器（markdown SOP，冷层冻结 `<available_skills>` 目录 + `load_skill` 工具按需取回正文）
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
│   │   └── worker.py                       make_bus_handler：粘合 graph + outbound
│   ├── wecom_aibot/                        WeCom 智能机器人 WS 适配器（Phase-3 G）
│   │   ├── debounce.py                     500ms 合并窗口
│   │   ├── client.py                       WecomAibotClient（持久 WS + 指数退避重连）
│   │   └── outbound.py                     WecomAibotOutbound（Redis pub/sub 跨进程出站）
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
    │   └── retriever.py                    KnowledgeRetriever（Milvus hybrid 检索 + rerank 精排）
    ├── memory/                             用户级记忆（不含 checkpointer）
    │   ├── working.py                      user_profile 工作记忆缓存
    │   ├── long_term.py                    read_user_profile（PG）
    │   └── memory_extractor.py             promote_to_long_term（session-end固化）+ extract_from_messages（短会话提取）
    ├── models/
    │   ├── factory.py                      get_chat_model / get_embedding（OpenAI-compat）
    │   └── llm_caller.py                   LLMCaller（30s 超时 + 同家族降级）
    ├── hooks/
    │   ├── session.py                      on_session_start：profile 注入
    ├── tools/
    │   ├── calculator.py                   安全 AST 求值（白名单算子）
    │   ├── recall_memory.py                pgvector cosine + 时间衰减（per-customer）
    │   ├── search.py                       薄包装层 → rag/retriever.py
    │   └── subagent.py                     通用子 agent，shared / independent 上下文
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

### 4.2.1 `MemoryConsumer` + `IdleWatcher` ([backend/app/bus/memory_bus.py](../backend/app/bus/memory_bus.py))

三条全局（非分片）记忆任务流，替代原 `asyncio.create_task` fire-and-forget：

- `memory:promote` — session-end 固化（`promote_to_long_term`），进程崩溃后 PEL 自动重投。
- `memory:consolidate` — 月度情节巩固（ACT-R 激活值剪枝）。
- `memory:extract` — 空闲预提取：`IdleWatcher` 每 60s 扫 `memory:idle_watch` sorted set，
  找 5min 未活动的 session 入队，预填 Redis working memory，丰富 session-end 提取输入。

消费方式：consumer group `memory_workers`，count=1，失败写 DLQ 后 XACK。

### 4.3 `Debouncer` ([backend/app/wecom_aibot/debounce.py](../backend/app/wecom_aibot/debounce.py))

合并 500ms 内同一身份的连续消息，asyncio task 定时器 + Redis hash 双层。

### 4.4 `RedisCheckpointer` ([backend/v/agents/checkpointer.py](../backend/v/agents/checkpointer.py))

热路径 LangGraph checkpointer。每个 thread 一个 hash，整体 TTL 1800s。

### 4.5 `open_pg_checkpointer` ([backend/v/agents/pg_checkpointer.py](../backend/v/agents/pg_checkpointer.py))

冷路径，基于 langgraph 官方 `AsyncPostgresSaver`。`search_path=agent,public`，让 checkpoints 表落到 `agent` schema。

### 4.6 `KnowledgeRetriever` ([backend/v/rag/retriever.py](../backend/v/rag/retriever.py))

共享知识语料库（Milvus `knowledge_chunks` collection）的唯一查询入口。**单次 hybrid 检索 + rerank 精排**：dense cosine + BM25 经 `WeightedRanker` 融合取候选池（默认 20 条）→ cross-encoder reranker 重排截 top_k。reranker 两种 transport：`LocalReranker`（自托管 sidecar，bge-reranker-v2-m3 @ :8767）/ `RemoteReranker`（云 API）；挂了降级为 hybrid 原序，不崩。**无瀑布流 / 置信门控 / LLM 重写**——agentic RAG 里 query 已是 agent LLM 从上下文写的，再用检索器 LLM 重写一遍是冗余（也是旧架构 25-49s 延迟来源），改由 rerank 做质量提升。详见 `backend/eval/README.md`。

### 4.7 `SkillRegistry` ([backend/v/skills/registry.py](../backend/v/skills/registry.py))

启动时从配置目录加载所有 markdown SOP。采用 Anthropic 式**渐进式披露**：cold 层注入冻结的 `<available_skills>` 目录（`render_catalog`，仅技能名 + 描述），模型自选后调用 `load_skill(name)` 工具取回正文（`get` + 落到对话热区）。目录不做 top-K 截断——它是冻结的 cold 层，列全部技能才能让缓存前缀稳定。

### 4.8 `AppSettings` ([backend/v/configs/base.py](../backend/v/configs/base.py))

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
| `SkillSettings` | `SKILL_` |

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

`AGENT_TOOLS = [calculator, search, recall_memory, subagent]`

所有工具使用 `@tool(parse_docstring=True)` — docstring 是 LLM 可见的工具描述（唯一真相源）。

| 工具 | 文件 | 作用 |
| --- | --- | --- |
| `calculator` | [tools/calculator.py](../backend/v/tools/calculator.py) | 安全 AST 求值 |
| `search` | [tools/search.py](../backend/v/tools/search.py) → [rag/retriever.py](../backend/v/rag/retriever.py) | 共享知识库语义召回 |
| `recall_memory` | [tools/recall_memory.py](../backend/v/tools/recall_memory.py) | per-customer 事件记忆 + 时间衰减 |
| `subagent` | [tools/subagent.py](../backend/v/tools/subagent.py) | 单轮 LLM 子任务调用 |

## 7. 部署形态

两个常驻进程：

```
1. FastAPI 主进程            uv run fastapi dev backend/app/main.py
   ├── Bus consumer task（背景 asyncio）

2. WeCom 智能机器人 Worker    uv run python -m backend.app.wecom_aibot_worker
   ├── 持久 WebSocket（重连 + 心跳）
   ├── 入站 frame → SystemMessage → bus
   └── 订阅 Redis pub/sub `wecom_aibot:outbound` 转发出站
```

## 8. CI/CD

[.github/workflows/ci.yml](../.github/workflows/ci.yml)：push + PR 触发 `ruff check` + `ruff format --check` + `pytest --cov=backend --cov-fail-under=80`。

测试体系：

- `backend/test/unit/`：模块级（fakeredis + mock asyncpg + respx）
- `backend/test/e2e/`：跨模块（真实 PG 容器 + LangGraph + 完整 lifespan）
- 当前覆盖率 82.28%，402 个测试。

测试集组成（粗略）：

- 配置 / 工具 / 渠道：~60
- 总线 / 网关 / 路由：~25
- 智能层（state / nodes / graph / hooks）：~35
- 记忆（checkpointer / migration / extractor / working / long_term）：~30
- RAG（retriever / search tool）：~20
- Skill：~39
- e2e（入站到回复 / pg checkpointer）：~0（e2e suite currently minimal）
