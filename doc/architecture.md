# 架构与实现方法

## 1. 整体定位

外部客户服务 Agent 平台，通过企业微信（WeCom）+ 飞书（Feishu）接入 C 端客户。Phase-1 已实现**被动答疑**主链路（客户消息 → AI 回复）；**主动触达**和**人工接管**是 Phase-2 范围。

## 2. 分层

```
┌──────────────────────────────────────────────────────────────┐
│  /backend/app/  ── 外壳层（Shell & Gateway）                  │
│  纯 I/O：HTTP、签名校验、解密、归一化、入站排队、出站投递      │
│  ─────────────────────────────────────────────────────────   │
│  /backend/v/   ── 智能层（Agent Engine）                      │
│  纯逻辑：LangGraph 编排、LLM 推理、记忆管理、模型工厂          │
└──────────────────────────────────────────────────────────────┘

依赖单向：app → v���v 不得引入 app。
```

依赖单向性由代码组织保证：`/backend/v/` 内任何文件 import `backend.app.*` 都视为违规（Phase-2 加 import-linter 自动检查）。

## 3. 目录树（已实现部分）

```
backend/
├── app/                            外壳 & 网关
│   ├── main.py                     FastAPI 入口 + lifespan（资源启停）
│   ├── gateway/
│   │   ├── middleware.py           RequestIdMiddleware：请求 id + 结构化日志
│   │   └── routers/
│   │       └── health.py           GET /health（PG + Redis 状态）
│   ├── store/
│   │   ├── postgres.py             asyncpg 池 + jsonb/vector codecs
│   │   └── redis.py                redis.asyncio 客户端工厂
│   ├── bus/
│   │   ├── messages.py             SystemMessage（Pydantic，frozen）
│   │   ├── shard.py                RedisStreamShard：mmh3 路由 + Streams 读写
│   │   ├── producer.py             BusProducer.enqueue
│   │   ├── consumer.py             BusConsumer：每 shard 一个 task 严格串行
│   │   └── worker.py               make_bus_handler：粘合 graph + outbound
│   └── channels/
│       ├── base.py                 ChannelAdapter ABC
│       ├── debounce.py             Debouncer：500ms 合并窗口
│       ├── wecom/                  WeCom：signature/crypto/router/outbound
│       └── feishu/                 Feishu：同上
└── v/                              Agent Engine
    ├── configs/
    │   └── base.py                 9 个 BaseSettings + get_settings()
    ├── utils/
    │   └── logging.py              structlog + bind_request 上下文管理
    ├── agents/
    │   ├── state.py                CustomerServiceState（add_messages reducer）
    │   ├── nodes.py                enter / agent / exit
    │   ├── edges.py                节点常量（线性路由）
    │   └── graph.py                build_graph(checkpointer)
    ├── memory/
    │   ├── checkpointer.py         RedisCheckpointer（BaseCheckpointSaver）
    │   ├── working.py              user_profile 缓存读写
    │   └── long_term.py            read_user_profile（PG 只读）
    ├── models/
    │   ├── factory.py              get_chat_model / get_embedding
    │   └── llm_caller.py           LLMCaller（30s 超时 + 同家族降级）
    └── hooks/
        └── session.py              on_session_start：profile 注入
```

**Phase-2 占位目录**（已存在但为空）：`backend/v/{cron, hooks, mcp, skills, tools}`、`backend/app/operator`。

## 4. 关键复用基元

### 4.1 `LLMCaller` ([backend/v/models/llm_caller.py](../backend/v/models/llm_caller.py))

所有 LLM 调用的唯一入口。

- **接口**：`async def chat(role, messages, *, structured=None, tools=None) -> LLMResult`
- **路由**：role 决定 model id；Phase-1 实际只用 `main_primary`，其余 role 返回 fallback model。
- **超时**：`asyncio.wait_for(timeout=settings.llm.timeout_seconds)`（默认 30s）。
- **降级触发**：`asyncio.TimeoutError` / `openai.APITimeoutError` / `RateLimitError` (429) / `InternalServerError` (5xx)。
- **降级策略**：同家族单次重试（pro → flash），跨家族（DeepSeek → Qwen chat）暂不开启。
- **可观测性**：每次调用 emit `llm.invoked`（model / role / latency_ms / fallback_used）；降级时额外 emit `llm.primary_failed_falling_back`。

### 4.2 `RedisStreamShard` ([backend/app/bus/shard.py](../backend/app/bus/shard.py))

固定 64 个 shard（可配，`BUS_SHARD_COUNT`）。

- **路由**：`mmh3.hash(f"{channel}:{channel_user_id}")` % shard_count，**确定性**（不同进程结果一致）。
- **Stream key 格式**：`{prefix}:{shard_idx}`，例如 `bus:42`。
- **DLQ**：`{prefix}:dlq` 共享一条死信流。
- **消费组**：通过 `ensure_group` 幂等创建，遇 `BUSYGROUP` 静默忽略。

### 4.3 `Debouncer` ([backend/app/channels/debounce.py](../backend/app/channels/debounce.py))

合并 500ms 内同一 `(channel, channel_user_id)` 的连续消息。

- **存储**：Redis 哈希 `debounce:{channel}:{user}`（text + dedup_key + 元数据）。
- **定时器**：每 `observe()` 重置一个 asyncio 任务，500ms 后 flush；新消息到来则 cancel + 重新调度。
- **合并语义**：text 用 `\n` 拼接、dedup_key 用 `|` 拼接，按到达顺序。
- **flush 输出**：调用注入的 `dispatch(merged_message)`，本项目里 `dispatch` 是 `BusProducer.enqueue`。

### 4.4 `RedisCheckpointer` ([backend/v/memory/checkpointer.py](../backend/v/memory/checkpointer.py))

LangGraph `BaseCheckpointSaver` 的极简 Redis 实现。

- **键设计**：每个 thread 一个 hash `ckpt:{thread_id}`，所有 checkpoint 字段（type/blob/meta/parent）都在这个 hash 里；`latest` 字段指向最新 checkpoint id。
- **TTL**：每次 `aput` 调用都 `EXPIRE` 刷新到 `WORKING_MEMORY_TTL_SECONDS`（默认 1800s）。
- **序列化**：用 LangGraph 自带的 `JsonPlusSerializer`（dumps_typed → (type_str, bytes)）。
- **限制**：`alist` 仅按 checkpoint id 字典序排，不支持 `filter` 内细分；Phase-1 线性图够用，Phase-2 接入 Postgres 持久 checkpointer 时一并加强。

### 4.5 `AppSettings` ([backend/v/configs/base.py](../backend/v/configs/base.py))

9 个独立 BaseSettings 类（按 env_prefix 分隔），由 `get_settings()` 组合，`@lru_cache` 缓存进程一次。

- 每个子类 `_env_file=".env"` + `case_sensitive=False` + `extra="ignore"`。
- 子类列表：`RuntimeSettings` (APP_) / `LLMSettings` (LLM_) / `EmbeddingSettings` (EMBEDDING_) / `DBSettings` (POSTGRES_) / `RedisSettings` (REDIS_) / `MemorySettings` (MEMORY_) / `BusSettings` (BUS_) / `WecomSettings` (WECOM_) / `FeishuSettings` (FEISHU_)。

### 4.6 结构化日志 ([backend/v/utils/logging.py](../backend/v/utils/logging.py))

- `structlog` + dev / json 双模式（`APP_ENV=dev` 时输出 key=value，否则 JSON）。
- `bind_request(request_id, channel, channel_user_id, session_id)` 上下文管理器：在进入块期间所有日志自动带这些字段。
- 所有外壳 + 智能层调用链统一通过 `get_logger(name)`。

## 5. 数据库 schema

单 Postgres 实例，三个 schema（详见 [data_flow.md](data_flow.md#5-记忆分层)）：

| schema | 用途 | Phase-1 状态 |
| --- | --- | --- |
| `agent` | Agent 自身记忆/会话/profile/episodes（pgvector(1024) HNSW cosine） | 表创建完成；Phase-1 仅读 user_profile，写入是 Phase-2 |
| `dw` | 业务数据仓库（dim_region/customer/product/date + fact_order，2025 Q1 共 231 行种子） | 完整迁移自 legacy MySQL，已加 4 个 FK |
| `meta` | NL2SQL 语义元数据 | 表创建完成；空数据，Phase-2 注册时填充 |

迁移文件：[scripts/sql/](../scripts/sql/) 按文件名前缀数字顺序应用：

- `000_extensions.sql`：`vector` + `pgcrypto`
- `010_schema_agent.sql`
- `020_schema_dw.sql`
- `030_schema_meta.sql`

应用脚本：[scripts/db_apply.sh](../scripts/db_apply.sh)，`make db-migrate` 调用。

每张 `agent.*` 表都预留 `tenant_id UUID NULL` 列，单租户起步、未来切多租户不破坏 schema。

## 6. 部署形态

Phase-1：单进程 FastAPI（uvicorn / fastapi dev）。

`make dev` → `fastapi dev backend/app/main.py` → 触发 `lifespan`：

1. 加载 settings + 配置 logging
2. 开 PG 池 + Redis 客户端
3. 构造 BusShard + BusProducer + BusConsumer + Debouncer
4. 构造 WeComCrypto/Outbound + FeishuCrypto/Outbound
5. 构造 RedisCheckpointer + 编译 graph + 构造 LLMCaller
6. `asyncio.create_task(consumer.run(handler))` 启动 bus 消费者
7. 挂载 WeCom + Feishu 路由

`docker/docker-compose.yml` 提供本地基础设施（pgvector pg16 + redis 7-alpine）。生产部署是 Phase-2 范畴。

## 7. CI/CD

[.github/workflows/ci.yml](../.github/workflows/ci.yml)：push + PR 时跑 `ruff check`、`ruff format --check`、`pytest --cov=backend --cov-fail-under=80`，附带 service container（pgvector + redis）。

测试体系：

- `backend/test/unit/`：模块级测试（fakeredis + asyncpg mock + respx）。
- `backend/test/e2e/`：跨模块集成（迁移 e2e 跑真实 PG，inbound→reply e2e 跑全栈但 mock LLM/outbound）。
- `pytest -m e2e` 仅跑 e2e；默认 `pytest` 跑两类。
- 覆盖率门槛 80%，当前 87.44%（182 测试）。
