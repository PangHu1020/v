# v-agent-platform

LangGraph + FastAPI + Redis Streams + ARQ + PostgreSQL 上的企微 / 飞书外部客服 AI 平台。

主要工作流：

- **被动答疑**（reactive）：客户在企微 / 飞书发消息 → 网关验签 → 500ms 去抖归一化 → Redis Streams（按 `(channel, channel_user_id)` 分片，单分片串行）→ Worker → LangGraph Agent → 回写 channel。
- **主动触达**（proactive）：ARQ 定时任务（物流到货 / 广告推送 / 复购提醒）→ LangGraph 生成 → 写入 working memory → 推 channel。
- **人工接管**（handoff）：`transfer_to_human` 工具触发 LangGraph `interrupt()` → 状态从 Redis 迁移到 Postgres checkpointer（去 TTL）→ Slack 推快照 → 人工回复经 Slack 转回 channel；点击 resume 按钮 → `Command(resume=...)` → 状态迁回 Redis。

详细架构 / 数据流 / 调用链：[doc/architecture.md](doc/architecture.md) · [doc/data_flow.md](doc/data_flow.md) · [doc/call_chain.md](doc/call_chain.md) · [doc/gaps.md](doc/gaps.md)。

## 0. 前置条件

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) — 包管理 / 运行时
- Docker + Docker Compose（用于跑本地 Postgres + Redis）
- `psql` CLI（用于跑 schema 迁移；macOS `brew install libpq` / Ubuntu `apt install postgresql-client`）

## 1. 安装依赖

```bash
uv sync
```

会创建 `.venv/` 并按 `uv.lock` 装齐依赖。后续所有命令都用 `uv run ...` 自动落到这个环境。

## 2. 起 Postgres + Redis

```bash
docker compose -f docker/docker-compose.yml up -d postgres redis
```

只起这俩；`app` service 是 profiled 的（`profiles: ["app"]`），不会被 `up -d` 默认拉起。等到都 healthy 再继续：

```bash
docker compose -f docker/docker-compose.yml ps
```

## 3. 灌 schema

```bash
bash scripts/db_apply.sh
```

按数字顺序执行 [scripts/sql/](scripts/sql/) 下的 SQL：
- `000_extensions.sql` — `pgvector` 扩展
- `010_schema_agent.sql` — `agent` 库（working memory / checkpointer / user_profile / memory_episodes）
- `020_schema_dw.sql` — `dw` 业务库
- `030_schema_meta.sql` — `meta` NL2SQL 元数据库
- `050_session_memory_ttl.sql` — session memory TTL 列

脚本读 `POSTGRES_DSN` 环境变量，缺省走 `postgresql://postgres:postgres@localhost:5432/agent`。

## 4. 配 `.env`

```bash
cp .env.example .env
```

按需填值。最少需要：

| 分组 | 关键字段 |
| --- | --- |
| LLM | `LLM_*_API_KEY` / `LLM_*_BASE_URL` / `LLM_*_PRIMARY_MODEL` |
| Embedding | `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL` |
| Postgres | `POSTGRES_DSN` |
| Redis | `REDIS_URL` |
| WeCom HTTP webhook | `WECOM_CORP_ID` / `WECOM_AGENT_ID` / `WECOM_SECRET` / `WECOM_TOKEN` / `WECOM_AES_KEY` |
| WeCom 智能机器人 WS | `WECOM_AIBOT_WS_URL` / `WECOM_AIBOT_BOT_ID` / `WECOM_AIBOT_SECRET` |
| Feishu | `FEISHU_APP_ID` / `FEISHU_APP_SECRET` / `FEISHU_ENCRYPT_KEY` / `FEISHU_VERIFY_TOKEN` |
| Slack 接管 | `SLACK_BOT_TOKEN` / `SLACK_SIGNING_SECRET` / `SLACK_HANDOFF_CHANNEL` |

只起被动答疑链 + WeCom HTTP，那 Feishu / Slack / WeCom 智能机器人 一组留空即可——对应 channel 不会被启用。

## 5. 启动三个进程

平台跑起来需要 **三个独立进程**，按依赖顺序：

### 5.1 FastAPI 主进程（HTTP 网关 + 反应式 bus worker）

```bash
uv run fastapi dev backend/app/main.py
```

监听 `8000`。承担：

- WeCom / Feishu HTTP webhook 收件
- Slack 接管回调
- 内部 admin API
- 进程内的 bus worker（消费 Redis Streams → 调 `/v/agents`）

生产用 `fastapi run`（同名子命令，无 reload）。

### 5.2 ARQ 主动触达 worker

```bash
uv run arq backend.app.cron_worker.WorkerSettings
```

承担：

- 定时任务（每天 09:30 跑复购提醒扫描）
- 按需任务（`notify_logistics_delivered` / `push_ad` / `consolidate_session` / `extract_session_memory`）

任何进程都可以 `redis_pool.enqueue_job("notify_logistics_delivered", ...)` 把任务塞进来。

### 5.3 WeCom 智能机器人 WS worker（仅在用 `wecom_aibot` channel 时启）

```bash
uv run python -m backend.app.wecom_aibot_worker
```

独立进程是因为它持有一条**长连 WebSocket**：

- 入站：`aibot_msg_callback` 帧 → 归一化为 `SystemMessage` → 进 bus → 走和 webhook 完全相同的下游链。
- 出站：订阅 Redis pub/sub `wecom_aibot:outbound`；任何进程（FastAPI / ARQ / Slack 接管）都可以经 `WecomAibotOutbound.send_text(...)` publish payload，由这个进程经 WS 发出。

`WECOM_AIBOT_WS_URL` 留空时此进程会立即退出（disabled-channel guard）。

## 6. 验证

跑通的最小冒烟：

```bash
# 单元测试 + 覆盖率（CI 也跑这个）
uv run pytest --cov=backend --cov-fail-under=80

# 静态检查
uv run ruff check . --fix && uv run ruff format .
```

## 7. 开发约定速查

- 严格分层：`/backend/app/` 只做 I/O / 路由 / 归一化；`/backend/v/` 只做 Agent 逻辑。`/v/` 不许 import `/app/`。
- 不要绕过 bus 直发回信——回信由 channel 适配器**直发**，不入 bus（bus 只走反应式入站）。
- 不要把 LLM 调用写进 `/app/`。所有 LLM 走 `/v/agents/` 或 `/v/models/`。
- 提交格式：Angular Conventional Commits（`feat:` / `fix:` / `refactor:` / `test:` / `docs:` / `chore:`）。
- 全量规则在三份 CLAUDE.md：[根](CLAUDE.md) / [/backend/app/](backend/app/CLAUDE.md) / [/backend/v/](backend/v/CLAUDE.md)。
