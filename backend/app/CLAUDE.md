# Project Description and Function Overview
This module (`/backend/app/`) is the API gateway, channel adapter layer, message bus, operator-side adapter, and global infrastructure manager for the customer-service agent platform.

- **What it does**: HTTP webhook ingestion, platform-specific signature verification and decryption, normalization of heterogeneous payloads into unified `SystemMessage`, **500ms per-user debounce**, Redis Streams enqueue (with consumer groups), Slack operator adapter for human handoff, and singleton initialization for PostgreSQL and Redis.
- **What it does NOT do**: NO LLM reasoning, NO agent state transitions, NO prompt engineering, NO memory orchestration. All intelligence lives in `/backend/v/`.

# Project structure and corresponding functions
- `/backend/app/main.py`: FastAPI entrypoint. Owns lifespan startup/shutdown for DB pools and Redis clients.
- `/backend/app/gateway/`: HTTP entry point. FastAPI routers, unified exception handlers, middlewares. Internal admin APIs use unified JWT/API-Key auth here.
- `/backend/app/channels/`: **Customer-side** Platform Adapters (WeCom, Feishu only in MVP). Handles platform-specific signature/decryption, normalizes payloads to `SystemMessage`, performs 500ms debounce per `(channel, channel_user_id)`, and pushes to `/app/bus`. Adapters are also responsible for **outbound delivery** when the agent or operator emits a reply.
- `/backend/app/operator/`: **Operator-side** Adapters (Slack in MVP). NOT a customer channel. Responsibilities:
  - Push handoff alerts (conversation snapshot + context) to Slack on agent `interrupt()`.
  - Receive operator replies and route them back to the original customer channel for delivery (and persist as `AIMessage{author_type:"human_agent"}` via `/v/memory`).
  - Receive resume button callbacks → trigger `Command(resume=...)` against the suspended graph.
- `/backend/app/bus/`: **Message Bus.** Redis Streams (NOT ARQ — ARQ lives in `/v/cron`). Stream is sharded by `(channel, channel_user_id)` for **strict per-shard serial processing**. Consumer groups handle worker scaling. Reactive inbound only — outbound replies do not go through this bus.
- `/backend/app/store/`: **Global Infrastructure Manager.** Initializes singletons:
  - PostgreSQL async pool (`asyncpg`) covering `agent`, `dw`, `meta` schemas.
  - Redis async client (shared by Streams bus, working memory, and ARQ).
  - DOES NOT initialize MySQL — MySQL is forbidden project-wide.

# Technology stack and versions
- **Language**: Python 3.11+
- **Web Framework**: FastAPI
- **Bus**: Redis Streams (consumer groups)
- **DB Driver**: `asyncpg` (raw) + SQLAlchemy 2.x async ORM where modeling helps; pgvector for vector columns.
- **HTTP Client**: `httpx` (async)
- **Package Management**: `uv`

# Architecture constraints
- **Customer Message Flow (Strict)**: External Webhook → `/gateway/routers` → `/channels/{platform}` (verify + decrypt + debounce + normalize) → `/bus` (Streams enqueue) → Worker (in-process or separate) consumes → calls `/v/agents`. Outbound replies travel **directly** from `/v/agents` (or operator) → `/channels/{platform}` send API. They do NOT re-enter the bus.
- **Operator Message Flow (Strict)**: `/v/hooks` on-interrupt → `/operator/slack/` push alert; Slack callback → `/operator/slack/` receives → either:
  - Operator reply: relay to `/channels/{platform}` outbound + persist via `/v/memory`.
  - Resume button: invoke `Command(resume=...)` on the LangGraph runtime.
- **Auth Separation**:
  - Internal admin routes in `/gateway/`: unified JWT/API-Key middleware.
  - Customer webhooks: per-platform signature validation **inside each `/channels/{platform}/`** (algorithms differ — WeCom uses AES + signature, Feishu uses encrypted callback).
  - Operator webhooks: per-platform signature validation **inside `/operator/{platform}/`** (Slack request signing).
- **Resource Management**: All DB and Redis clients initialized in FastAPI `lifespan`, passed via `Depends`. Never instantiate connections inside route handlers.
- **User Identity (MVP)**: primary key is `(channel, channel_user_id)` everywhere. No cross-channel merging.

# Coding conventions
- **Naming**: classes `CamelCase`; functions, variables, route paths `snake_case`.
- **Documentation**: Google-Style docstrings on all functions, especially channel normalizers (document the inbound payload shape and the produced `SystemMessage` fields).
- **Async First**: `async def` for all routes and I/O. Never block the event loop.
- **Dependency Injection**: use FastAPI `Depends` exclusively for DB sessions, Redis clients, and queue clients.

# Forbidden patterns
- **No Agent Logic**: NEVER import LangGraph nodes or write LLM interaction code in `/app/`. `/app/` only enqueues to bus or invokes interfaces exposed by `/v/`.
- **No Blocking Code**: NEVER use synchronous HTTP/DB clients. Use `httpx` and `asyncpg`.
- **No MySQL**: never add `aiomysql`, `pymysql`, or any MySQL driver. Legacy `dw.sql`/`meta.sql` are migrated into Postgres schemas during initial seeding.
- **No Hardcoded Secrets**: AppIDs, AppSecrets, AES keys, signing secrets, DB URIs all come from `.env` via `pydantic-settings`.
- **No Global Mutable State**: use `app.state` or DI; never rely on module-level mutable variables.
- **No Outbound Through Bus**: replies/notifications go directly to channel adapters. The bus is reactive-inbound only.

# Key file paths
- Entrypoint: `/backend/app/main.py`
- DB & Redis init: `/backend/app/store/`
- Bus logic: `/backend/app/bus/`
- Customer adapters: `/backend/app/channels/`
- Operator adapter: `/backend/app/operator/slack/`

# Test and build commands
- **Run local dev server**: `fastapi dev backend/app/main.py`
- **Lint & Format**: `ruff check . --fix && ruff format .`
- **Run Unit Tests**: `pytest`
