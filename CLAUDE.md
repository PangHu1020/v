# Project Description and Function Overview
A state-driven AI agent platform for **external customer service over enterprise IM** (WeCom 智能机器人 WebSocket), built on LangGraph, FastAPI, Redis Streams, ARQ, and PostgreSQL (with pgvector).

- **Primary mode**: reactive customer-service inquiries (被动答疑).
- **Secondary mode**: proactive scheduled outreach (logistics notifications, ad pushes, repurchase reminders).
- **Tertiary mode**: human handoff via LangGraph `interrupt()` with Slack as the operator-side channel.
- **Tenancy**: single-tenant in MVP; multi-tenancy is **deferred but not blocked** (do not bake in single-tenant assumptions that would force rewrites later).

As an AI developer (Claude Code) working in this monorepo, your primary objective is to maintain strict architectural boundaries and adhere to a test-driven development lifecycle.

# Development Workflow & CI/CD Constraints (CRITICAL)
You MUST strictly follow this execution sequence. Do NOT skip steps:

1. **Implementation & Unit Testing**: Write feature code AND its `pytest` unit tests in the matching `test/` directory.
2. **Verification**: Run tests. Do not proceed until all pass and coverage on changed files is **≥ 80%** (project-wide gate, enforced in CI).
3. **Documentation**: Update API docs and the relevant `CLAUDE.md` files to reflect new capabilities, parameters, or architectural shifts.
4. **Version Control**: Only after tests pass and docs are updated, commit using Angular Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, `test:`).

**Automated CI/CD Pipeline** (`.github/workflows/`):
- Triggers on push and pull request.
- Runs `ruff check`, `ruff format --check`, `pytest --cov` (coverage gate ≥ 80%).
- The pipeline is the ultimate gatekeeper. No broken code reaches `main`.

# Core System Workflows (How Data Moves)
NEVER bypass layers to create shortcuts.

**1. Reactive Workflow (Customer Initiates Inquiry)**
`Customer` → `WeCom 智能机器人 WS` → `wecom_aibot_worker` (**500ms debounce + normalize** to `SystemMessage`) → `/app/bus` (Redis Streams, sharded by `(channel, channel_user_id)`, **strict per-shard serial**) → `Worker` consumes → calls `/v/agents` (LangGraph) → Agent uses `/v/tools` → reply pushed back via Redis pub/sub → `WecomAibotClient` → `Customer`.

**2. Proactive Workflow (System Initiates Outreach)**
`/v/cron` (ARQ scheduled task) → calls `/v/agents` to generate context-aware message → writes to `/v/memory` (CRITICAL — proactive output is part of conversation history) → pushes to `/app/bus` → `Channel` → `Customer`.
MVP scenarios: **logistics delivery notification**, **ad-hoc ad push**, **repurchase reminder**.

**3. Human Handoff Workflow (Escalation)**
Agent invokes `transfer_to_human` tool → LangGraph `interrupt()` suspends the graph → on-interrupt hook **migrates state from Redis (hot) to Postgres checkpointer (cold), removing TTL** → alert with conversation snapshot pushed to **Slack** via `/app/operator/slack/` → during suspension:
- Customer's continued messages are stored as `HumanMessage` in the durable checkpointer **but do not re-enter the graph**.
- Operator's replies (sent from Slack) are relayed to the customer via the original channel AND stored as `AIMessage` with metadata `{author_type: "human_agent", operator_id: ...}`.
- Operator clicks the **Slack resume button** → triggers `Command(resume=...)` → on-resume hook migrates state back to Redis with fresh TTL → graph resumes with full updated history.

# Project structure and corresponding functions
- `/backend/app/`: **Shell & Gateway.** HTTP traffic, webhook signature validation, cross-platform normalization, bus ingestion, operator-side Slack adapter, infrastructure (DB/Redis) lifecycle.
- `/backend/v/`: **Agent Engine.** LangGraph orchestration, LLM reasoning, hierarchical memory, tools, MCP/Skill integration, model factory, cron logic.
- `/backend/test/`: Unit + integration test suites mirroring `/app/` and `/v/`.
- `/backend/eval/`: LLM evaluation harness (offline, not part of CI gate).
- `/docker/`: Compose files for Redis, PostgreSQL (with `pgvector`), and the app.
- `/scripts/`: Migration (raw SQL until schema stabilizes — Alembic deferred), seeding, admin scripts.
- `/doc/`: Architecture diagrams and API docs.

# Architecture Constraints (Global Rules)
- **Strict Layering**: `/backend/app/` is pure I/O and routing. `/backend/v/` is pure logic and intelligence. No cross-contamination.
- **Unidirectional Dependency**: `/app/` depends on `/v/`'s interfaces. `/v/` MUST NOT import from `/app/`.
- **Infrastructure Injection**: PostgreSQL pool, Redis client, MySQL is **forbidden** (see below). Initialized in `/app/store/` during FastAPI lifespan, passed to `/v/` via dependency injection or app context.
- **Single Postgres Instance, Multiple Schemas**:
  - `agent` schema: agent memory, sessions, checkpointer, user_profile, event_memory, knowledge_chunk (with pgvector).
  - `dw` schema: business data warehouse (migrated from legacy `dw.sql`, originally MySQL).
  - `meta` schema: NL2SQL semantic metadata (migrated from legacy `meta.sql`, originally MySQL).
- **Bus = Redis Streams; Cron = ARQ**: do not conflate them. Streams handle reactive ordering and replay; ARQ handles delayed/scheduled jobs (cron + memory consolidation).
- **User Identity (MVP)**: primary key is `(channel, channel_user_id)`. Cross-channel merging is deferred; reserve a `user_alias` table for future use.

# Coding conventions
- **Global Naming**: modules and files use `snake_case`. Classes use `CamelCase`.
- **Typing**: strict Python type hints universally.
- **Documentation**: all public APIs, LangGraph nodes, Channel adapters, and tools MUST have Google-Style docstrings.
- **Environment Management**: NEVER commit `.env`. New configs go in `.env.example` and the matching Pydantic settings model in `/backend/v/configs/`.
- **Time**: all timestamps stored in UTC; presentation-layer conversion only.

# Forbidden patterns
- **No Synchronous Bottlenecks**: never use `requests`, `psycopg2`, or `pymysql`. Use `httpx`, `asyncpg`. (MySQL is forbidden — see DB unification.)
- **No Circular Imports**: between `/app/` and `/v/`.
- **No Direct LLM Bypassing**: never put OpenAI/Anthropic/DeepSeek calls in `/app/`. All LLM interactions go through `/v/agents/` or `/v/models/`.
- **No MySQL**: legacy `dw.sql` / `meta.sql` are the only legacy MySQL artifacts and are migrated to Postgres on import.
- **No Hardcoded Secrets**: API keys, DB URIs, and webhook secrets are loaded via `pydantic-settings`.

# Test and build commands
- **Install dependencies**: `uv sync`
- **Lint & Format**: `ruff check . --fix && ruff format .`
- **Run All Tests**: `pytest --cov=backend --cov-fail-under=80`
- **Run Local Dev Server**: `fastapi dev backend/app/main.py`
