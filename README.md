<div align="center">

# v-agent-platform

**State-Driven Multi-Agent Customer Service over Enterprise IM**

A LangGraph customer-service agent for WeCom AiBot (WeChat Work bot), with cascade RAG retrieval, hierarchical memory, and an offline evaluation harness.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange)
![Milvus](https://img.shields.io/badge/Milvus-2.5-00bfa5)
![Redis](https://img.shields.io/badge/Redis-Streams-d82c20)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791)
![Coverage](https://img.shields.io/badge/coverage-82%25-green)

[Quick Start](#-quick-start) | [Features](#️-features) | [Architecture](#️-architecture) | [Evaluation](#-evaluation)

</div>

> [!NOTE]
> Single-tenant MVP. Multi-tenancy is deferred but not blocked.

## What is v-agent-platform?

v-agent-platform is a reactive customer-service agent that answers customer inquiries over WeCom Aibot (WeChat Work intelligent bot) via a persistent WebSocket. Inbound messages are debounced, ordered through a sharded Redis Streams bus, then handled by a LangGraph agent that retrieves grounded answers from a Milvus knowledge base and remembers each customer across sessions.

**Key highlights:**

- LangGraph state machine: `enter → compress → intent → agent → [tools]* → reflect → exit`
- Three-stage cascade RAG: dense → hybrid (BM25+dense) → LLM structural rewrite, with tuned thresholds
- Hierarchical memory: Redis working memory (hot) → Postgres `event_memory` (30-day vector recall) → `user_profile` (permanent)
- Token-efficient memory consolidation: extract mid-session at compression, promote at session end — no redundant LLM passes
- Strict per-shard serial ordering by `(channel, channel_user_id)`
- Built-in tools auto-described from docstrings: `calculator`, `search`, `recall_memory`, `subagent`
- Three-tier offline eval harness: retrieval (recall/MRR/nDCG), generation (RAGAS), system (latency + token cost)

## Who is this for?

Anyone building or studying **production-shaped Agentic systems over IM**: the strict app/engine layering, the bus + worker concurrency model, the hierarchical-memory lifecycle, and the evaluation harness are all decoupled and readable. A good reference for LangGraph orchestration, cascade RAG design, and how to evaluate a RAG pipeline end to end.

---

## Contents

- [🗞️ Features](#️-features)
- [📽️ Architecture](#️-architecture)
- [📁 Project Structure](#-project-structure)
- [📖 Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [Install](#install)
  - [Infrastructure](#infrastructure)
  - [Schema](#schema)
  - [Configuration](#configuration)
  - [Run](#run)
- [📊 Evaluation](#-evaluation)
- [🗝️ Tech Stack](#️-tech-stack)
- [⚠️ Security Notice](#️-security-notice)
- [📝 License](#-license)

---

## 🗞️ Features

| Category | Details |
|---|---|
| **Channel** | WeCom AiBot persistent WebSocket; exponential-backoff reconnect + 30s heartbeat |
| **Inbound bus** | Redis Streams, sharded by `(channel, channel_user_id)`; strict per-shard serial; DLQ on handler failure |
| **Debounce** | 500ms idle window merges message bursts into one turn before the bus |
| **Agent graph** | LangGraph: enter → compress → intent classify → agent → tool loop → reflect → exit |
| **Cascade RAG** | Stage-1 dense cosine → Stage-2 hybrid (dense+BM25 WeightedRanker) → Stage-3 LLM structural rewrite + hybrid |
| **Tools** | `calculator` (safe AST), `search` (Milvus knowledge recall), `recall_memory` (per-customer episodic), `subagent` (isolated single-shot delegate) |
| **Memory** | Redis working memory (TTL) + Postgres `event_memory` (pgvector, 30-day) + `user_profile` (permanent JSONB) |
| **Consolidation** | Mid-session extraction at token threshold; session-end promotion (working memory → profile + events), or extract-from-history for short sessions |
| **Reflection** | Post-answer fact-check against tool results; bounded retry on hallucination |
| **Observability** | structlog structured logging + optional LangSmith tracing (env-gated) |
| **Evaluation** | retrieval (recall@k / MRR / nDCG by difficulty + stage), generation (RAGAS 4 metrics), system (latency p50/p95 + token cost) |

---

## 📽️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  /backend/app/  ── Shell & Gateway                            │
│  WS frame normalization, debounce, Redis Streams bus,          │
│  bus worker (glue), infra lifecycle (Postgres/Redis/Milvus)    │
│  ────────────────────────────────────────────────────────     │
│  /backend/v/    ── Agent Engine                               │
│  LangGraph orchestration, LLM reasoning, RAG retrieval,        │
│  hierarchical memory, tools, skills, model factory             │
└──────────────────────────────────────────────────────────────┘
            Unidirectional dependency:  app → v   (v never imports app)
```

**Reactive flow:**

```
Customer → WeCom AiBot WS → wecom_aibot_worker (debounce + normalize → SystemMessage)
  → Redis Streams (sharded, per-shard serial) → bus worker → LangGraph agent
  → tools (search / recall_memory / calculator / subagent)
  → reply via Redis pub/sub → WecomAibotClient → Customer
```

On 30-min silence a new session is minted, and the previous session's working memory is promoted to long-term storage inline (no external queue).

Detailed docs: [doc/architecture.md](doc/architecture.md) · [doc/data_flow.md](doc/data_flow.md) · [doc/call_chain.md](doc/call_chain.md) · [doc/gaps.md](doc/gaps.md)

---

## 📁 Project Structure

```
v-main/
├── backend/
│   ├── app/                          # Shell & Gateway (pure I/O, routing, infra)
│   │   ├── main.py                   # FastAPI entry: lifespan, bus consumer task, health routes
│   │   ├── wecom_aibot_worker.py     # Standalone WS worker process (inbound frames + pub/sub outbound)
│   │   ├── wecom_aibot/              # WeCom AiBot adapter
│   │   │   ├── client.py             # Persistent WS client (reconnect + heartbeat)
│   │   │   ├── debounce.py           # 500ms merge window (Redis hash + asyncio timer)
│   │   │   └── outbound.py           # Redis pub/sub → WS send bridge
│   │   ├── bus/                      # Redis Streams reactive bus
│   │   │   ├── shard.py              # mmh3 routing by (channel, channel_user_id)
│   │   │   ├── producer.py / consumer.py
│   │   │   └── worker.py             # make_bus_handler: glue graph + outbound + session-end consolidation
│   │   ├── gateway/                  # RequestIdMiddleware + /livez /readyz /health
│   │   └── store/                    # asyncpg pool + redis.asyncio factories
│   │
│   └── v/                            # Agent Engine (pure logic)
│       ├── agents/                   # state, nodes, edges, graph, checkpoints (Redis hot / PG cold)
│       │   ├── compression_node.py   # Mid-session memory extraction + history trim
│       │   └── intent_reflect.py     # Intent classifier + reflection (hallucination check)
│       ├── rag/retriever.py          # KnowledgeRetriever: Milvus 3-stage cascade
│       ├── memory/                   # working (Redis) / event_memory (pgvector) / user_profile / extractor
│       ├── tools/                    # calculator, search, recall_memory, subagent
│       ├── models/                   # LLMCaller (timeout + same-family fallback) + factory
│       ├── skills/                   # markdown SOP loader + keyword matcher
│       ├── hooks/                    # on_session_start, tool_guard
│       └── configs/base.py           # Pydantic settings (one class per env prefix)
│
│   ├── eval/                         # Offline evaluation harness
│   │   ├── retrieval/                # run_eval.py (cascade) + run_ablation.py + metrics.py
│   │   ├── generate/                 # run_generate_eval.py (RAGAS 4 metrics)
│   │   ├── system/                   # run_system_eval.py (latency + token cost)
│   │   ├── gen/                      # LLM data generators (catalog / QA / difficulty scorer)
│   │   ├── seed_milvus.py            # Embed corpus → Milvus collection
│   │   └── data/                     # products.jsonl / faq.jsonl / qa.jsonl / reports
│   │
│   └── test/                         # pytest unit + e2e suites (402 tests, 82% coverage)
│
├── docker/docker-compose.yml         # postgres(pgvector) + redis + etcd + minio + milvus 2.5 + app
├── scripts/sql/                      # Raw SQL migrations (Alembic deferred)
├── doc/                              # architecture / data_flow / call_chain / gaps
├── CLAUDE.md                         # Repo-wide engineering rules (+ per-layer CLAUDE.md)
└── README.md
```

---

## 📖 Quick Start

### Prerequisites

- `Python ≥ 3.11`
- [uv](https://docs.astral.sh/uv/) — package manager / runner
- Docker + Docker Compose (local Postgres + Redis + Milvus)
- `psql` CLI (for schema migration)
- An OpenAI-compatible LLM + embedding endpoint (default: DashScope compatible-mode — DeepSeek chat + Qwen `text-embedding-v4`)

### Install

```bash
uv sync
```

Creates `.venv/` and installs from `uv.lock`. Run everything with `uv run ...`.

### Infrastructure

```bash
docker compose -f docker/docker-compose.yml up -d postgres redis etcd minio milvus
docker compose -f docker/docker-compose.yml ps   # wait until healthy
```

The `app` service is profiled (`profiles: ["app"]`) and is not started by default.

### Schema

```bash
bash scripts/db_apply.sh
```

Runs [scripts/sql/](scripts/sql/) in order: pgvector extension → `agent` schema (working memory / checkpointer / user_profile / event_memory / knowledge_chunk) → `dw` business warehouse → `meta` NL2SQL metadata. Reads `POSTGRES_DSN`, defaulting to `postgresql://postgres:postgres@localhost:5432/agent`.

### Configuration

Two layers, by design:

| File | Holds | Precedence |
|---|---|---|
| `config.yaml` | **non-secret tunables** — model names, thresholds, TTLs, shard counts, RAG weights | lowest (above code defaults) |
| `.env` | **secrets + deployment infra** — API keys, DSNs, bot credentials | overrides YAML |

Final precedence (highest wins): **env var > `.env` > `config.yaml` > code default**. Edit `config.yaml` to change behaviour without touching code; keep secrets out of it.

```bash
cp config.example.yaml config.yaml    # tunables — edit freely
cp .env.example .env                  # secrets — fill in API keys / credentials
```

The YAML is *sectioned by reflection*: each top-level key maps to an `AppSettings` field (`llm`, `rag`, `memory`, `bus`, …) — no hand-written wiring, so a new settings class needs no extra config plumbing. Point at a different file with `APP_CONFIG_FILE=/path/to/config.yaml`.

`.env` (secrets) minimum required:

| Group | Keys |
|---|---|
| LLM | `LLM_BASE_URL_DEEPSEEK` / `LLM_API_KEY_DEEPSEEK` / `LLM_BASE_URL_QWEN` / `LLM_API_KEY_QWEN` |
| Postgres | `POSTGRES_DSN` |
| Redis | `REDIS_URL` |
| Milvus | `MILVUS_TOKEN` (cloud only) |
| WeCom AiBot | `WECOM_AIBOT_WS_URL` / `WECOM_AIBOT_BOT_ID` / `WECOM_AIBOT_SECRET` |
| LangSmith (optional) | `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` |

`config.yaml` tunables (see `config.example.yaml`): `llm.main_primary/fallback`, `embedding.model`, `memory.*` TTLs + compression thresholds, `bus.shard_count/debounce_ms`, `rag.min_score/stage2_min_score/weights`, `milvus.uri/collection_name`.

Leaving `WECOM_AIBOT_WS_URL` empty disables the WS worker (it exits immediately on start).

### Run

Two processes:

**1. FastAPI main process** (HTTP gateway + in-process bus worker):

```bash
uv run fastapi dev backend/app/main.py      # listens on :8000
```

**2. WeCom AiBot WS worker** (holds the persistent WebSocket):

```bash
uv run python -m backend.app.wecom_aibot_worker
```

Inbound `aibot_msg_callback` frames → normalized to `SystemMessage` → bus → same downstream as any channel. Outbound replies are published to Redis pub/sub `wecom_aibot:outbound` and forwarded over the WS by this worker.

Before first use, seed the knowledge base:

```bash
uv run python -m backend.eval.seed_milvus
```

### Verify

```bash
uv run pytest --cov=backend --cov-fail-under=80   # 402 tests, ≥80% gate
uv run ruff check . --fix && uv run ruff format .
```

Health probes on the FastAPI process: `GET /livez` (always 200), `GET /readyz` (503 if Postgres/Redis down), `GET /health` (legacy).

---

## 📊 Evaluation

A three-tier offline harness over a 150-doc corpus (120 products + 30 FAQ) and 337 difficulty-balanced QA pairs. See [backend/eval/README.md](backend/eval/README.md) for full results and parameter tuning.

```bash
# Retrieval — recall@k / MRR / nDCG, by difficulty + cascade stage
uv run python -m backend.eval.retrieval.run_eval

# Retrieval ablation — dense vs hybrid variants vs rewrite+hybrid
uv run python -m backend.eval.retrieval.run_ablation --no-rewrite

# Generation — RAGAS: Faithfulness / ContextRecall / AnswerRelevancy / AnswerCorrectness
uv run python -m backend.eval.generate.run_generate_eval --gen-model qwen3-8b --no-think

# System — E2E latency (p50/p95/p99) + token consumption + cost estimate
uv run python -m backend.eval.system.run_system_eval
```

Headline retrieval result (337 queries, top_k=5): **hit@5 = 0.982**, cascade stage split ≈ 60% : 30% : 10% (dense : hybrid : rewrite).

---

## 🗝️ Tech Stack

<details>
<summary>1. Agent Orchestration (click me)</summary>

LangGraph state machine in `backend/v/agents/graph.py`:

```
enter → compress → intent → agent → route_after_agent ─┬→ tools → agent → ...
                                                       └→ reflect ─┬→ agent (retry)
                                                                   └→ exit → END
```

- `enter`: stacks the per-turn system prompt — base persona, `<customer_profile>`, `<recent_events>`, session working memory, matched SOPs.
- `compress`: when `messages` exceed the token threshold, an LLM extracts `working_memories` (→ Redis) + `event_memories` (→ Postgres), then trims history behind a `<compressed_history>` marker.
- `intent`: flash-tier LLM classifies the turn (refund / logistics / complaint / general).
- `agent`: main-tier LLM with bound tools; loops through `ToolNode` until no more tool calls.
- `reflect`: fact-checks the reply against tool results; bounded retry on detected hallucination.

`LLMCaller` (`backend/v/models/llm_caller.py`) is the single entry point for all LLM calls: 30s timeout, same-family fallback (primary → fallback), structured output via Pydantic, and `RunnableConfig` propagation for nested LangSmith traces.

</details>

<details>
<summary>2. Cascade RAG (click me)</summary>

`KnowledgeRetriever` (`backend/v/rag/retriever.py`) runs a three-stage cascade against a single Milvus collection (`knowledge_chunks`):

```
query
  ▼ Stage-1: dense cosine search          score ≥ 0.76  → return (~60%)
  ▼ Stage-2: hybrid WeightedRanker         score ≥ 0.41  → return (~30%)
  ▼ Stage-3: LLM structural rewrite + hybrid             → return (~10%)
```

- Collection schema: `source_type`, `source_id`, `text` (BM25-analyzed), `embedding` (FLOAT_VECTOR 1024), `sparse_embedding` (auto from BM25 Function), `metadata` (JSON).
- Indexes: HNSW (M=16, efConstruction=64, COSINE) + SPARSE_INVERTED_INDEX (BM25).
- Stage-2/3 thresholds were tuned on the eval corpus; note hybrid WeightedRanker scores live on a different scale (~0.40–0.90) than dense cosine (~0.60–0.95).

The `search` tool is a thin `@tool` wrapper over this retriever; the seeder reuses the exact same collection setup for schema parity.

</details>

<details>
<summary>3. Hierarchical Memory (click me)</summary>

Three temperature tiers, all sentence-shaped via `MemoryEntry`:

- **Working memory** (Redis, TTL = session window): per-session list, injected into the prompt each turn.
- **Event memory** (Postgres `agent.event_memory`, pgvector, 30-day TTL): cross-session episodic recall via the `recall_memory` tool (recency-weighted cosine).
- **User profile** (Postgres `agent.user_profile`, JSONB, permanent): one row per `(channel, channel_user_id)`, fully injected at `on_session_start`.

**Consolidation is token-efficient by design:**
- *Mid-session* (token threshold) — `compression_node` extracts working + event memories from the dropped history. No profile write (session still active).
- *Session-end* (next message after silence) — `_promote_prev_session` fires fire-and-forget: if working memory is non-empty it promotes directly (no extra LLM call); if empty (short session) it runs `extract_from_messages()` over the checkpoint history, then promotes.

</details>

<details>
<summary>4. Bus & Concurrency (click me)</summary>

- `RedisStreamShard`: `mmh3.hash(f"{channel}:{channel_user_id}") % shard_count` (default 64 shards) → strict per-shard serial; different shards run concurrently.
- `Debouncer`: merges sub-500ms bursts from the same identity into one turn before the bus (Redis hash + asyncio timer).
- `BusConsumer`: one asyncio task per shard, `XREADGROUP` with DLQ on handler failure.
- Outbound replies do **not** re-enter the bus — they go straight back through the channel adapter (the bus is reactive-inbound only).
- LangGraph checkpointer: Redis on the hot path (`RedisCheckpointer`, TTL), with a langgraph-official `AsyncPostgresSaver` available for cold storage.

</details>

<details>
<summary>5. Models & Infra (click me)</summary>

- **LLM / Embedding**: OpenAI-compatible via `langchain-openai` `ChatOpenAI(base_url=...)`. Default DashScope compatible-mode — DeepSeek chat models + Qwen `text-embedding-v4` (1024-dim). `check_embedding_ctx_length=False` so DashScope receives raw strings, not token ids.
- **Postgres**: single instance, multiple schemas — `agent` (memory/sessions/checkpoints/pgvector), `dw` (business warehouse), `meta` (NL2SQL metadata). asyncpg only; psycopg2 forbidden.
- **Milvus 2.5**: vector + BM25 hybrid; deployed via docker-compose with etcd + minio.
- **Redis**: bus, working memory, checkpointer hot path, pub/sub outbound.
- **Config**: `pydantic-settings`, one class per env prefix, composed by a cached `get_settings()`.

</details>

<details>
<summary>6. Evaluation Harness (click me)</summary>

`backend/eval/` is split into three sub-modules plus data generators:

- `retrieval/` — `run_eval.py` (end-to-end cascade: recall/MRR/nDCG/hit by difficulty + stage attribution) and `run_ablation.py` (dense vs hybrid weight variants vs rewrite+hybrid, run independently).
- `generate/` — `run_generate_eval.py` scores answers with RAGAS 0.4 (Faithfulness / ContextRecall / AnswerRelevancy / AnswerCorrectness). The judge LLM is fixed to DeepSeek for apples-to-apples comparison; `--gen-model` overrides only the generation model (e.g. `qwen3-8b --no-think`).
- `system/` — `run_system_eval.py` runs the full graph per query, capturing latency p50/p95/p99 and token counts (via a LangChain callback), with a configurable price table for cost estimation.
- `gen/` — LLM generators that build the catalog, QA pairs, and difficulty scores; `seed_milvus.py` embeds the corpus into Milvus with the production schema.

</details>

---

## ⚠️ Security Notice

This platform is an **MVP / learning project** intended for **trusted local or internal-network** use. It does not ship production-grade hardening:

- **No gateway authentication.** Internal admin/health routes are open to anyone who can reach the process.
- **Secrets in plaintext `.env`.** `LLM_API_KEY_*`, `POSTGRES_DSN`, WeCom bot secrets are read from `.env` via `pydantic-settings`. Never commit `.env`; rotate keys manually.
- **Default credentials** in `.env.example` / `docker-compose.yml` (e.g. `postgres:postgres`) must be changed before any non-local deployment.
- **Plain transport / open CORS** by default. Terminate TLS and restrict origins behind a reverse proxy if exposed beyond localhost.
- **Milvus / Postgres / Redis** are exposed without network-level access control by default — use firewall rules or Docker network isolation.

> [!CAUTION]
> Do not expose this directly to the public internet without adding authentication, TLS, and access controls.

---

## 📝 License

Open source under the [MIT License](./LICENSE).
