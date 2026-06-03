# Project Description and Function Overview
The agent engine (`/backend/v/`) for a LangGraph-based external customer-service system over WeCom 智能机器人 WebSocket.

- **What it does**: ONLY core closed-loop agent logic — LangGraph state orchestration, intent reasoning, tool invocation, RAG retrieval, Skill integration, hierarchical memory management (Redis hot + Postgres cold + pgvector), and ARQ-driven proactive cron tasks.
- **What it does NOT do**: NO API gateway auth, NO web routing, NO webhook signature handling, NO frontend rendering. Those belong to `/backend/app/`.

# Project structure and corresponding functions
Strict directory boundaries:

- **Upstream Interaction (External — `/backend/app/`)**:
  - `/app/wecom_aibot`: WeCom 智能机器人 WS client (Debouncer, WecomAibotClient, WecomAibotOutbound).
  - `/app/bus`: Redis Streams reactive-inbound bus (strict per-shard serial).

- **Core Logic (`/backend/v/`)**:
  - `/agents`: LangGraph brain. State definitions (`state.py`), nodes (`nodes.py`), edges (`edges.py`), graph assembly (`graph.py`). Exposes invocation interfaces consumed by bus workers and cron.
  - `/agents/background/`: **Background analysts** (NOT graph nodes, NOT subagents). Async coroutines triggered by hooks: `summarizer.py` for context compression. Runs out-of-band so the main graph is not blocked.
  - `/memory`: Hierarchical memory manager.
    - `working.py`: Redis-backed working memory (TTL 30 min). Single source of truth for active-session state and LangGraph checkpointer (hot path).
    - `session.py`: Postgres-backed session memory (consolidated working memory of completed/expiring sessions).
    - `long_term.py`: Postgres long-term memory split into two tables — see Memory Hierarchy below.
    - `memory_extractor.py`: async coroutine that extracts profile + episodic memories from session memory using DeepSeek v4 flash with structured Pydantic output. Handles **dedup, conflict, and obsolescence** on write to `user_profile` and `memory_episodes`.
  - `/tools`: Built-in tools, all implemented via LangChain `@tool(parse_docstring=True)` so the docstring IS the tool description exposed to the LLM. Each tool's "when to use" lives in its own docstring, not in the system prompt.
    - `calculator` — safe AST eval (no `eval()`, whitelisted ops, exponent ≤ 64).
    - `search` — thin tool wrapper over `rag/retriever.py`; generic semantic recall over `agent.knowledge_chunk` (products / FAQ / policies). Does NOT recency-rank.
    - `recall_memory` — on-demand semantic query against `agent.event_memory` (recency-weighted cosine) for THIS customer's prior conversations.
    - `subagent` — focused single-shot delegate with its own context window. Function intentionally NOT pinned.
  - `/rag`: Retrieval layer. `retriever.py` owns the pgvector SQL (filtered + unfiltered branches), top_k clamping, and result formatting. Tools and future endpoints import from here — no logic duplication.
  - `/skills`: Skill management. Two formats supported: pure markdown SOP, and `SKILL.md` + executable scripts. Sources: internal repo + community registry. **Executable scripts run only from internal-repo or whitelisted community sources; un-whitelisted community skills degrade to markdown-only mode.**
  - `/models`: LLM and embedding factory. OpenAI-compatible adapters via `langchain-openai` `ChatOpenAI(base_url=...)`. Per-task routing read from `.env`. Same-family fallback only (e.g., `pro → flash`); cross-family fallback deferred. Trigger conditions: 30-second timeout (one shot), HTTP 5xx, HTTP 429.
  - `/hooks`: Lifecycle interceptors and centralized exception handling. Specific responsibilities:
    - `on_session_start`: inject full `user_profile` row.
    - `on_context_threshold`: trigger `summarizer` async + schedule ARQ delayed `consolidate_session(session_id)` job at `TTL - Δ`.
    - `on_session_end`: final consolidation + extract long-term memory.
  - `/configs`: Pydantic settings models per module. All parameters initialized here.
  - `/cron`: ARQ task definitions. **MVP scenarios**: `logistics_delivery_notification`, `ad_hoc_ad_push`, `repurchase_reminder`. Also hosts `consolidate_session` delayed-job worker.
  - `/utils`: Cross-cutting helpers (logging, retry, structured-output decoders).

# Technology stack and versions
- **Language**: Python 3.11+
- **Agent Framework**: LangGraph + LangChain
- **Package Management**: `uv`
- **DB / Cache**: PostgreSQL (with `pgvector`) for persistence; Redis for working memory, bus, and ARQ.
- **Validation**: Pydantic v2

# Architecture constraints
- **Passive Invocation**: `/v/` exposes interfaces; it is invoked by bus workers or ARQ. Never define FastAPI routes or auth here.
- **Hierarchical Memory Lifecycle (CRITICAL)**:
  - **Working Memory (Redis)**: current-session `messages` + LangGraph state. TTL 30 min. **Single source of truth on the hot path.** New session_id is created when 30-min silence elapses (long-term memory injection bridges continuity).
  - **Session Memory (Postgres)**: consolidated summary of a session. Written by hook-triggered `summarizer` when (a) context exceeds token threshold, or (b) ARQ delayed job fires near TTL expiry, or (c) `on_session_end`. Cleared/archived after long-term extraction.
  - **Long-term Memory (Postgres, two tables)**:
    - `user_profile`: structured, **one row per `(channel, channel_user_id)`** (preferences, member level, recent-order summary, etc.). Fully injected into system prompt at every `on_session_start`.
    - `event_memory`: vectorized (Qwen `text-embedding-v4`, 1024-dim Matryoshka), pgvector HNSW index. Queried on-demand via the `recall_memory` tool with recency-weighted cosine ranking. Distinct from `agent.knowledge_chunk` (which `search` reads): event_memory is per-customer episodic; knowledge_chunk is the shared corpus.
    - Writes use `memory_extractor` with DeepSeek v4 flash + Pydantic structured output. **Dedup, conflict, and obsolescence are mandatory** — never blindly append.
- **Subagent Has No Common Abstraction**: do NOT introduce a `BaseSubAgent`. Background analysts (summarizer, memory_extractor) are async coroutines. The `subagent` tool is a focused single-shot LLM call. They share only the `LLMCaller` utility.
- **Skill**: install / update / list / pin operations are handled in `/skills`. Trust boundary enforced when loading executable skills.
- **Cron (ARQ)**: scheduled outreach + delayed memory consolidation. Cron-generated outbound messages MUST be persisted to working memory before being pushed to the channel — they are part of conversation history, not fire-and-forget.

# Built-in Tools (LangChain `@tool(parse_docstring=True)`)
All tool descriptions live in their docstrings (single source of truth — the system prompt does NOT re-document them). Adding/changing a tool means editing the docstring; the LLM sees it via `parse_docstring=True`.
- `calculator` — deterministic arithmetic over a whitelisted AST.
- `search` — semantic recall over `agent.knowledge_chunk` (products / FAQ / policies). Optional `source_type` filter. Returns chunks tagged `[source_type:source_id]`.
- `recall_memory` — semantic recall over `agent.event_memory` for the current customer (recency-weighted cosine). Customer-scoped; cannot leak across users.
- `subagent` — single-shot delegate with isolated context. Function intentionally unrestricted: the main agent invokes it whenever an independent reasoning context helps. Subagents may register their own tools (e.g., NL2SQL one would carry `query_metric_meta` + read-only `execute_sql`), but the parent's docstring does NOT pin a fixed use case.

# LLM & Embedding Routing
All providers must be OpenAI-compatible (use `ChatOpenAI(base_url=...)`).

Routing keys read from `.env` (see `.env.example` for full list):

```
LLM_MAIN_PRIMARY=deepseek-chat-v4-pro
LLM_MAIN_FALLBACK=deepseek-chat-v4-flash      # same-family only in MVP
LLM_SUMMARY=deepseek-chat-v4-flash
LLM_MEMORY_EXTRACT=deepseek-chat-v4-flash
EMBEDDING_MODEL=qwen-text-embedding-v4
EMBEDDING_DIM=1024                            # Matryoshka — keep 1024 for HNSW
```

Failure → fallback triggers: 30-second timeout (single attempt, no retry on primary), HTTP 5xx, HTTP 429.

# Coding conventions
- **Naming**: classes `CamelCase` (e.g., `CustomerServiceState`); functions/variables `snake_case`. LangGraph node functions end with `_node`; routing edge functions start with `route_`.
- **Documentation**: Google-Style docstrings on all functions and classes. Document parameters, return types, and exceptions explicitly.
- **Async First**: every I/O operation is `async`/`await`.
- **Error Handling**: NEVER use blanket `try/except Exception`. Tool failures, LLM errors, and external-service errors are caught in `/hooks` (e.g., `post_tool` injects failures back into graph state).
- **Logging**: log every state transition, tool invocation, fallback trigger, and error. Use structured logs (key=value or JSON) for downstream parsing.

# Forbidden patterns
- **No Hardcoded Parameters**: zero magic numbers, URLs, prompts, or thresholds in business logic. Everything comes from `/configs`.
- **No Hardcoded Secrets**: API keys, DB credentials, webhook secrets all come from `.env` via `pydantic-settings`.
- **No State Mutation**: never overwrite the incoming `state` dict in a node. Return a new dict containing only incremental updates. The `messages` key MUST use `add_messages` reducer to append.
- **No Manual LLM Output Parsing**: never `json.loads()` an LLM response. Use Structured Output bound to Pydantic models.
- **No Subagent Abstraction**: do NOT create a `BaseSubAgent` parent. Background analysts and the subagent tool are different things and must remain so.
- **No Synchronous Drivers**: never use `psycopg2`, `redis-py` sync client, or `requests`. Use `asyncpg`, `redis.asyncio`, `httpx`.

# Key file paths
- Graph state: `/backend/v/agents/state.py`
- Memory layer: `/backend/v/memory/`
- Nodes & lifecycle hooks: `/backend/v/agents/`, `/backend/v/hooks/`
- Model factory: `/backend/v/models/`
- Global exceptions: `/backend/v/exceptions/`

# Test and build commands
- **Install**: `uv sync` (or `uv pip install -e .`), the default environment path is `/mnt/data/conda_envs/agent`
- **Lint & Format**: `ruff check . --fix && ruff format .`
- **Run Unit Tests**: `pytest`
- **Coverage Gate**: `pytest --cov=backend/v --cov-fail-under=80`
