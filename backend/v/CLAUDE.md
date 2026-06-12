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
  - `/memory`: Hierarchical memory manager (V2 — three bounded tiers).
    - `working.py`: Redis-backed working memory (TTL 30 min). Single source of truth for active-session state (hot path).
    - `event_memory.py`: episodic tier (`agent.event_memory`, append-only events + pgvector). `insert_episodic_candidates` (V2 write, tier='raw') + `read_recent_event_memories`.
    - `user_memory.py`: semantic tier (`agent.user_memory`, bi-temporal overwritable attributes). `upsert_user_memory` (three-step supersede) + `rebuild_profile_cache` (projects active rows into the `user_profile` JSONB cache).
    - `long_term.py`: reads the `user_profile` JSONB cache.
    - `policy_gate.py`: deterministic write gate — importance floor + PII mask + episodic same-subject dedup.
    - `memory_extractor.py`: `promote_to_long_term` — the single session-end durable-write orchestrator (one LLM call → both tiers).
    - `consolidation.py`: `consolidate_user` — monthly episodic forgetting (cluster→summary + ACT-R activation prune); lazy-triggered on session start.
  - `/tools`: Built-in tools, all implemented via LangChain `@tool(parse_docstring=True)` so the docstring IS the tool description exposed to the LLM. Each tool's "when to use" lives in its own docstring, not in the system prompt.
    - `calculator` — safe AST eval (no `eval()`, whitelisted ops, exponent ≤ 64).
    - `search` — thin tool wrapper over `rag/retriever.py`; generic semantic recall over `agent.knowledge_chunk` (products / FAQ / policies). Does NOT recency-rank.
    - `recall_memory` — on-demand semantic query against `agent.event_memory` (recency-weighted cosine) for THIS customer's prior conversations.
    - `subagent` — focused single-shot delegate with its own context window. Function intentionally NOT pinned.
    - `load_skill` — progressive-disclosure loader: returns the full SOP body for a skill named in the cold-layer `<available_skills>` catalog. The body lands in the conversation (hot, append-only) instead of the system prompt, keeping the cached prefix stable.
  - `/rag`: Retrieval layer. `retriever.py` owns the Milvus cascade pipeline (dense → hybrid → LLM-rewrite), top_k clamping, and result formatting. Tools and future endpoints import from here — no logic duplication. See `backend/eval/README.md` for eval results and parameter tuning.
  - `/mcp`: Model Context Protocol client. Connects to external MCP servers (stdio / streamable-HTTP / SSE), discovers their tools, and exposes them as LangChain `StructuredTool`s named `{server_id}__{tool_name}`. `config.py` (server schema + JSON parser), `oauth.py` (OAuth 2.0 client_credentials token provider — cached + auto-refreshed), `client.py` (one `ClientSession` per server; resolves OAuth bearer at connect), `cache.py` (L1 in-mem + L2 Redis tool-result cache; write-class tools opt out), `registry.py` (lifecycle + tool wrapping). Disabled when `MCP_SERVERS_JSON` is empty.
  - `/skills`: Skill management. Two formats supported: pure markdown SOP, and `SKILL.md` + executable scripts. Sources: internal repo + community registry. **Executable scripts run only from internal-repo or whitelisted community sources; un-whitelisted community skills degrade to markdown-only mode.**
  - `/models`: LLM and embedding factory. OpenAI-compatible adapters via `langchain-openai` `ChatOpenAI(base_url=...)`. Per-task routing read from `.env`. Same-family fallback only (e.g., `pro → flash`); cross-family fallback deferred. Trigger conditions: 30-second timeout (one shot), HTTP 5xx, HTTP 429.
  - `/hooks`: Lifecycle interceptors. Specific responsibilities:
    - `on_session_start`: inject full `user_profile` row.
    - `on_session_end`: final consolidation + extract long-term memory.
  - `/configs`: Pydantic settings models per module. All parameters initialized here.
  - `/utils`: Cross-cutting helpers (logging, retry, structured-output decoders).

# Technology stack and versions
- **Language**: Python 3.11+
- **Agent Framework**: LangGraph + LangChain
- **Package Management**: `uv`
- **DB / Cache**: PostgreSQL (with `pgvector`) for persistence; Redis for working memory, bus, and ARQ.
- **Validation**: Pydantic v2

# Architecture constraints
- **Passive Invocation**: `/v/` exposes interfaces; it is invoked by bus workers or ARQ. Never define FastAPI routes or auth here.
- **Hierarchical Memory Lifecycle (CRITICAL — memory V2)**: three clearly-bounded tiers.
  - **Working memory (Redis, session-scoped)**: current `messages` + LangGraph state + distilled session signal. TTL 30 min. **Single source of truth on the hot path.** New session_id minted after 30-min silence.
    - **Mid-session** (token threshold): `compression_node` runs ONE LLM call → a structured `conversation_state` (current_topic / events / actions_taken / unresolved_questions / key_facts) + working/event sentences. The sentences fold into **Redis working memory only** (NO mid-session Postgres write); the dropped head is replaced by a `<compressed_history>` block rendering the state for seamless resumption.
  - **Episodic memory (`agent.event_memory`, append-only events)**: "what happened" — complaints, enquiries, orders. Vectorized (Qwen `text-embedding-v4`, 1024-dim), pgvector HNSW. Recalled by the `recall_memory` tool (recency+importance cosine; bumps `access_count` per hit). Events never conflict — only append, dedup (same-subject cosine), and **monthly consolidation**: closed-month per-subject clusters are summarised into one `tier='summary'` row and low-**activation** raws (`importance + ln(1+access) − ln(1+age)`, ACT-R) are pruned — use-it-or-lose-it. Lazy trigger: fired fire-and-forget when a returning customer starts a session.
  - **User memory (`agent.user_memory`, bi-temporal, overwritable)**: "what the customer is like" — preferences / constraints / patterns. One active row per `attr_key`; an overwrite supersedes (old row `status='superseded'`, `valid_to`, `superseded_by` — never deleted, the chain is the audit log). Arbitration: stated>inferred>confidence>recency. No embedding; projected into the `user_profile` JSONB cache (rebuilt on each write) and injected whole at `on_session_start`. Never forgotten — only flagged stale via `last_confirmed_at`.
  - **Session-end consolidation** (`promote_to_long_term`, fire-and-forget on session mint): ONE `memory_extract` LLM call over working memory (or the checkpoint transcript for a short session) → `MemoryExtraction` → policy gate (importance floor + PII mask) → bi-temporal upsert of user candidates + episodic append (embed-once → dedup). **This is the single durable-write site** — all PII is masked here before storage (`utils/pii.py`).
- **Subagent Has No Common Abstraction**: do NOT introduce a `BaseSubAgent`. Background analysts (summarizer, memory_extractor) are async coroutines. The `subagent` tool is a focused single-shot LLM call. They share only the `LLMCaller` utility.
- **Skill**: install / update / list / pin operations are handled in `/skills`. Trust boundary enforced when loading executable skills. **Injection model (cold/hot)**: `enter_node` emits the system prompt in volatility tiers so the provider's prefix cache stays warm — a COLD `SystemMessage` (persona + channel + frozen `<available_skills>` catalog; customer-independent, cross-session-cacheable) followed by a WARM one (profile + recent_events + session_memory; per-customer, frozen until the next compression). The conversation history is appended after, append-only. Skills are NOT keyword-matched into the prompt; the model selects from the catalog and pulls bodies on demand via the `load_skill` tool (hot layer).

# Built-in Tools (LangChain `@tool(parse_docstring=True)`)
All tool descriptions live in their docstrings (single source of truth — the system prompt does NOT re-document them). Adding/changing a tool means editing the docstring; the LLM sees it via `parse_docstring=True`.
- `calculator` — deterministic arithmetic over a whitelisted AST.
- `search` — semantic recall over `agent.knowledge_chunk` (products / FAQ / policies). Optional `source_type` filter. Returns chunks tagged `[source_type:source_id]`.
- `recall_memory` — semantic recall over `agent.event_memory` for the current customer (recency-weighted cosine). Customer-scoped; cannot leak across users.
- `subagent` — single-shot delegate with isolated context. Function intentionally unrestricted: the main agent invokes it whenever an independent reasoning context helps. Subagents may register their own tools (e.g., NL2SQL one would carry `query_metric_meta` + read-only `execute_sql`), but the parent's docstring does NOT pin a fixed use case.
- `load_skill` — fetches a SOP body by name from the frozen `<available_skills>` catalog (Anthropic-style progressive disclosure). The catalog (names + descriptions only) sits in the cold system layer; the model calls this tool when it needs the full procedure, and the body is appended to the conversation rather than injected into the prompt.

# LLM & Embedding Routing
All providers must be OpenAI-compatible (use `ChatOpenAI(base_url=...)`).

Routing keys read from `.env` (see `.env.example` for full list):

```
LLM_MAIN_PRIMARY=deepseek-v4-flash
LLM_MAIN_FALLBACK=deepseek-v3.1
EMBEDDING_MODEL=text-embedding-v4
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
