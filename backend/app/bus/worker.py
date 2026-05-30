"""Bus consumer handler: glue between debounced inbound messages and the graph.

For each :class:`SystemMessage` consumed off the bus:

1. Resolve the session id (mint a new UUID after 30 minutes of silence).
2. **If the session is suspended (operator handoff active)**, forward the
   customer message to the Slack alert thread and return; do NOT invoke
   the graph.
3. Otherwise run ``on_session_start`` to populate ``user_profile``.
4. Build the initial ``CustomerServiceState`` with a single ``HumanMessage``
   for the new turn.
5. Invoke the compiled graph with ``thread_id=session_id``.
6. **If the graph paused at a ``transfer_to_human`` interrupt**, run the
   ``on_interrupt`` hook (migrate hot->cold, post Slack alert, mark
   suspended) instead of dispatching a normal reply.
7. Otherwise pull the last ``AIMessage`` from the resulting state and
   dispatch it via the channel-specific outbound adapter.
8. Refresh the ``last_seen`` timestamp.

Outbound replies do NOT re-enter the bus — they flow directly to the
channel adapter (per the architectural rule in ``backend/app/CLAUDE.md``).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import asyncpg
import redis.asyncio as redis_async
from langchain_core.messages import AIMessage, HumanMessage

from backend.app.bus.messages import SystemMessage
from backend.v.agents.checkpoints.redis import RedisCheckpointer
from backend.v.agents.state import CustomerServiceState
from backend.v.hooks.handoff import (
    HandoffNotifier,
    extract_interrupt,
    is_suspended,
    on_interrupt,
)
from backend.v.hooks.session import on_session_start
from backend.v.models.llm_caller import LLMCaller
from backend.v.utils.logging import bind_request, get_logger
from backend.v.utils.reply import split_reply_segments

_log = get_logger("bus.worker")

# Outbound dispatch: channel slug -> async (channel_user_id, text) -> None.
SendFn = Callable[[str, str], Awaitable[None]]
SendRegistry = dict[str, SendFn]


async def _resolve_session_id(
    redis: redis_async.Redis,
    *,
    channel: str,
    channel_user_id: str,
    silence_seconds: int,
) -> tuple[str, bool]:
    """Return ``(session_id, was_minted)``.

    Mints a new session id when no session exists or when ``last_seen`` is
    older than ``silence_seconds``. Long-term memory injection at session
    start bridges continuity across the new session boundary.
    """
    last_seen_key = f"last_seen:{channel}:{channel_user_id}"
    session_key = f"session:{channel}:{channel_user_id}"

    last_seen_raw = await redis.get(last_seen_key)
    if last_seen_raw:
        try:
            last_seen = float(last_seen_raw.decode("ascii"))
        except (UnicodeDecodeError, ValueError):
            last_seen = 0.0
        if (time.time() - last_seen) < silence_seconds:
            existing = await redis.get(session_key)
            if existing:
                return existing.decode("utf-8"), False

    session_id = str(uuid.uuid4())
    await redis.set(session_key, session_id, ex=silence_seconds * 2)
    return session_id, True


async def _persist_session_row(
    pool: asyncpg.Pool,
    *,
    session_id: str,
    channel: str,
    channel_user_id: str,
) -> None:
    """Insert an ``agent.session`` row for a freshly minted session.

    Idempotent on concurrent retries (``ON CONFLICT DO NOTHING``). Required
    so downstream operations (e.g., ``on_interrupt`` flipping status to
    ``suspended``) can reference the session by id.
    """
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO agent.session (session_id, channel, channel_user_id) "
            "VALUES ($1, $2, $3) "
            "ON CONFLICT (session_id) DO NOTHING",
            session_id,
            channel,
            channel_user_id,
        )


async def _record_last_seen(
    redis: redis_async.Redis,
    *,
    channel: str,
    channel_user_id: str,
    silence_seconds: int,
) -> None:
    last_seen_key = f"last_seen:{channel}:{channel_user_id}"
    await redis.set(last_seen_key, str(time.time()), ex=silence_seconds * 2)


def _last_ai_message(messages: list) -> str | None:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            content = m.content
            if isinstance(content, str) and content:
                return content
            if isinstance(content, list):
                # LangChain content can be a list of content parts; flatten text bits.
                parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                return "".join(parts).strip() or None
    return None


def make_bus_handler(
    *,
    graph: Any,
    pool: asyncpg.Pool,
    redis: redis_async.Redis,
    llm_caller: LLMCaller,
    sends: SendRegistry,
    silence_seconds: int,
    cache_ttl_seconds: int,
    slack_outbound: HandoffNotifier | None = None,
    redis_ckpt: RedisCheckpointer | None = None,
    pg_ckpt: Any | None = None,
    embedder: Any | None = None,
    skill_registry: Any | None = None,
    skill_top_k: int = 3,
    recent_events_to_inject: int = 0,
    compression_threshold_tokens: int = 0,
    compression_keep_recent_messages: int = 4,
    token_model: str | None = None,
    consolidate_callable: Any | None = None,
    consolidate_ctx: dict[str, Any] | None = None,
    emotion_detector: Any | None = None,
    emotion_threshold: float = 0.80,
) -> Callable[[SystemMessage], Awaitable[None]]:
    """Build a bus consumer handler bound to the runtime dependencies.

    Args:
        slack_outbound: Operator-side adapter that satisfies
            :class:`HandoffNotifier`. When ``None`` the handler skips the
            suspended-session and interrupt branches and behaves like
            Phase-1 (no handoff).
        redis_ckpt / pg_ckpt: Required when ``slack_outbound`` is set, used
            by the on-interrupt hook to migrate the thread between hot and
            cold checkpointers.
        embedder: Phase-2 P3. Forwarded into graph config so the
            ``recall_memory`` tool can embed queries on demand.
        skill_registry: Phase-2 P4. Forwarded into graph config so
            ``enter_node`` can match the customer's message against
            loaded skills and inline matching SOPs into the system
            prompt. ``None`` disables skill injection.
        skill_top_k: Cap on injected skills per turn.
        recent_events_to_inject: Phase-3 Group C. Number of medium-term
            event-memory rows to load at session start; ``0`` disables.
        compression_threshold_tokens: Phase-3 Group C. Mid-session
            compression fires when the prompt exceeds this many tokens;
            ``0`` disables compression entirely.
        compression_keep_recent_messages: Trailing messages to keep
            verbatim when compression fires.
        token_model: Optional model name for the tokenizer; defaults to
            ``cl100k_base`` via :mod:`backend.v.utils.tokens`.
        consolidate_callable: Phase-3 Group C. The async function the
            compression node calls when it needs to write 会话记忆 +
            事件记忆 (typically ``consolidate_session``).
        consolidate_ctx: ARQ-style context dict passed to
            ``consolidate_callable`` (must contain ``pool``, ``redis``,
            ``llm_caller``, ``ttl_seconds``, ``event_ttl_days``).
    """

    handoff_enabled = slack_outbound is not None and redis_ckpt is not None and pg_ckpt is not None

    async def handle(msg: SystemMessage) -> None:
        with bind_request(
            channel=msg.channel,
            channel_user_id=msg.channel_user_id,
        ):
            session_id, minted = await _resolve_session_id(
                redis,
                channel=msg.channel,
                channel_user_id=msg.channel_user_id,
                silence_seconds=silence_seconds,
            )
            if minted:
                await _persist_session_row(
                    pool,
                    session_id=session_id,
                    channel=msg.channel,
                    channel_user_id=msg.channel_user_id,
                )
            with bind_request(session_id=session_id):
                # 1) Suspended-session: forward to Slack thread, skip graph.
                if handoff_enabled and await is_suspended(session_id, redis=redis):
                    assert slack_outbound is not None  # narrow for type checkers
                    thread_ts = await slack_outbound.get_thread_for_session(session_id)
                    if thread_ts:
                        await slack_outbound.post_customer_message(
                            thread_ts=thread_ts,
                            text=msg.text,
                        )
                        _log.info("bus.worker.forwarded_to_slack", thread_ts=thread_ts)
                    else:
                        _log.warning("bus.worker.suspended_but_no_thread")
                    await _record_last_seen(
                        redis,
                        channel=msg.channel,
                        channel_user_id=msg.channel_user_id,
                        silence_seconds=silence_seconds,
                    )
                    return

                # 2) Normal path.
                turn_started = time.perf_counter()

                t0 = time.perf_counter()
                bootstrap = await on_session_start(
                    pool=pool,
                    redis=redis,
                    channel=msg.channel,
                    channel_user_id=msg.channel_user_id,
                    cache_ttl_seconds=cache_ttl_seconds,
                    recent_events_limit=recent_events_to_inject,
                    session_id=session_id,
                )
                session_start_ms = int((time.perf_counter() - t0) * 1000)
                profile = bootstrap["profile"]
                recent_events = bootstrap["recent_events"]
                working_memory = bootstrap["working_memory"]

                input_state: CustomerServiceState = {
                    "messages": [HumanMessage(content=msg.text)],
                    "session_id": session_id,
                    "channel": msg.channel,
                    "channel_user_id": msg.channel_user_id,
                    "user_profile": profile,
                    "recent_events": recent_events,
                    "working_memory": working_memory,
                    "interrupt_payload": None,
                }
                config = {
                    "configurable": {
                        "thread_id": session_id,
                        "llm_caller": llm_caller,
                        # Phase-2 P3: recall_memory tool needs the PG pool +
                        # embedder + the customer's identity. We thread them
                        # through configurable so any tool the agent calls
                        # in this turn can reach them without a global.
                        "pg_pool": pool,
                        "embedder": embedder,
                        "channel": msg.channel,
                        "channel_user_id": msg.channel_user_id,
                        # Phase-2 P4: enter_node consults this registry to
                        # match SOPs against the customer's message.
                        "skill_registry": skill_registry,
                        "skill_top_k": skill_top_k,
                        # Phase-3 Group C: compression_node levers + sync
                        # consolidation callable / ctx so it can compress
                        # mid-session without leaving the graph.
                        "compression_threshold_tokens": compression_threshold_tokens,
                        "compression_keep_recent_messages": compression_keep_recent_messages,
                        "token_model": token_model,
                        # Phase-3 Group F: emotion pre-emption.
                        "emotion_detector": emotion_detector,
                        "emotion_threshold": emotion_threshold,
                        "consolidate_callable": consolidate_callable,
                        "consolidate_ctx": consolidate_ctx,
                    },
                    # LangSmith filtering levers — surface in the trace UI.
                    "run_name": f"turn:{msg.channel}:{msg.channel_user_id}",
                    "tags": [f"channel:{msg.channel}", f"session:{session_id}"],
                    "metadata": {
                        "channel": msg.channel,
                        "channel_user_id": msg.channel_user_id,
                        "session_id": session_id,
                        "session_minted": minted,
                        "dedup_key": msg.dedup_key,
                    },
                }

                _log.info(
                    "bus.worker.invoke_graph",
                    session_minted=minted,
                    text_len=len(msg.text),
                )
                t1 = time.perf_counter()
                final_state = await graph.ainvoke(input_state, config=config)
                graph_ms = int((time.perf_counter() - t1) * 1000)

                # 3) Interrupt: graph paused at transfer_to_human.
                if handoff_enabled:
                    interrupt_payload = await extract_interrupt(final_state)
                    if interrupt_payload is not None:
                        assert slack_outbound is not None
                        assert redis_ckpt is not None
                        assert pg_ckpt is not None
                        await on_interrupt(
                            session_id=session_id,
                            interrupt_payload=interrupt_payload,
                            channel=msg.channel,
                            channel_user_id=msg.channel_user_id,
                            customer_text=msg.text,
                            redis=redis,
                            pg_pool=pool,
                            redis_ckpt=redis_ckpt,
                            pg_ckpt=pg_ckpt,
                            slack_outbound=slack_outbound,
                        )
                        await _record_last_seen(
                            redis,
                            channel=msg.channel,
                            channel_user_id=msg.channel_user_id,
                            silence_seconds=silence_seconds,
                        )
                        return

                # 4) Normal reply.
                reply = _last_ai_message(final_state.get("messages", []))
                send_ms = 0
                segments_sent = 0
                if reply:
                    send = sends.get(msg.channel)
                    if send is None:
                        _log.error("bus.worker.no_send_registered", channel=msg.channel)
                    else:
                        segments = split_reply_segments(reply)
                        t2 = time.perf_counter()
                        for i, seg in enumerate(segments):
                            if i > 0:
                                await asyncio.sleep(0.25)
                            await send(msg.channel_user_id, seg)
                        send_ms = int((time.perf_counter() - t2) * 1000)
                        segments_sent = len(segments)
                        _log.info(
                            "bus.worker.replied",
                            reply_len=len(reply),
                            segments=segments_sent,
                        )
                else:
                    _log.warning("bus.worker.no_ai_message")

                await _record_last_seen(
                    redis,
                    channel=msg.channel,
                    channel_user_id=msg.channel_user_id,
                    silence_seconds=silence_seconds,
                )

                _log.info(
                    "bus.worker.turn_complete",
                    total_ms=int((time.perf_counter() - turn_started) * 1000),
                    session_start_ms=session_start_ms,
                    graph_ms=graph_ms,
                    send_ms=send_ms,
                    text_len=len(msg.text),
                    reply_len=len(reply or ""),
                    segments_sent=segments_sent,
                )

    return handle
