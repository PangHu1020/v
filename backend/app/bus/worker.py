"""Bus consumer handler: glue between debounced inbound messages and the graph."""

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
from backend.v.agents.state import CustomerServiceState
from backend.v.hooks.session import on_session_start
from backend.v.models.llm_caller import LLMCaller
from backend.v.utils.logging import bind_request, get_logger
from backend.v.utils.reply import split_reply_segments

_log = get_logger("bus.worker")

SendFn = Callable[[str, str], Awaitable[None]]
SendRegistry = dict[str, SendFn]


async def _resolve_session_id(
    redis: redis_async.Redis,
    *,
    channel: str,
    channel_user_id: str,
    silence_seconds: int,
) -> tuple[str, bool, str | None]:
    """Return ``(session_id, was_minted, prev_session_id)``.

    ``prev_session_id`` is the expired session id when a new one is minted;
    ``None`` on first-ever message or when the session is still active.
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
                return existing.decode("utf-8"), False, None

    # Mint a new session; read previous id before overwriting.
    prev_raw = await redis.get(session_key)
    prev_session_id = prev_raw.decode("utf-8") if prev_raw else None
    session_id = str(uuid.uuid4())
    await redis.set(session_key, session_id, ex=silence_seconds * 2)
    return session_id, True, prev_session_id


async def _persist_session_row(
    pool: asyncpg.Pool,
    *,
    session_id: str,
    channel: str,
    channel_user_id: str,
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO agent.session (session_id, channel, channel_user_id) "
            "VALUES ($1, $2, $3) ON CONFLICT (session_id) DO NOTHING",
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
    await redis.set(
        f"last_seen:{channel}:{channel_user_id}",
        str(time.time()),
        ex=silence_seconds * 2,
    )


def _last_ai_message(messages: list) -> str | None:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            content = m.content
            if isinstance(content, str) and content:
                return content
            if isinstance(content, list):
                parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                return "".join(parts).strip() or None
    return None


async def _promote_prev_session(
    session_id: str,
    *,
    pool: asyncpg.Pool,
    redis: redis_async.Redis,
    llm_caller: Any,
    embedder: Any,
) -> None:
    """Fire-and-forget: promote expired session's working memory to long-term."""
    try:
        from backend.v.memory.memory_extractor import promote_to_long_term

        result = await promote_to_long_term(
            {"pool": pool, "redis": redis, "llm_caller": llm_caller, "embedder": embedder},
            session_id=session_id,
        )
        _log.info("bus.worker.session_end_promoted", session_id=session_id, result=result)
    except Exception as exc:
        _log.error(
            "bus.worker.session_end_promote_failed",
            session_id=session_id,
            error=type(exc).__name__,
        )


def make_bus_handler(
    *,
    graph: Any,
    pool: asyncpg.Pool,
    redis: redis_async.Redis,
    llm_caller: LLMCaller,
    sends: SendRegistry,
    silence_seconds: int,
    cache_ttl_seconds: int,
    settings: Any | None = None,
    embedder: Any | None = None,
    skill_registry: Any | None = None,
    skill_top_k: int = 3,
    recent_events_to_inject: int = 0,
    compression_threshold_tokens: int = 0,
    compression_keep_recent_messages: int = 4,
    token_model: str | None = None,
) -> Callable[[SystemMessage], Awaitable[None]]:
    """Build a bus consumer handler bound to the runtime dependencies."""

    async def handle(msg: SystemMessage) -> None:
        with bind_request(channel=msg.channel, channel_user_id=msg.channel_user_id):
            session_id, minted, prev_session_id = await _resolve_session_id(
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
                # Previous session just expired — promote its working memory to
                # long-term storage (fire-and-forget; don't block the new turn).
                if prev_session_id and llm_caller and embedder:
                    asyncio.create_task(  # noqa: RUF006  fire-and-forget by design
                        _promote_prev_session(
                            prev_session_id,
                            pool=pool,
                            redis=redis,
                            llm_caller=llm_caller,
                            embedder=embedder,
                        )
                    )
            with bind_request(session_id=session_id):
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

                input_state: CustomerServiceState = {
                    "messages": [HumanMessage(content=msg.text)],
                    "session_id": session_id,
                    "channel": msg.channel,
                    "channel_user_id": msg.channel_user_id,
                    "user_profile": bootstrap["profile"],
                    "recent_events": bootstrap["recent_events"],
                    "working_memory": bootstrap["working_memory"],
                }
                config = {
                    "configurable": {
                        "thread_id": session_id,
                        "llm_caller": llm_caller,
                        "pg_pool": pool,
                        "embedder": embedder,
                        "redis": redis,
                        "channel": msg.channel,
                        "channel_user_id": msg.channel_user_id,
                        "skill_registry": skill_registry,
                        "skill_top_k": skill_top_k,
                        "compression_threshold_tokens": compression_threshold_tokens,
                        "compression_keep_recent_messages": compression_keep_recent_messages,
                        "token_model": token_model,
                        "settings": settings,
                    },
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

                _log.info("bus.worker.invoke_graph", session_minted=minted, text_len=len(msg.text))
                t1 = time.perf_counter()
                final_state = await graph.ainvoke(input_state, config=config)
                graph_ms = int((time.perf_counter() - t1) * 1000)

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
