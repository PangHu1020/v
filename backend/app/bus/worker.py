"""Bus consumer handler: glue between debounced inbound messages and the graph.

For each :class:`SystemMessage` consumed off the bus:

1. Resolve the session id (mint a new UUID after 30 minutes of silence).
2. Run ``on_session_start`` to populate ``user_profile``.
3. Build the initial ``CustomerServiceState`` with a single ``HumanMessage``
   for the new turn.
4. Invoke the compiled graph with ``thread_id=session_id``.
5. Pull the last ``AIMessage`` from the resulting state and dispatch it
   via the channel-specific outbound adapter.
6. Refresh the ``last_seen`` timestamp.

Outbound replies do NOT re-enter the bus — they flow directly to the
channel adapter (per the architectural rule in ``backend/app/CLAUDE.md``).
"""

from __future__ import annotations

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
            if isinstance(content, str):
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
) -> Callable[[SystemMessage], Awaitable[None]]:
    """Build a bus consumer handler bound to the runtime dependencies."""

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
            with bind_request(session_id=session_id):
                profile = await on_session_start(
                    pool=pool,
                    redis=redis,
                    channel=msg.channel,
                    channel_user_id=msg.channel_user_id,
                    cache_ttl_seconds=cache_ttl_seconds,
                )

                input_state: CustomerServiceState = {
                    "messages": [HumanMessage(content=msg.text)],
                    "session_id": session_id,
                    "channel": msg.channel,
                    "channel_user_id": msg.channel_user_id,
                    "user_profile": profile,
                    "interrupt_payload": None,
                }
                config = {
                    "configurable": {
                        "thread_id": session_id,
                        "llm_caller": llm_caller,
                    }
                }

                _log.info(
                    "bus.worker.invoke_graph",
                    session_minted=minted,
                    text_len=len(msg.text),
                )
                final_state = await graph.ainvoke(input_state, config=config)
                reply = _last_ai_message(final_state.get("messages", []))

                if reply:
                    send = sends.get(msg.channel)
                    if send is None:
                        _log.error("bus.worker.no_send_registered", channel=msg.channel)
                    else:
                        await send(msg.channel_user_id, reply)
                        _log.info("bus.worker.replied", reply_len=len(reply))
                else:
                    _log.warning("bus.worker.no_ai_message")

                await _record_last_seen(
                    redis,
                    channel=msg.channel,
                    channel_user_id=msg.channel_user_id,
                    silence_seconds=silence_seconds,
                )

    return handle
