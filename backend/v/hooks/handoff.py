"""Handoff lifecycle hooks (Phase-2 P0).

Two hooks bracket the ``transfer_to_human`` interrupt:

- :func:`on_interrupt` — runs after the bus worker observes that the most
  recent ``graph.ainvoke`` left the thread paused at an ``interrupt(...)``
  call. Migrates the thread from Redis (hot) to Postgres (cold), flips the
  session status to ``suspended``, and posts the Slack alert. After this
  point the worker bypasses the graph for incoming customer messages and
  forwards them to Slack instead (see Phase-2 P0.D).
- :func:`on_resume` — fires when the operator clicks the "Resume AI"
  button on the Slack alert. Migrates the thread back to Redis, flips the
  status to ``active``, drains any operator messages collected during the
  suspension into a structured ``decision`` payload, and resumes the
  graph with :class:`langgraph.types.Command`. The agent's final reply is
  then sent to the customer.

Operator messages received during suspension are appended to a Redis list
``operator_log:{session_id}`` so they survive process restarts and are
available verbatim to the resumed agent.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

import asyncpg
import redis.asyncio as redis_async
from langgraph.types import Command

from backend.v.agents.checkpoints.migration import migrate_cold_to_hot, migrate_hot_to_cold
from backend.v.agents.checkpoints.redis import RedisCheckpointer
from backend.v.models.llm_caller import LLMCaller
from backend.v.utils.logging import bind_request, get_logger
from backend.v.utils.reply import split_reply_segments

_log = get_logger("hooks.handoff")

SendRegistry = dict[str, Callable[[str, str], Awaitable[None]]]
"""Channel slug -> async ``(channel_user_id, text) -> None``. Mirrors the
type used in :mod:`backend.app.bus.worker`; redeclared here to keep
``/backend/v/`` independent of ``/backend/app/`` per the layering rule."""


class HandoffNotifier(Protocol):
    """Protocol for the operator-side adapter. ``SlackOutbound`` (in
    ``backend.app.operator.slack``) satisfies this implicitly via duck
    typing; defining it here lets ``/v/`` stay free of ``/app/`` imports."""

    async def post_handoff_alert(
        self,
        *,
        session_id: str,
        channel: str,
        channel_user_id: str,
        customer_text: str,
        transfer_reason: str,
    ) -> str: ...


OPERATOR_LOG_TTL_SECONDS = 7 * 24 * 60 * 60
"""Operator messages collected during suspension live for 7 days. After
that we assume the operator stopped responding and the handoff is dead."""


def _session_status_key(session_id: str) -> str:
    return f"session_status:{session_id}"


def _operator_log_key(session_id: str) -> str:
    return f"operator_log:{session_id}"


async def extract_interrupt(result: Any) -> dict[str, Any] | None:
    """Return the interrupt payload from an ``ainvoke`` result, else ``None``.

    LangGraph 1.x surfaces pending interrupts on the invocation result via
    a ``__interrupt__`` key carrying a list of ``Interrupt`` objects. We
    inspect the first one (the only one in a single-tool-call cycle); the
    payload is whatever was passed to :func:`langgraph.types.interrupt`.
    """
    if not isinstance(result, dict):
        return None
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    value = getattr(first, "value", None)
    if isinstance(value, dict):
        return value
    if value is None:
        return None
    return {"value": value}


async def on_interrupt(
    *,
    session_id: str,
    interrupt_payload: dict[str, Any],
    channel: str,
    channel_user_id: str,
    customer_text: str,
    redis: redis_async.Redis,
    pg_pool: asyncpg.Pool,
    redis_ckpt: RedisCheckpointer,
    pg_ckpt: Any,
    slack_outbound: HandoffNotifier,
) -> str:
    """Execute the handoff suspend procedure.

    Returns:
        The Slack thread ``ts`` of the alert message (for downstream tests
        / observability).
    """
    with bind_request(channel=channel, channel_user_id=channel_user_id, session_id=session_id):
        await migrate_hot_to_cold(session_id, redis_ckpt=redis_ckpt, pg_ckpt=pg_ckpt)

        await redis.set(
            _session_status_key(session_id),
            b"suspended",
            ex=OPERATOR_LOG_TTL_SECONDS,
        )
        async with pg_pool.acquire() as conn:
            await conn.execute(
                "UPDATE agent.session SET status = 'suspended', "
                "last_activity_at = now() WHERE session_id = $1",
                session_id,
            )

        thread_ts = await slack_outbound.post_handoff_alert(
            session_id=session_id,
            channel=channel,
            channel_user_id=channel_user_id,
            customer_text=customer_text,
            transfer_reason=str(interrupt_payload.get("reason", "(未提供)")),
        )
        _log.info("hooks.handoff.suspended", thread_ts=thread_ts)
        return thread_ts


async def append_operator_log(
    session_id: str,
    text: str,
    *,
    redis: redis_async.Redis,
) -> None:
    """Persist an operator message into the per-session log.

    Triggered by the Slack router on a thread reply. The list is replayed
    into :class:`langgraph.types.Command`'s ``resume`` payload when the
    operator finally clicks "Resume AI".
    """
    await redis.rpush(_operator_log_key(session_id), text.encode("utf-8"))
    await redis.expire(_operator_log_key(session_id), OPERATOR_LOG_TTL_SECONDS)


async def is_suspended(session_id: str, *, redis: redis_async.Redis) -> bool:
    raw = await redis.get(_session_status_key(session_id))
    return bool(raw and raw == b"suspended")


async def on_resume(
    *,
    session_id: str,
    channel: str,
    channel_user_id: str,
    redis: redis_async.Redis,
    pg_pool: asyncpg.Pool,
    redis_ckpt: RedisCheckpointer,
    pg_ckpt: Any,
    graph: Any,
    llm_caller: LLMCaller,
    sends: SendRegistry,
) -> str | None:
    """Lift the interrupt and let the agent finish the turn.

    Returns the AI's final reply text (also already dispatched to the
    customer's channel) for observability; ``None`` if no AI message was
    produced after resume.
    """
    with bind_request(channel=channel, channel_user_id=channel_user_id, session_id=session_id):
        await migrate_cold_to_hot(session_id, pg_ckpt=pg_ckpt, redis_ckpt=redis_ckpt)

        # Drain operator log into the resume payload.
        log_key = _operator_log_key(session_id)
        raw_msgs = await redis.lrange(log_key, 0, -1)
        operator_messages = [m.decode("utf-8") for m in raw_msgs]
        await redis.delete(log_key)

        decision = {
            "type": "resume",
            "operator_messages": operator_messages,
        }

        await redis.set(
            _session_status_key(session_id),
            b"active",
            ex=30 * 60,
        )
        async with pg_pool.acquire() as conn:
            await conn.execute(
                "UPDATE agent.session SET status = 'active', "
                "last_activity_at = now() WHERE session_id = $1",
                session_id,
            )

        config = {
            "configurable": {
                "thread_id": session_id,
                "llm_caller": llm_caller,
            }
        }
        _log.info(
            "hooks.handoff.resuming",
            operator_message_count=len(operator_messages),
        )
        final_state = await graph.ainvoke(Command(resume=decision), config=config)

        # Find the last AIMessage with non-empty content (non-tool-call).
        from langchain_core.messages import AIMessage  # local import keeps module import light

        reply: str | None = None
        for m in reversed(final_state.get("messages", [])):
            if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
                reply = m.content if isinstance(m.content, str) else None
                if reply:
                    break

        if reply:
            send = sends.get(channel)
            if send is None:
                _log.error("hooks.handoff.resume.no_send_registered", channel=channel)
            else:
                segments = split_reply_segments(reply)
                for i, seg in enumerate(segments):
                    if i > 0:
                        await asyncio.sleep(0.25)
                    await send(channel_user_id, seg)
                _log.info(
                    "hooks.handoff.resume.replied",
                    reply_len=len(reply),
                    segments=len(segments),
                )
        else:
            _log.warning("hooks.handoff.resume.no_ai_reply")
        return reply
