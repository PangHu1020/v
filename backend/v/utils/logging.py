"""Structured logging configured around ``structlog``.

In ``dev`` mode emits human-readable key=value lines. In any other ``ENV`` value
emits JSON for downstream log aggregation. ``bind_request`` is a contextmanager
that scopes request-correlation fields onto the structlog contextvars so every
nested log call automatically carries them.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog


def configure(level: str = "INFO", *, json: bool = False) -> None:
    """Configure structlog and the stdlib root logger.

    Args:
        level: Logging level name (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``).
        json: If ``True`` emit JSON lines, otherwise human-readable key=value.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
        force=True,
    )

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=False))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger."""
    return structlog.get_logger(name)


@contextmanager
def bind_request(
    request_id: str | None = None,
    *,
    channel: str | None = None,
    channel_user_id: str | None = None,
    session_id: str | None = None,
    **extra: Any,
) -> Iterator[None]:
    """Bind request-correlation fields onto the structlog context for the block.

    Args:
        request_id: Originating request id, typically generated at the gateway.
        channel: Channel slug (``wecom`` / ``feishu``).
        channel_user_id: Per-channel user identifier.
        session_id: Active session id, if known.
        **extra: Additional fields to bind for the duration of the block.
    """
    fields: dict[str, Any] = {}
    if request_id is not None:
        fields["request_id"] = request_id
    if channel is not None:
        fields["channel"] = channel
    if channel_user_id is not None:
        fields["channel_user_id"] = channel_user_id
    if session_id is not None:
        fields["session_id"] = session_id
    fields.update(extra)

    structlog.contextvars.bind_contextvars(**fields)
    try:
        yield
    finally:
        structlog.contextvars.unbind_contextvars(*fields.keys())
