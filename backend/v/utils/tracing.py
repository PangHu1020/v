"""LangSmith tracing bootstrap.

LangChain / LangGraph auto-instrument when the standard ``LANGSMITH_*``
environment variables are set in the process. Calling
:func:`configure_langsmith` early in startup (FastAPI lifespan, ARQ
worker, the wecom_aibot worker) is enough to enable tracing for every
``ChatOpenAI`` / ``compiled_graph`` invocation that follows.

When ``api_key`` is empty or ``tracing`` is false, this is a no-op.
"""

from __future__ import annotations

import os

from backend.v.configs.base import LangSmithSettings
from backend.v.utils.logging import get_logger

_log = get_logger("utils.tracing")


def configure_langsmith(settings: LangSmithSettings) -> bool:
    """Set the env vars LangChain reads for LangSmith export.

    Returns ``True`` if tracing was enabled, ``False`` otherwise. Safe to
    call multiple times — each call re-asserts the env vars.
    """
    if not settings.tracing or not settings.api_key:
        os.environ.pop("LANGSMITH_TRACING", None)
        return False

    # LangChain reads both legacy LANGCHAIN_* and the newer LANGSMITH_*
    # names. Set the LANGSMITH_* keys (current SDK convention) and fall
    # back to the legacy ones for older transitive deps.
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.api_key
    os.environ["LANGSMITH_PROJECT"] = settings.project
    os.environ["LANGSMITH_ENDPOINT"] = settings.endpoint
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.endpoint

    _log.info(
        "tracing.langsmith.enabled",
        project=settings.project,
        endpoint=settings.endpoint,
    )
    return True
