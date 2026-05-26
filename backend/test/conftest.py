"""Root conftest: shared fixtures across unit and e2e tests.

Group-A scope: settings cache invalidation only. Later groups extend this with
``async_pg_pool``, ``redis_client``, ``mock_llm``, ``mock_channel_send``.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from backend.v.configs.base import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """Reset the ``get_settings`` LRU cache between tests.

    Settings are read once per process and cached. Without this fixture,
    monkeypatched env vars in one test would leak into the next.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
