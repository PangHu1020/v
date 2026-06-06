"""Root conftest: shared fixtures across unit and e2e tests.

Group-A scope: settings cache invalidation only. Later groups extend this with
``async_pg_pool``, ``redis_client``, ``mock_llm``, ``mock_channel_send``.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from backend.v.configs.base import _load_yaml, get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Reset settings + YAML caches between tests and isolate config.

    Settings are read once per process and cached. Without this fixture,
    monkeypatched env vars in one test would leak into the next.

    We also point ``APP_CONFIG_FILE`` at a path that does not exist so tests
    run against pure code defaults regardless of any local ``config.yaml``,
    and clear the path-keyed ``_load_yaml`` cache so a YAML file written by
    one test is never served stale to another.
    """
    monkeypatch.setenv("APP_CONFIG_FILE", "/nonexistent/test-config.yaml")
    get_settings.cache_clear()
    _load_yaml.cache_clear()
    yield
    get_settings.cache_clear()
    _load_yaml.cache_clear()
