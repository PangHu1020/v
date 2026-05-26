"""Global infrastructure: PostgreSQL pool and Redis client.

Initialized once during the FastAPI lifespan and injected into route handlers
and ``/v/`` modules via ``Depends`` (or app context). Both ``/app/`` and
``/v/`` modules consume connections from here; neither layer instantiates its
own pool.
"""

from backend.app.store.postgres import (
    acquire_with_schema,
    close_pool,
    create_pool,
    pg_health,
)
from backend.app.store.redis import close_client, create_client, redis_health

__all__ = [
    "acquire_with_schema",
    "close_client",
    "close_pool",
    "create_client",
    "create_pool",
    "pg_health",
    "redis_health",
]
