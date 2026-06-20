"""Seed ``agent.knowledge_chunk`` from ``dw.dim_product``.

Run-once (idempotent) bootstrap so the ``search`` tool has something to
recall against on day one. Each ``dw.dim_product`` row becomes one
``knowledge_chunk`` row with ``source_type='product'``,
``source_id=product_id``, and a hand-rolled Chinese description that
fronts the embedding:

    "iPhone 15 Pro 类别:手机数码 品牌:苹果 编号:P001"

This tightly-packed text gives the embedding model the signal it needs
without bloating tokens. ``ON CONFLICT (source_type, source_id) DO
UPDATE`` makes the script safe to re-run after schema or product
catalog changes.

Usage::

    uv run python -m scripts.seed_knowledge_chunks
    # or
    /path/to/python -m scripts.seed_knowledge_chunks
"""

from __future__ import annotations

import asyncio
import json
import sys

import asyncpg

from backend.app.store import close_pool, create_pool
from backend.v.configs import get_settings
from backend.v.models.factory import get_embedding
from backend.v.utils.logging import configure as configure_logging
from backend.v.utils.logging import get_logger

_log = get_logger("scripts.seed_knowledge_chunks")

_BATCH_SIZE = 25  # Qwen batch limit is generous; keep small to stay observable.


def _build_text(row: asyncpg.Record) -> str:
    return (
        f"{row['product_name']} 类别:{row['category']} 品牌:{row['brand']} 编号:{row['product_id']}"
    )


def _build_metadata(row: asyncpg.Record) -> dict[str, str]:
    return {
        "product_id": row["product_id"],
        "product_name": row["product_name"],
        "category": row["category"],
        "brand": row["brand"],
    }


async def _seed(pool: asyncpg.Pool, embedder) -> int:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT product_id, product_name, category, brand "
            "FROM dw.dim_product ORDER BY product_id"
        )
    if not rows:
        _log.warning("seed.no_products")
        return 0

    written = 0
    for start in range(0, len(rows), _BATCH_SIZE):
        batch = rows[start : start + _BATCH_SIZE]
        texts = [_build_text(r) for r in batch]
        vectors = await embedder.aembed_documents(texts)
        if len(vectors) != len(batch):
            raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(batch)} inputs")
        async with pool.acquire() as conn:
            async with conn.transaction():
                for row, text, vec in zip(batch, texts, vectors, strict=True):
                    await conn.execute(
                        """
                        INSERT INTO agent.knowledge_chunk
                            (source_type, source_id, text, embedding, metadata)
                        VALUES ($1, $2, $3, $4::vector, $5::jsonb)
                        ON CONFLICT (source_type, source_id) DO UPDATE
                        SET text = EXCLUDED.text,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata,
                            updated_at = now()
                        """,
                        "product",
                        row["product_id"],
                        text,
                        # asyncpg + pgvector accept either pgvector.Vector or a
                        # textual representation; the textual form keeps this
                        # script free of optional codec setup.
                        "[" + ",".join(f"{x:.6f}" for x in vec) + "]",
                        json.dumps(_build_metadata(row), ensure_ascii=False),
                    )
                    written += 1
        _log.info("seed.batch_done", batch=start // _BATCH_SIZE + 1, written=written)
    return written


async def main() -> None:
    settings = get_settings()
    configure_logging(level=settings.runtime.log_level, json=False)
    if not settings.embedding.api_key or not settings.embedding.base_url:
        _log.error("seed.missing_embedding_credentials")
        sys.exit(2)

    embedder = get_embedding(settings.llm, settings.embedding)
    pool = await create_pool(settings.db.dsn)
    try:
        written = await _seed(pool, embedder)
        _log.info("seed.complete", chunks_upserted=written)
    finally:
        await close_pool(pool)


if __name__ == "__main__":
    asyncio.run(main())
