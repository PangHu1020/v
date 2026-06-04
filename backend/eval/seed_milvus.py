"""Seed the Milvus ``knowledge_chunks`` collection from the eval artifacts.

Reads ``data/products.jsonl`` + ``data/faq.jsonl``, builds the canonical chunk
text for each (``build_product_text`` / ``build_faq_text``), embeds it with the
**same** Qwen embedder the production retriever uses, and inserts into the
collection the retriever queries.

Schema parity is guaranteed by reusing ``KnowledgeRetriever._setup_collection_sync``
— the seeder never hand-rolls a schema. The BM25 ``Function`` on the collection
auto-derives ``sparse_embedding`` from ``text``, so we only insert
``source_type`` / ``source_id`` / ``text`` / ``embedding`` / ``metadata``.

The collection is **dropped and recreated** on each run so the corpus is a clean,
reproducible function of the committed JSONL (Milvus ``auto_id`` would otherwise
accumulate duplicates across runs).

Usage::

    uv run python -m backend.eval.seed_milvus
"""

from __future__ import annotations

import asyncio

from backend.eval.common import (
    build_faq_text,
    build_product_text,
    get_embedder,
    load_faqs,
    load_products,
)
from backend.v.configs import get_settings
from backend.v.rag.retriever import KnowledgeRetriever
from backend.v.utils.logging import configure as configure_logging
from backend.v.utils.logging import get_logger

_log = get_logger("eval.seed_milvus")

_EMBED_BATCH = 10  # DashScope text-embedding-v4 caps batch size at 10


def _build_rows() -> list[dict]:
    """Flatten products + FAQ into insertable rows (text + metadata, no vector)."""
    rows: list[dict] = []
    for p in load_products():
        rows.append(
            {
                "source_type": "product",
                "source_id": p.product_id,
                "text": build_product_text(p),
                "metadata": {
                    "product_id": p.product_id,
                    "product_name": p.product_name,
                    "category": p.category,
                    "brand": p.brand,
                    "price": p.price,
                },
            }
        )
    for f in load_faqs():
        rows.append(
            {
                "source_type": "faq",
                "source_id": f.faq_id,
                "text": build_faq_text(f),
                "metadata": {"faq_id": f.faq_id, "topic": f.topic},
            }
        )
    return rows


async def _embed_all(embedder, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH):
        batch = texts[start : start + _EMBED_BATCH]
        vectors.extend(await embedder.aembed_documents(batch))
        _log.info("seed.embedded", done=len(vectors), total=len(texts))
    return vectors


async def main() -> None:
    configure_logging(level="INFO", json=False)
    settings = get_settings()
    rows = _build_rows()
    if not rows:
        raise SystemExit("no data — run expand_catalog first")

    embedder = get_embedder()
    vectors = await _embed_all(embedder, [r["text"] for r in rows])
    if len(vectors) != len(rows):
        raise RuntimeError(f"{len(vectors)} vectors for {len(rows)} rows")

    retriever = KnowledgeRetriever()
    client = retriever._get_client(settings.milvus.uri, settings.milvus.token)
    coll = settings.milvus.collection_name

    # Drop + recreate for a clean, reproducible corpus.
    if client.has_collection(coll):
        client.drop_collection(coll)
        _log.info("seed.dropped", collection=coll)
    retriever._setup_collection_sync(client, coll)

    insert_rows = [
        {
            "source_type": r["source_type"],
            "source_id": r["source_id"],
            "text": r["text"],
            "embedding": vec,
            "metadata": r["metadata"],
        }
        for r, vec in zip(rows, vectors, strict=True)
    ]
    client.insert(collection_name=coll, data=insert_rows)
    client.flush(coll)
    _log.info("seed.inserted", collection=coll, count=len(insert_rows))

    stats = client.get_collection_stats(coll)
    print(f"seeded {len(insert_rows)} chunks into '{coll}': {stats}")


if __name__ == "__main__":
    asyncio.run(main())
