"""Shared helpers for the RAG evaluation harness.

This module centralizes everything the generators, the Milvus seeder, and the
eval runner have in common:

- Paths to the frozen data artifacts (``data/*.jsonl``).
- ``Product`` / ``FaqEntry`` / ``QaItem`` dataclasses describing the on-disk
  JSONL schemas.
- ``build_product_text`` / ``build_faq_text`` — the canonical chunk-text
  renderers. The seeder embeds exactly these strings, so the eval measures
  retrieval over the same text the production ``search`` tool would see.
- ``load_jsonl`` / ``dump_jsonl`` IO.
- ``get_embedder`` / ``get_llm_caller`` thin wrappers over the model factory.

Nothing here calls an LLM or touches Milvus at import time, so it is cheap to
import from tests.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

DATA_DIR = Path(__file__).parent / "data"
PRODUCTS_PATH = DATA_DIR / "products.jsonl"
FAQ_PATH = DATA_DIR / "faq.jsonl"
QA_PATH = DATA_DIR / "qa.jsonl"

# Milvus VARCHAR(text) is capped at 4096; keep a safety margin.
MAX_CHUNK_CHARS = 3500

# The six categories from the seed catalog (dw.dim_product).
CATEGORIES = [
    "手机数码",
    "家用电器",
    "鞋靴",
    "服饰",
    "食品饮料",
    "休闲零食",
]

# Query difficulty tiers used by the QA generator and reported by the harness.
TIERS = ["plain", "colloquial", "synonym", "noisy", "multi_condition"]


@dataclass
class Product:
    """One product chunk. ``source_type='product'``, ``source_id=product_id``."""

    product_id: str
    product_name: str
    category: str
    brand: str
    price: float
    description: str
    specs: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Product:
        return cls(
            product_id=d["product_id"],
            product_name=d["product_name"],
            category=d["category"],
            brand=d["brand"],
            price=float(d["price"]),
            description=d["description"],
            specs=dict(d.get("specs") or {}),
        )


@dataclass
class FaqEntry:
    """One FAQ / policy chunk. ``source_type='faq'``, ``source_id=faq_id``."""

    faq_id: str
    topic: str
    question: str
    answer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FaqEntry:
        return cls(
            faq_id=d["faq_id"],
            topic=d["topic"],
            question=d["question"],
            answer=d["answer"],
        )


@dataclass
class QaItem:
    """One evaluation item: a query plus its gold source id(s)."""

    qa_id: str
    query: str
    gold_source_type: str
    gold_source_ids: list[str]
    tier: str
    answer: str = ""
    difficulty: str = ""  # easy / medium / hard (set by score_difficulty)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QaItem:
        return cls(
            qa_id=d["qa_id"],
            query=d["query"],
            gold_source_type=d["gold_source_type"],
            gold_source_ids=list(d["gold_source_ids"]),
            tier=d["tier"],
            answer=d.get("answer", ""),
            difficulty=d.get("difficulty", ""),
        )


def build_product_text(p: Product) -> str:
    """Render a product into the chunk text that gets embedded + BM25-indexed.

    Packs the discriminative signal (name, brand, category, price, the spec
    key/values, and the sales description) into one compact Chinese string.
    Truncated to ``MAX_CHUNK_CHARS`` so it always fits the Milvus VARCHAR.
    """
    spec_str = " ".join(f"{k}:{v}" for k, v in p.specs.items())
    parts = [
        p.product_name,
        f"品牌:{p.brand}",
        f"类别:{p.category}",
        f"售价:{p.price:g}元",
        f"编号:{p.product_id}",
    ]
    if spec_str:
        parts.append(f"规格:{spec_str}")
    if p.description:
        parts.append(p.description)
    return " ".join(parts)[:MAX_CHUNK_CHARS]


def build_faq_text(f: FaqEntry) -> str:
    """Render a FAQ entry into chunk text (topic + question + answer)."""
    return f"{f.topic} {f.question} {f.answer}"[:MAX_CHUNK_CHARS]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts. Missing file -> empty list."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dicts to JSONL (one compact object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_products() -> list[Product]:
    return [Product.from_dict(d) for d in load_jsonl(PRODUCTS_PATH)]


def load_faqs() -> list[FaqEntry]:
    return [FaqEntry.from_dict(d) for d in load_jsonl(FAQ_PATH)]


def load_qa() -> list[QaItem]:
    return [QaItem.from_dict(d) for d in load_jsonl(QA_PATH)]


def get_embedder() -> Any:
    """Build the same embedder the production retriever uses."""
    from backend.v.configs import get_settings
    from backend.v.models.factory import get_embedding

    s = get_settings()
    return get_embedding(s.llm, s.embedding)


def get_llm_caller() -> Any:
    """Build an LLMCaller bound to the configured DeepSeek/Qwen models."""
    from backend.v.configs import get_settings
    from backend.v.models.llm_caller import LLMCaller

    return LLMCaller(get_settings().llm)


class TokenCounter(BaseCallbackHandler):
    """LangChain callback tallying prompt + completion tokens across a run.

    Attach to ``config["callbacks"]`` of a ``graph.ainvoke`` call to capture
    token usage for the whole turn (all LLM calls inside the graph). Reads
    ``generation_info["token_usage"]`` (what the OpenAI-compatible providers
    emit). Shared by the system + conversational eval harnesses.
    """

    def __init__(self) -> None:
        super().__init__()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        for gen_list in response.generations:
            for gen in gen_list:
                usage = (getattr(gen, "generation_info", None) or {}).get("token_usage") or {}
                self.prompt_tokens += usage.get("prompt_tokens", 0)
                self.completion_tokens += usage.get("completion_tokens", 0)
