"""ragas 0.4 metrics setup using InstructorLLM (required by collections metrics)."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("langchain_community.chat_models.vertexai", MagicMock())

from openai import AsyncOpenAI  # noqa: E402
from ragas.embeddings import embedding_factory  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics.collections import (  # noqa: E402
    AnswerCorrectness,
    AnswerRelevancy,
    ContextRecall,
    Faithfulness,
)


def build_ragas_metrics(
    *,
    judge_base_url: str,
    judge_api_key: str,
    judge_model: str,
    embed_base_url: str,
    embed_api_key: str,
    embed_model: str,
):
    """Return configured RAGAS metric instances.

    The judge LLM and embedder can be any OpenAI-compatible endpoint.
    Typically both point at DashScope but can differ.
    """
    llm_client = AsyncOpenAI(base_url=judge_base_url, api_key=judge_api_key)
    emb_client = AsyncOpenAI(base_url=embed_base_url, api_key=embed_api_key)
    llm = llm_factory(model=judge_model, client=llm_client)
    emb = embedding_factory(model=embed_model, client=emb_client)
    return [
        Faithfulness(llm=llm),
        ContextRecall(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=emb),
        AnswerCorrectness(llm=llm, embeddings=emb),
    ]


__all__ = ["build_ragas_metrics"]
