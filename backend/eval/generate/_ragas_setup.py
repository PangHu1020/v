"""shared ragas + LLM bootstrap used by generate eval."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# ragas 0.4 imports langchain_community.chat_models.vertexai which has moved.
sys.modules.setdefault("langchain_community.chat_models.vertexai", MagicMock())

from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics.collections import (  # noqa: E402
    AnswerCorrectness,
    AnswerRelevancy,
    ContextRecall,
    Faithfulness,
)


def build_ragas_metrics(llm_caller_llm, embedder):
    """Return configured RAGAS metric instances."""
    llm_w = LangchainLLMWrapper(llm_caller_llm)
    emb_w = LangchainEmbeddingsWrapper(embedder)

    metrics = [
        Faithfulness(llm=llm_w),
        ContextRecall(llm=llm_w),
        AnswerRelevancy(llm=llm_w, embeddings=emb_w),
        AnswerCorrectness(llm=llm_w),
    ]
    return metrics


__all__ = ["build_ragas_metrics"]
