"""BGE Reranker v2-m3 sidecar service on port 8767.

Listens for POST /rerank with {"query": str, "passages": [str], "top_n": int}.
Returns ranked passages with scores. Uses the local model at /mnt/zh/project/bge-reranker-v2-m3.

Run in conda vllm env:
    source /data/anaconda3/bin/activate vllm
    python scripts/start_reranker.py
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

# Suppress transformers/tokenizers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("tokenizers").setLevel(logging.ERROR)

MODEL_PATH = "/mnt/zh/project/bge-reranker-v2-m3"
PORT = 8767
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="BGE Reranker v2-m3")

# Global model instance (loaded once at startup)
_model: CrossEncoder | None = None


class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1)
    passages: list[str] = Field(..., min_items=1)
    top_n: int = Field(default=10, ge=1)


class RerankResponse(BaseModel):
    results: list[dict[str, Any]]  # [{"index": int, "score": float, "text": str}, ...]


@app.on_event("startup")
def load_model() -> None:
    global _model
    print(f"Loading {MODEL_PATH} on {DEVICE}...")
    _model = CrossEncoder(MODEL_PATH, device=DEVICE, max_length=512)
    print("Model loaded.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    if _model is None:
        raise RuntimeError("Model not loaded")

    # Build (query, passage) pairs
    pairs = [[req.query, passage] for passage in req.passages]
    # Score them
    scores = _model.predict(pairs, batch_size=32)

    # Rank by score desc
    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True,
    )[: req.top_n]

    results = [
        {"index": idx, "score": float(score), "text": req.passages[idx]} for idx, score in ranked
    ]
    return RerankResponse(results=results)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
