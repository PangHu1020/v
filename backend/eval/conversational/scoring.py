"""Scoring helpers for conversational retrieval eval.

The judging mechanism: the ``search`` tool renders results as lines like
``- [product:P001] ...`` (see ``backend/v/rag/retriever.py:_format``). After a
dialogue runs, we scan every tool message for these markers, collect the union
of retrieved product ids, and a case is a hit if that union intersects the
gold set (any-of).
"""

from __future__ import annotations

import re

# Matches the source markers the search tool emits: ``[product:P001]``,
# ``[faq:FAQ003]``. We only score product retrieval here.
_SOURCE_MARKER = re.compile(r"\[(product|faq):([^\]]+)\]")


def extract_retrieved_ids(tool_outputs: list[str], *, source_type: str = "product") -> list[str]:
    """Extract retrieved source ids of ``source_type`` from tool output texts.

    Args:
        tool_outputs: Raw ToolMessage contents collected across the dialogue.
        source_type: Which marker kind to keep (``product`` / ``faq``).

    Returns:
        Ordered, de-duplicated list of retrieved ids (first-seen order).
    """
    seen: dict[str, None] = {}
    for text in tool_outputs:
        if not text:
            continue
        for kind, sid in _SOURCE_MARKER.findall(text):
            if kind == source_type and sid not in seen:
                seen[sid] = None
    return list(seen)


def score_hit(retrieved_ids: list[str], gold_ids: list[str]) -> bool:
    """any-of: True if any retrieved id is in the gold set."""
    gold = set(gold_ids)
    return any(r in gold for r in retrieved_ids)
