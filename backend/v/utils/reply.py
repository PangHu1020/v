"""Reply post-processing shared by inbound worker and handoff resume.

The main agent prompt instructs the model to use a literal blank line
(``\\n\\n``) as an explicit segment delimiter when a single response is
better split into multiple chat bubbles (e.g., a step-by-step procedure).
``split_reply_segments`` materializes that contract on the dispatch
side: callers iterate the returned list and emit one outbound message
per segment, mirroring how a human agent would type and send several
short messages in succession.
"""

from __future__ import annotations

import re

# One or more blank lines (allowing intermediate whitespace) act as the
# segment boundary. Single ``\n`` inside a paragraph is preserved.
_SEGMENT_BOUNDARY = re.compile(r"\n\s*\n+")

# Hard cap on outbound bubbles per reply. Excess segments are merged
# back into the last allowed segment so no content is lost.
MAX_SEGMENTS = 4


def split_reply_segments(reply: str, *, max_segments: int = MAX_SEGMENTS) -> list[str]:
    """Split an AI reply into discrete outbound segments.

    Returns a list of stripped, non-empty segments capped at
    ``max_segments``. When the model produces more segments than the cap,
    the tail is merged (joined with ``\\n\\n``) into the last segment so
    no content is dropped.

    A reply without any blank-line boundary collapses to a single-element
    list, so callers can use the same iteration regardless of whether the
    model chose to segment its output.
    """
    if not reply:
        return []
    parts = [p.strip() for p in _SEGMENT_BOUNDARY.split(reply) if p and p.strip()]
    if not parts:
        return []
    if len(parts) <= max_segments:
        return parts
    head = parts[: max_segments - 1]
    tail = "\n\n".join(parts[max_segments - 1 :])
    head.append(tail)
    return head
