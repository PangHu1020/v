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


def split_reply_segments(reply: str) -> list[str]:
    """Split an AI reply into discrete outbound segments.

    Returns a list of stripped, non-empty segments. A reply without any
    blank-line boundary collapses to a single-element list, so callers
    can use the same iteration regardless of whether the model chose to
    segment its output.
    """
    if not reply:
        return []
    parts = _SEGMENT_BOUNDARY.split(reply)
    return [p.strip() for p in parts if p and p.strip()]
