"""Phase-1 graph edges.

The Phase-1 graph is linear: ``enter -> agent -> exit -> END``. Routing
edges (``route_*``) for ``transfer_to_human`` and tool dispatch land in
later phases.
"""

from __future__ import annotations

ENTER = "enter"
AGENT = "agent"
EXIT = "exit"
