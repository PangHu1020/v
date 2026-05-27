"""Cron / proactive task definitions (Phase-2 P2).

Owns the four async tasks that run outside the reactive customer-message
flow:

- :mod:`tasks.logistics`         — order delivery notifications.
- :mod:`tasks.ad_hoc_ad`         — manual / batched ad pushes.
- :mod:`tasks.repurchase`        — periodic repurchase reminders for
  customers whose past purchases are due for a refill.
- :mod:`tasks.consolidate_session` — LLM-summarize an expiring session
  and persist into ``agent.session_memory``; triggered by hooks before
  the working-memory TTL fires (closing the Phase-1 gap of session
  memory writes never happening).

Channel-direct outbound delivery is shared via :func:`proactive.deliver_proactive`.
The ARQ worker entry point lives in :mod:`backend.app.cron_worker` because
it imports both ``/app/`` channel adapters and ``/v/`` task functions;
core task logic stays here in ``/v/`` per the layering rule.
"""

from backend.v.cron.proactive import (
    PROACTIVE_LOG_KEY_PREFIX,
    deliver_proactive,
    record_proactive,
)
from backend.v.cron.tasks.ad_hoc_ad import push_ad
from backend.v.cron.tasks.consolidate_session import consolidate_session
from backend.v.cron.tasks.logistics import notify_logistics_delivered
from backend.v.cron.tasks.repurchase import (
    build_repurchase_targets,
    send_repurchase_reminders,
)

__all__ = [
    "PROACTIVE_LOG_KEY_PREFIX",
    "build_repurchase_targets",
    "consolidate_session",
    "deliver_proactive",
    "notify_logistics_delivered",
    "push_ad",
    "record_proactive",
    "send_repurchase_reminders",
]
