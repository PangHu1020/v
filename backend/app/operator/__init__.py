"""Internal operator-side adapters.

Phase-2 introduces Slack as the operator channel for human handoff. Slack
is NOT a customer channel — it never speaks to end customers directly. Its
job is:

1. Receive handoff alerts when an Agent invokes ``transfer_to_human``.
2. Let an operator reply in the alert thread; we relay those replies back
   to the original customer channel.
3. Provide a "Resume" button that lifts the LangGraph interrupt.

Future operator channels (e.g., a webapp dashboard) live alongside Slack
under ``operator/`` rather than mixed into customer ``channels/``.
"""
