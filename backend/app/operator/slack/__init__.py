"""Slack operator adapter."""

from backend.app.operator.slack.outbound import SlackOutbound
from backend.app.operator.slack.signature import verify_signature

__all__ = ["SlackOutbound", "verify_signature"]
