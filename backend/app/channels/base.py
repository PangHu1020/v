"""Common interface for customer-side platform adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from backend.app.bus.messages import SystemMessage


class ChannelAdapter(ABC):
    """Contract that every platform adapter (WeCom, Feishu) must satisfy.

    Inbound flow: the gateway router receives the raw webhook, calls
    :meth:`verify` to validate the signature, then :meth:`decrypt_normalize`
    to produce a :class:`SystemMessage`. The router then runs the message
    through the debouncer before pushing onto the bus.

    Outbound flow: ``/v/agents`` (or the operator adapter) calls :meth:`send`
    with the target identity and reply text. Replies do not go through the
    bus.
    """

    slug: str

    @abstractmethod
    def verify(
        self,
        *,
        signature: str,
        timestamp: str,
        nonce: str,
        body: str | bytes,
    ) -> bool:
        """Return ``True`` iff the signature matches the per-platform algorithm."""

    @abstractmethod
    def decrypt_normalize(self, body: str | bytes) -> SystemMessage:
        """Decrypt the platform-shaped payload and return a normalized message."""

    @abstractmethod
    async def send(self, channel_user_id: str, text: str) -> None:
        """Deliver a reply to the customer via the platform's outbound API."""
