"""Customer-side channel adapters (WeCom + Feishu).

Each platform handles its own signature verification and AES decryption,
normalizes to :class:`backend.app.bus.messages.SystemMessage`, and pushes
through the channel-layer 500ms debounce before reaching the bus.
"""

from backend.app.channels.base import ChannelAdapter
from backend.app.channels.debounce import Debouncer

__all__ = ["ChannelAdapter", "Debouncer"]
