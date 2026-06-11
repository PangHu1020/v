"""PII detection and masking for memory writes.

The memory write pipeline's policy gate runs every candidate fact through
:func:`mask_pii` before it is persisted, so durable storage never holds raw
personally-identifiable values. The goal is compliance-friendly retention:
keep the *shape* of the fact ("客户提供了收货地址") for recall and audit, drop
the sensitive payload.

Masking, not deletion, is deliberate — a masked token still lets the agent
know a phone/address exists and ask the customer to re-confirm, without the
system durably storing the value. Two strengths:

- **Partial mask** (phone, ID card, bank card): keep enough to recognise /
  disambiguate ("138****1234"), redact the middle. The customer can confirm
  "尾号 1234 那个" without us storing the full number.
- **Full replacement** (street address): collapse to a placeholder phrase,
  because a partial address is both useless for recall and still leaky.

All detection is regex-based and conservative — it errs toward leaving text
untouched rather than over-masking legitimate content (e.g. an order id that
merely looks long). Patterns target the Chinese customer-service domain
(mainland mobile numbers, 18-digit 居民身份证, UnionPay-length cards).
"""

from __future__ import annotations

import re

# Mainland mobile: 1 + [3-9] + 9 digits, not embedded in a longer digit run.
_PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
# Resident ID card: 17 digits + final digit-or-X checksum.
_ID_CARD_RE = re.compile(r"(?<!\d)(\d{17}[\dXx])(?!\d)")
# Bank / UnionPay card: 16–19 digit run. Checked AFTER id-card so an 18-digit
# id isn't mis-tagged as a card.
_BANK_CARD_RE = re.compile(r"(?<!\d)(\d{16,19})(?!\d)")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Street address: anchored on a leading 省/市/区/县 administrative token and a
# trailing 号/室/栋/单元 building token, so we don't swallow ordinary prose.
_ADDRESS_RE = re.compile(
    r"[一-龥]{2,}(?:省|市|区|县|镇|乡|街道|路|街|村)"
    r"[一-龥\d\-]{0,40}?(?:号|室|栋|幢|单元|楼|层)"
)

_ADDRESS_PLACEHOLDER = "【已脱敏地址】"


def _mask_middle(value: str, *, keep_head: int, keep_tail: int) -> str:
    """Mask the middle of ``value``, keeping head/tail for recognisability."""
    if len(value) <= keep_head + keep_tail:
        return "*" * len(value)
    stars = "*" * (len(value) - keep_head - keep_tail)
    return f"{value[:keep_head]}{stars}{value[-keep_tail:]}"


def mask_pii(text: str) -> str:
    """Return ``text`` with detected PII masked.

    Detection order matters: email and id-card run before the bank-card rule so
    a longer structured token isn't greedily mis-tagged. Each kind is masked to
    preserve just enough for human recognition, except addresses which are fully
    replaced.

    Args:
        text: Free-text fact content (a candidate memory sentence).

    Returns:
        The same text with phone / id-card / bank-card / email / address spans
        masked. Returns the input unchanged when nothing matches.
    """
    if not text:
        return text

    text = _EMAIL_RE.sub(lambda m: _mask_email(m.group(0)), text)
    # Address before the digit rules so address-embedded digits don't trip them.
    text = _ADDRESS_RE.sub(_ADDRESS_PLACEHOLDER, text)
    text = _ID_CARD_RE.sub(lambda m: _mask_middle(m.group(1), keep_head=4, keep_tail=4), text)
    text = _BANK_CARD_RE.sub(lambda m: _mask_middle(m.group(1), keep_head=0, keep_tail=4), text)
    text = _PHONE_RE.sub(lambda m: _mask_middle(m.group(1), keep_head=3, keep_tail=4), text)
    return text


def _mask_email(value: str) -> str:
    local, _, domain = value.partition("@")
    masked_local = local[0] + "*" * max(1, len(local) - 1) if local else "*"
    return f"{masked_local}@{domain}"


def contains_pii(text: str) -> bool:
    """True if ``text`` contains any detectable PII. Cheap pre-check for gating."""
    if not text:
        return False
    return any(
        rx.search(text) for rx in (_EMAIL_RE, _ADDRESS_RE, _ID_CARD_RE, _BANK_CARD_RE, _PHONE_RE)
    )
