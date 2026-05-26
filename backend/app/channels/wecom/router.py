"""WeCom webhook router.

Two endpoints:

- ``GET /webhook/wecom``: URL verification echo. WeCom sends ``msg_signature``,
  ``timestamp``, ``nonce``, and an encrypted ``echostr``. We verify the
  signature, decrypt ``echostr``, and return its plaintext.
- ``POST /webhook/wecom``: encrypted event delivery. The XML body contains
  ``<Encrypt>...</Encrypt>``. We verify the signature against that ciphertext,
  decrypt, parse, normalize to :class:`SystemMessage`, and forward through
  the debouncer.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from backend.app.bus.messages import SystemMessage
from backend.app.channels.debounce import Debouncer
from backend.app.channels.wecom.crypto import WecomCrypto, WecomCryptoError
from backend.app.channels.wecom.signature import verify_signature
from backend.v.configs.base import WecomSettings
from backend.v.utils.logging import bind_request, get_logger

_log = get_logger("channels.wecom.router")


def _parse_xml(text: str) -> dict[str, str]:
    """Parse a WeCom message XML envelope into a flat string dict."""
    try:
        root = ET.fromstring(text)  # noqa: S314  # WeCom XML is signed + decrypted by us; not user-supplied
    except ET.ParseError as exc:
        raise ValueError(f"malformed XML: {exc}") from exc
    return {child.tag: (child.text or "") for child in root}


def _extract_encrypt(xml_text: str) -> str:
    """Pull just the ``<Encrypt>`` element value from an outer envelope."""
    fields = _parse_xml(xml_text)
    encrypt = fields.get("Encrypt")
    if not encrypt:
        raise ValueError("envelope is missing <Encrypt>")
    return encrypt


def build_router(
    settings: WecomSettings,
    crypto: WecomCrypto,
    debouncer: Debouncer,
) -> APIRouter:
    """Construct the WeCom router bound to the given settings + crypto + debouncer."""
    router = APIRouter(prefix="/webhook/wecom", tags=["wecom"])

    @router.get("", response_class=PlainTextResponse)
    async def url_verification(
        msg_signature: str = Query(...),
        timestamp: str = Query(...),
        nonce: str = Query(...),
        echostr: str = Query(...),
    ) -> str:
        """Reply to WeCom's URL verification challenge."""
        if not verify_signature(
            token=settings.token,
            timestamp=timestamp,
            nonce=nonce,
            encrypted=echostr,
            signature=msg_signature,
        ):
            raise HTTPException(status_code=401, detail="bad signature")
        try:
            return crypto.decrypt(echostr)
        except WecomCryptoError as exc:
            _log.error("wecom.url_verify.decrypt_failed", error=str(exc))
            raise HTTPException(status_code=400, detail="decrypt failed") from exc

    @router.post("", response_class=PlainTextResponse)
    async def receive_event(
        request: Request,
        msg_signature: str = Query(...),
        timestamp: str = Query(...),
        nonce: str = Query(...),
    ) -> str:
        """Accept an encrypted event; ack immediately (200 ``success``)."""
        body = (await request.body()).decode("utf-8")
        try:
            encrypt = _extract_encrypt(body)
        except ValueError as exc:
            _log.error("wecom.envelope.parse_failed", error=str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if not verify_signature(
            token=settings.token,
            timestamp=timestamp,
            nonce=nonce,
            encrypted=encrypt,
            signature=msg_signature,
        ):
            raise HTTPException(status_code=401, detail="bad signature")

        try:
            inner = crypto.decrypt(encrypt)
        except WecomCryptoError as exc:
            _log.error("wecom.event.decrypt_failed", error=str(exc))
            raise HTTPException(status_code=400, detail="decrypt failed") from exc

        try:
            inner_fields = _parse_xml(inner)
        except ValueError as exc:
            _log.error("wecom.event.xml_failed", error=str(exc))
            raise HTTPException(status_code=400, detail="bad inner XML") from exc

        msg_type = inner_fields.get("MsgType", "")
        if msg_type != "text":
            # Phase-1 only handles text. Voice/image are out of scope.
            _log.info("wecom.event.skip_non_text", msg_type=msg_type)
            return "success"

        sys_msg = SystemMessage(
            channel="wecom",
            channel_user_id=inner_fields["FromUserName"],
            text=inner_fields.get("Content", ""),
            dedup_key=inner_fields.get("MsgId", "")
            or f"wecom-{inner_fields['FromUserName']}-{timestamp}",
            received_at=datetime.now(UTC),
        )

        with bind_request(
            request_id=request.state.request_id,
            channel="wecom",
            channel_user_id=sys_msg.channel_user_id,
        ):
            await debouncer.observe(sys_msg)
            _log.info("wecom.event.observed", dedup_key=sys_msg.dedup_key)

        return "success"

    return router
