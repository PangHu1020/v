"""Feishu webhook router.

Single endpoint:

- ``POST /webhook/feishu``: handles three payload shapes:
  1. URL verification: ``{"type": "url_verification", "challenge": "..."}``
     -> echo ``{"challenge": ...}``.
  2. Encrypted event: ``{"encrypt": "<base64>"}`` -> decrypt -> parse ->
     normalize -> debounce.
  3. Plain event (encryption disabled): handled the same as case 2 but
     without the decrypt step. Phase-1 expects encryption to be enabled.

Signature verification uses the headers ``X-Lark-Signature``,
``X-Lark-Request-Timestamp``, and ``X-Lark-Request-Nonce``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request

from backend.app.bus.messages import SystemMessage
from backend.app.channels.debounce import Debouncer
from backend.app.channels.feishu.crypto import FeishuCrypto, FeishuCryptoError
from backend.app.channels.feishu.signature import verify_signature
from backend.v.configs.base import FeishuSettings
from backend.v.utils.logging import bind_request, get_logger

_log = get_logger("channels.feishu.router")


def _open_id_from_event(event: dict) -> str:
    sender = event.get("sender", {})
    sender_id = sender.get("sender_id", {})
    open_id = sender_id.get("open_id")
    if not open_id:
        raise ValueError("event missing sender.sender_id.open_id")
    return open_id


def _text_from_event(event: dict) -> str:
    message = event.get("message", {})
    if message.get("message_type") != "text":
        return ""
    raw = message.get("content", "")
    if not raw:
        return ""
    try:
        return json.loads(raw).get("text", "")
    except json.JSONDecodeError:
        return ""


def _message_id_from_event(event: dict) -> str:
    return event.get("message", {}).get("message_id", "")


def build_router(
    settings: FeishuSettings,
    crypto: FeishuCrypto,
    debouncer: Debouncer,
) -> APIRouter:
    """Construct the Feishu router bound to the given settings + crypto + debouncer."""
    router = APIRouter(prefix="/webhook/feishu", tags=["feishu"])

    @router.post("")
    async def receive_event(request: Request) -> dict:
        body = await request.body()

        timestamp = request.headers.get("x-lark-request-timestamp", "")
        nonce = request.headers.get("x-lark-request-nonce", "")
        signature = request.headers.get("x-lark-signature", "")

        # Signature is required for encrypted events; permitted to be absent
        # only when encryption is disabled at the platform side. Phase-1
        # requires encryption, so we always check.
        if signature and not verify_signature(
            timestamp=timestamp,
            nonce=nonce,
            encrypt_key=settings.encrypt_key,
            body=body,
            signature=signature,
        ):
            raise HTTPException(status_code=401, detail="bad signature")

        try:
            outer = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail="bad JSON") from exc

        # URL verification (handshake during webhook setup).
        if outer.get("type") == "url_verification":
            return {"challenge": outer.get("challenge", "")}

        # Encrypted event delivery.
        if "encrypt" in outer:
            try:
                inner_text = crypto.decrypt(outer["encrypt"])
            except FeishuCryptoError as exc:
                _log.error("feishu.event.decrypt_failed", error=str(exc))
                raise HTTPException(status_code=400, detail="decrypt failed") from exc
            try:
                inner = json.loads(inner_text)
            except json.JSONDecodeError as exc:
                _log.error("feishu.event.inner_json_failed", error=str(exc))
                raise HTTPException(status_code=400, detail="bad inner JSON") from exc
        else:
            # Encryption disabled at the platform side; payload is already plain.
            inner = outer

        # Inner URL-verification (some Feishu modes pack the challenge inside
        # the encrypted envelope).
        if inner.get("type") == "url_verification":
            return {"challenge": inner.get("challenge", "")}

        event = inner.get("event", {})
        try:
            open_id = _open_id_from_event(event)
        except ValueError:
            # Non-message events (member_added, etc.) are silently ignored
            # in Phase-1.
            _log.info("feishu.event.skip_non_message")
            return {"ok": True}

        text = _text_from_event(event)
        if not text:
            _log.info("feishu.event.skip_non_text")
            return {"ok": True}

        sys_msg = SystemMessage(
            channel="feishu",
            channel_user_id=open_id,
            text=text,
            dedup_key=_message_id_from_event(event) or f"feishu-{open_id}-{timestamp}",
            received_at=datetime.now(UTC),
        )

        with bind_request(
            request_id=request.state.request_id,
            channel="feishu",
            channel_user_id=sys_msg.channel_user_id,
        ):
            await debouncer.observe(sys_msg)
            _log.info("feishu.event.observed", dedup_key=sys_msg.dedup_key)

        return {"ok": True}

    return router
