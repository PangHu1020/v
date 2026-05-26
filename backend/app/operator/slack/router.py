"""Slack inbound router.

Two endpoints:

- ``POST /operator/slack/events``: Slack Events API. Handles
  ``url_verification`` handshake and forwards operator's thread replies to
  the registered ``on_operator_message`` callback. Bot-authored messages
  are filtered out so we don't echo our own forwarded customer messages.
- ``POST /operator/slack/interactivity``: Slack Block Kit interactivity.
  Currently handles only the ``resume_session`` button which fires the
  registered ``on_resume`` callback with the embedded session id.

Signature verification runs on both endpoints.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from urllib.parse import parse_qs

import redis.asyncio as redis_async
from fastapi import APIRouter, HTTPException, Request

from backend.app.operator.slack.signature import verify_signature
from backend.v.utils.logging import bind_request, get_logger

_log = get_logger("operator.slack.router")

ResumeFn = Callable[[str], Awaitable[None]]
"""``async (session_id) -> None``"""

OperatorMessageFn = Callable[[str, str], Awaitable[None]]
"""``async (session_id, text) -> None``"""


def build_router(
    *,
    signing_secret: str,
    redis: redis_async.Redis,
    on_resume: ResumeFn,
    on_operator_message: OperatorMessageFn,
) -> APIRouter:
    """Construct the Slack router bound to the resume and message callbacks."""
    router = APIRouter(prefix="/operator/slack", tags=["slack"])

    async def _ensure_signed(request: Request) -> bytes:
        body = await request.body()
        ts = request.headers.get("x-slack-request-timestamp", "")
        sig = request.headers.get("x-slack-signature", "")
        if not verify_signature(
            signing_secret=signing_secret,
            timestamp=ts,
            body=body,
            signature=sig,
        ):
            raise HTTPException(status_code=401, detail="bad signature")
        return body

    @router.post("/events")
    async def events(request: Request) -> dict:
        body = await _ensure_signed(request)
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail="bad JSON") from exc

        if payload.get("type") == "url_verification":
            return {"challenge": payload.get("challenge", "")}

        if payload.get("type") != "event_callback":
            return {"ok": True}

        event = payload.get("event", {})
        if event.get("type") != "message":
            return {"ok": True}
        # Filter: ignore bot-authored, edited, and thread root messages.
        if event.get("bot_id") or event.get("subtype"):
            return {"ok": True}
        thread_ts = event.get("thread_ts")
        if not thread_ts:
            return {"ok": True}

        session_raw = await redis.get(f"slack_thread:{thread_ts}")
        if not session_raw:
            _log.warning("slack.events.unknown_thread", thread_ts=thread_ts)
            return {"ok": True}
        session_id = session_raw.decode("utf-8")
        text = event.get("text", "")
        if not text:
            return {"ok": True}

        with bind_request(
            request_id=request.state.request_id,
            session_id=session_id,
        ):
            _log.info("slack.events.operator_message", text_len=len(text))
            await on_operator_message(session_id, text)
        return {"ok": True}

    @router.post("/interactivity")
    async def interactivity(request: Request) -> dict:
        body = await _ensure_signed(request)
        # Body is form-encoded ``payload=<urlencoded JSON>``.
        form = parse_qs(body.decode("utf-8"))
        raw_payload = form.get("payload", [""])[0]
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="bad payload JSON") from exc

        if payload.get("type") != "block_actions":
            return {"ok": True}

        for action in payload.get("actions", []):
            if action.get("action_id") != "resume_session":
                continue
            session_id = action.get("value")
            if not session_id:
                continue
            with bind_request(
                request_id=request.state.request_id,
                session_id=session_id,
            ):
                _log.info("slack.interactivity.resume_clicked")
                await on_resume(session_id)
        return {"ok": True}

    return router
