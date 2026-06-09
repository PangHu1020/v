"""Standalone mock MCP server — order / logistics / human-handoff tools.

A real HTTP MCP server (FastMCP, streamable-http transport) protected by
OAuth 2.0 client_credentials. It exists so the agent's MCP integration can be
exercised end-to-end against a genuine OAuth + JSON-RPC surface; swap it for a
real server later by only changing the ``url`` / credentials in config.

Endpoints
---------
- ``POST /token`` — OAuth token endpoint. Accepts ``grant_type=client_credentials``
  + ``client_id`` + ``client_secret``; returns a short-lived bearer token.
- ``/mcp``        — the MCP streamable-http endpoint, bearer-gated.

Tools
-----
- ``query_order(order_id)``       — order status + amount (hardcoded fake data).
- ``query_logistics(order_id)``   — shipment status + tracking no + timeline (fake).
- ``create_handoff_ticket(reason, customer_note)`` — registers a human-handoff
  ticket (write-class: not cached) and returns ``{ticket_id, status}``.

All data is hardcoded — no DB. Run::

    uv run python -m scripts.mock_mcp_server          # listens on :9100
    MOCK_MCP_PORT=9100 MOCK_MCP_CLIENT_ID=agent \\
      MOCK_MCP_CLIENT_SECRET=secret uv run python -m scripts.mock_mcp_server
"""

from __future__ import annotations

import json
import os
import secrets
import time
import uuid

from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from pydantic import AnyHttpUrl
from starlette.requests import Request
from starlette.responses import JSONResponse

# --- credentials (override via env) ---------------------------------------
CLIENT_ID = os.environ.get("MOCK_MCP_CLIENT_ID", "agent")
CLIENT_SECRET = os.environ.get("MOCK_MCP_CLIENT_SECRET", "secret")
PORT = int(os.environ.get("MOCK_MCP_PORT", "9100"))
HOST = os.environ.get("MOCK_MCP_HOST", "127.0.0.1")
ISSUER = f"http://{HOST}:{PORT}"
TOKEN_TTL_SECONDS = 3600
REQUIRED_SCOPE = "tools:call"

# --- in-memory token store: token -> expires_at(epoch) --------------------
_ISSUED: dict[str, float] = {}
# --- in-memory handoff tickets --------------------------------------------
_TICKETS: list[dict] = []

# --- hardcoded fake order data --------------------------------------------
_ORDERS: dict[str, dict] = {
    "SO1001": {"status": "已发货", "amount": 8999.0, "product": "iPhone 15 Pro", "qty": 1},
    "SO1002": {"status": "已签收", "amount": 3299.0, "product": "美的空调 KFR-35GW", "qty": 1},
    "SO1003": {"status": "待付款", "amount": 1390.0, "product": "亚瑟士 GEL-KAYANO 30", "qty": 1},
    "SO1004": {"status": "已退款", "amount": 150.0, "product": "乐事原味薯片 150g", "qty": 6},
}
_LOGISTICS: dict[str, dict] = {
    "SO1001": {
        "carrier": "顺丰速运",
        "tracking_no": "SF1234567890",
        "status": "运输中",
        "timeline": ["2026-06-01 已揽收", "2026-06-02 到达深圳转运中心", "2026-06-03 派送中"],
    },
    "SO1002": {
        "carrier": "京东物流",
        "tracking_no": "JD9876543210",
        "status": "已签收",
        "timeline": ["2026-05-28 已揽收", "2026-05-29 已签收（本人）"],
    },
}


class _StaticTokenVerifier(TokenVerifier):
    """Verifies opaque bearer tokens issued by our /token endpoint."""

    async def verify_token(self, token: str) -> AccessToken | None:
        expires_at = _ISSUED.get(token)
        if expires_at is None or time.time() >= expires_at:
            _ISSUED.pop(token, None)
            return None
        return AccessToken(
            token=token,
            client_id=CLIENT_ID,
            scopes=[REQUIRED_SCOPE],
            expires_at=int(expires_at),
        )


mcp = FastMCP(
    "mock-shop",
    host=HOST,
    port=PORT,
    token_verifier=_StaticTokenVerifier(),
    auth=AuthSettings(
        issuer_url=AnyHttpUrl(ISSUER),
        resource_server_url=AnyHttpUrl(f"{ISSUER}/mcp"),
        required_scopes=[REQUIRED_SCOPE],
    ),
)


@mcp.tool(description="查询订单状态与金额。参数 order_id，例如 SO1001。")
def query_order(order_id: str) -> str:
    o = _ORDERS.get(order_id.strip().upper())
    if not o:
        return json.dumps({"found": False, "order_id": order_id}, ensure_ascii=False)
    return json.dumps({"found": True, "order_id": order_id, **o}, ensure_ascii=False)


@mcp.tool(description="查询订单物流轨迹。参数 order_id，例如 SO1001。")
def query_logistics(order_id: str) -> str:
    o = _LOGISTICS.get(order_id.strip().upper())
    if not o:
        return json.dumps(
            {"found": False, "order_id": order_id, "note": "暂无物流信息"}, ensure_ascii=False
        )
    return json.dumps({"found": True, "order_id": order_id, **o}, ensure_ascii=False)


@mcp.tool(description="登记人工介入工单。参数 reason（原因）、customer_note（客户原话）。")
def create_handoff_ticket(reason: str, customer_note: str = "") -> str:
    ticket = {
        "ticket_id": f"TK{uuid.uuid4().hex[:8].upper()}",
        "reason": reason,
        "customer_note": customer_note,
        "status": "open",
    }
    _TICKETS.append(ticket)
    return json.dumps(ticket, ensure_ascii=False)


# --- OAuth token endpoint (mounted on the same Starlette app) -------------
async def _token_endpoint(request: Request) -> JSONResponse:
    form = await request.form()
    grant = form.get("grant_type")
    cid = form.get("client_id")
    secret = form.get("client_secret")
    if grant != "client_credentials":
        return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)
    if cid != CLIENT_ID or secret != CLIENT_SECRET:
        return JSONResponse({"error": "invalid_client"}, status_code=401)
    token = secrets.token_urlsafe(32)
    _ISSUED[token] = time.time() + TOKEN_TTL_SECONDS
    return JSONResponse(
        {"access_token": token, "token_type": "Bearer", "expires_in": TOKEN_TTL_SECONDS}
    )


def build_app():
    """Return the Starlette app: MCP streamable-http + the /token route."""
    app = mcp.streamable_http_app()
    app.add_route("/token", _token_endpoint, methods=["POST"])
    return app


def main() -> None:
    import uvicorn

    print(f"mock MCP server on {ISSUER}  (client_id={CLIENT_ID})")
    print(f"  token:  POST {ISSUER}/token")
    print(f"  mcp:    {ISSUER}/mcp  (bearer-gated)")
    uvicorn.run(build_app(), host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
