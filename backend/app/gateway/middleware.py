"""Cross-cutting middleware for the gateway layer."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from backend.v.utils.logging import bind_request, get_logger

_log = get_logger("gateway.access")


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Stamp every request with a UUID, bind it onto the structlog context.

    The id is read from the ``X-Request-Id`` header when present (so upstream
    proxies can keep their correlation id), otherwise generated. The same id
    is echoed back on the response.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id

        with bind_request(request_id=request_id):
            _log.info(
                "http.request",
                method=request.method,
                path=request.url.path,
            )
            response = await call_next(request)
            _log.info(
                "http.response",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
            )
        response.headers["x-request-id"] = request_id
        return response
