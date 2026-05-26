"""HTTP gateway: routers, middleware, and the FastAPI app entrypoint glue.

Webhooks for customer channels (``/wecom/*``, ``/feishu/*``) are mounted by
their respective ``backend.app.channels`` modules; this package owns the
admin / health surface and the global request-id middleware.
"""
