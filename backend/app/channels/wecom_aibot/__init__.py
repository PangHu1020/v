"""WeCom 智能机器人 (WebSocket) channel adapter.

Unlike the HTTP webhook adapter under ``backend.app.channels.wecom``,
this package speaks the persistent-WebSocket protocol of WeCom's
intelligent-bot interface. The WS connection is owned by a single
standalone worker process (:mod:`backend.app.wecom_aibot_worker`); other
processes publish outbound replies through a Redis pub/sub channel which
the worker subscribes to and forwards over its socket.
"""
