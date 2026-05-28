# 调用链：一条 WeCom 客户消息的完整跟踪

跟踪一条来自 WeCom 客户的文本消息「订单 ORD123 状态」从 HTTP 入站到 AI 回复落地的全部函数调用，带文件路径和关键行号。

跟踪基于 [backend/test/e2e/test_inbound_to_reply.py:test_full_round_trip](../backend/test/e2e/test_inbound_to_reply.py)，可视为可运行的"活文档"——任何代码改动都会被这个测试验证或暴露。

Phase-2 后图变成 `enter → agent → [tools? → agent]* → exit → END`（条件循环），且 worker 在调用图前后多了 **挂起检查** 与 **interrupt 检查** 两个分支。Phase-2 后涉及的工具调用 / 接管路径在文末单列。

---

## 阶段 0：客户端发出加密 webhook

WeCom 平台对配置的回调 URL 发起 POST：

```
POST /webhook/wecom?msg_signature=<sha1>&timestamp=<ts>&nonce=<n>
Content-Type: text/xml

<xml>
  <ToUserName>corp_id</ToUserName>
  <Encrypt><![CDATA[<base64 ciphertext>]]></Encrypt>
</xml>
```

ciphertext 解密后是另一段内层 XML，含 `<FromUserName>`、`<MsgType>`、`<Content>`、`<MsgId>` 等字段。

---

## 阶段 1：FastAPI 入口 + 中间件

[backend/app/main.py:create_app](../backend/app/main.py) → Uvicorn 接收 HTTP，路由到 `RequestIdMiddleware`

[backend/app/gateway/middleware.py:RequestIdMiddleware.dispatch](../backend/app/gateway/middleware.py)

- 读取 `X-Request-Id` header 或 mint 一个 UUID
- 进入 `bind_request(request_id=...)` 上下文
- 调用 `call_next(request)` 进入路由

---

## 阶段 2：WeCom 路由处理

[backend/app/channels/wecom/router.py:receive_event](../backend/app/channels/wecom/router.py)

```python
@router.post("", response_class=PlainTextResponse)
async def receive_event(request: Request, msg_signature, timestamp, nonce):
    body = (await request.body()).decode("utf-8")
```

### 2.1 解析外层 XML 取 `<Encrypt>` 字段

[backend/app/channels/wecom/router.py:_extract_encrypt](../backend/app/channels/wecom/router.py) → `xml.etree.ElementTree.fromstring`。

### 2.2 验证签名

[backend/app/channels/wecom/signature.py:verify_signature](../backend/app/channels/wecom/signature.py)

```python
parts = sorted([token, timestamp, nonce, encrypted])
expected = sha1("".join(parts)).hexdigest()
return hmac.compare_digest(expected, msg_signature)
```

不匹配返回 401。

### 2.3 AES 解密

[backend/app/channels/wecom/crypto.py:WecomCrypto.decrypt](../backend/app/channels/wecom/crypto.py)

- base64 decode → AES-256-CBC decrypt → PKCS7 unpad
- envelope 头：`random(16) | msg_len(4 BE) | msg | corp_id`
- 校验 `corp_id`，返回内层 XML

### 2.4 解析内层 XML & 构造 SystemMessage

```python
sys_msg = SystemMessage(
    channel="wecom",
    channel_user_id="ext-acceptance",
    text="订单 ORD123 状态",
    dedup_key="1234567",  # 来自 <MsgId>
    received_at=datetime.now(UTC),
)
```

### 2.5 推入 debouncer

```python
with bind_request(request_id=..., channel="wecom", channel_user_id="ext-acceptance"):
    await debouncer.observe(sys_msg)
return "success"
```

---

## 阶段 3：Debouncer 合并窗口

[backend/app/channels/debounce.py:Debouncer.observe](../backend/app/channels/debounce.py) — HSET 累积 + asyncio task 定时器；500ms 后 `_flush_after_window` 触发 `dispatch`（即 `BusProducer.enqueue`）。

---

## 阶段 4：进入 Bus

[backend/app/bus/producer.py:BusProducer.enqueue](../backend/app/bus/producer.py) → `mmh3.hash("wecom:ext-acceptance") % 64` → `XADD bus:{shard}` 一行。HTTP 200 早在阶段 2 就返回了，整条入站异步分离。

---

## 阶段 5：Bus Consumer 异步处理

[backend/app/bus/consumer.py:BusConsumer._run_shard](../backend/app/bus/consumer.py) `XREADGROUP block_ms=5000` → batch → `_dispatch` → 反序列化 `SystemMessage` → `bind_request` → 调 handler。失败入 DLQ + ack；成功 ack。

---

## 阶段 6：Worker handler

handler 由 [backend/app/bus/worker.py:make_bus_handler](../backend/app/bus/worker.py) 构造的闭包。Phase-2 后多了挂起 / interrupt 两个分支。

### 6.1 解析 session_id

[backend/app/bus/worker.py:_resolve_session_id](../backend/app/bus/worker.py) — 30 分钟以内复用，否则 mint UUID。**首次 mint 时还会 INSERT `agent.session` 行**（`ON CONFLICT DO NOTHING`），让 on_interrupt 的 UPDATE 有目标。

### 6.2 ⚠️ 挂起检查（Phase-2 P0）

[backend/v/hooks/handoff.py:is_suspended](../backend/v/hooks/handoff.py)

```python
if handoff_enabled and await is_suspended(session_id, redis=redis):
    thread_ts = await slack_outbound.get_thread_for_session(session_id)
    if thread_ts:
        await slack_outbound.post_customer_message(thread_ts=thread_ts, text=msg.text)
    await _record_last_seen(...)
    return  # ← 不调 graph
```

挂起期间客户继续发消息 → 直接转发到 Slack alert thread。

### 6.3 加载 user_profile

[backend/v/hooks/session.py:on_session_start](../backend/v/hooks/session.py) — Redis 缓存 → PG 兜底 → 写回缓存。

### 6.4 构造 state + config 并调 graph

```python
input_state = {
    "messages": [HumanMessage("订单 ORD123 状态")],
    "session_id": session_id,
    "channel": "wecom",
    "channel_user_id": "ext-acceptance",
    "user_profile": {"member_level": "黄金"},
    "interrupt_payload": None,
}
config = {
    "configurable": {
        "thread_id": session_id,
        "llm_caller": llm_caller,
        "pg_pool": pool,           # recall_memory 用
        "embedder": embedder,       # recall_memory 用
        "channel": msg.channel,
        "channel_user_id": msg.channel_user_id,
        "skill_registry": skill_registry,  # enter_node 用
        "skill_top_k": 3,
    }
}
final_state = await graph.ainvoke(input_state, config=config)
```

### 6.5 ⚠️ Interrupt 检查（Phase-2 P0）

[backend/v/hooks/handoff.py:extract_interrupt](../backend/v/hooks/handoff.py)

```python
interrupt_payload = await extract_interrupt(final_state)
# 读 final_state["__interrupt__"][0].value
if interrupt_payload is not None:
    await on_interrupt(...)  # 见阶段 10
    return  # ← 不发普通回复
```

---

## 阶段 7：LangGraph 执行

[backend/v/agents/graph.py:build_graph](../backend/v/agents/graph.py) 构造的图：

```
enter → agent → route_after_agent → ┬→ tools → agent → ... → exit → END
                                    └→ exit → END
```

配 [RedisCheckpointer](../backend/v/agents/checkpointer.py)，热路径。

### 7.1 ckpt.aget_tuple（首轮 → None）

`HGET ckpt:{thread_id} latest` → 首次 `None` → 从初始 state 开始。

### 7.2 enter_node

[backend/v/agents/nodes.py:enter_node](../backend/v/agents/nodes.py)

```python
if any(isinstance(m, SystemMessage) for m in messages):
    return {}                                             # 后续轮跳过
prompt = _system_prompt(profile, channel)                 # 基础客服 prompt + 客户档案
skill_section = _maybe_skill_section(skill_registry,      # Phase-2 P4
                                     query=last_human_text,
                                     channel=channel,
                                     top_k=3)
full = f"{prompt}\n\n{skill_section}" if skill_section else prompt
return {"messages": [SystemMessage(content=full)]}
```

skill_section 内容形如：

```
以下 SOP 与本次咨询相关，按优先级排列，请优先遵循：

## refund_sop（退款流程）
1. 询问订单号
2. 确认订单状态 ...
```

### 7.3 ckpt.aput

`HSET ckpt:{thread_id} ck:{cid}:type ... ck:{cid}:blob ... latest <cid>` + `EXPIRE 1800`.

### 7.4 agent_node

[backend/v/agents/nodes.py:agent_node](../backend/v/agents/nodes.py)（通过 `build_graph` 内的闭包包装传 `tools`）

```python
result = await caller.chat(
    "main_primary",
    list(state["messages"]),
    tools=[transfer_to_human, recall_memory, subagent, *mcp_tools],
)
return {"messages": [result.message]}
```

### 7.5 LLMCaller.chat

[backend/v/models/llm_caller.py:LLMCaller.chat](../backend/v/models/llm_caller.py)

- 解析 role → model id（`deepseek-chat-v4-pro`）
- 用 `factory.get_chat_model(...)` 构造 `ChatOpenAI(base_url=DEEPSEEK_BASE, ...)` 并 `bind_tools(tools)`
- `await asyncio.wait_for(runnable.ainvoke(messages), timeout=30)`
- 返回 `LLMResult(message=..., model=..., fallback_used=False, latency_ms=...)`

降级规则见 [data_flow.md §2.4](data_flow.md)。

### 7.6 路由判断

[backend/v/agents/edges.py:route_after_agent](../backend/v/agents/edges.py)

```python
last = state["messages"][-1]
return "tools" if isinstance(last, AIMessage) and last.tool_calls else "exit"
```

#### 7.6.a 命中 tools 分支

`ToolNode([transfer_to_human, recall_memory, subagent, *mcp_tools])` 拿到 AIMessage 里的 `tool_calls`，按 name 路由到对应工具：

- `transfer_to_human` → 调 `langgraph.types.interrupt(...)` → **整个图暂停**，返回到阶段 6.5。
- `recall_memory` → embed 查询 → pgvector cosine + 时间衰减 → 返回 top-K 文本。
- `subagent` → 单轮 LLMCaller 调用（带或不带父 messages）。
- `{server_id}__{tool}` → MCP `client.call_tool(...)`，结果走 L1+L2 缓存。

每个工具结果写为 `ToolMessage`，state.messages 追加，回到 `agent_node`。

LLM 看到工具结果后，可能再调工具或产出最终回复。直到不再产工具调用 → exit。

#### 7.6.b exit 分支

[backend/v/agents/nodes.py:exit_node](../backend/v/agents/nodes.py) — 只日志，无 state 更新 → END。

### 7.7 ckpt.aput（每次节点结束）

每跳一个节点都写一次 checkpoint，`latest` 滚动指向最新。`EXPIRE` 续约。

---

## 阶段 8：取最后 AIMessage 并出站

[backend/app/bus/worker.py:_last_ai_message](../backend/app/bus/worker.py) — 倒序找第一个有 content 且**没有 tool_calls** 的 AIMessage。

```python
send = sends["wecom"]
await send("ext-acceptance", "您好，已收到您的咨询...")
```

### 8.1 WecomOutbound.send_text

[backend/app/channels/wecom/outbound.py:WecomOutbound.send_text](../backend/app/channels/wecom/outbound.py)

- 进程内 `asyncio.Lock` 拿 access_token（缓存命中直接用，否则 `gettoken`）
- POST `/cgi-bin/message/send?access_token=...` 携带 `touser/msgtype=text/agentid/text.content`

测试里被 `respx` mock；生产真打 WeCom API。

### 8.2 记录 last_seen

```python
await redis.set("last_seen:wecom:ext-acceptance", str(time.time()), ex=3600)
```

---

## 阶段 9：Bus consumer 完成 ack

回到 [BusConsumer._dispatch](../backend/app/bus/consumer.py)：`XACK bus:42 workers <msg_id>`。

---

## 阶段 10（仅当阶段 6.5 命中）：人工接管

[backend/v/hooks/handoff.py:on_interrupt](../backend/v/hooks/handoff.py)

```python
async def on_interrupt(...):
    # 1. migrate hot → cold
    await migrate_hot_to_cold(session_id, redis_ckpt=..., pg_ckpt=...)
    # 2. 标记 suspended
    await redis.set(f"session_status:{session_id}", b"suspended", ex=...)
    await pool.execute("UPDATE agent.session SET status='suspended' WHERE session_id=$1", session_id)
    # 3. 推 Slack 告警（postMessage with Block Kit + Resume button）
    thread_ts = await slack_outbound.post_handoff_alert(
        session_id=..., channel=..., channel_user_id=...,
        customer_text=..., transfer_reason=interrupt_payload["reason"],
    )
    # SlackOutbound 内部：
    #   chat_postMessage → SET slack_thread:{ts}=session_id
    #                      SET session_thread:{session_id}={channel_id, ts}
    return thread_ts
```

挂起期间客户消息走阶段 6.2，操作员消息走 [/operator/slack/events](../backend/app/operator/slack/router.py) → `on_operator_message` → 同时写 `operator_log:{session_id}` 列表 + 通过 channel adapter 转发给客户。

操作员点 Slack "Resume AI" 按钮 → `/operator/slack/interactivity` → `on_resume`：

```python
async def on_resume(session_id, ...):
    await migrate_cold_to_hot(session_id, ...)
    await redis.set(f"session_status:{session_id}", b"active", ex=1800)
    await pool.execute("UPDATE agent.session SET status='active' ...", session_id)

    raw_msgs = await redis.lrange(f"operator_log:{session_id}", 0, -1)
    operator_messages = [m.decode() for m in raw_msgs]
    await redis.delete(f"operator_log:{session_id}")

    decision = {"type": "resume", "operator_messages": operator_messages}
    final_state = await graph.ainvoke(Command(resume=decision), config=...)
    # interrupt() 内返回 decision → tool 返回格式化字符串 → agent 看到 ToolMessage →
    # agent 产出 final AIMessage

    reply = _last_ai_message(final_state["messages"])
    await sends[channel](channel_user_id, reply)
```

详见 [data_flow.md §4](data_flow.md)。

---

## 阶段 11（变体）：WeCom 智能机器人 WebSocket（Phase-3 G）

智能机器人入站不走 HTTP webhook，而是常驻 WS。阶段 0–2 被替换为以下路径，阶段 3 起完全相同。

### 11.1 启动

[backend/app/wecom_aibot_worker.py](../backend/app/wecom_aibot_worker.py) 作为独立进程启动（`python -m backend.app.wecom_aibot_worker`），加载 `WECOM_AIBOT_WS_URL` 等配置；为空则立即退出（部署上跳过此渠道无成本）。

### 11.2 WS 长连接 + 帧解码

[backend/app/channels/wecom_aibot/client.py:WecomAibotClient.run](../backend/app/channels/wecom_aibot/client.py)

- `websockets.connect(url)`，30s 心跳 ping，断线指数回退（1s→60s）重连
- 每条入站 JSON 帧 → 提取 `from_user`、`text`、`msg_id` → 构造 `SystemMessage(channel="wecom_aibot", ...)`
- `await debouncer.observe(sys_msg)` —— 与 HTTP 渠道共用同一个 debouncer + bus producer

### 11.3 出站（Redis pub/sub 跨进程）

阶段 8 的 `sends["wecom_aibot"]` 是 [WecomAibotOutbound.send_text](../backend/app/channels/wecom_aibot/outbound.py)：在 worker handler 进程内不能直接握 WS 句柄（句柄在 wecom_aibot_worker 进程里），所以 `send_text` 仅做 `redis.publish("wecom_aibot:outbound", json)`；wecom_aibot_worker 的 pub/sub task 收到后调 `WecomAibotClient.send_text(...)` 真正写 WS 帧。

阶段 3–7、9 完全复用 HTTP 路径；阶段 6.5 的 interrupt / 阶段 10 的接管同样适用，操作员回复仍通过 `sends[channel]` 即上面这条 Redis 通道送回 WS。

---

## 总结：函数调用栈深度

最简单的一轮（无工具调用、无接管）大约 **17 个跨模块函数调用**。

| 阶段 | 耗时（粗估） |
| --- | --- |
| HTTP → bus（同步路径） | < 50ms |
| Debounce 窗口 | 500ms（几乎都是等） |
| Bus 出队 + 反序列化 | < 5ms |
| Session resolution + on_session_start | < 10ms（缓存命中 < 1ms） |
| LangGraph turn（含 ckpt 写） | LLM 调用占 99%（DeepSeek 通常 1-3s） |
| Outbound HTTP | 100-500ms |

工具调用增加：

| 工具 | 增量 |
| --- | --- |
| `recall_memory` | embed (~50-200ms) + 一次 pgvector + 一次 update last_accessed_at（~10-50ms） |
| `subagent` | 一次额外 LLM 调用（DeepSeek flash，1-2s） |
| MCP（缓存命中） | < 1ms |
| MCP（缓存未命中） | server 端往返（取决于 transport，stdio < 10ms / HTTP 100ms+） |
| `transfer_to_human` | 工具内部立即 interrupt，graph 暂停；`on_interrupt` 串行做迁移 + Slack 告警（~200-500ms） |

LLM 调用始终是绝对瓶颈。
