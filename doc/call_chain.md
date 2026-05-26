# 调用链：一条 WeCom 客户消息的完整跟踪

跟踪一条来自 WeCom 客户的文本消息「订单 ORD123 状态」从 HTTP 入站到 AI 回复落地的全部函数调用，带文件路径和关键行号。

跟踪基于 [backend/test/e2e/test_inbound_to_reply.py:test_full_round_trip](../backend/test/e2e/test_inbound_to_reply.py)，可视为可运行的"活文档"——任何代码改动都会被这个测试验证或暴露。

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

[backend/app/main.py:create_app](../backend/app/main.py)
↓ Uvicorn 接收 HTTP，路由到 `RequestIdMiddleware`

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

[backend/app/channels/wecom/router.py:_extract_encrypt](../backend/app/channels/wecom/router.py)
↓ 调用 `_parse_xml(body)` → `xml.etree.ElementTree.fromstring`
↓ 找到 `Encrypt` 子元素

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

- base64 decode → ciphertext bytes
- AES-256-CBC decrypt（key 来自 `EncodingAESKey + "="` 的 base64 解码，IV 是 key 前 16 字节）
- PKCS7 unpad
- 解析 envelope 头：`random(16) | msg_len(4 BE) | msg | corp_id`
- 校验 `corp_id` 与配置匹配
- 返回 `msg`（即内层 XML）

### 2.4 解析内层 XML

[backend/app/channels/wecom/router.py:_parse_xml](../backend/app/channels/wecom/router.py)

提取 `MsgType`、`FromUserName`、`Content`、`MsgId` 等字段。

非 `text` 类型直接返回 `success`，不入 bus（Phase-1 不处理图片/语音）。

### 2.5 构造 SystemMessage

```python
sys_msg = SystemMessage(
    channel="wecom",
    channel_user_id="ext-acceptance",
    text="订单 ORD123 状态",
    dedup_key="1234567",  # 来自 <MsgId>
    received_at=datetime.now(UTC),
)
```

### 2.6 推入 debouncer

```python
with bind_request(request_id=..., channel="wecom", channel_user_id="ext-acceptance"):
    await debouncer.observe(sys_msg)
return "success"
```

---

## 阶段 3：Debouncer 合并窗口

[backend/app/channels/debounce.py:Debouncer.observe](../backend/app/channels/debounce.py)

```python
key = f"debounce:wecom:ext-acceptance"
prior = await redis.hgetall(key)  # 空（首条消息）
merged_text = "订单 ORD123 状态"
merged_dedup = "1234567"
await redis.hset(key, mapping={text, dedup_key, channel, ...})
await redis.pexpire(key, 500 * 4)  # 防泄漏

# 启动 500ms 后的 flush
self._timers[key] = asyncio.create_task(self._flush_after_window(key))
```

如果 500ms 内有第二条消息进来，会取消上一个 task 并重启计时（这是 [backend/test/e2e/test_inbound_to_reply.py:test_burst_collapses_into_one_reply](../backend/test/e2e/test_inbound_to_reply.py) 验证的行为）。

500ms 后 [Debouncer._flush_after_window](../backend/app/channels/debounce.py)：

```python
await asyncio.sleep(0.5)
pending = await redis.hgetall(key)
await redis.delete(key)
merged = SystemMessage(...)
await self._dispatch(merged)  # 这是注入的 channel_to_bus 函数
```

---

## 阶段 4：进入 Bus

`channel_to_bus` 在 [backend/app/main.py:lifespan](../backend/app/main.py) 里定义为闭包：

```python
async def channel_to_bus(message: SystemMessage) -> None:
    await producer.enqueue(message)
```

[backend/app/bus/producer.py:BusProducer.enqueue](../backend/app/bus/producer.py)

```python
key = self._shard.stream_key("wecom", "ext-acceptance")
# 内部：mmh3.hash("wecom:ext-acceptance") % 64 → shard_idx
# key 格式 "bus:{shard_idx}"
payload = message.model_dump_json().encode("utf-8")
entry_id = await self._shard.xadd(key, {b"json": payload})
```

[backend/app/bus/shard.py:RedisStreamShard.xadd](../backend/app/bus/shard.py)
→ Redis `XADD bus:42 * json <serialized>`

返回的 entry_id 用于日志，函数到此返回；HTTP 响应早在 webhook 路由阶段已经回 200，整个入站完成。

---

## 阶段 5：Bus Consumer 异步处理

启动时 [backend/app/main.py:lifespan](../backend/app/main.py) 已经 `asyncio.create_task(consumer.run(handler))`，每个 shard 一个 task。

### 5.1 BusConsumer._run_shard

[backend/app/bus/consumer.py:BusConsumer._run_shard](../backend/app/bus/consumer.py)

```python
while not self._stop_event.is_set():
    entries = await self._shard.xreadgroup(
        group="workers", consumer="<host>:<id>",
        keys=["bus:42"], count=1, block_ms=5000,
    )
    if not entries:
        await asyncio.sleep(0.01)  # fakeredis 不阻塞，避免 hot-spin
        continue
    for stream_key, batch in entries:
        for msg_id, fields in batch:
            await self._dispatch(stream_key, msg_id, fields, handler)
```

### 5.2 BusConsumer._dispatch

[backend/app/bus/consumer.py:BusConsumer._dispatch](../backend/app/bus/consumer.py)

- 反序列化 `SystemMessage.model_validate_json(fields[b"json"])`
- 反序列化失败 → `_send_to_dlq` + `xack`
- 成功 → `bind_request(...)` 上下文 → 调用 handler
- handler 抛异常 → `_send_to_dlq` + `xack`（保证不阻塞 stream）
- 最后 `xack`

---

## 阶段 6：Worker handler

handler 由 [backend/app/bus/worker.py:make_bus_handler](../backend/app/bus/worker.py) 构造，闭包绑定 graph、pool、redis、llm_caller、sends 等运行时依赖。

### 6.1 解析 session_id

[backend/app/bus/worker.py:_resolve_session_id](../backend/app/bus/worker.py)

```python
last_seen_raw = await redis.get("last_seen:wecom:ext-acceptance")  # 首条 → None
# silence 检查跳过
session_id = str(uuid.uuid4())  # mint
await redis.set("session:wecom:ext-acceptance", session_id, ex=3600)
return session_id, True  # minted=True
```

### 6.2 加载 user_profile

[backend/v/hooks/session.py:on_session_start](../backend/v/hooks/session.py)

- 先查 Redis 缓存 `profile:wecom:ext-acceptance` → miss
- [backend/v/memory/long_term.py:read_user_profile](../backend/v/memory/long_term.py) → `SELECT profile FROM agent.user_profile WHERE channel=$1 AND channel_user_id=$2`
  - 测试中 mock 返回 `{"member_level": "黄金"}`
- 写回 Redis 缓存（TTL 1800s）
- 返回 `{"member_level": "黄金"}`

### 6.3 构造 state 并调用 graph

```python
input_state = {
    "messages": [HumanMessage("订单 ORD123 状态")],
    "session_id": session_id,
    "channel": "wecom",
    "channel_user_id": "ext-acceptance",
    "user_profile": {"member_level": "黄金"},
    "interrupt_payload": None,
}
config = {"configurable": {"thread_id": session_id, "llm_caller": llm_caller}}
final_state = await graph.ainvoke(input_state, config=config)
```

---

## 阶段 7：LangGraph 执行

由 [backend/v/agents/graph.py:build_graph](../backend/v/agents/graph.py) 构造的线性图：`enter → agent → exit → END`，配 [RedisCheckpointer](../backend/v/memory/checkpointer.py)。

### 7.1 ckpt.aget_tuple

LangGraph 先查 [backend/v/memory/checkpointer.py:RedisCheckpointer.aget_tuple](../backend/v/memory/checkpointer.py)：

```
HGET ckpt:{thread_id} latest  → None（首次）
return None
```

→ 从初始 state 开始执行。

### 7.2 enter_node

[backend/v/agents/nodes.py:enter_node](../backend/v/agents/nodes.py)

```python
messages = state.get("messages", [])  # [HumanMessage(...)]
if any(isinstance(m, SystemMessage) for m in messages):
    return {}  # 跳过
prompt = _system_prompt(profile, channel)
# prompt 内容：客服角色 + channel + "已知客户档案: 会员等级：黄金"
return {"messages": [SystemMessage(content=prompt)]}
```

LangGraph 用 `add_messages` reducer 把 SystemMessage 追加到 messages 头部（实际是按 ID 合并，结果是 [Sys, Human]）。

### 7.3 ckpt.aput（after enter_node）

`HSET ckpt:{thread_id} ck:{cid}:type ... ck:{cid}:blob ... latest <cid>`
`EXPIRE ckpt:{thread_id} 1800`

### 7.4 agent_node

[backend/v/agents/nodes.py:agent_node](../backend/v/agents/nodes.py)

```python
caller = config["configurable"]["llm_caller"]
result = await caller.chat("main_primary", state["messages"])
return {"messages": [result.message]}
```

### 7.5 LLMCaller.chat

[backend/v/models/llm_caller.py:LLMCaller.chat](../backend/v/models/llm_caller.py)

- `_resolve("main_primary")` → `"deepseek-chat-v4-pro"`
- `_fallback_model_for("main_primary")` → `"deepseek-chat-v4-flash"`
- 进 `_invoke`：
  - [backend/v/models/factory.py:get_chat_model](../backend/v/models/factory.py) 构造 `ChatOpenAI(model=pro, base_url=DEEPSEEK_BASE, api_key=...)`
  - `runnable = chat`（无 structured / tools）
  - `result = await asyncio.wait_for(runnable.ainvoke(messages), timeout=30)`
  - 返回 `LLMResult(message=AIMessage("您好，已收到您的咨询..."), model="...-pro", fallback_used=False, latency_ms=...)`

如果 primary 抛 `APITimeoutError` / `RateLimitError` / `InternalServerError` / `asyncio.TimeoutError`：

- log `llm.primary_failed_falling_back`
- 改用 fallback model 重新 `_invoke`
- 都失败 → 抛原始 primary 错误

测试里 `llm_caller.chat` 被 `AsyncMock` 替代，直接返回固定 `LLMResult`。

### 7.6 ckpt.aput（after agent_node）

再写一次 checkpoint：包含 [Sys, Human, AI] 完整 messages。`latest` 指向新 cid。

### 7.7 exit_node

[backend/v/agents/nodes.py:exit_node](../backend/v/agents/nodes.py)
仅日志，无 state update。

### 7.8 END

LangGraph 把整个 thread 的 final state 返回给 caller。

---

## 阶段 8：取最后 AIMessage 并出站

[backend/app/bus/worker.py:_last_ai_message](../backend/app/bus/worker.py)

倒序遍历 messages 找第一个 `AIMessage`，返回其 `content`。

[backend/app/bus/worker.py:make_bus_handler](../backend/app/bus/worker.py) 闭包内：

```python
send = sends["wecom"]  # = WecomOutbound.send_text
await send("ext-acceptance", "您好，已收到您的咨询...")
```

### 8.1 WecomOutbound.send_text

[backend/app/channels/wecom/outbound.py:WecomOutbound.send_text](../backend/app/channels/wecom/outbound.py)

```python
token = await self._access_token()
# 内部：asyncio.Lock + 内存 token cache
# - 缓存命中 → 直接返回
# - miss → GET /cgi-bin/gettoken?corpid=...&corpsecret=... → 缓存

await self._client.post(
    "/cgi-bin/message/send?access_token=...",
    json={
        "touser": "ext-acceptance",
        "msgtype": "text",
        "agentid": "1000002",
        "text": {"content": "您好，..."},
    },
)
```

测试里这个 HTTP 调用被 `respx` 截获并断言 payload 正确。生产环境真正 hit 企业微信 API。

### 8.2 记录 last_seen

```python
await redis.set(
    "last_seen:wecom:ext-acceptance",
    str(time.time()),
    ex=3600,
)
```

---

## 阶段 9：Bus consumer 完成 ack

回到 [BusConsumer._dispatch](../backend/app/bus/consumer.py)：

```python
await self._shard.xack("bus:42", "workers", msg_id)
```

shard 上的下一条消息可以处理了。

---

## 总结：函数调用栈深度

完整一次往返大约 **17 个跨模块函数调用**。耗时分布（粗估）：

| 阶段 | 耗时 |
| --- | --- |
| HTTP → bus（同步路径） | < 50ms |
| Debounce 窗口 | 500ms（几乎都是等） |
| Bus 出队 + 反序列化 | < 5ms |
| Session resolution + on_session_start | < 10ms（命中缓存 < 1ms） |
| LangGraph turn（含两次 ckpt 写） | LLM 调用占 99%（DeepSeek 通常 1-3s） |
| Outbound HTTP | 100-500ms |

**LLM 调用是绝对瓶颈**——其余都在毫秒级。Phase-2 加 `recall_memory` 工具后会引入额外的向量检索（~10-50ms），但仍是 LLM 主导。
