# 调用链：一条 WeCom 智能机器人消息的完整跟踪

跟踪一条来自 WeCom 智能机器人客户的文本消息「订单 ORD123 状态」从 WS 入站到 AI 回复落地的全部函数调用。

Phase-3 G 后只有 WeCom 智能机器人 WS 渠道，HTTP webhook 渠道（WeCom / Feishu）已移除。

---

## 阶段 0：客户端发出 WS 帧

WeCom 智能机器人服务端向 `WECOM_AIBOT_WS_URL` 推送 JSON 帧：

```json
{
  "cmd": "aibot_msg_callback",
  "header": {"req_id": "req-abc"},
  "body": {
    "msgid": "1234567",
    "from": {"userid": "ext-acceptance"},
    "chatid": "ext-acceptance",
    "msgtype": "text",
    "text": {"content": "订单 ORD123 状态"}
  }
}
```

---

## 阶段 1：WS Worker 帧解码

[backend/app/wecom_aibot_worker.py](../backend/app/wecom_aibot_worker.py) 作为独立进程运行。

[backend/app/wecom_aibot/client.py:WecomAibotClient.run](../backend/app/wecom_aibot/client.py)

- `websockets.connect(url)`，30s 心跳 ping，断线指数退避（1s→60s）重连
- 入站 frame 解析 → 提取 `chatid`, `text`, `msgid` → 构造：

```python
sys_msg = SystemMessage(
    channel="wecom_aibot",
    channel_user_id="ext-acceptance",
    text="订单 ORD123 状态",
    dedup_key="1234567",
    received_at=datetime.now(UTC),
)
```

---

## 阶段 2：Debouncer 合并窗口

[backend/app/wecom_aibot/debounce.py:Debouncer.observe](../backend/app/wecom_aibot/debounce.py)

HSET 累积 + asyncio task 定时器；500ms 后 `_flush_after_window` 触发 `dispatch`（即 `BusProducer.enqueue`）。

---

## 阶段 3：进入 Bus

[backend/app/bus/producer.py:BusProducer.enqueue](../backend/app/bus/producer.py)

`mmh3.hash("wecom_aibot:ext-acceptance") % 64` → `XADD bus:{shard}`

---

## 阶段 4：Bus Consumer 异步处理

[backend/app/bus/consumer.py:BusConsumer._run_shard](../backend/app/bus/consumer.py)

`XREADGROUP block_ms=5000` → 反序列化 `SystemMessage` → 调 handler。失败入 DLQ + ack；成功 ack。

---

## 阶段 5：Worker handler

handler 由 [backend/app/bus/worker.py:make_bus_handler](../backend/app/bus/worker.py) 构造的闭包。

### 5.1 解析 session_id

[backend/app/bus/worker.py:_resolve_session_id](../backend/app/bus/worker.py) — 30 分钟以内复用，否则 mint UUID + INSERT `agent.session`。

### 5.2 挂起检查

[backend/v/hooks/handoff.py:is_suspended](../backend/v/hooks/handoff.py)

```python
if handoff_enabled and await is_suspended(session_id, redis=redis):
    thread_ts = await slack_outbound.get_thread_for_session(session_id)
    if thread_ts:
        await slack_outbound.post_customer_message(thread_ts=thread_ts, text=msg.text)
    return  # ← 不调 graph
```

### 5.3 加载 user_profile

[backend/v/hooks/session.py:on_session_start](../backend/v/hooks/session.py) — Redis 缓存 → PG 兜底。

### 5.4 构造 state + config 并调 graph

```python
config = {
    "configurable": {
        "thread_id": session_id,
        "llm_caller": llm_caller,
        "pg_pool": pool,
        "embedder": embedder,
        "channel": msg.channel,
        "channel_user_id": msg.channel_user_id,
        "skill_registry": skill_registry,
    }
}
final_state = await graph.ainvoke(input_state, config=config)
```

### 5.5 Interrupt 检查

[backend/v/hooks/handoff.py:extract_interrupt](../backend/v/hooks/handoff.py)

```python
if interrupt_payload := await extract_interrupt(final_state):
    await on_interrupt(...)
    return
```

---

## 阶段 6：LangGraph 执行

[backend/v/agents/graph.py:build_graph](../backend/v/agents/graph.py)

```
enter → compress → intent → agent → route_after_agent → ┬→ tools → agent → ...
                                                        └→ reflect → exit → END
```

配 [RedisCheckpointer](../backend/v/agents/checkpointer.py)，热路径。

### 6.1 enter_node → intent_node → agent_node

intent_node（flash LLM）分类意图 → agent_node（primary LLM）生成回复或工具调用。

### 6.2 工具分支

`ToolNode([calculator, search, recall_memory, subagent, transfer_to_human])`：

- `search(query)` → `KnowledgeRetriever.retrieve(pool, embedder, query)` → pgvector cosine over `agent.knowledge_chunk`
- `recall_memory(query)` → pgvector cosine + 时间衰减 over `agent.event_memory`（per-customer）
- `calculator(expr)` → 安全 AST 求值
- `subagent(task)` → 单轮 LLM 子任务
- `transfer_to_human(reason)` → `interrupt()` → 图暂停

### 6.3 reflection_node

检查 AI 回复是否引用了工具结果以外的具体事实；重试上限 2 次。

---

## 阶段 7：取最后 AIMessage 并出站

[backend/app/bus/worker.py:_last_ai_message](../backend/app/bus/worker.py)

```python
await sends["wecom_aibot"]("ext-acceptance", "您好，已收到您的咨询...")
```

[backend/app/wecom_aibot/outbound.py:WecomAibotOutbound.send_text](../backend/app/wecom_aibot/outbound.py)

→ `redis.publish("wecom_aibot:outbound", {channel_user_id, text})`

wecom_aibot_worker 内 pub/sub task 收到 → `WecomAibotClient.send_text(...)` 真正写 WS 帧。

---

## 阶段 8：Bus consumer 完成 ack

`XACK bus:42 workers <msg_id>`

---

## 阶段 9（仅当阶段 5.5 命中）：人工接管

[backend/v/hooks/handoff.py:on_interrupt](../backend/v/hooks/handoff.py)

```
1. migrate_hot_to_cold → Redis hash → PG checkpointer
2. UPDATE agent.session SET status='suspended'
3. SET session_status:{session_id}=suspended
4. slack.post_handoff_alert → Block Kit 卡片 + Resume 按钮
```

挂起期间：客户消息 → Slack thread 转发；操作员回复 → append_operator_log + send 给客户。

操作员点 Resume → `on_resume` → migrate_cold_to_hot → `Command(resume=...)` → graph.ainvoke → sends["wecom_aibot"]。

详见 [data_flow.md §4](data_flow.md)。

---

## 总结：函数调用栈深度

最简单的一轮（无工具调用、无接管）大约 **15 个跨模块函数调用**。

| 阶段 | 耗时（粗估） |
| --- | --- |
| WS frame → debounce → bus | < 10ms（帧解码） + 500ms（debounce 等待） |
| Bus 出队 + 反序列化 | < 5ms |
| Session resolution + on_session_start | < 10ms（缓存命中 < 1ms） |
| LangGraph turn（含 ckpt 写） | LLM 调用占 99%（DeepSeek 通常 1-3s） |
| Outbound pub/sub + WS 写 | < 50ms |

工具调用增加：

| 工具 | 增量 |
| --- | --- |
| `search` | embed (~50-200ms) + 一次 pgvector (~10-50ms) |
| `recall_memory` | embed (~50-200ms) + pgvector + 时间衰减重排 (~10-50ms) |
| `calculator` | < 1ms（纯 CPU） |
| `subagent` | 一次额外 LLM 调用（DeepSeek flash，1-2s） |
| `transfer_to_human` | 工具内 interrupt，`on_interrupt` 串行：迁移 + Slack 告警（~200-500ms） |
