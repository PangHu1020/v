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

### 5.1 解析 session_id 并触发 session-end 固化

[backend/app/bus/worker.py:_resolve_session_id](../backend/app/bus/worker.py) — 30 分钟以内复用，否则 mint UUID + INSERT `agent.session`。

新 session mint 时（`minted=True`），`_promote_prev_session(prev_session_id)` 作为 fire-and-forget task 异步触发：将上一个 session 的 working memory 固化为 user_profile + event_memory（有 working memory 直接 promote；无则 extract_from_messages(history) 后 promote）。

### 5.2 加载 user_profile

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

---

## 阶段 6：LangGraph 执行

[backend/v/agents/graph.py:build_graph](../backend/v/agents/graph.py)

```
enter → compress → intent → agent → route_after_agent → ┬→ tools → agent → ...
                                                        └→ reflect → exit → END
```

配 [RedisCheckpointer](../backend/v/agents/checkpointer.py)，热路径。

### 6.1 enter_node → intent_node → agent_node

enter_node 按**冷热分层**注入 system prompt（提升 prefix 缓存命中率）：
- **COLD** SystemMessage：persona + channel + 冻结的 `<available_skills>` 目录（仅技能名 + 描述）。与客户无关，跨会话可缓存。
- **WARM** SystemMessage：user_profile + recent_events + session_memory。按客户区分，冻结到下次压缩。
- 对话历史 append 在后，append-only。

技能不再按关键词预注入正文——模型从目录自选，按需调用 `load_skill(name)` 工具取回正文（落到对话热区）。

intent_node（flash LLM）分类意图 → agent_node（primary LLM）生成回复或工具调用。

### 6.2 工具分支

`ToolNode([calculator, search, recall_memory, subagent, load_skill])`：

- `search(query)` → `KnowledgeRetriever.retrieve(pool, embedder, query)` → pgvector cosine over `agent.knowledge_chunk`
- `recall_memory(query)` → pgvector cosine + 时间衰减 over `agent.event_memory`（per-customer）
- `calculator(expr)` → 安全 AST 求值
- `subagent(task)` → 单轮 LLM 子任务
- `load_skill(name)` → 从 `<available_skills>` 目录取回该 SOP 正文（渐进式披露，正文 append 到对话）



compression_node 还负责中段记忆提取：一次 LLM 调用同时产出 conversation_state（结构化对话状态）+ working_memories → Redis + event_memories → PG；截断 head 后注入 `<compressed_history>` 渲染 conversation_state 实现无缝衔接；不更新 user_profile（会话仍活跃）。

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

