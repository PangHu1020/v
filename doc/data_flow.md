# 数据流

## 1. 三大工作流

| 流向 | 触发 | 状态 | 见 § |
| --- | --- | --- | --- |
| **被动答疑**（Reactive） | 客户在 IM 平台发消息 | ✅ | §2 |
| **主动触达**（Proactive） | ARQ 定时任务 / 事件触发 | ✅ | §3 |
| **人工接管**（Handoff） | Agent 调用 `transfer_to_human` 工具 | ✅ | §4 |

## 2. 被动答疑详细数据流

### 2.1 顶层 ASCII

```
   Customer
      │  (消息)
      ▼
┌─────────────────┐
│ WeCom / Feishu  │  ← 加密 + 签名（AES-CBC + SHA1/SHA256）
└────────┬────────┘
         │ HTTP POST webhook
         ▼
┌──────────────────────────────────┐
│ /backend/app/gateway/middleware  │  RequestIdMiddleware
└────────┬─────────────────────────┘
         ▼
┌──────────────────────────────────┐
│ /backend/app/channels/{platform} │
│  1. signature.verify_signature   │
│  2. crypto.{Wecom|Feishu}Crypto  │
│  3. XML/JSON parse               │
│  4. 构造 SystemMessage           │
└────────┬─────────────────────────┘
         ▼
┌──────────────────────────────────┐
│ /backend/app/channels/debounce   │  500ms 静默窗口、合并 burst
└────────┬─────────────────────────┘
         ▼ (after 500ms idle)
┌────────────────────────────��─────┐
│ /backend/app/bus/producer        │  XADD bus:{shard}
└────────┬─────────────────────────┘
         ▼
┌────────────────┐    ┌─────────────────────────────────────────┐
│ Redis Streams  │ ◀▶│ /backend/app/bus/consumer               │
│ (sharded)      │    │   每 shard 一个 asyncio task，严格串行   │
└────────────────┘    └────────┬────────────────────────────────┘
                               ▼
                      ┌──────────────────────────────────────┐
                      │ /backend/app/bus/worker              │
                      │   make_bus_handler 闭包              │
                      │  ① _resolve_session_id               │
                      │  ② is_suspended? → forward to Slack  │
                      │  ③ on_session_start                  │
                      │  ④ graph.ainvoke                     │
                      │  ⑤ extract_interrupt? → on_interrupt │
                      │  ⑥ _last_ai_message + send           │
                      │  ⑦ _record_last_seen                 │
                      └────────┬─────────────────────────────┘
                               ▼
                      ┌──────────────────────────────────────────┐
                      │ LangGraph                                │
                      │ enter → agent → [tools? → agent]* → exit │
                      └────────┬─────────────────────────────────┘
                               │ AIMessage
                               ▼
                      ┌──────────────────────────────────┐
                      │ channel-direct outbound          │  （**不**经 bus）
                      └────────┬─────────────────────────┘
                               ▼
                            Customer
```

### 2.2 关键时序细节

**(1) HTTP → Bus**（要求 < 1 秒响应）

WeCom / Feishu 平台对 webhook 响应有 5 秒超时；超时即重投。

```
T0      入站 webhook
T0+5ms  RequestIdMiddleware 绑定上下文
T0+10ms verify_signature（HMAC，常量时间比较）
T0+15ms crypto.decrypt（AES）
T0+20ms XML / JSON parse + 构造 SystemMessage
T0+25ms Debouncer.observe（HSET + 取消重排定时器）
T0+30ms HTTP 200 "success" 返回
```

**(2) Debounce 窗口**

```
T0+25ms   observe(msg1)：HSET pending=msg1，启动定时器 500ms 后 flush
T0+200ms  observe(msg2)：HSET pending=msg1+msg2，cancel 旧定时器，重启 500ms
T0+450ms  observe(msg3)：HSET pending=msg1+msg2+msg3，再次 cancel + 重启
T0+950ms  定时器到期：HGETALL pending → BusProducer.enqueue → HDEL
```

**(3) Bus 消费 → 回复**

```
[每 shard 一个 asyncio task]
loop:
  XREADGROUP block_ms=5000 → batch (count=1)
  if batch is empty: sleep 10ms, retry  # fakeredis 不阻塞，real Redis 走 BLOCK
  for each msg_id, fields in batch:
    deserialize SystemMessage from fields[b"json"]
    bind_request(channel, channel_user_id)
    handler(msg)
    XACK msg_id  # 不论 handler 成功还是抛异常都 ack；handler 失败入 DLQ
```

handler 内部按顺序：

```
1. session_id, minted = _resolve_session_id(redis, channel, user_id, silence=1800)
   - 30 分钟以内：复用现有 session_id
   - 否则：mint 新 uuid（长记忆兜底连续性）
   - minted=True → INSERT agent.session 行（ON CONFLICT DO NOTHING）
2. **挂起检查**（Phase-2 P0）：
   if is_suspended(session_id):
       slack.post_customer_message(thread_ts, msg.text)
       return  # 不 invoke graph
3. profile = on_session_start(...) → user_profile 注入到 state
4. config = {
       "configurable": {
           "thread_id": session_id,
           "llm_caller": ...,
           "pg_pool": pool,            # recall_memory 工具用
           "embedder": embedder,
           "channel": ..., "channel_user_id": ...,
           "skill_registry": ...,      # enter_node 用
           "skill_top_k": 3,
       }
   }
5. final_state = await graph.ainvoke(input_state, config=config)
6. **interrupt 检查**（Phase-2 P0）：
   payload = extract_interrupt(final_state)
   if payload:
       on_interrupt(...)  # 迁移 hot→cold + Slack 告警 + 标记 suspended
       return
7. reply = _last_ai_message(final_state.messages)
8. sends[channel](channel_user_id, reply)
9. _record_last_seen(...)
```

### 2.3 出站为何不走 bus？

CLAUDE.md 明确规定：bus 仅 reactive-inbound。直接走 channel adapter 的好处：

- 入站流不被回执污染。
- 同一会话回执必然在同一 worker（per-shard 串行），无需额外排队。
- 失败语义直接：`send_text` 抛出 → handler 落 DLQ。

### 2.4 工具调用循环（Phase-2 P0+P1+P3 后）

`agent_node` 给 LLM 绑定的工具列表 = `[transfer_to_human, recall_memory, subagent, *mcp_tools]`，绑定方式见 `LLMCaller.chat(role, msgs, tools=...)`。LLM 输出 `AIMessage(tool_calls=[...])` 时，`route_after_agent` 路由到 `ToolNode`，`ToolNode` 用同一份工具列表分发执行，工具结果作为 `ToolMessage` 写回 state，再回到 `agent`。直到 LLM 不再产出 tool_calls 才退出 → `exit`。

工具说明：

| 工具 | 行为 |
| --- | --- |
| `transfer_to_human(reason)` | 调用 `langgraph.types.interrupt({"type": "transfer_to_human", "reason": ...})`，**suspend 整个图** |
| `recall_memory(query, top_k=5)` | 走 pgvector cosine + 时间衰减重排（half_life=30d）召回 episodes |
| `subagent(task, context_mode=...)` | 单轮 LLM 子任务调用；shared 模式带父 messages 末尾 10 条 |
| `{server_id}__{tool}` | MCP 工具，结果走 L1+L2 缓存（写类按配置 opt-out） |

## 3. 主动触达（Phase-2 P2）

ARQ worker 是独立进程（`make cron`）。

### 3.1 任务类型

| 任务 | 触发方式 | 内容 |
| --- | --- | --- |
| `notify_logistics_delivered(channel, user_id, order_id, tracking_number, courier?)` | 上游业务系统 enqueue | 模板化「订单已签收 + 运单号」 |
| `push_ad(channel, user_id, text)` | 营销系统 enqueue（已准备好的文案） | 直接转发 |
| `_scheduled_repurchase_run` | cron 每天 09:30 | 扫 `dw.fact_order` ⨝ `agent.user_profile` 找 7-30 天前买消耗品的客户 |
| `consolidate_session(session_id)` | 由 hooks 提交（context 阈值 / TTL 临近 / on_session_end） | LLM 总结一次会话 → 写 `agent.session_memory` |
| `extract_session_memory(session_id)` | `consolidate_session` 完成后链式提交 | 总结 → 结构化输出 → upsert `agent.user_profile` + 向量化插入 `agent.memory_episodes` |

### 3.2 主动投递公共流程

任意主动消息（物流通知 / 广告 / 复购）都走 `deliver_proactive`：

```
deliver_proactive(channel, user_id, text, purpose):
  1. RPUSH proactive_log:{channel}:{user}  → 待 on_session_start drainer 接（hookup 后续补）
  2. EXPIRE 30 days
  3. sends[channel](user_id, text)
```

### 3.3 复购任务的两阶段

```
build_repurchase_targets(pool):
  SELECT up.channel, up.channel_user_id, p.product_name, d.year/month/day
  FROM dw.fact_order o
  JOIN dw.dim_product p   ON p.product_id = o.product_id
  JOIN dw.dim_date d      ON d.date_id = o.date_id
  JOIN agent.user_profile up
       ON up.profile->>'dw_customer_id' = o.customer_id   ← 关键 join
  WHERE p.category = ANY('食品饮料','休闲零食')
    AND make_date(...) BETWEEN now()-30d AND now()-7d
  LIMIT 200

→ List[{channel, channel_user_id, product_name, order_date}]

send_repurchase_reminders(ctx, targets):
  for target in targets:
      deliver_proactive(...)
```

`profile->>'dw_customer_id'` 是 agent 体系和数仓体系的桥梁（Phase-3 P3 在 `memory_extractor` 中由 LLM 维护）。

## 4. 人工接管（Phase-2 P0）

### 4.1 触发链路

```
客户复杂问题 → agent_node 决定调 transfer_to_human(reason) →
  ToolNode 执行 → tool 内部 langgraph.types.interrupt(...) → 整图暂停 →
    final_state["__interrupt__"] 携带 payload 返回到 worker handler →
      extract_interrupt(final_state) 不为 None →
        on_interrupt 接管：
          1. migrate_hot_to_cold(session_id)：Redis hash → PG checkpointer，删 Redis 键
          2. UPDATE agent.session SET status='suspended'
          3. SET session_status:{session_id}=suspended（Redis 缓存，is_suspended() 用）
          4. slack.post_handoff_alert(...)：Block Kit 卡片 + Resume 按钮 → 拿到 thread_ts
          5. SET slack_thread:{thread_ts} ↔ session_thread:{session_id}（双向映射）
```

### 4.2 挂起期间消息走向

```
客户继续发消息 → debounce → bus consumer → handler.is_suspended() == True →
  slack.get_thread_for_session(session_id) → thread_ts →
  slack.post_customer_message(thread_ts, text)
  ─ LLM 不调 ─

操作员在 Slack thread 里回复 → Slack 推送 message event 到 /operator/slack/events →
  router 验签 → 查 slack_thread:{thread_ts} 拿到 session_id →
  on_operator_message callback：
    1. append_operator_log(session_id, text)（Redis list，7 天 TTL）
    2. sends[channel](channel_user_id, text)：直接转发给客户
```

### 4.3 恢复链路

```
操作员点 "Resume AI" → /operator/slack/interactivity → block_actions →
  action_id="resume_session", value=session_id →
  on_resume callback：
    1. migrate_cold_to_hot(session_id)：PG → Redis
    2. UPDATE agent.session SET status='active'
    3. SET session_status:{session_id}=active
    4. LRANGE operator_log:{session_id} → drained = [...]
    5. graph.ainvoke(Command(resume={"type":"resume","operator_messages":drained}), config=...)
       ─ interrupt() 内部返回 decision → tool 返回格式化字符串 → agent 看到 ToolMessage →
       agent 产出 final AIMessage
    6. 取 final AIMessage 的 content → sends[channel](user_id, text)
    7. 删 operator_log key
```

## 5. Session 生命周期

```
首条消息          30 分钟内活跃         30+ 分钟静默        新消息到达
   │                 │                       │                    │
   ▼                 ▼                       ▼                    ▼
mint session_id  reuse session_id     last_seen 过期        mint 新 session_id
INSERT agent.    GET 命中             session 键自然过期    （+ 长记忆注入兜底连续性）
  session 行     SET last_seen        TTL = 1800 * 2       SET 新 session
```

跨 session 的连续性靠**长期记忆**保证：

- `user_profile` 每次 `on_session_start` 全量注入��� SystemMessage（已实现）。
- `memory_episodes` 通过 `recall_memory` 工具按需召回（Phase-2 P3 实现）。
- 抽取链：`consolidate_session` 写 session_memory → `extract_session_memory` 写 user_profile + memory_episodes（Phase-2 P3 实现）。

新开 session、靠结构化记忆桥接是有意为之：30 分钟前的对话上下文 ① 大概率与当前问题无关，② 占 token 浪费成本，③ 可能引入误解。

## 6. Bus 串行/并发语义

- 同一 `(channel, channel_user_id)` 的所有消息必落到同一 shard。
- 每个 shard 由 BusConsumer 中的**一个** asyncio task 消费，串行处理。
- 不同 shard 的消息并发处理（最多 64 路并发）。
- 同一会话内 N 条消息，被 debouncer 合并前已经顺序串行；合并后只有 1 条进 bus。
- 跨会话 / 跨用户消息互相独立。

## 7. 记忆分层（Phase-2 P3 后完整）

```
┌────────────────── Working Memory ──────────────────┐
│ Redis  ckpt:{thread_id}  TTL=1800s                 │  ← LangGraph state（热路径）
│ Redis  profile:{ch}:{u}  TTL=1800s                 │  ← user_profile 缓存
│ Redis  session:{ch}:{u}  TTL=3600s                 │  ← 当前 session_id
│ Redis  last_seen:{ch}:{u} TTL=3600s                │  ← 静默判���
│ Redis  session_status:{session_id}                 │  ← active / suspended（P2 P0）
│ Redis  proactive_log:{ch}:{u} TTL=30d              │  ← 主动消息历史（P2 P2）
│ Redis  operator_log:{session_id} TTL=7d            │  ← 人工回复缓存（P2 P0）
└────────────────────────────────────────────────────┘
       ▼ on_interrupt: 整 thread 迁移
┌──────────── Cold Working Memory ────────────┐
│ PG  agent.checkpoints + checkpoint_blobs    │  ← 由 langgraph 官方 saver 管理
│     + checkpoint_writes + checkpoint_       │     仅 suspended 期间使用
│       migrations                            │
└─────────────────────────────────────────────┘
       ▼ consolidate_session（hook 触发 ARQ）
┌─────────────── Session Memory ──────────────┐
│ PG  agent.session_memory                    │  ← LLM 总结 + 结构化 metadata
│     summary, token_count, metadata JSONB    │     (intents/key_facts/sentiment/unresolved)
└─────────────────────────────────────────────┘
       ▼ extract_session_memory（链式 ARQ）
┌────────────── Long-term Memory ─────────────┐
│ PG  agent.user_profile  PK (ch, user_id)    │  ← 结构化偏好（LLM 合并）
│ PG  agent.memory_episodes  vector(1024)     │  ← Qwen embedding，HNSW cosine
│     + importance + tags + last_accessed_at  │     recall_memory 时间衰减召回
└─────────────────────────────────────────────┘
```

抽取链特性：

- `consolidate_session` 用 LLM（DeepSeek flash）+ Pydantic structured output 产出 `SessionSummary{narrative, intents, key_facts, sentiment, unresolved}`。
- `extract_session_memory` 接力，再调 LLM 输入 (existing_profile, session_summary) → output (merged_profile, episodes[])。**LLM 直接产出"合并后的完整 profile"**，避免在代码里写合并规则。
- episodes 用 Qwen `text-embedding-v4` 向量化（1024-dim Matryoshka，与 `pgvector(1024)` 列对齐）。
- 召回时按 `cosine_similarity * exp(-age_days / 30)` 重排，30 天半衰期。

## 8. MCP 工具调用流（Phase-2 P1）

```
启动时（main.py lifespan）：
  parse_servers(MCP_SERVERS_JSON) → [MCPServerConfig, ...]
  registry = MCPRegistry(configs, cache, ...)
  await registry.connect_all():
    for cfg in configs:
        client = MCPClient(cfg)
        client.connect():
            stack = AsyncExitStack()
            transport = stdio_client(...) | streamablehttp_client(...) | sse_client(...)
            session = ClientSession(transport)
            await session.initialize()
        listing = client.list_tools()
        for tool in listing.tools:
            registry._tools.append(_build_langchain_tool(client, tool))
              ← StructuredTool with args_schema 由 inputSchema 动态生成

build_graph(checkpointer, extra_tools=registry.tools)：
  AGENT_TOOLS + extra_tools 一起绑给 LLM 也一起喂给 ToolNode
```

工具调用：

```
agent_node → LLM 决定调 "products__get_status(order_id='ORD123')" →
ToolNode 路由到对应 StructuredTool →
  内部闭包：
    1. 非写类工具：cache.get(server_id, tool_name, args)；命中 → 直接返回
    2. client.call_tool(name, args)  # JSON-RPC over transport
    3. 非写类：cache.put(...)
    4. _extract_text(result) → 字符串返回给 ToolNode
ToolNode 包装为 ToolMessage → state.messages 追加 → agent 继续看到工具结果
```

写类工具（按 `MCPServerConfig.write_tools` 配置）跳过缓存。
