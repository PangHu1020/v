# 数据流

## 1. 三大工作流概览

| 流向 | 触发 | Phase-1 状态 | 文件锚点 |
| --- | --- | --- | --- |
| **被动答疑**（Reactive） | 客户在 IM 平台发消息 | ✅ 已实现 | 见 §2 |
| **主动触达**（Proactive） | ARQ 定时任务（物流通知 / 广告 / 复购提醒） | ❌ Phase-2 | 见 [gaps.md](gaps.md) |
| **人工接管**（Handoff） | Agent 调用 `transfer_to_human` 工具 | ❌ Phase-2 | 见 [gaps.md](gaps.md) |

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
│ /backend/app/gateway/middleware  │  ← RequestIdMiddleware（请求 id + 结构化日志上下文）
└────────┬────────────────────────┘
         ▼
┌──────────────────────────────────┐
│ /backend/app/channels/{platform} │
│  1. signature.verify_signature   │  ← 401 if mismatch
│  2. crypto.{Wecom|Feishu}Crypto  │  ← AES 解密
│  3. XML/JSON parse               │
│  4. 构造 SystemMessage           │
└────────┬─────────────────────────┘
         ▼
┌──────────────────────────────────┐
│ /backend/app/channels/debounce   │  Debouncer.observe()
│  500ms 静默窗口、合并 burst       │
└────────┬─────────────────────────┘
         │ (500ms 后)
         ▼
┌──────────────────────────────────┐
│ /backend/app/bus/producer        │  BusProducer.enqueue()
└────────┬─────────────────────────┘
         │  XADD bus:{shard}
         ▼
┌────────────────┐    ┌─────────────────────┐
│ Redis Streams  │ ◀▶│ /backend/app/bus/   │
│ (sharded by    │    │   consumer (sync_   │
│  user_id hash) │    │   per-shard task)   │
└────────────────┘    └────────┬────────────┘
                               │ XREADGROUP
                               ▼
                      ┌──────────────────────────────────┐
                      │ /backend/app/bus/worker          │
                      │   make_bus_handler 闭包          │
                      │  1. _resolve_session_id          │
                      │  2. on_session_start             │
                      │  3. graph.ainvoke                │
                      │  4. _last_ai_message + send      │
                      │  5. _record_last_seen            │
                      └────────┬─────────────────────────┘
                               ▼
                      ┌──────────────────────────────────┐
                      │ LangGraph: enter → agent → exit  │
                      │  enter: 注入 SystemMessage       │
                      │  agent: LLMCaller.chat           │
                      │  exit:  log metrics              │
                      └────────┬─────────────────────────┘
                               │ AIMessage
                               ▼
                      ┌──────────────────────────────────┐
                      │ channel-direct outbound          │
                      │  WecomOutbound.send_text         │
                      │  FeishuOutbound.send_text        │
                      │  （**不**经 bus）                 │
                      └────────┬─────────────────────────┘
                               ▼
                            Customer
```

### 2.2 关键时序细节

**(1) HTTP → Bus**（同步路径，要求 < 1 秒响应）

WeCom / Feishu 平台对 webhook 响应有 5 秒超时；超时即重投。所以路由层要快：

```
T0      入站 webhook
T0+5ms  RequestIdMiddleware 绑定上下文
T0+10ms verify_signature（HMAC，常量时间比较，~µs 级）
T0+15ms crypto.decrypt（AES，~ms 级）
T0+20ms XML / JSON parse + 构造 SystemMessage
T0+25ms Debouncer.observe（HSET + 取消重排定时器）
T0+30ms HTTP 200 "success" 返回
```

WeCom 路由返回 `"success"` 字面量；Feishu 路由返回 `{"ok": true}`。两者都不等待下游处理。

**(2) Debounce 窗口**

```
T0+25ms   observe(msg1)：HSET pending=msg1，启动定时器 500ms 后 flush
T0+200ms  observe(msg2)：HSET pending=msg1+msg2，cancel 旧定时器，重启 500ms
T0+450ms  observe(msg3)：HSET pending=msg1+msg2+msg3，再次 cancel + 重启
T0+950ms  定时器到期：HGETALL pending，组装 merged SystemMessage，调用 dispatch（即 BusProducer.enqueue），HDEL
```

注意：`flush_after_window` 是 asyncio task；如果在 500ms 内没有新消息，定时器自然到期，**不会** cancel。

**(3) Bus 消费 → 回复**

```
[每 shard 一个 asyncio task]
loop:
  XREADGROUP block_ms=5000 → batch (count=1)
  if batch is empty: sleep 10ms, retry  # fakeredis 不阻塞，real Redis 走 BLOCK
  for each msg_id, fields in batch:
    deserialize SystemMessage from fields[b"json"]
    bind_request(channel, channel_user_id)
    handler(msg)  # make_bus_handler 闭包
      ↓
    XACK msg_id
```

handler 内部（顺序串行）：

```
1. session_id, minted = _resolve_session_id(redis, channel, user_id, silence=1800)
   - GET last_seen:{ch}:{u}  → 时间戳
   - 若 < 1800s 前：复用 GET session:{ch}:{u} 的现有 session_id
   - 否则：mint 新 uuid，SETEX session:{ch}:{u}
2. profile = on_session_start(pool, redis, channel, user_id, ttl=1800)
   - GET profile:{ch}:{u} 缓存 → 命中直接返回
   - 否则 SELECT profile FROM agent.user_profile，写回缓存
3. graph.ainvoke({messages: [HumanMessage(text)], session_id, channel, user_id, user_profile, ...},
                 config={"configurable": {"thread_id": session_id, "llm_caller": ...}})
   - LangGraph 内部 ckpt.aget_tuple → 已有 thread state？
     - 有：merge new HumanMessage 进既有 messages
     - 无：从空开始，触发 enter_node 注入 SystemMessage
   - enter_node → agent_node → exit_node → END
   - 每个节点结束 ckpt.aput 写一次 checkpoint，刷新 TTL
4. reply = _last_ai_message(final_state.messages)
5. sends[channel](channel_user_id, reply)
   - WecomOutbound.send_text 或 FeishuOutbound.send_text
   - 内部：access_token cache（asyncio.Lock）→ POST /messages
6. SETEX last_seen:{ch}:{u} time.time()
```

### 2.3 出站为何不走 bus？

CLAUDE.md 明确规定：bus 仅用于 reactive-inbound。出站直接走 channel adapter 的好处：

- 入站流不会被回执污染。
- 回执无序就好（同一会话回执必然在同一 worker，单 worker 串行调用 send_text 即���）。
- 失败处理更直白：`send_text` 抛出后 worker 在 try/except 内决定降级（Phase-2 加重试 + DLQ）。

## 3. Session 生命周期

```
首条消息          30 分钟内活跃                30+ 分钟静默             新消息到达
   │                 │                           │                       │
   ▼                 ▼                           ▼                       ▼
mint session_id   reuse session_id          last_seen 过期         mint 新 session_id
SET session:..    GET session:.. 命中       session 键自然过期      （+ 长记忆注入兜底）
SET last_seen     SET last_seen           （TTL = 1800 * 2）       SET 新 session
                                                                    SET 新 last_seen
```

跨 session 的连续性靠**长期记忆（user_profile + memory_episodes）**保证：

- `user_profile` 每次 `on_session_start` 全量注入到 SystemMessage（已实现）。
- `memory_episodes` 通过 `recall_memory` 工具按需召回（**Phase-2**）。

为什么不直接复用旧 session？因为 30 分钟前的对话上下文：① 大概率与当前问题无关，② 占 token 浪费成本，③ 可能引入误解。新开 session、靠结构化记忆桥接是更干净的设计。

## 4. Bus 的串行/并发语义

- 同一 `(channel, channel_user_id)` 的所���消息，必落到同一 shard。
- 每个 shard 由 BusConsumer 中的**一个** asyncio task 消费，串行处理。
- 不同 shard 的消息并发处理（最多 64 路并发）。
- 同一会话内 N 条消息（被 debouncer 合并前的）顺序保证；debouncer 合并后只有 1 条进 bus。
- 跨会话 / 跨用户消息互相独立。

这意味着：

- 同一用户连发，bus 端不会 race condition。
- 不同用户高并发（< 64 并发用户）能完全并行；超过则按 shard 排队。
- 64 这个数字来自 `BUS_SHARD_COUNT`，按 max in-flight conversations 估算。

## 5. 记忆分层

```
┌────────────────── Working Memory ──────────────────┐
│ Redis  ckpt:{thread_id}  TTL=1800s                 │  ← LangGraph state，每 turn 自动续期
│ Redis  profile:{ch}:{u}  TTL=1800s                 │  ← user_profile 快照，免每 turn 查 PG
│ Redis  session:{ch}:{u}  TTL=3600s                 │  ← 当前 session_id
│ Redis  last_seen:{ch}:{u} TTL=3600s                │  ← 静默窗口判断
└────────────────────────────────────────────────────┘
                    ▼ on_session_end / 阈值触发（Phase-2）
┌─────────────────── Session Memory ──────────────────┐
│ PG  agent.session_memory                            │  ← 一次会话的 LLM 总结
└─────────────────────────────────────────────────────┘
                    ▼ memory_extractor（Phase-2）
┌────────────────── Long-term Memory ─────────────────┐
│ PG  agent.user_profile  (channel, user_id) PK       │  ← 结构化偏好/画像
│ PG  agent.memory_episodes  vector(1024) HNSW cos    │  ← 向量化情节
└─────────────────────────────────────────────────────┘
```

Phase-1 实现：所有 working memory 路径 + long-term memory 的**读取**（user_profile 注入）。
Phase-1 未实现：session memory 写入、long-term 写入、memory_episodes 召回工具、热冷迁移。
