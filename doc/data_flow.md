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
┌─────────────────────────────────────┐
│ WeCom 智能机器人 WS endpoint         │
└────────────┬────────────────────────┘
             │ JSON frame (cmd=aibot_msg_callback)
             ▼
┌─────────────────────────────────────────────────┐
│ wecom_aibot_worker 进程                          │
│  WecomAibotClient:                              │
│   1. 解析帧 → 提取 text / chatid / msgid         │
│   2. 构造 SystemMessage                          │
└────────────┬────────────────────────────────────┘
             ▼
┌──────────────────────────────────┐
│ /backend/app/wecom_aibot/        │
│ debounce.py  500ms 静默窗口       │
└────────────┬─────────────────────┘
             ▼ (after 500ms idle)
┌──────────────────────────────────┐
│ /backend/app/bus/producer        │  XADD bus:{shard}
└────────────┬─────────────────────┘
             ▼
┌────────────────┐    ┌─────────────────────────────────────────┐
│ Redis Streams  │ ◀▶│ /backend/app/bus/consumer               │
│ (sharded)      │    │   每 shard 一个 asyncio task，严格串行   │
└────────────────┘    └────────┬────────────────────────────────┘
                               ▼
                      ┌──────────────────────────────────────┐
                      │ /backend/app/bus/worker              │
                      │  ① _resolve_session_id               │
                      │  ② is_suspended? → forward to Slack  │
                      │  ③ on_session_start                  │
                      │  ④ graph.ainvoke                     │
                                            │  ⑥ _last_ai_message + send           │
                      │  ⑦ _record_last_seen                 │
                      └────────┬─────────────────────────────┘
                               ▼
                      ┌────────────────────────────────────────────────┐
                      │ LangGraph                                      │
                      │ enter → compress → intent → agent → reflect   │
                      │       → [tools? → agent]*  → exit             │
                      └────────┬───────────────────────────────────────┘
                               │ AIMessage
                               ▼
                      ┌──────────────────────────────────┐
                      │ WecomAibotOutbound.send_text      │
                      │ redis.publish → wecom_aibot_worker│
                      │ → WecomAibotClient WS 写帧        │
                      └──────────────────────────────────┘
                               ▼
                            Customer
```

### 2.2 关键时序细节

**(1) WS frame → Bus**

```
T0      入站 WS frame
T0+2ms  帧解析 + SystemMessage 构造
T0+5ms  Debouncer.observe（HSET + 取消重排定时器）
T0+505ms 定时器到期 → BusProducer.enqueue → XADD
```

**(2) Debounce 窗口**

```
T0       observe(msg1)：HSET pending=msg1，启动定时器 500ms
T0+200ms observe(msg2)：HSET pending=msg1+msg2，cancel + 重启
T0+700ms 定时器到期：HGETALL → BusProducer.enqueue → HDEL
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
7. reply = _last_ai_message(final_state.messages)
8. sends[channel](channel_user_id, reply)
9. _record_last_seen(...)
```

### 2.3 出站为何不走 bus？

CLAUDE.md 明确规定：bus 仅 reactive-inbound。直接走 channel adapter 的好处：

- 入站流不被回执污染。
- 同一会话回执必然在同一 worker（per-shard 串行）。
- 失败语义直接：`send_text` 抛出 → handler 落 DLQ。

### 2.4 工具调用循环

`AGENT_TOOLS = [calculator, search, recall_memory, subagent, transfer_to_human]`

工具说明：

| 工具 | 行为 | 实现 |
| --- | --- | --- |
| `calculator(expr)` | 安全 AST 求值（白名单算子，exponent ≤ 64） | [tools/calculator.py](../backend/v/tools/calculator.py) |
| `search(query, top_k=5, source_type=None)` | 语义召回 `agent.knowledge_chunk` (pgvector cosine)；逻辑在 [rag/retriever.py](../backend/v/rag/retriever.py) | [tools/search.py](../backend/v/tools/search.py) |
| `recall_memory(query, top_k=5)` | 召回该客户的 `agent.event_memory` (pgvector cosine + 时间衰减重排) | [tools/recall_memory.py](../backend/v/tools/recall_memory.py) |
| `subagent(task, context_mode=...)` | 单轮 LLM 子任务调用；shared 模式带父 messages 末尾 10 条 | [tools/subagent.py](../backend/v/tools/subagent.py) |

### 2.5 出站通道

WS 连接只在 `wecom_aibot_worker` 进程里，FastAPI 主进程 / ARQ worker / Slack handoff 通过 Redis pub/sub 发布，worker 统一写 WS：

```
sends["wecom_aibot"](user_id, text)
  → WecomAibotOutbound.send_text()
  → redis.publish("wecom_aibot:outbound", {channel_user_id, text})
  ────────────────────────────────────────────────────────────
  wecom_aibot_worker 内的 pub/sub 订阅 task:
    on message → WecomAibotClient.send_text(WS frame)
```

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

- `user_profile` 每次 `on_session_start` 全量注入到 SystemMessage（已实现）。
- `event_memory` 通过 `recall_memory` 工具按需召回。
- 抽取链：`consolidate_session` → `extract_session_memory` 写 user_profile + event_memory。

新开 session、靠结构化记忆桥接是有意为之：30 分钟前的对话上下文 ① 大概率与当前问题无关，② 占 token 浪费成本，③ 可能引入误解。

## 6. Bus 串行/并发语义

- 同一 `(channel, channel_user_id)` 的所有消息必落到同一 shard。
- 每个 shard 由 BusConsumer 中的**一个** asyncio task 消费，串行处理。
- 不同 shard 的消息并发处理（最多 64 路并发）。
- 同一会话内 N 条消息，被 debouncer 合并前已经顺序串行；合并后只有 1 条进 bus。
- 跨会话 / 跨用户消息互相独立。

## 7. 记忆分层与异步提取（Phase-3 Group C/H 完整）

### 7.1 三层记忆架构

```
┌────────────────── Working Memory (热路径) ──────────────────┐
│ Redis  ckpt:{thread_id}         TTL=1800s                   │  ← LangGraph checkpointer
│ Redis  working:{session_id}     TTL=1800s                   │  ← 本会话待提取的记忆条目 (MemoryEntry[])
│ Redis  profile:{ch}:{u}         TTL=1800s                   │  ← user_profile 缓存
│ Redis  session:{ch}:{u}         TTL=3600s                   │  ← 当前 session_id
│ Redis  last_seen:{ch}:{u}       TTL=3600s                   │  ← 静默判断
│ Redis  proactive_log:{ch}:{u}   TTL=30d                     │  ← 主动消息历史
└──────────────────────────────────────────────────────────────┘
       ▼ on_interrupt: 整 thread 迁移 (transfer_to_human 触发)
┌──────────── Cold Checkpointer (挂起期间) ───────��────┐
│ PG  agent.checkpoints + checkpoint_blobs            │  ← LangGraph 官方 PostgresSaver
│     + checkpoint_writes + checkpoint_migrations     │     仅 suspended session 使用
└──────────────────────────────────────────────────────┘
       ▼ consolidate_session (ARQ 延时任务，context 阈值 / TTL 临近时触发)
┌─────────────── Event Memory (中期，30 天) ────────────��─┐
│ PG  agent.event_memory                                  │  ← 一行 = 一条可召回的记忆片段
│     content TEXT, embedding vector(1024), expires_at   │     Qwen text-embedding-v3, HNSW cosine
│     importance, tags[], created_at                      │     recall_memory 工具按 cosine * exp(-age/30d) 召回
└──────────────────────────────────────────────────────────┘
       ▼ promote_to_long_term (会话结束时一次性，读 working → 写 profile + event)
┌────────────── Long-term Memory (永久，去重合并) ─────────────┐
│ PG  agent.user_profile  PK (channel, channel_user_id)      │  ← 结构化偏好 (JSONB)
│     customer_name, preferred_language, member_level, ...   │     on_session_start 全量注入 SystemMessage
│     + extras {}, notes TEXT                                 │
│ PG  agent.knowledge_chunk  vector(1024)                     │  ← 共享知识语料库（产品 / FAQ / 政策）
│     (source_type, source_id) UNIQUE, HNSW cosine           │     search 工具语义召回，无 recency 权重
└─────────────────────────────────────────────────────────────┘
```

### 7.2 异步记忆提取流程详解

记忆提取分为**两阶段**，均由 ARQ 后台任务驱动，与主对话流程完全解耦：

#### 阶段一：consolidate_session（会话活跃期的增量提取）

**触发时机**（三选一，哪个先到触发哪个）：
1. **Context 阈值**：`on_context_threshold` hook 检测到 checkpointer 中 messages 数 ≥ 配置阈值（默认 30 条）。
2. **TTL 临近**：session last_seen 距离 TTL（1800s）还剩 Δ 时间时（如剩 300s），提前触发。
3. **会话结束**：`on_session_end` hook（目前未实现独立 hook，等同于 30 min 静默后下次新消息到达时检测）。

**实现路径**：
```
hooks/某处 → ctx["arq_pool"].enqueue(
    "consolidate_session",
    session_id=session_id,
    defer_by=timedelta(seconds=...),  # TTL 临近时有 defer，context 阈值时立即
)
```

**执行逻辑**（[backend/v/cron/tasks/consolidate_session.py](../backend/v/cron/tasks/consolidate_session.py)）：
```python
async def consolidate_session(ctx, *, session_id):
    # 1. 读取 Redis checkpointer 中的完整 messages 历史
    checkpointer = RedisCheckpointer(redis, session_id)
    messages = await checkpointer.aget_tuple(config)
    transcript = _format_history(messages)  # 转纯文本对话记录

    # 2. 调 LLM（DeepSeek v4 flash）+ Pydantic structured output
    result: ExtractionResult = await llm_caller.chat(
        "summary", prompt, structured=ExtractionResult
    )
    # ExtractionResult = {
    #   profile_updates: dict,        # 本阶段**忽略**（留给 promote_to_long_term）
    #   working_memories: [MemoryEntry],  # 短期工作记忆，写 Redis
    #   event_memories: [MemoryEntry],    # 中期事件记忆，写 PG + embed
    # }

    # 3. 双写：
    # 3a. working_memories → Redis list working:{session_id} (TTL 1800s)
    for entry in result.working_memories:
        await append_working_memory(redis, session_id, entry)

    # 3b. event_memories → agent.event_memory (带 embedding)
    #     每条 content 调 embedder.aembed_documents([content])
    #     INSERT ... expires_at = now() + 30 days
    events_inserted = await insert_event_memories(
        pool, channel, channel_user_id, session_id,
        result.event_memories, embedder
    )

    # 4. 标记会话 status='consolidated'（幂等）
    await conn.execute(
        "UPDATE agent.session SET status='consolidated' WHERE session_id=$1",
        session_id
    )
    return {"working_inserted": len(result.working_memories), "events_inserted": events_inserted}
```

**关键特性**：
- `working_memories` 是**临时便签**：供本会话后续轮次快速访问，TTL 与 session 同步过期。
- `event_memories` 是**持久记忆**：30 天 TTL，可被 `recall_memory` 工具跨会话召回，每条带 `embedding vector(1024)` + `importance` + `tags[]`。
- `profile_updates` 在这一阶段**被忽略**——会话活跃时不修改 user_profile，避免频繁写冲突和中间态污染。

#### 阶段二：promote_to_long_term（会话结束后的终局合并）

**触发时机**：
会话彻底结束（30 分钟静默后，Redis working memory 自然过期）时，由外部逻辑（目前可能是手动触发或更上层的 session closer）调用。

**实现路径**：
```
某处检测到 session 已 consolidated 且已过期 →
ctx["arq_pool"].enqueue("extract_session_memory", session_id=...)
# extract_session_memory 是 promote_to_long_term 的别名
```

**执行逻辑**（[backend/v/memory/memory_extractor.py](../backend/v/memory/memory_extractor.py)）：
```python
async def promote_to_long_term(ctx, *, session_id):
    # 1. 读取身份
    channel, channel_user_id = await _read_session_identity(pool, session_id)

    # 2. 读取 Redis working:{session_id} 中的所有 MemoryEntry
    working = await read_working_memory(redis, session_id)
    if not working:
        return None  # 无内容，幂等退出

    # 3. 读取现有 user_profile
    existing = await _read_existing_profile(pool, channel, channel_user_id)

    # 4. 调 LLM（DeepSeek v4 flash）做**合并决策**
    prompt = [
        SystemMessage(LONG_TERM_PROMOTION_SYSTEM_PROMPT),
        HumanMessage(f"<existing_profile>{json(existing)}</existing_profile>\n"
                     f"<working_memory>{json(working)}</working_memory>")
    ]
    result: ExtractionResult = await llm_caller.chat(
        "memory_extract", prompt, structured=ExtractionResult
    )
    # 这次的 ExtractionResult:
    #   profile_updates: dict,  ← **这次会用**，LLM 输出"合并后的完整 profile"
    #   event_memories: [MemoryEntry],  ← 额外的长期记忆片段（去重后的）

    # 5. 合并 profile（代码层浅合并 + LLM 输出深合并）
    merged = _merge_profile(existing, result.profile_updates)
    if merged != existing:
        await conn.execute(
            """INSERT INTO agent.user_profile (channel, channel_user_id, profile)
               VALUES ($1, $2, $3)
               ON CONFLICT (channel, channel_user_id) DO UPDATE
                 SET profile = EXCLUDED.profile, updated_at = now()""",
            channel, channel_user_id, merged
        )

    # 6. event_memories 再次 embed + INSERT（去重：UNIQUE (source_type, source_id) 或内容哈希）
    events_inserted = await insert_event_memories(...)

    # 7. 删除 Redis working:{session_id}（已提取完毕，避免重复处理）
    await delete_working_memory(redis, session_id)

    return {"profile_updated": 1 if merged != existing else 0, "events_inserted": events_inserted}
```

**关键特性**：
- **LLM 产出合并后的完整 profile**：不在代码里写 if-else 合并规则，而是让 LLM 理解语义冲突（如"偏好顺丰" vs "最近改用京东"），产出最新的正确状态。
- **去重 + 过时标记**：`_merge_profile` 逻辑 + LLM prompt 明确要求"conflict → recency wins; obsolete info → drop"。
- **event_memories 二次提取**：从 working memory 中进一步筛选**跨会话仍有价值的条目**（如投诉记录、特殊需求），写入 30 天 TTL 的 event_memory。
- **幂等性**：working memory 一旦删除，重跑 promote_to_long_term 立即返回 None。

### 7.3 召回路径

**on_session_start**（每轮开始前）：
```python
profile = await read_user_profile(pool, channel, channel_user_id)
# → 全量注入 <customer_profile> 到 SystemMessage

recent_events = await read_recent_event_memories(pool, channel, channel_user_id, limit=5)
# → <recent_events> 注入（最近 5 条，按 created_at DESC）

working_memory = await read_working_memory(redis, session_id)
# → <working_memory> 注入（本会话累积的临时记忆）
```

**recall_memory 工具**（主动召回）：
```python
@tool("recall_memory", parse_docstring=True)
async def recall_memory(query: str, config: RunnableConfig) -> str:
    pool, embedder = config["configurable"]["pg_pool"], config["configurable"]["embedder"]
    query_vec = await embedder.aembed_query(query)
    rows = await conn.fetch(
        """SELECT content, created_at, importance, tags,
                  embedding <=> $1::vector AS distance
           FROM agent.event_memory
           WHERE channel=$2 AND channel_user_id=$3 AND expires_at > now()
           ORDER BY distance LIMIT $4""",
        query_vec, channel, channel_user_id, top_k
    )
    # 时间衰减重排：score = (1 - distance) * exp(-age_days / 30)
    # 返回 top_k 条格式化文本
```

**search 工具**（知识库召回，与客户无关）：
```python
@tool("search", parse_docstring=True)
async def search(query: str, top_k: int = 5, source_type: str | None = None) -> str:
    query_vec = await embedder.aembed_query(query)
    if source_type:
        sql = "... WHERE source_type=$2 ORDER BY embedding <=> $1::vector LIMIT $3"
        rows = await conn.fetch(sql, query_vec, source_type, top_k)
    else:
        sql = "... ORDER BY embedding <=> $1::vector LIMIT $2"
        rows = await conn.fetch(sql, query_vec, top_k)
    # 返回 "[source_type:source_id] text" 格式，无 recency 权重
```

### 7.4 完整时序图

```
T0        客户首条消息 → mint session_id → on_session_start (profile + recent_events 注入)
T0+30s    agent 调工具、回复 → working memory 无感写入 Redis (暂不触发提取)
T0+5min   第 15 轮对话 → messages 数达阈值 30 → on_context_threshold hook
              → enqueue consolidate_session(session_id, defer_by=0)
T0+5min+2s ARQ worker 执行 consolidate_session:
              - LLM 总结前 15 轮 → working_memories (3 条) + event_memories (1 条)
              - Redis RPUSH working:{s_id} × 3
              - PG INSERT agent.event_memory × 1 (embed + expires_at=now()+30d)
              - UPDATE session status='consolidated'
T0+10min  继续对话，working memory 累积到 5 条（3 条来自上次提取 + 2 条新增）
T0+30min  客户静默，Redis ckpt + working 键自然过期（TTL 1800s）
T0+35min  外部 closer 检测到 session 已 consolidated 且过期
              → enqueue extract_session_memory(session_id)
T0+35min+3s ARQ worker 执行 promote_to_long_term:
              - 读 Redis working:{s_id}（5 条）+ PG existing profile
              - LLM 合并决策 → profile_updates (dw_customer_id, preferred_courier 更新)
                             → event_memories (1 条额外长期记忆)
              - UPSERT agent.user_profile
              - INSERT agent.event_memory × 1
              - DEL Redis working:{s_id}（清理完毕）
T0+60min  客户再次发消息 → mint 新 session_id
              → on_session_start 读到刚才更新的 profile + 前面写入的 2 条 event_memory
              → LangGraph 继承了上一会话的"记忆"，但 working memory 是全新的空列表
```

### 7.5 设计权衡

| 维度 | 决策 | 理由 |
| --- | --- | --- |
| **为何两阶段？** | consolidate 活跃期增量，promote 结束后终局 | 避免活跃期频繁写 profile 造成冲突；终局时 LLM 可以看到完整会话做最优合并 |
| **working memory 为何在 Redis？** | TTL 自动过期 + 与 session 同生命周期 | 无需手动清理；会话结束后这些临时记忆自然消失，不污染长期存储 |
| **event_memory 30 天 TTL** | 平衡召回价值与存储成本 | 多数客服场景，1 个月前的具体对话细节已无召回价值；profile 保留结构化要点 |
| **profile_updates 为何延迟到 promote？** | 减少并发写、避免中间态 | 会话中客户可能改口（"顺丰" → "算了还是京东"），等会话结束再由 LLM 产出最终态 |
| **LLM 负责合并而非代码？** | 语义冲突只有 LLM 能理解 | "客户说不吃辣" vs "今天点了麻辣锅" → LLM 判断是临时例外还是偏好变更 |
| **recall_memory 与 search 分离？** | 一个查客户历史，一个查共享知识 | recall 带 recency decay（旧事件权重低），search 不带（产品信息无新旧）|

