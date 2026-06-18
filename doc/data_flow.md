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
1. session_id, minted, prev_session_id = _resolve_session_id(redis, ...)
   - 30 分钟以内：复用现有 session_id
   - 否则：mint 新 uuid；prev_session_id = 刚过期的 session id
   - minted=True → INSERT agent.session + fire-and-forget _promote_prev_session(prev_session_id)
     → 如 working memory 非空：直接 promote_to_long_term（无额外 LLM 调用）
     → 如 working memory 空（短对话）：extract_from_messages(history) → promote_to_long_term
3. profile = on_session_start(...) → user_profile 注入到 state
4. config = {
       "configurable": {
           "thread_id": session_id,
           "llm_caller": ...,
           "pg_pool": pool,            # recall_memory 工具用
           "embedder": embedder,
           "channel": ..., "channel_user_id": ...,
           "skill_registry": ...,      # enter_node 的 cold 层目录 + load_skill 工具用
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

`AGENT_TOOLS = [calculator, search, recall_memory, subagent]`

工具说明：

| 工具 | 行为 | 实现 |
| --- | --- | --- |
| `calculator(expr)` | 安全 AST 求值（白名单算子，exponent ≤ 64） | [tools/calculator.py](../backend/v/tools/calculator.py) |
| `search(query, top_k=5, source_type=None)` | 语义召回 `agent.knowledge_chunk`（Milvus 单次 hybrid 检索 dense+BM25 → rerank 精排）；逻辑在 [rag/retriever.py](../backend/v/rag/retriever.py) | [tools/search.py](../backend/v/tools/search.py) |
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

## 3. 主动触达（Phase-2 P2，未实现）

主动触达功能（物流通知、复购推送等）为规划中功能，当前未实现，已删除 cron/ARQ 相关代码。

### 3.1 记忆任务流（已实现，替代原 ARQ 方案）

记忆提取/固化/巩固原设计为 ARQ 后台任务，现已改为**可靠的 Redis Streams 任务流**，由
[backend/app/bus/memory_bus.py](../backend/app/bus/memory_bus.py) 实现。

**三条专用流**（单一全局，非分片）：

| 流 | 生产者 | 消费者 | 触发时机 |
|---|---|---|---|
| `memory:promote` | `enqueue_promote(redis, session_id)` | `MemoryConsumer` | 新 session mint（上一个 session 过期时） |
| `memory:consolidate` | `enqueue_consolidate(redis, channel, user_id)` | `MemoryConsumer` | 同上（客户回访时懒触发月度巩固） |
| `memory:extract` | `IdleWatcher` 自动入队 | `MemoryConsumer` | 会话空闲 5min（预提取到 working memory） |

**消费方式**：`MemoryConsumer` 使用 XREADGROUP consumer group `memory_workers`，count=1 严格串行，
失败写 DLQ（`memory:dlq`）后 XACK，进程重启后 PEL 中未 ACK 的任务自动重投。

**空闲检测**：`IdleWatcher` 每 60s 扫描 Redis sorted set `memory:idle_watch`（每轮 turn 后
`mark_turn_active` 以当前时间戳 ZADD），找出 5min 未活动的 session 入队 `memory:extract`，
预提取历史消息到 working memory，让 session-end promotion 得到更丰富的输入。

### 3.2 主动投递公共流程（规划中）

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

## 7. 记忆分层与异步提取（记忆 V2）

### 7.1 三层记忆架构

V2 把"扁平的 MemoryEntry 句子"拆成**语义/情节二分**的清晰边界：

```
┌────────────────── 工作记忆 Working (热路径, 会话级) ──────────────────┐
│ Redis  ckpt:{thread_id}         TTL=1800s   ← LangGraph checkpointer  │
│ Redis  working:{session_id}     TTL=1800s   ← 本会话待固化的记忆 + 摘要 │
│ Redis  profile:{ch}:{u}         TTL=1800s   ← user_profile 缓存        │
│ Redis  session/last_seen:{ch}:{u}           ← session_id / 静默判断    │
└────────────────────────────────────────────────────────────────────────┘
       ▼ promote_to_long_term（session mint 时 fire-and-forget，唯一持久写入点）
┌────────────── 情节记忆 Episodic (append-only 事件) ──────────────┐
│ PG  agent.event_memory   一行 = 一件"发生过的事"                  │
│     content, embedding vector(1024), subject, source, confidence │
│     tier(raw|summary), period, access_count, activation          │
│     recall_memory 工具召回（cosine·recency·importance；命中 +access）│
│     月度巩固：同 subject 成簇 → 一条 summary + 按 activation 删 raw │
└────────────────────────────────────────────────────────────────────┘
┌────────────── 用户记忆 User (双时态, 可覆盖) ──────────────┐
│ PG  agent.user_memory   一个 attr_key 一条 active 行         │
│     attr_key/attr_value/kind(preference|constraint|pattern) │
│     source/confidence, status(active|superseded)            │
│     valid_from/valid_to/superseded_by  ← 改写=三步 supersede │
│     无 embedding；投影进 user_profile JSONB 缓存，启动全量注入│
│ PG  agent.user_profile  ← 当前态物化缓存（每次写入后重建）   │
└──────────────────────────────────────────────────────────────┘
       （另：agent.knowledge_chunk 是共享知识库，search 工具用，与客户记忆无关）
```

**边界判据**：会变的当前状态（偏好顺丰、过敏、月初下单）→ 用户记忆，可覆盖、冲突只发生在这里；
发生过的带时间的事（投诉 SO123、问过尺码）→ 情节记忆，只追加、永不冲突。

### 7.2 写入流程（全异步，单一持久写入点）

所有持久写入集中在 **session-end**（`promote_to_long_term`，session mint 时 fire-and-forget），
中段压缩只写 Redis、不碰 PG：

**阶段一：compression_node 中段压缩（token 阈值触发，仅 Redis）**
```
compression_node：
  1. 一次 memory_extract LLM 调用 → ExtractionResult：
     conversation_state（结构化对话状态）+ working_memories + event_memories
  2. 两类句子**合并 fold 进 Redis working:{session_id}**（不写 PG）
  3. 截断 head，注入 <compressed_history>（渲染 conversation_state + 当前 working memory）
  无 PG 写入——持久化全部留给 session-end
```

**阶段二：session-end 固化（`promote_to_long_term`，唯一持久写入点）**
```
1. 读身份 + 读 Redis working memory（空则用 checkpoint transcript 作 fallback_messages）
2. 一次 memory_extract LLM 调用 → MemoryExtraction：
   user_candidates（属性，带 attr_key/attr_value/kind/source/confidence）
   + episodic_candidates（事件，带 subject）
3. policy_gate（确定性，无 LLM）：重要性下限 → PII 脱敏（utils/pii.py 掩码手机/身份证/地址）
4. 用户候选：逐条 upsert_user_memory（key 冲突走三步 supersede：
   旧行 status=superseded+valid_to+superseded_by，新行 active；
   仲裁 stated>inferred>confidence>recency）→ 重建 user_profile 缓存
5. 情节候选：embed 一次 → 同 subject cosine 去重 → append (tier='raw', expires_at NULL)
6. 删 Redis working memory（幂等：重跑即 no-op）
```

**结构化对话状态（无缝衔接）**：`conversation_state` 与持久记忆是两件事——它是本会话延续性快照，
压缩后接手的模型读它即可知道"在聊什么、做了什么、还有什么没解决"。和中段提取**合并为一次 LLM 调用**。

### 7.2.1 遗忘：月度巩固（`consolidate_user`，惰性触发）

情节记忆 append-only，靠**月度巩固**封顶增长。回访客户开启新会话时 fire-and-forget 触发：
```
对已封闭月份（period < 当前月）中 tier='raw' 且未巩固的行：
  按 (period, subject) 成簇（≥min_cluster_size 才处理）→
  LLM 压成一条 tier='summary' 行（新 embedding）→
  按 ACT-R 激活值 activation = w_imp·importance + w_acc·ln(1+access) − w_age·ln(1+age) 剪枝：
    低于阈值的 raw 删除；高激活的存活并 link 到 summary（不再被重复巩固，仍可召回）
```
"用进废退"：被 `recall_memory` 频繁命中的事件 `access_count` 高 → 激活高 → 即使 importance 低也存活。
稳态每客户 ~数百带向量行，全局有界，不靠 cron。

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

**recall_memory 工具**（主动召回，本轮不改排序逻辑）：
```python
@tool("recall_memory", parse_docstring=True)
async def recall_memory(query: str, config: RunnableConfig) -> str:
    pool, embedder = config["configurable"]["pg_pool"], config["configurable"]["embedder"]
    query_vec = await embedder.aembed_query(query)
    rows = await conn.fetch(
        """SELECT content, created_at, importance, keywords,
                  embedding <=> $1::vector AS distance
           FROM agent.event_memory
           WHERE channel=$2 AND channel_user_id=$3 AND (expires_at IS NULL OR expires_at > now())
           ORDER BY distance LIMIT $4""",
        query_vec, channel, channel_user_id, top_k
    )
    # 时间衰减 + importance 重排（cosine · exp(-age/half_life) · (0.5+0.5·importance)）
    # 命中后 touch：UPDATE ... SET last_accessed_at=now(), access_count=access_count+1
    #   ↑ access_count 喂给月度巩固的 ACT-R 激活值（用进废退）
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
T0+30s    agent 调工具、回复 → compression_node 检查 token 数
T0+5min   [IdleWatcher 检测到 5min 无新消息]
              → XADD memory:extract {session_id, channel, user_id}
              → MemoryConsumer 消费 → _extract_to_working:
                  读 checkpoint 消息 → memory_extract LLM → 写 working:{s_id}（预提取）
T0+Xmin   token 阈值触发 compression_node（如果会话足够长）:
              → LLM 抽取 conversation_state + working/event 句子 → fold 进 working:{s_id}
              → 截断 head 消息，注入 <compressed_history>
T0+30min  客户静默，session TTL 倒计时中
T0+30min  新消息到达（或新客户消息触发 session mint）:
              minted=True → 旧 session 过期
              → await enqueue_promote(redis, session_id=prev_session_id)
                  XADD memory:promote {session_id}
              → await enqueue_consolidate(redis, channel=..., user_id=...)
                  XADD memory:consolidate {channel, user_id}
T0+30min+Δ MemoryConsumer 消费 memory:promote:
              → _promote_prev_session → promote_to_long_term:
                  读 working:{s_id}（5 条预提取 + 压缩句子）
                  → 一次 memory_extract LLM → MemoryExtraction
                  → policy_gate（重要性下限 + PII 脱敏）
                  → user_memory 双时态 upsert（key 冲突 supersede）
                  → episodic insert（tier=raw, embed + cosine 去重）
                  → 重建 user_profile 缓存 → DEL working:{s_id}
              → XACK（幂等：working memory 已删，重跑无副作用）
T0+30min+Δ MemoryConsumer 消费 memory:consolidate（背景，不影响对话）:
              → _consolidate_user_memory → consolidate_user:
                  扫已封闭月份 raw 情节 → 按 subject 成簇 → LLM 月度摘要
                  → 按 ACT-R 激活值删低价值 raw
T0+60min  客户再次发消息 → mint 新 session_id
              → on_session_start 读到刚才更新的 user_profile + 新 event_memory
              → agent 继承了上一会话的"记忆"，working memory 全新空列表
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

