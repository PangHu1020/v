# RAG 检索评测报告

## 语料库

| 项目 | 数值 |
|------|------|
| 文档总数 | 150（120 产品 + 30 FAQ） |
| 产品品类 | 手机数码 / 家用电器 / 鞋靴 / 服饰 / 食品饮料 / 休闲零食（各 20 条） |
| 每条产品字段 | 名称、品牌、品类、售价、卖点描述、3-5 个关键规格 |
| FAQ 主题 | 退货退款 / 换货 / 物流 / 运费 / 会员 / 积分 / 保修 / 发票 / 支付 / 优惠券（各 3 条） |
| 存储 | Milvus 2.5.14，collection `knowledge_chunks`，dense HNSW + BM25 稀疏索引 |
| Embedding | Qwen `text-embedding-v4`，1024 dim，DashScope compatible-mode |

## 评测数据集

| 项目 | 数值 |
|------|------|
| 总量 | 337 条（平衡后） |
| 难度分布 | easy 67 / medium 203 / hard 67（约 20:60:20，正态） |
| 难度定义 | LLM 语义打分：1=含精确关键词，2=需近义推理，3=噪音/多约束叠加 |
| 查询风格（tier） | plain / colloquial / synonym / noisy / multi_condition |
| Gold label | 每条 query 锚定一个文档（`gold_source_ids` = 单条 `source_id`） |
| 生成方式 | LLM 按文档逐条生成（DeepSeek v4-flash），参考答案同步生成 |

## 检索架构：单次 hybrid + rerank（替代旧的三级瀑布流）

> **架构变更**：旧版是三级瀑布流（dense → hybrid → LLM 重写 + 相对 margin 门控）。
> 但本系统是 **agentic RAG**——`search(query)` 的 query 是 agent LLM 从对话上下文写的，
> 旧 stage-3 用检索器的 LLM 再重写一遍 = LLM 改 LLM 的冗余，也是 25-49s 延迟来源。
> 已**彻底废掉瀑布流**（连同 `gate.py` / `calibrate.py` / `run_ablation.py`），改为单一形态：

```
query (agent LLM 写好的)
  |
  v [可选] 标量过滤: search(category=…, price_min=…, price_max=…)
  |   命中 metadata["category"] / price 区间，先把目录缩到子集再排序（不传则不过滤）
  v 单次 hybrid search: dense cosine + BM25, WeightedRanker(0.5/0.5)
  |   取候选池 top-N（RAG_RERANK_CANDIDATES，默认 20）
  v rerank: cross-encoder 重排 (query, passage) 对，截 top_k
      LocalReranker（自托管 sidecar）/ RemoteReranker（云 API）
      reranker 挂掉 -> 降级为 hybrid 原序（不崩）
```

> **标量过滤**：客户说"一千多的手机"时，agent 在 `search` 填 `category="手机数码", price_max=1300`，
> 经 Milvus 布尔 expr 命中 product `metadata` 的 category/price 把目录缩到匹配子集再排序。
> 对话评测里它把 7 个"检索到但 gold 掉出 top-k"的 miss（gold 按品类+价格定义，纯语义 rerank
> 会正确地把跨子类项压下去）从 0-1/4 救到 4/4。客户诉求模糊时留空，避免过窄过滤误杀。

无门控、无 stage、无 LLM 重写。检索质量从"靠门控决定升级 + LLM 救场"
转为"宽候选池 + cross-encoder 精排"。

rerank 后端按 **URL 自动推断**（不再有 `RAG_RERANK_MODE` 开关）：
- `RAG_RERANK_URL` 指向 `localhost`/`127.0.0.1` → `LocalReranker`（自托管 sidecar，bge-reranker-v2-m3，`scripts/start_reranker.py` 起在 :8767），零 token。
- 其他 host → `RemoteReranker`（云 rerank API，带 `RAG_RERANK_API_KEY`）。

## 参数说明

### Milvus 配置（`MILVUS_*`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MILVUS_URI` | `http://localhost:19530` | Milvus 服务地址，Standalone 或 Cloud |
| `MILVUS_TOKEN` | `""` | Milvus Cloud API key，本地留空 |
| `MILVUS_COLLECTION_NAME` | `knowledge_chunks` | 知识库 collection 名 |

### RAG 检索参数（`RAG_*`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `RAG_MIN_K` | `3` | 至少返回的结果数 |
| `RAG_DENSE_WEIGHT` | `0.5` | Hybrid 中 dense 分支权重 |
| `RAG_BM25_WEIGHT` | `0.5` | Hybrid 中 BM25 稀疏分支权重 |
| `RAG_RERANK_ENABLED` | `true` | 是否对 hybrid 候选做 rerank 精排 |
| `RAG_RERANK_URL` | `http://localhost:8767/rerank` | reranker endpoint（localhost→sidecar，其他→云 API，自动推断） |
| `RAG_RERANK_MODEL` | `bge-reranker-v2-m3` | 模型名（仅 remote 用） |
| `RAG_RERANK_API_KEY` | `""` | remote 模式 bearer key（secret，走 .env） |
| `RAG_RERANK_CANDIDATES` | `20` | 送进 reranker 的 hybrid 候选数 |
| `RAG_RERANK_API_KEY` | `""` | remote 模式 bearer key（secret，走 env） |

### Collection Schema

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INT64, auto | 主键 |
| `source_type` | VARCHAR(64) | `"product"` / `"faq"` |
| `source_id` | VARCHAR(128) | `P001`、`FAQ003` 等 |
| `text` | VARCHAR(4096) | 嵌入文本，同时作为 BM25 分析字段 |
| `embedding` | FLOAT_VECTOR(1024) | Qwen text-embedding-v4 |
| `sparse_embedding` | SPARSE_FLOAT_VECTOR | BM25 Function 自动生成 |
| `metadata` | JSON | 结构化属性（品牌、价格、品类等） |

索引：HNSW（M=16，efConstruction=64，metric=COSINE）+ SPARSE_INVERTED_INDEX（BM25）

## 评测模块结构

```
backend/eval/
  gen/           数据生成（catalog / QA / difficulty scorer）
  retrieval/     检索质量评测（hybrid + rerank）
  generate/      生成质量评测（RAGAS 四指标）
  system/        系统性能评测（E2E 延迟 + token 成本）
  data/          共享数据目录（products/faq/qa/reports）
  common.py      共享数据类 + IO 工具
  seed_milvus.py Milvus 数据导入
```

## 数据准备

| 脚本 | 功能 |
|------|------|
| `gen/expand_catalog.py` | LLM 扩写商品目录 → `data/products.jsonl` + `data/faq.jsonl` |
| `gen/gen_qa.py` | 生成 tiered QA → `data/qa.jsonl` |
| `gen/gen_medium_qa.py` | 补充 medium 难度 QA，平衡分布 |
| `gen/score_difficulty.py` | LLM 语义难度打分（easy/medium/hard） |
| `seed_milvus.py` | Embed 语料 → Milvus `knowledge_chunks` |

```bash
LLM_TIMEOUT_SECONDS=180 uv run python -m backend.eval.gen.expand_catalog
LLM_TIMEOUT_SECONDS=180 uv run python -m backend.eval.gen.gen_qa
LLM_TIMEOUT_SECONDS=180 uv run python -m backend.eval.gen.gen_medium_qa --count 80
LLM_TIMEOUT_SECONDS=180 uv run python -m backend.eval.gen.score_difficulty
uv run python -m backend.eval.seed_milvus
```

## 检索评测（`retrieval/`）

| 脚本 | 功能 |
|------|------|
| `retrieval/run_eval.py` | hybrid+rerank 检索 recall/MRR/nDCG/hit，按 difficulty/tier 分维度 |

```bash
uv run python -m backend.eval.retrieval.run_eval
```

## 生成质量评测（`generate/`）

RAGAS 四指标：**Faithfulness** / **ContextRecall** / **AnswerRelevancy** / **AnswerCorrectness**

每条 QA 用真实 Milvus retriever 召回上下文，LLM 生成回答，与 `qa.jsonl` 参考答案对比。

```bash
LLM_TIMEOUT_SECONDS=180 uv run python -m backend.eval.generate.run_generate_eval
uv run python -m backend.eval.generate.run_generate_eval --limit 50         # smoke
uv run python -m backend.eval.generate.run_generate_eval --difficulty hard   # 仅困难
```

## 系统性能评测（`system/`）

E2E 延迟（mean/p50/p95/p99）+ token 消耗 + 成本估算（¥/轮）。
运行完整 `graph.ainvoke`（enter→intent→agent→reflect）per query，LangChain callback 统计 token。
成本单价在 `system/run_system_eval.py::PRICE_PER_1M` 中配置（默认 deepseek-v4-flash ¥0.5/¥1.5 per 1M）。

```bash
uv run python -m backend.eval.system.run_system_eval
uv run python -m backend.eval.system.run_system_eval --limit 50
```

## 对话式商品检索评测（`conversational/`）

单轮 QA（上面那套）有个根本局限：每条 query 自带完整锚点，检索想错都难（hit@5≈0.98 是数据太干净，不是检索强）。真实客户是**多轮、口语化、信息挤牙膏、夹带情绪/跑题**的，而且"三千左右拍照好的手机"这种宽泛诉求有**多个正确商品**。

这套评测端到端测**对话驱动的商品检索**：

- **LLM user simulator** 扮客户（`user_sim.py`），按 persona（预算紧/急性子/不懂行/带怨气/送礼/惜字如金/话痨/反复改主意）碎片化吐露需求 + 夹噪声。
- **真跑 graph**：每轮 `graph.ainvoke`，agent 自己从多轮上下文构造 query → 检索。同时考"能力A（上下文 query 构造）+ 能力B（检索匹配）"。
- **any-of 多 gold**：gold 集由**结构化约束可判定枚举**（category + price-band，代码扫 `products.jsonl`，完备不漏标）。对话过程中 agent 任一轮 `search` 检索到 gold 集任一商品即 pass。
- **判定**：从对话轨迹所有 ToolMessage 正则提取 `[product:Pxxx]` 标记 → 与 gold 求交。

| 项目 | 数值 |
|------|------|
| 数据集 | `data/conv_cases.jsonl`，约束枚举生成（`gen_conv_cases.py`） |
| case 数 | 30（6 品类 × 5 价格带，每带 1 persona） |
| gold 集大小 | 4-6（any-of，约束完备枚举） |
| persona | 8 种（budget_tight / in_a_hurry / clueless / frustrated / gifting / terse / rambler / wishy_washy） |
| 对话轮数 | ≤6 轮/case（user_sim 觉得满足或达上限即止） |

```bash
# 生成数据集（离线，只读 products.jsonl + LLM 造句，~30 次调用）
uv run python -m backend.eval.conversational.gen_conv_cases --per-constraint 1

# 起 reranker sidecar（hit 依赖精排；conda 环境见 scripts/start_reranker.py 顶注）
python scripts/start_reranker.py                       # bge-reranker-v2-m3 @ :8767

# 跑评测（需 Milvus + reranker 在线；真跑 graph，每 case 多轮 LLM）
#   RAGAS judge 用独立端点（与被测 agent 解耦），配 EVAL_JUDGE_* in .env；--no-ragas 可跳过
uv run python -m backend.eval.conversational.run_conv_eval --out report.json
uv run python -m backend.eval.conversational.run_conv_eval --limit 2   # smoke
```

报告（一次 graph 运行同时产出全部反馈指标，按 persona/category 分桶）：

| 指标 | 含义 | 怎么测 |
|------|------|--------|
| 检索命中率 hit (any-of) | 多轮中任一 search 是否命中 gold 商品集 | ToolMessage 提取 `[product:Pxxx]` ∩ gold |
| 检索命中率（仅调了 search 的 case） | 排除 agent 从不调 search 的 case 后的命中率 | hit / tool_calls>0 的子集 |
| 平均 gold 覆盖率 | 每 case 检索到的 gold 占其 gold 集比例 | retrieved ∩ gold / |gold| 求均 |
| **任务完成率** | 客户诉求是否被实质满足（非答非所问/悬而未决） | LLM-as-judge（`judge.py`）读 transcript + hidden_need |
| **工具调用成功率** | 工具调用未报错的比例 | 统计 ToolMessage 中 `[tool_error]`/`[tool_guard]` 前缀 |
| **RAGAS Faithfulness** | 末轮回复是否基于检索内容（抓"不检索凭空编"幻觉） | judge LLM，输入 final_response + retrieved_contexts |
| **RAGAS AnswerRelevancy** | 回复是否切合客户诉求 | judge LLM + embedding，输入 hidden_need + final_response |
| **RAGAS ContextRecall** | 检索内容是否覆盖 gold 事实 | judge LLM，输入 retrieved_contexts + gold 合成 reference |
| **RAGAS AnswerCorrectness** | 回复与 gold 商品的吻合度 | judge LLM + embedding，输入 final_response + reference |
| 轮次 / 命中轮 | 平均对话轮数、首次命中在第几轮 | transcript user 轮计数 |
| **token 消耗** | 每 case prompt+completion token | `TokenCounter` 回调（与 system eval 共用）挂在 graph callbacks |
| **延迟 p50/p95** | 整 case 多轮墙钟 | per-case 计时 |

RAGAS 四指标由独立 judge LLM 离线打分（`EVAL_JUDGE_*`，与被测 agent 解耦避免互扰），reference 从 gold 商品**机械合成**（代码枚举，无 LLM 评判，可复现）。只对"有回复且有检索内容"的 case 打分——agent 从不调 search 的 case 排除，避免把 agent 行为缺陷混进生成质量。

未命中 case 的完整轨迹存档供人工 debug。失败（API/欠费）显式区分于"0 命中"，不静默吞。

> 人工接管率**未纳入**：handoff 机制在项目早期已删除（无 `transfer_to_human` 工具），
> 没有接管事件可统计。要测得先把 handoff 作为 feature 重建。
> 单轮 RAGAS（固定 query、固定 reference）仍保留在 `generate/`，作为不受多轮 agent 行为干扰的对照轴。
