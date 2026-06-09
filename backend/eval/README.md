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

## 单一检索策略对比（Ablation）

`run_ablation.py` 独立运行每种策略，无级联退出。

| 策略 | hit@1 | hit@3 | hit@5 | mean_ms | 说明 |
|------|-------|-------|-------|---------|------|
| **dense** | 0.891 | 0.976 | 0.979 | ~5.6s | 纯 cosine 向量检索 |
| **hybrid_5_5** | 0.891 | 0.976 | 0.979 | ~5.5s | dense 0.5 + BM25 0.5 |
| **hybrid_7_3** | 0.891 | 0.976 | 0.979 | ~5.4s | dense 主导 |
| **hybrid_3_7** | 0.881 | 0.967 | 0.973 | ~5.6s | BM25 主导，精确名略降 |
| **rewrite_hybrid** | 0.848 | 0.957 | 0.964 | ~48s | LLM 结构化重写 + hybrid |

**结论**：在当前 150-doc 小语料上，dense 已足够，BM25 无统计增量。rewrite 对 `noisy` tier 有 +3pp 改善（0.908 → 0.938），但单独运行时 hit@1 因重写偏移而略低（0.848）；在级联模式中仅对 stage-1/2 失败的查询触发，才能发挥优势。

## 三级瀑布流检索（端到端）

每一级用**相对 margin 置信门控**（`backend/v/rag/gate.py`）决定是否退出：top-1 要过绝对 `floor`，且与第二名的相对差 `(top1−top2)/top1` 要 ≥ `rel_margin`。相对 margin 尺度无关——dense cosine（~0.6–0.95）和未归一化的 hybrid 融合排名分（~0.4–0.9）共用同一套逻辑。

```
query
  │
  ▼ stage-1: dense cosine search
  │ 门控退出：top1 ≥ 0.70 → 返回（~77%）
  │
  ▼ stage-2: hybrid WeightedRanker (dense 0.5 + BM25 0.5)
  │ 门控退出：(top1−top2)/top1 ≥ 0.04 → 返回（~7%）
  │
  ▼ stage-3: LLM 结构化重写 + hybrid search
    返回（~16%）
```

门控参数由 `backend.eval.retrieval.calibrate` 在 337-QA 集上按 Youden's J 自动校准（非手拍），换模型/语料后重跑即可。

### 端到端评测结果（337 条，top_k=5）

| 指标 | hit@1 | hit@3 | hit@5 | MRR@5 | nDCG@5 |
|------|-------|-------|-------|-------|--------|
| **整体** | 0.855 | 0.970 | **0.985** | 0.912 | 0.931 |
| easy | 0.970 | 1.000 | 1.000 | 0.985 | 0.989 |
| medium | 0.833 | 0.980 | 0.990 | 0.904 | 0.926 |
| hard | 0.806 | 0.910 | 0.955 | 0.866 | 0.889 |
| noisy | 0.841 | 0.905 | 0.952 | 0.881 | 0.899 |
| product | 0.934 | 0.962 | 0.981 | 0.951 | 0.959 |
| faq | 0.720 | 0.984 | 0.992 | 0.847 | 0.884 |

Stage 拦截分布：260 : 24 : 53 ≈ **77% : 7% : 16%**

### 关键观察

- **相对 margin 门控 vs 旧硬阈值**：noisy tier hit@1 从 0.762 → 0.841（+7.9pp），整体 hit@1 0.846 → 0.855、hit@5 0.982 → 0.985。两级门控 FPR=0（gold 不在结果里时绝不早退）。
- stage-1 用绝对 floor（dense cosine 这层 floor 就是最好判别器），stage-2 用纯相对 margin（��合排名分未归一化，绝对阈值无意义）——这正是旧的 `0.41` magic number 修不掉的问题。
- FAQ hit@1=0.720 弱于产品（0.934），但 hit@3 即达 0.984——相关文档被检索到但排名靠后。
- hard / noisy 主要靠 stage-3 rewrite 兜底。

## 参数说明

### Milvus 配置（`MILVUS_*`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MILVUS_URI` | `http://localhost:19530` | Milvus 服务地址，Standalone 或 Cloud |
| `MILVUS_TOKEN` | `""` | Milvus Cloud API key，本地留空 |
| `MILVUS_COLLECTION_NAME` | `knowledge_chunks` | 知识库 collection 名 |

### RAG 门控参数（`RAG_*`）

每级门控两个参数，由 `calibrate` 自动写入；手动调见 `config.yaml`。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `RAG_STAGE1_FLOOR` | `0.70` | Stage-1 top-1 绝对下限（dense cosine） |
| `RAG_STAGE1_REL_MARGIN` | `0.0` | Stage-1 相对分离度下限（这层 floor 主导，故为 0） |
| `RAG_STAGE2_FLOOR` | `0.0` | Stage-2 绝对下限（hybrid 分数未归一化，故为 0） |
| `RAG_STAGE2_REL_MARGIN` | `0.04` | Stage-2 相对分离度 `(top1−top2)/top1` 下限 |
| `RAG_MIN_K` | `3` | 至少要返回的结果数，低于此数不退出 |
| `RAG_DENSE_WEIGHT` | `0.5` | Hybrid 中 dense 分支权重 |
| `RAG_BM25_WEIGHT` | `0.5` | Hybrid 中 BM25 稀疏分支权重 |

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
  retrieval/     检索质量评测（cascade pipeline / ablation）
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
| `retrieval/run_eval.py` | 级联检索 recall/MRR/nDCG/hit，按 difficulty/tier/stage 分维度 |
| `retrieval/run_ablation.py` | 单策略对比（dense / hybrid variants / rewrite+hybrid） |
| `retrieval/calibrate.py` | 按 Youden's J 校准每级门控的 floor + rel_margin，`--write` 写入 config.yaml |

```bash
uv run python -m backend.eval.retrieval.calibrate --write   # 校准门控参数
uv run python -m backend.eval.retrieval.run_eval
uv run python -m backend.eval.retrieval.run_ablation --no-rewrite
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
