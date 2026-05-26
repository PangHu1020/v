# 项目文档

Phase-1 完整文档索引。Phase-2 推进过程中各文档需同步更新。

| 文件 | 内容 |
| --- | --- |
| [architecture.md](architecture.md) | 模块边界、目录树、复用基元、配置约定 |
| [data_flow.md](data_flow.md) | 三大工作流的数据走向、记忆生命周期、Session 切换规则 |
| [call_chain.md](call_chain.md) | 一条客户消息从入站到回复的完整函数级调用链（含文件/行号） |
| [gaps.md](gaps.md) | Phase-1 已知不足 + Phase-2 待办（按优先级） |

约定：

- 所有路径都是相对仓库根（`/mnt/zh/project/v-main/`）。
- 文件引用格式 `path:line`，例如 [backend/v/agents/nodes.py:62](../backend/v/agents/nodes.py#L62)。
- 三份 CLAUDE.md（[根](../CLAUDE.md) / [/backend/app/](../backend/app/CLAUDE.md) / [/backend/v/](../backend/v/CLAUDE.md)）描述**规则**；本目录描述**事实**——已经实现了什么、还差什么。两边冲突时以 CLAUDE.md 为准。
