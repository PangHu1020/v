# 项目文档

Phase-2 收官后的完整文档索引。

| 文件 | 内容 |
| --- | --- |
| [architecture.md](architecture.md) | 模块边界、目录树、复用基元、配置约定 |
| [data_flow.md](data_flow.md) | 三大工作流（被动 / 主动 / 接管）+ 工具链 + 记忆生命周期 |
| [call_chain.md](call_chain.md) | 一条客户消息从入站到回复的完整函数级调用链 |
| [gaps.md](gaps.md) | 当前已知不足 + Phase-3 待办（按优先级） |

约定：

- 所有路径都是相对仓库根（`/mnt/zh/project/v-main/`）。
- 文件引用格式 `path:line`，例如 [backend/v/agents/nodes.py:62](../backend/v/agents/nodes.py#L62)。
- 三份 CLAUDE.md（[根](../CLAUDE.md) / [/backend/app/](../backend/app/CLAUDE.md) / [/backend/v/](../backend/v/CLAUDE.md)）描述**规则**；本目录描述**事实**——已经实现了什么、还差什么。两边冲突时以 CLAUDE.md 为准。

阶段进度：

- **Phase-1**（被动答疑主链）：已完成（`b624e3c..c0e677a`）
- **Phase-2 P0**（Slack 人工接管）：已完成（`d566fae..7d2aff7`）
- **Phase-2 P1**（MCP 客户端）：已完成（`00f40b9..7a970a3`）
- **Phase-2 P2**（ARQ 主动触达 + 会话总结）：已完成（`e60739b`）
- **Phase-2 P3**（长期记忆抽取 + 召回）：已完成（`08cac81`）
- **Phase-2 通用 subagent + Skill 加载器**：已完成（`f372192`、`c671fe7`）
- **Phase-3**（可观测性 / 性能 / 多租户）：见 [gaps.md](gaps.md)
