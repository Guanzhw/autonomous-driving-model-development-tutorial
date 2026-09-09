# Review Protocol

本目录把项目审查固定成两个独立角色。它们不是“给文档润色”的同一个 reviewer 的两个名字，而是要回答不同问题：

- **Reviewer**：这次改动是否正确、完整、可运行、可解释，并且覆盖真实岗位所需的工程接口？
- **Devil's Advocate**：这次改动是否让项目看起来比实际更完整？学习者或招聘经理会在哪些地方被误导？

两者可以由不同 agent、不同人，或在没有多 agent 环境时由同一维护者分两次独立执行；但必须先分别记录原始发现，再合并结论。

## 触发条件

以下任一情况必须触发双角色审查：

- 新增、删除或重排 Notebook；
- 修改模型、数据、评测、安全状态机或部署 profiling；
- 增加/升级依赖，尤其是 PyTorch、Hugging Face、仿真器或公开 checkpoint；
- 修改 README、HTML、岗位画像或任何“达到某水平”的公开表述；
- 完成一个阶段或 capstone；
- 修复一次审查报告中的 P0/P1 问题。

纯拼写修正可以跳过完整审查，但如果改变了技术含义，仍需触发。

## 执行顺序

```text
变更 → 本地验证 → Reviewer 与 Devil's Advocate 独立审查
    → 合并发现 → 维护者决策 → 修复/记录/延期
    → 更新 PROJECT_REFERENCE.md → 再次验证
```

审查者应优先阅读 `PROJECT_REFERENCE.md`，然后看变更 diff；不能只根据 README 推断 Notebook 已经运行。

## 统一输出格式

每条发现至少包含：

```text
ID: R-xxx 或 DA-xxx
Priority: P0 / P1 / P2
Location: 文件路径、Notebook 名称或章节
Finding: 可验证的问题
Evidence: 运行结果、代码、文档或缺失证据
Impact: 对学习者、岗位匹配或公开可信度的影响
Minimal fix: 最小可行修复
Status: fix-now / record-and-park / reject
```

优先级定义：

| 级别 | 定义 | 处理要求 |
|---|---|---|
| P0 | 会造成错误技术结论、无法运行、依赖破坏，或公开声明明显超过证据 | 合并前修复，或撤回该声明/功能 |
| P1 | 显著降低岗位训练价值、可复现性、指标可信度或路线完整性 | 当前里程碑修复，或在参考文档登记明确 owner/计划 |
| P2 | 教学表达、可维护性或扩展性问题 | 排入后续 backlog，不得伪装成已完成 |

## 合并门禁

一项改动只有在以下条件同时满足时才可以标记为完成：

- 代码/Notebook 能按文档运行；
- 指标和数据生成过程可解释，失败案例没有被删除；
- 公开声明与证据等级一致；
- Reviewer 没有未处理的 P0，且 P1 有修复或明确记录；
- Devil's Advocate 没有指出未标注的过度承诺、错误路线或“toy result = L4 evidence”；
- `PROJECT_REFERENCE.md` 的路线、缺口和变更日志已同步。

GitHub Actions 的 structural check 会自动检查 nbformat、Python syntax、14 个 canonical Notebook 的文档链接和审查资产；它不代替本地的全量 code-cell 执行。涉及模型或依赖的变更，仍需在本地运行 CPU smoke test，并把结果写进 PR 或审查报告。`course/` 是核心路线，`labs/` 是选修路线；旧 `notebooks/` 只保留兼容说明。

## 当前审查资产

- [Reviewer role](roles/reviewer.md)
- [Devil's Advocate role](roles/devils-advocate.md)
- [Initial dual review](initial-dual-review.md)（本次 L4 更新的首轮结果）
