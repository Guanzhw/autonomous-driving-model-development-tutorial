# Review Protocol

对模型、指标、依赖、课程路线或公开声明的非小型变更，执行两个独立角色审查。实现者负责处理结论并保留最终判断。

- [Reviewer](roles/reviewer.md)：数值、控制、运行、复现与指标是否正确。
- [Devil's Advocate](roles/devils-advocate.md)：解释、练习与证据能否支撑学习目标。

## 执行顺序

先阅读 PROJECT_REFERENCE 和 diff，再检查代码、运行相关实验。当前路线修改需要执行两个主线 notebook；归档移动需验证 14 份 notebook，影响执行路径时运行旧链（含 05/10 的 PyTorch）。审查者分别记录发现后，由主 agent 修复或登记未解决问题。

每项发现包含：ID、Priority（P0/P1/P2）、Location、Finding、Evidence、Impact、Minimal fix、Status。不能把尚未运行的内容写成通过。

P0 是无法运行、错误技术结论或公开声明超出证据，交付前修复。P1 是实质影响教学或复现的问题，应修复或明确列出限制和后续处理。P2 是局部表达或维护建议。

## 完成条件

文档命令与 notebook 可运行；失败保留；控制输出实际进入环境；指标可从轨迹核算；当前实现与计划区分明确；双角色发现已回应；维护参考与检查通过。CI 的结构检查和控制测试提供自动反馈，实际本机验证与教学试学另行记录。

[早期审查记录](initial-dual-review.md)保留历史背景，其旧路径和岗位目标按当时版本解释。
