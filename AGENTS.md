# Working Agreement for Agents

先阅读 [PROJECT_REFERENCE.md](PROJECT_REFERENCE.md)，确认当前实现、计划与证据。项目面向有初步深度学习经验、没有 RL 背景的学习者，从驾驶闭环逐步进入具身智能。

## 修改范围

- `course/first_loop/` 是当前可执行主线；后续路线先写清问题和验收再实现。
- `reference/legacy/` 保留原 11 课与 3 个 Lab。已知错误标注必须随生成器保留；不能把它们重新标为完整主线。
- 新主线复用 MetaDrive 的动力学与交互，控制器输出必须进入 `env.step`，指标来自实际轨迹。
- `src/ad_tutorial/scene.py`、`bev_model.py` 服务归档材料，保留历史坐标/时间与 artifact 以供复查。
- 变更路线或依赖时同步 README、course/README、单元说明、index.html、requirements 和维护参考。
- Notebook 通过对应 `scripts/build_*.py` 生成，编辑生成源；生成必须可重复。

## 验证和审查

- 运行结构与链接检查、受影响测试、当前两个 notebook 和 documented CLI。
- 移动归档材料或修改其生成源时，检查全部 14 份语法与链接；影响运行路径时按旧 00→10 执行并单独核对 05/10 的 PyTorch 路径。原始领域缺陷按归档说明处理，不把执行通过称为技术正确。
- 新增模型、指标、依赖、路线或公开表述，需 Reviewer 与 Devil's Advocate 独立审查；实现者保留最终判断并逐条回应。
- 在 review/ 中记录实际命令、结果与未验证事项，更新 PROJECT_REFERENCE.md，运行 git diff --check。

## 实现与复核

- 非小型实现默认委派独立 `gpt-5.6-luna`（high 用于非简单任务）；主 agent 整合、检查与验证。
- 非简单变更由独立 `gpt-5.6-terra` 审查；使用独立上下文 `fork_turns: none`。需要完整对话时用 `fork_turns: all` 并继承模型。
- 数据、坐标、时间、单位在实际输入边界验证一次。避免没有实际消费者的抽象。
- 保留失败、缺失数据和不确定性；让说明与运行证据相符。
