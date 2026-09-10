# Project Reference

## 目标和当前交付

从自动驾驶入门，再进入机器人操作与更广泛具身智能。学习者具有初步深度学习训练经验，尚无 RL 背景；每周约 26 小时。课程组织依据问题、因果实验与解释能力。

2026-09-10 重构将当前实现收敛为两个 notebook 组成的第一单元：路线跟踪、执行延迟与恢复。底层车辆推进由 MetaDrive 完成；观测、控制、实际动作与下一状态相连。后续 24 周方向在 [course/README.md](course/README.md) 中明确标为计划。

## 目录与依赖

| 位置 | 职责 |
|---|---|
| README.md | 学习入口、范围与下一步 |
| course/first_loop/ | 当前单元解释、实验与练习 |
| src/ad_tutorial/driving.py | 实际驾驶闭环、控制和轨迹指标 |
| scripts/run_first_unit.py | 一次可复现的对照实验 |
| scripts/build_first_unit.py | 当前 notebook 的生成源 |
| requirements-driving.txt | 第一单元环境依赖（复用基础 notebook 依赖，加固定模拟器版本） |
| reference/ | 精选教材、术语、数据与评测 |
| reference/legacy/ | 原 14 份材料及具体缺陷说明 |
| scripts/build_legacy_materials.py | 归档 notebook 生成与醒目标注 |
| src/ad_tutorial/scene.py、bev_model.py | 归档场景和 BEV 教学模型 |
| requirements.txt、requirements-ml.txt 等 | 归档材料及其可选扩展依赖 |
| review/ | 独立评审与实际验证记录 |

起步环境使用 Python 3.11。单元依赖中固定 MetaDrive 版本，训练/真实数据依赖按需要在独立环境安装。新单元不消费历史 `urban_cut_in` 文件，不复用旧报告作为新结果。

## 完成标准

一次修改必须能解释其因果关系：控制目标或控制参数变化会影响动作，再影响环境状态；延迟队列中的发出动作与实际执行动作分别记录。结果包含完整轨迹、指标来源、配置和失败回放。有限时长的存活不等同于完成整条路线。

代码与 notebook 实际运行、相同配置复现、指标从轨迹重算、不同初始条件的局限、中文解释和练习，分别检查。学习效果需由学习者试学反馈继续评估。

## 维护命令

使用第一单元环境，按顺序执行：

```powershell
python scripts/build_first_unit.py
python scripts/build_legacy_materials.py
python scripts/validate_project.py
python -m pytest tests -q
python scripts/run_first_unit.py
python scripts/execute_notebooks.py
git diff --check
```

归档运行路径发生变化时，在含 PyTorch 的归档环境执行 `python scripts/execute_notebooks.py --legacy`。生成器不应引入随机 notebook id 或运行输出；重复生成不应出现 diff。

## 证据与边界

- 新单元属于仿真机制实验，初始使用真值状态；视觉感知、学习型规划与复杂交通尚未接入。
- 真实道路数据、公开 benchmark 和机器人硬件均待后续建设；外部资料链接是阅读入口。
- 归档第 06/07 章混用了他车预测与自车轨迹，第 07 章 rollout 不消费 selected trajectory，第 10 章仅汇总旧 artifact；已从主线隔离并标注。
- 归档 BEV occupancy 可由 LiDAR 输入复制；camera_points 是点集近似。保留代码是为机制学习和审查，不据此声称具备融合或真实 BEV 能力。
- 仿真渲染、依赖版本、主机与运行时间以实际验证记录为准；GPU 型号不等于已完成 GPU 训练。

## 变更记录

| 日期 | 变化 | 验证 |
|---|---|---|
| 2026-09-09 | 原 11 core + 3 labs，共享点集与 artifact | 历史状态见 Git 与 review/initial-dual-review.md |
| 2026-09-10 | 以 MetaDrive 第一单元重构主线；归档原材料；中文课程与教材路线；因果测试与新验证流程 | 9项测试、2份当前/14份归档 notebook 执行；[验证记录](review/2026-09-10-validation.md)与其中的独立审查 |
