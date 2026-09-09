# Project Reference

> 这是本项目的长期维护参考。后续修改路线、Notebook、依赖、公开声明或审查流程时，先读本文件；完成修改后同步更新本文件的状态、缺口和变更日志。

- **项目**：Autonomous Driving Model Development Tutorial
- **仓库**：`Guanzhw/autonomous-driving-model-development-tutorial`
- **目标**：帮助已有深度学习基础的学习者进入 L4 智能驾驶模型开发岗位
- **当前版本**：L4-oriented model development，20 个交互式 Notebook
- **上次核对**：2026-09-09
- **远端基线**：commit `5cf4203`（L4 路线与 Transformer 依赖分层更新）
- **主入口**：[README](README.md)、[HTML 知识地图](index.html)、[Notebook Track](notebooks/README.md)
- **审查入口**：[review/README.md](review/README.md)
- **首轮报告**：[review/initial-dual-review.md](review/initial-dual-review.md)

## 1. 项目定义

本项目是一个 landing tutorial 和作品集孵化器，不是量产自动驾驶栈、真实车辆验证平台或安全认证材料。

学习者最终应该能展示：

1. 能把 ODD、坐标系、传感器输入、时间预算、模型输出和降级状态写成明确接口；
2. 能实现并评估感知、融合、时序、定位、预测、规划或控制中的至少一个模型模块；
3. 能把数据质量、场景切片、log replay、closed-loop 指标和 failure analysis 接起来；
4. 能报告模型效果、鲁棒性、p50/p95/p99 延迟、资源约束和已知失效边界；
5. 能把合成教学实验迁移到公开数据集、公开仿真器或真实项目约束，而不是把 toy result 当成 L4 证据。

## 2. 不变的路线判断

| 判断 | 项目约束 |
|---|---|
| L4 主线优先 | 先学 3D/时空感知、多传感器融合、定位、预测/规划、数据闭环、评测、安全和部署，再追逐更大的 foundation model。 |
| 前沿模型是第二曲线 | VLM、VLA、WA/WAM、World Model、π0 用来建立研究差异化，不替代坐标、时序、闭环和安全基础。 |
| 系统接口必须显式 | 即使讨论 end-to-end，也要保留传感器、坐标/时间、状态估计、数据、评测和 safety envelope 的接口。 |
| 合成结果必须诚实标注 | Notebook 的合成数据用于理解机制和测试失败模式；没有真实 benchmark 运行记录时，不宣称达到 nuScenes、nuPlan、NAVSIM、Waymo 或量产水平。 |
| 预训练模型不是完成证明 | 下载 checkpoint 或调用 `transformers` 不等于掌握模型开发；必须解释输入输出、训练/冻结策略、评测、失效和部署边界。 |
| 不强行填补不可验证内容 | 对暂时无法实现或没有证据的能力，标记为 roadmap / open gap，不用伪造结果、夸大岗位匹配或半完成实现掩盖。 |

## 3. 当前学习地图

| 阶段 | Notebook | 学习结果 |
|---|---|---|
| 系统契约 | `00` | ODD、sensor contract、latency budget、状态和降级接口 |
| 几何与融合 | `01–02` | SE(3)、标定、投影、BEV 和模态缺失/错位鲁棒性 |
| learned model | `03–05` | action chunk、corner-case monitor、PyTorch Transformer/BEV query、训练和延迟 |
| 3D 与时序 | `06–07` | LiDAR/BEV occupancy、tracking、outlier、timestamp alignment |
| 预测与控制 | `08–09` | 轨迹指标、约束规划、闭环控制 |
| 定位与地图 | `16` | drift、GNSS outage、map matching、relocalization |
| 场景与数据闭环 | `17`、`10–12` | log replay、scenario sweep、sensor bundle、closed-loop metrics、corner-case mining |
| 安全与部署 | `18`、`15` | safety state machine、degraded mode、量化和 p50/p95/p99 |
| 综合交付 | `19` | ODD、场景覆盖、碰撞/fallback/舒适性/延迟的综合报告 |
| 前沿分支 | `13–14` | VLM structured conditions、VLA/WA/π0 interface；建立在主干完成之后 |

学习地图的权威细节以 [notebooks/README.md](notebooks/README.md) 为准；HTML 用于导航和解释，不应独立定义另一套路线。

## 4. 依赖边界

| 层 | 文件 | 用途 |
|---|---|---|
| Core | `requirements.txt` | NumPy/SciPy/pandas/Matplotlib/Jupyter；支持几何、数据、评测和系统接口实验 |
| Learned model | `requirements-ml.txt` | 在 Core 上加入 PyTorch，支持 Notebook `03` 的可选 learned 分支和 Notebook `05` |
| Frontier | `requirements-frontier.txt` | 在 Learned model 上加入 Hugging Face `transformers`、processor/checkpoint 相关依赖 |

`transformers` 是预训练 Transformer/VLM 生态库，不是 attention 数学本身，也不是所有 BEV、tracking、planning 模型的必要依赖。Notebook `05` 用 PyTorch 的 `torch.nn.TransformerEncoder` 直接展示 token、query、self-attention、训练、鲁棒性和 latency；只有加载公开 VLM/VLA backbone 时才进入 Frontier 层。

## 5. 证据等级

维护者在文档、Notebook 和项目简介中使用以下标签：

| 标签 | 含义 | 可以说什么 |
|---|---|---|
| `toy-mechanism` | 合成数据、最小模型、机制实验 | 能解释接口、趋势和失败模式 |
| `open-benchmark-ready` | 已有数据/场景适配和明确 split，但尚未在本项目中跑出结果 | 可以说明复现实验计划和指标定义 |
| `open-benchmark-result` | 有固定 commit、环境、数据版本、脚本和结果文件 | 可以报告可复现的公开 benchmark 结果 |
| `system-evidence` | 有运行时、资源、闭环、安全策略和回归证据 | 可以讨论工程 trade-off；仍不能自动推出道路安全认证 |

当前新增 L4 Notebook 主要属于 `toy-mechanism`，Notebook `19` 是交付格式演练，不应被描述成真实 L4 验证。

## 6. 维护规则

每次修改路线、Notebook、依赖或公开声明时：

1. 更新对应 Notebook、`notebooks/README.md`、`README.md`、`index.html` 和本文件中受影响的部分；
2. 明确新增内容的证据等级和已知限制；
3. 运行全部 Notebook 的 nbformat、AST 和 code-cell CPU 验证，以及 HTML 链接检查；
4. 让 Reviewer 和 Devil's Advocate 独立审查，不能由实现者自己把“已实现”当成“已验证”；
5. 把剩余问题写入审查报告或 GitHub issue，不隐藏在 prose 中；
6. 在变更日志中记录日期、commit、变更、证据和未解决问题。

最小本地检查：

```bash
git diff --check
python -m py_compile scripts/build_l4_update.py
python -m jupyter nbconvert --to notebook --execute notebooks/<changed_notebook>.ipynb --stdout >/dev/null
```

全量执行和审查流程见 [review/README.md](review/README.md)。

## 7. 当前明确缺口

这些是项目下一阶段的真实工作项，而不是本版本可以默认为已完成的能力：

- 至少一个公开数据集的端到端适配、固定 split、坐标/时间清洗和可复现实验记录；
- nuPlan/NAVSIM/CARLA 等公开 runner 的 adapter，以及真实 failure replay artifact；
- C++/CUDA/TensorRT 或等价 runtime profiling，包含显存、吞吐、batch=1 延迟和版本回归；
- 更接近生产的 tracking/localization/planning 组件和跨模块接口测试；
- 数据版本、场景 schema、hard-negative mining 和模型/数据闭环的自动化 CI；
- 按岗位方向拆分的 portfolio project：perception/fusion、prediction/planning、data/evaluation、runtime/safety；
- 对 VLM/VLA/World Model 分支增加真实 checkpoint 的加载、冻结/微调、结构化输出约束和闭环对照实验。

缺口存在并不意味着当前教程无效；它决定了项目描述中应该使用“机制教程/公开 benchmark 准备/工程化 roadmap”哪一种表述。

## 8. 变更日志

| 日期 | 版本/commit | 变更 | 证据与遗留问题 |
|---|---|---|---|
| 2026-09-09 | `5cf4203` | 增加 L4 主干 Notebook `00`、`16–19`；把 `05` 升级为 PyTorch Transformer/BEV Query；拆分 Core/ML/Frontier 依赖。 | 20 个 Notebook 已通过本地 CPU 执行验证；仍以合成教学实验为主。 |
| 2026-09-09 | pending | 建立项目参考、Reviewer/Devil's Advocate 角色和 PR 审查门禁。 | 初始双角色审查报告待完成；后续以报告中的 P0/P1/P2 为维护队列。 |
