# Project Reference

长期维护本项目时先读本文件；它记录路线、证据等级、目录职责、依赖边界和已知缺口。

- **项目**：Autonomous Driving Model Development Tutorial
- **仓库**：`Guanzhw/autonomous-driving-model-development-tutorial`
- **目标**：帮助已有深度学习基础的学习者进入 L4 智能驾驶模型开发岗位
- **当前版本**：11 个核心课程 + 3 个 Advanced Lab；一个 shared urban cut-in artifact chain
- **上次核对**：2026-09-09
- **远端基线**：`6c1155b`（AD domain bridge 版本）
- **唯一学习入口**：[README](README.md)；核心 syllabus 为 [course/README.md](course/README.md)
- **维护入口**：[review/README.md](review/README.md)

## 1. 项目定义和不变前提

这是 landing tutorial 和作品集孵化器，不是量产自动驾驶栈、真实车辆验证平台或 safety certification material。

学习者已经具备：Python、PyTorch、常见深度学习模型、反向传播、训练/验证、基本概率统计和读写 notebook 的能力。项目不重复完整通用深度学习入门；它补的是智驾领域背景：ego/actor/scene/ODD、传感器和 frame、SE(3)、时间同步、BEV/occupancy、tracking、localization、prediction、planning、closed-loop、数据闭环、安全和 runtime。

## 2. 结构决策

### 2.1 11 + 3，而不是 26 个独立 demo

| Canonical chapter | 合并来源/主题 | 教学主问题 |
|---|---|---|
| `course/00` System & ODD | 系统全景 + ODD/契约 + 数据/安全入口 | 系统究竟在解决什么问题？ |
| `course/01` Sensors & Geometry | sensors/frames/time + SE(3)/projection | 观测如何进入共同世界？ |
| `course/02` BEV & Fusion | BEV/occupancy + fusion + LiDAR | 为什么选择这种空间表示？ |
| `course/03` Temporal State | temporal bridge + tracking/alignment | observation 如何成为 state？ |
| `course/04` Localization & Mapping | localization/mapping | ego pose 如何保持可用？ |
| `course/05` Learnable BEV Model | 重写原 Transformer baseline | 如何训练真正有 AD 空间语义的 query？ |
| `course/06` Prediction | prediction bridge + metrics | 多模态 future 如何度量并供 planner 使用？ |
| `course/07` Planning & Closed Loop | planning/control bridge + closed loop | action 如何改变下一帧？ |
| `course/08` Data & Evaluation | bundle + replay + metrics + mining | failure 如何进入可回归数据闭环？ |
| `course/09` Safety & Runtime | corner monitor + state machine + profiling | accuracy 之外的发布门禁是什么？ |
| `course/10` Capstone | L4 model development capstone | 如何把 artifact 串成可面试交付？ |

Advanced Labs：`flow_matching_action_chunk`、`vlm_structured_driving_conditions`、`vla_world_action_interface`。它们不能抢在 geometry/state/planning/safety 主线之前。

### 2.2 Shared scene 和 artifact 是课程主线

`src/ad_tutorial/scene.py` 是唯一的 toy scene 生成和 BEV 编码入口。坐标约定为 ego `x forward, y left, z up`；所有核心课应复用 `urban_cut_in`，不要重新发明互不兼容的 cube/token/episode。

`artifacts/urban_cut_in/` 是运行时输出，不提交模型权重或大数据。当前链路：

```text
00 contract
 → 01 geometry
 → 02 BEV dataset
 → 03 temporal state + 04 localization
 → 05 learned checkpoint
 → 06 prediction
 → 07 planner
 → 08 evaluation
 → 09 safety/runtime
 → 10 capstone
```

Chapter 05 必须使用 Chapter 02 的 BEV grid；Chapter 10 必须加载 Chapter 05 checkpoint，并检查其配置/输出，而不是只比较手写 policy。

### 2.3 README / HTML / compatibility 文件的职责

- `README.md`：唯一对学习者的总入口和路线摘要；
- `course/README.md`：11 课的 canonical syllabus、前置 artifact 和产出；
- `labs/README.md`：3 个 optional labs；
- `index.html`：薄 landing page，只做导航；不写第二套教学路线；
- `notebooks/README.md`：旧路径兼容说明，不再列课程清单；
- `PROJECT_REFERENCE.md`、`review/`：维护者文档，不作为学习入口。

## 3. 依赖边界

| 层 | 文件 | 用途 |
|---|---|---|
| Core | `requirements.txt` | NumPy/SciPy/pandas/Matplotlib/Jupyter；支持 00–04、06–09 和 labs 机制实验 |
| Learned model | `requirements-ml.txt` | Core + PyTorch；Chapter 05 和 Chapter 10 checkpoint |
| Real-data checkpoint | `requirements-real-data.txt` | Core + nuScenes devkit；Chapter 01/02/08 optional sample |
| Frontier | `requirements-frontier.txt` | Learned model + Hugging Face `transformers` 等；真实 VLM/VLA 扩展 |

`transformers` 不属于所有 AD 模型的必需依赖。Chapter 05 用 `torch.nn.TransformerEncoder` 与 `MultiheadAttention` 学习 token/query/attention/训练/runtime；只有加载预训练 VLM/VLA backbone 才安装 Frontier 层。

## 4. 证据等级

| 标签 | 含义 | 允许的公开表述 |
|---|---|---|
| `toy-mechanism` | 合成数据、最小模型、机制实验 | 能解释接口、趋势和失败模式 |
| `open-benchmark-ready` | 有公开数据/runner adapter、固定 split/命令，但未提交结果 | 可以说明复现实验计划 |
| `open-benchmark-result` | 有固定 commit、环境、数据版本、脚本和结果文件 | 可以报告该公开 benchmark 的结果 |
| `system-evidence` | 有 runtime、资源、闭环、安全策略和回归证据 | 可以讨论 trade-off，仍不自动推出道路安全认证 |

当前 11 课和 labs 的默认结果是 `toy-mechanism`。nuScenes mini checkpoint 是 `open-benchmark-ready` scaffolding，除非在仓库中提交固定 sample 的输出，否则不得称为已复现 benchmark。

## 5. 维护规则

每次修改课程、目录、依赖或公开声明时：

1. 更新受影响 notebook、`course/README.md`、`labs/README.md`、根 README、HTML 和本文件；
2. 维护 shared scene 的坐标/时间语义和 artifact schema，不创建平行 toy pipeline；
3. 明确证据等级、学习前提、数据许可和未解决问题；
4. 运行 `python scripts/build_course_notebooks.py`（若改了生成器），再运行结构校验；
5. 路线/依赖变化时按 14 个 canonical notebook 做 nbformat + AST 检查；按顺序执行所有不依赖外部数据的 code cells；Chapter 05/10 另做 PyTorch smoke test；
6. 运行 `git diff --check`；
7. 触发 Reviewer 与 Devil's Advocate 的独立审查；实现者不得替代任一角色签字；
8. 将 real-data、runtime、safety 和真实 runner 缺口写入 review/reference，不用 prose 掩盖。

最小命令：

```bash
python scripts/build_course_notebooks.py
python scripts/validate_project.py
git diff --check
```

## 6. 当前明确缺口

- nuScenes mini adapter 目前是 checkpoint scaffolding，仍需实际下载数据、固定 sample、输出图和结果 artifact；
- 尚未集成 nuPlan/NAVSIM/CARLA 的真实 runner 和公开 failure replay；
- runtime 仍是 Python/CPU 教学画像，缺少 C++/CUDA/TensorRT、固定车端硬件、显存和版本回归；
- safety state machine 是机制教学，不是 ISO 26262/SOTIF safety case；
- 真实 VLM/VLA/World Model checkpoint、冻结/微调、结构化输出和闭环对照属于 Advanced Lab 后续工作；
- 角色定向 portfolio（perception/fusion、prediction/planning、data/evaluation、runtime/safety）仍需各自补一个真实公开 benchmark。

## 7. 变更日志

| 日期 | 版本/commit | 变更 | 证据与遗留问题 |
|---|---|---|---|
| 2026-09-09 | `5cf4203` | L4 主线、PyTorch Transformer baseline、依赖分层。 | 早期 20 个 notebook；以合成机制为主。 |
| 2026-09-09 | `6c1155b` | 增加 00A–00F AD domain bridge、项目参考、双角色 review 门禁。 | 26 个 notebook；暴露出路线过度展开和重复 pipeline。 |
| 2026-09-09 | pending | 依据结构性评审收敛为 11 core + 3 labs；融合 domain bridge；统一 urban cut-in；重写 Chapter 05；capstone 加载 checkpoint；HTML 瘦身；加入 real-data checkpoint scaffolding。 | 当前默认证据仍为 `toy-mechanism`；待全量结构/AST/执行与双角色 follow-up review。 |
