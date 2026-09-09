# Autonomous Driving Model Development Tutorial

从 **已有深度学习背景** 到 **Autonomous Driving Model Engineer** 的 11 课实战教程。项目补充的是智驾领域背景、数据形态、坐标/时间语义、系统接口、闭环评测和部署约束；不重复完整的 Python、反向传播或通用深度学习入门。

## 你会构建什么

一条持续演化的 `urban cut-in` 实验链：

```text
raw camera/LiDAR
  → frame / calibration / time
  → BEV occupancy + risk
  → temporal agent state
  → learned BEV model
  → prediction
  → planning / closed loop
  → scenario replay / corner-case slice
  → safety gate / runtime
  → capstone evidence
```

Chapter 02 产生的 BEV 数据会被 Chapter 05 直接训练；Chapter 05 保存的 checkpoint 会被 Chapter 10 加载；中间的 state、prediction、planner、evaluation 和 safety artifact 会逐步落到 `artifacts/urban_cut_in/`。这使课程从“独立 demo 集合”变成一条可复核的模型开发闭环。

合成实验属于 `toy-mechanism`：用于理解接口、机制和失败模式，不等价于真实车辆验证、安全认证或公开 benchmark 结果。Chapter 01、02、05、08 都包含 nuScenes mini 的 optional real-data checkpoint；需要数据、许可证和额外依赖，不会伪造结果。

## 唯一学习入口

| 入口 | 用途 |
|---|---|
| [课程路线](course/README.md) | 11 个核心章节，唯一 syllabus |
| [Advanced Labs](labs/README.md) | 3 个主线完成后的选修实验 |
| [参考资料](reference/README.md) | glossary、metrics、datasets、公开课程和论文入口 |
| [维护参考](PROJECT_REFERENCE.md) | 证据等级、维护规则和审查记录；面向 maintainer |
| [Review Protocol](review/README.md) | Reviewer / Devil's Advocate 的审查门禁 |
| [HTML landing page](index.html) | 仅做项目介绍和导航，不定义另一套课程路线 |

## 核心课程

| # | Notebook | 主问题 | 关键产出 |
|---:|---|---|---|
| 00 | [System & ODD](course/00_system_and_odd.ipynb) | ODD、ego、actor、scene 和模块契约是什么？ | system contract |
| 01 | [Sensors & Geometry](course/01_sensors_geometry.ipynb) | 传感器如何通过 frame、SE(3)、标定和时间进入模型？ | projection + calibration experiment |
| 02 | [BEV & Sensor Fusion](course/02_bev_and_fusion.ipynb) | 为什么用 BEV/occupancy，错位如何传播？ | `02_bev_dataset.npz` |
| 03 | [Temporal State](course/03_temporal_state.ipynb) | observation 如何变成稳定的 agent state？ | tracking state artifact |
| 04 | [Localization & Mapping](course/04_localization_mapping.ipynb) | ego pose、漂移、GNSS outage 如何影响系统？ | localization report |
| 05 | [Learnable BEV Model](course/05_learnable_bev_model.ipynb) | 如何训练有空间语义的 Transformer BEV query？ | `05_bev_model.pt` |
| 06 | [Prediction](course/06_prediction.ipynb) | 多模态 agent future 如何评估并提供给 planner？ | ADE/FDE/miss-rate artifact |
| 07 | [Planning & Closed Loop](course/07_planning_closed_loop.ipynb) | 预测如何影响 ego trajectory 和下一帧输入？ | planner + rollout |
| 08 | [Data & Evaluation](course/08_data_and_evaluation.ipynb) | bundle → scenario → replay → metrics → corner slice 如何闭环？ | evaluation report |
| 09 | [Safety & Runtime](course/09_safety_runtime.ipynb) | uncertainty、degraded mode 和 p99 latency 如何成为发布门禁？ | safety/runtime gate |
| 10 | [Capstone](course/10_capstone.ipynb) | 如何加载前面 artifact，完成一次可面试的模型开发交付？ | end-to-end report |

## Advanced Labs（可选）

这些内容建立在 Chapter 06–10 的接口之上，不计入核心先修路线：

- [Flow Matching / Action Chunk](labs/flow_matching_action_chunk.ipynb)
- [VLM Structured Driving Conditions](labs/vlm_structured_driving_conditions.ipynb)
- [VLA / WA / π0 Interface](labs/vla_world_action_interface.ipynb)

## 依赖与运行

```bash
python -m venv .venv
source .venv/bin/activate          # Windows 使用 .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
jupyter lab
```

Chapter 05 和 Chapter 10 的 checkpoint 路径需要 PyTorch：

```bash
python -m pip install -r requirements-ml.txt
```

真实数据 checkpoint 需要单独下载并遵守 nuScenes 条款：

```bash
python -m pip install -r requirements-real-data.txt
export NUSCENES_ROOT=/path/to/nuscenes
```

Advanced Labs 的真实预训练 VLM/VLA 扩展才需要 Hugging Face 依赖：

```bash
python -m pip install -r requirements-frontier.txt
```

`transformers` 不是 BEV、tracking、planning 或 attention 数学本身的必需依赖；Chapter 05 直接使用 PyTorch `TransformerEncoder` 和 `MultiheadAttention`，便于学习架构和训练接口。只有加载预训练 VLM/VLA backbone 时，才进入 Frontier 依赖层。

## 推荐学习节奏

1. 先读本 README 和 [course/README.md](course/README.md)，确认你已具备深度学习基础。
2. 按 Chapter 00→10 顺序运行；不要先跳去 Flow Matching/VLM/VLA。
3. 每章检查它保存的 artifact 是否存在，并回答 notebook 末尾的 domain checkpoint。
4. Chapter 05 训练 checkpoint 后，再运行 Chapter 10；最后再选 Advanced Lab。
5. 将 toy 结果替换为 nuScenes mini 固定样本或公开 runner，并记录数据版本、坐标/时间约定、seed、配置、指标、latency 和 failure replay。

## 作品集完成标准

最终提交应至少包含：可 clone 的环境和运行命令、数据/场景 schema、模型 checkpoint、open-loop 与 closed-loop 指标、至少两个 ablation、一个 failure replay、p50/p95/p99 runtime 报告，以及清楚的“toy mechanism ≠ L4 safety case”边界声明。

更多维护约束见 [PROJECT_REFERENCE.md](PROJECT_REFERENCE.md)。
