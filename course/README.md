# Core Course · 11 Chapters

这是仓库唯一的核心 syllabus。前提是：你已经掌握 Python、PyTorch、常见深度学习模型、反向传播、训练/验证和基本概率统计；课程把时间花在自动驾驶领域背景和模型工程接口上。

## 教学结构

每章都遵循：

```text
领域问题 → shared urban cut-in scene → 最小接口 → 可运行实验
→ 失败/扰动 → artifact → 下一章
```

不要把这些 notebook 当作 11 个互不相关的 demo。除 Chapter 01 的 optional nuScenes checkpoint 外，默认无需下载大型数据；Chapter 02–10 会在 `artifacts/urban_cut_in/` 里逐步产生和消费文件。

## 路线与产出

| Chapter | Notebook | 依赖前章 artifact | 本章留下的证据 |
|---:|---|---|---|
| 00 | [System & ODD](00_system_and_odd.ipynb) | — | `00_system_contract.json` |
| 01 | [Sensors & Geometry](01_sensors_geometry.ipynb) | 00 的 scene contract | `01_geometry.json` |
| 02 | [BEV & Fusion](02_bev_and_fusion.ipynb) | 01 的 frame/time semantics | `02_bev_dataset.npz` + metadata |
| 03 | [Temporal State](03_temporal_state.ipynb) | shared scene | `03_temporal_state.npz` |
| 04 | [Localization & Mapping](04_localization_mapping.ipynb) | 00 的 ODD pose threshold | `04_localization.npz` |
| 05 | [Learnable BEV Model](05_learnable_bev_model.ipynb) | 02 的真实 BEV grid | `05_bev_model.pt` + runtime |
| 06 | [Prediction](06_prediction.ipynb) | 03 的 tracked state | `06_prediction.npz` |
| 07 | [Planning & Closed Loop](07_planning_closed_loop.ipynb) | 06 的 future modes | `07_planner.npz` |
| 08 | [Data & Evaluation](08_data_and_evaluation.ipynb) | 07 的 plan + all contracts | `08_eval_report.json` |
| 09 | [Safety & Runtime](09_safety_runtime.ipynb) | 05/08 的 evidence | `09_safety_runtime.json` |
| 10 | [Capstone](10_capstone.ipynb) | 00–09 全部 artifact | portfolio-ready report skeleton |

## 如何运行

从仓库根目录打开 JupyterLab，并按 00→10 执行。Chapter 05 和 10 需要：

```bash
python -m pip install -r requirements-ml.txt
```

如果只想检查结构，不需要 torch：

```bash
python scripts/validate_project.py
```

## Real-data checkpoint

Chapter 01、02 和 08 都有明确的 nuScenes mini checkpoint。它们默认打印“数据未提供”而不会伪造结果；真正运行时需要：

```bash
python -m pip install -r requirements-real-data.txt
export NUSCENES_ROOT=/path/to/nuscenes
```

运行后必须记录 dataset version、sample token、frame chain、timestamp policy、split 和结果文件。toy grid、toy metrics 和本仓库的 CPU latency 不能写成 nuScenes/量产证据。

## 每章完成标准

提交一张可解释图、一个参数/数据扰动、一个失败案例、所有 TODO 的解释，以及本章 artifact 的 schema。完成 Chapter 10 时，再补至少两个 ablation、一个 failure replay、open/closed-loop 对比和 p50/p95/p99 runtime 报告。
