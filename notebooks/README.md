# Interactive Notebook Track

这条路线把 HTML 教程中的关键接口做成可执行学习单元，并在原有机制教程之上补齐 L4 系统闭环。它假设学习者已经有深度学习基础，优先补自动驾驶领域背景。每个 Notebook 都包含概念解释、最小数学接口、可复现数据、参数扰动、失败案例和 TODO 习题。

## 运行方式

```bash
python -m pip install -r requirements.txt
jupyter lab
```

`05_training_evaluation_baseline.ipynb` 使用 PyTorch，需要先安装 `requirements-ml.txt`。当前 `13–14` 是不下载大型 checkpoint 的 VLM/VLA 机制教学；只有把它们扩展为真实 Hugging Face backbone、processor 或公开 checkpoint 时，才需要 `requirements-frontier.txt`。GitHub 可以直接渲染并阅读 ipynb；交互控件和训练实验建议在本地 JupyterLab 中运行。

## A. AD domain bridge：先补智驾背景

这 6 个 Notebook 不重新讲 Python、反向传播或通用优化器，而是把已有深度学习能力接到自动驾驶的问题空间。建议按顺序完成；每节约 1–3 小时，先理解领域接口，再进入后面的机制 Notebook。

| 顺序 | Notebook | 领域入口 | 最低产出 |
|---|---|---|---|
| 00A | [智能驾驶系统全景](00a_autonomous_driving_system_overview.ipynb) | ego、actor、scene、ODD、模块接口 | 画出一条 sensor→control→safety 链路 |
| 00B | [传感器、坐标系与时间](00b_sensors_frames_and_time.ipynb) | frame、pose、extrinsic、timestamp、ego-motion | 解释一个坐标或同步误差 |
| 00C | [BEV / Occupancy / Scene Representation](00c_bev_occupancy_and_scene_representation.ipynb) | occupancy、box、vector、map、agent state | resolution/dropout 对 occupancy 的影响 |
| 00D | [时序场景状态与 Tracking](00d_temporal_scene_state_and_tracking.ipynb) | observation、association、state、uncertainty | dropout/outlier 下的 track 报告 |
| 00E | [Prediction → Planning → Control](00e_prediction_planning_control_closed_loop.ipynb) | trajectory、action、constraint、closed-loop | open/closed-loop gap 对比 |
| 00F | [数据、安全、评测与部署](00f_data_safety_evaluation_deployment.ipynb) | scenario、slice、fallback、latency、regression | 一张 slice/gate 报告 |

每个 bridge notebook 都指向下一篇深入 Notebook，避免“读完术语但不知道为什么要学它”。

## B. 推荐顺序与最低产出

| 顺序 | Notebook | 核心问题 | 最低产出 |
|---|---|---|---|
| 00 | [ODD 与系统契约](00_odd_and_system_contract.ipynb) | ODD、接口、延迟预算、降级状态如何定义？ | ODD coverage + contract violation report |
| 01 | [SE(3) / 标定 / 投影](01_se3_calibration_projection.ipynb) | 传感器如何映射到统一坐标系？ | 投影图 + 标定误差曲线 |
| 02 | [BEV 融合鲁棒性](02_sensor_fusion_robustness.ipynb) | 模态缺失和时间错位如何破坏融合？ | dropout/错位 ablation |
| 03 | [Flow Matching / Action Chunk](03_flow_matching_action_chunk.ipynb) | 连续动作如何生成并执行？ | 轨迹图 + p50/p95 延迟 |
| 04 | [Corner Case Safety Monitor](04_corner_case_safety_monitor.ipynb) | 不确定时如何触发 fallback？ | TTC + recall/误报分析 |
| 05 | [PyTorch Transformer / BEV Query](05_training_evaluation_baseline.ipynb) | 如何训练多模态 token 模型？ | loss/accuracy + dropout robustness + latency |
| 06 | [LiDAR / BEV Occupancy](06_lidar_bev_occupancy.ipynb) | 点云如何体素化为 BEV？ | 点云图 + 分辨率/dropout ablation |
| 07 | [时序跟踪与时间对齐](07_temporal_tracking_alignment.ipynb) | dropout、outlier、timestamp offset 有何影响？ | track 图 + RMSE/gate 曲线 |
| 16 | [Localization 与 Mapping](16_localization_and_mapping.ipynb) | 漂移、GNSS outage 和地图匹配如何评估？ | pose RMSE + outage recovery |
| 08 | [轨迹预测指标](08_trajectory_prediction_metrics.ipynb) | 如何评估多模态未来？ | ADE/FDE/minADE/miss rate |
| 09 | [规划与闭环控制](09_planning_control_closed_loop.ipynb) | 轨迹如何变成可执行控制？ | closed-loop path + constraint analysis |
| 17 | [Scenario Runner / Log Replay](17_scenario_runner_log_replay.ipynb) | 如何构造可回归场景？ | scenario matrix + failure replay |
| 10 | [多传感器数据管线](10_dataset_pipeline_sensor_bundle.ipynb) | 如何组装可审计 sensor bundle？ | bundle 表 + 数据质量报告 |
| 11 | [Closed-loop 评测](11_closed_loop_evaluation_metrics.ipynb) | 如何同时报告安全、任务和舒适性？ | episode metrics + weight sensitivity |
| 12 | [Corner-case Mining](12_corner_case_mining_scenario_slices.ipynb) | 如何找到高风险 slice 和 hard cases？ | slice report + replay policy |
| 18 | [Safety State Machine](18_safety_state_machine_degraded_mode.ipynb) | 如何处理 fault、degraded mode 和 ODD exit？ | transition table + detection/recovery latency |
| 15 | [部署画像与量化](15_deployment_profiling_quantization.ipynb) | 如何报告真实系统效率？ | MACs + p50/p95/p99 + quant error |
| 19 | [L4 Model Development Capstone](19_l4_model_development_capstone.ipynb) | 如何交付完整模型开发证据？ | ODD/scenario/report/failure replay |
| 13 | [VLM Structured Conditions](13_vlm_structured_driving_conditions.ipynb) | VLM 如何输出可验证语义条件？ | schema + precision/recall/coverage |
| 14 | [VLA / WA / π0 Interface](14_vla_world_action_interface.ipynb) | action chunk 与 world-action 后果如何连接？ | BC MSE + 闭环 drift |

## 依赖分层

### 基础与系统接口

`00A–00F`、`00–04`、`06–12`、`15–19` 默认以 NumPy/SciPy/pandas/Matplotlib 讲解自动驾驶领域接口、几何、数据、评测和系统状态。这样可以在没有大型数据集和 GPU 的情况下验证接口与失败模式。

### PyTorch learned-model track

`05` 用 `torch.nn.TransformerEncoder` 实现一个带 camera/LiDAR token 和 learned BEV query 的小模型。它直接展示 token projection、modality embedding、self-attention、训练循环、模态 dropout 和 latency。学习架构时不需要 `transformers` 包。

### Hugging Face frontier track

`requirements-frontier.txt` 增加 `transformers`、`accelerate`、`safetensors`、`sentencepiece` 和 `einops`，用于把 `13–14` 扩展为预训练 VLM/VLA backbone、processor 和公开 checkpoint 实验。当前 notebook 的合成机制实验不需要它；它也不应替代定位、融合、规划、闭环评测和安全训练。

## 完成规则

每个 Notebook 都应留下：

1. 至少一张可解释的图；
2. 至少一个参数或数据扰动实验；
3. 至少一个失败案例；
4. 完成并解释所有 TODO，包括为什么你的实现不能直接上车。

对 `00A–00F`，还必须留下一个领域 checkpoint：用自己的话解释术语、输入输出、失效模式，以及它连接到哪一个后续模型/评测 Notebook。

Notebook 使用合成数据来保证可运行性；它们用于理解机制，不等价于在 nuScenes、Waymo、nuPlan 或 NAVSIM 上取得真实 benchmark 结果。进入真实数据阶段后，需要补充数据许可、坐标和时间契约、场景划分、闭环 runner、硬件 runtime 和版本回归。
