# Autonomous Driving Model Development Tutorial

面向已有深度学习基础、希望进入智能驾驶模型开发岗位的系统化学习项目。

项目采用一条 T 型路线：

- 主干：数据管线 → 3D/BEV 感知 → 多传感器融合 → 时序跟踪 → 预测/规划 → 闭环评测 → 安全与部署。
- 前沿分叉：VLM、VLA、World Model、WA/WAM、Physical Intelligence π0。
- 学习方式：HTML 知识地图 + 15 个可运行 Jupyter Notebook + 公开数据/代码资源。
- 每个阶段都要求：可运行代码、明确指标、参数实验和失败分析。

## 项目入口

| 入口 | 内容 |
|---|---|
| [HTML 总教程](index.html) | 岗位画像、系统全景、概念辨析、24 周路线、公开资源和作品集标准 |
| [Notebook Track 说明](notebooks/README.md) | 15 个 Notebook 的学习顺序、目标和交付物 |
| [requirements.txt](requirements.txt) | JupyterLab、NumPy、SciPy、pandas、Matplotlib 和 ipywidgets |

## Notebook 路线

### A. 机制基础：先把接口和失败模式看懂

| Notebook | 核心问题 |
|---|---|
| [01 · SE(3) / 标定 / 投影](notebooks/01_se3_calibration_projection.ipynb) | 不同传感器如何映射到一致坐标系？ |
| [02 · BEV 融合鲁棒性](notebooks/02_sensor_fusion_robustness.ipynb) | 模态缺失和时间错位如何破坏融合？ |
| [03 · Flow Matching / Action Chunk](notebooks/03_flow_matching_action_chunk.ipynb) | 连续动作如何生成并在有限步数内执行？ |
| [04 · Corner Case Safety Monitor](notebooks/04_corner_case_safety_monitor.ipynb) | 不确定时如何触发约束和 fallback？ |

### B. 自动驾驶主干：从训练到闭环

| Notebook | 核心问题 |
|---|---|
| [05 · 训练与评测基线](notebooks/05_training_evaluation_baseline.ipynb) | 如何建立可复现的训练、阈值和校准实验？ |
| [06 · LiDAR / BEV Occupancy](notebooks/06_lidar_bev_occupancy.ipynb) | 点云如何体素化并变成 BEV 表示？ |
| [07 · 时序跟踪与时间对齐](notebooks/07_temporal_tracking_alignment.ipynb) | dropout、outlier 和 timestamp offset 如何影响 tracking？ |
| [08 · 轨迹预测指标](notebooks/08_trajectory_prediction_metrics.ipynb) | 如何评估多模态预测而不是一条平均轨迹？ |
| [09 · 规划与闭环控制](notebooks/09_planning_control_closed_loop.ipynb) | open-loop path 如何在车辆闭环中变成可执行轨迹？ |
| [10 · 多传感器数据管线](notebooks/10_dataset_pipeline_sensor_bundle.ipynb) | 如何从 sample tables 组装可审计的 sensor bundle？ |
| [11 · Closed-loop 评测](notebooks/11_closed_loop_evaluation_metrics.ipynb) | 如何同时报告进度、碰撞、越界和舒适性？ |
| [12 · Corner-case Mining](notebooks/12_corner_case_mining_scenario_slices.ipynb) | 如何从场景属性中找到高风险 slice 和 hard cases？ |

### C. 前沿接口与工程落地

| Notebook | 核心问题 |
|---|---|
| [13 · VLM Structured Conditions](notebooks/13_vlm_structured_driving_conditions.ipynb) | VLM 如何输出可验证的驾驶条件，而不是直接控车？ |
| [14 · VLA / WA / π0 Interface](notebooks/14_vla_world_action_interface.ipynb) | action chunk、模仿学习和 world-action 后果预测如何连接？ |
| [15 · 部署画像与量化](notebooks/15_deployment_profiling_quantization.ipynb) | 如何报告 p50/p95/p99、MACs、吞吐和量化误差？ |

## 本地运行

    python -m venv .venv

    # Linux/macOS
    source .venv/bin/activate

    # Windows PowerShell
    # .venv\Scripts\Activate.ps1

    python -m pip install -r requirements.txt
    jupyter lab

Notebook 01–15 默认使用合成数据，保证无需下载大型数据集也能运行。Notebook 03 中的 PyTorch 神经速度场是可选分支；完成机制实验后，再接入 nuScenes、Waymo、nuPlan、NAVSIM、CARLA、LeRobot、OpenVLA 或 openpi。

## 学习顺序建议

1. 先阅读 HTML 的岗位画像、系统全景和先修知识。
2. 完成 01–04，建立坐标、融合、生成式策略和安全 monitor 的共同语言。
3. 完成 05–07，补齐训练评测、BEV 表示和时序跟踪。
4. 完成 08–11，把预测、规划、数据管线和闭环指标连起来。
5. 完成 12，建立 corner-case 数据挖掘与 slice evaluation。
6. 完成 13–14，再进入 VLM/VLA/WA/π0 的前沿接口。
7. 完成 15，形成可用于岗位面试的部署和 latency 证据。

## Notebook 完成标准

每个 Notebook 至少留下：

- 一张可解释的结果图；
- 一个参数或数据扰动实验；
- 一个失败案例及 root cause；
- 一个自己实现的 TODO；
- 主指标以及 p50/p95/p99 或等价的效率指标。

这些 Notebook 用合成数据理解机制，不等价于在 nuScenes、Waymo 或 NAVSIM 上取得真实 benchmark 结果。真实项目应继续补充数据许可、坐标/时间契约、训练配置、场景划分、闭环评测和部署环境。

## 作品集标准

每个较大的项目至少记录：

- 数据、坐标约定、随机种子和运行环境；
- 主指标与参数/误差分析；
- 实际运行时间、吞吐、p50/p95/p99 latency 或显存；
- 至少一个失败案例及 root cause；
- 至少一个 ablation，而不是只保留最优结果。

## 许可证与资料边界

本仓库的教程组织、Notebook 代码和实验设计用于学习。外部课程、论文、数据集、模型和代码仍遵循各自许可证；不要将受限数据、私有 checkpoint 或凭据提交到仓库。
