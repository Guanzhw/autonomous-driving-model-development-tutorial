# Autonomous Driving Model Development Tutorial

面向已有深度学习基础、希望进入智能驾驶模型开发岗位的系统化学习项目。当前路线已经升级为 **L4-oriented model development**：把模型放回数据、定位、规划、闭环评测、安全状态机和部署系统中学习。

## 路线判断

项目采用一条 T 型路线：

- **L4 主干**：ODD / 系统契约 → 几何与标定 → 3D/BEV 感知 → 多传感器融合 → 时序跟踪 → 定位与地图 → 预测/规划 → 场景回放与闭环评测 → 数据闭环 → 安全降级 → 部署。
- **前沿分叉**：VLM、VLA、World Model、WA/WAM、Physical Intelligence π0。
- **学习方式**：HTML 知识地图 + 20 个可运行 Jupyter Notebook + 公开数据/代码资源。
- **完成标准**：每个阶段都要有可运行代码、明确指标、参数实验、失败分析和可复现运行记录。

L4 在这里不是“更大的模型”，而是系统在定义好的 ODD 内能够完成任务、识别退化并安全退出。合成 notebook 用于理解机制，不等价于真实车辆验证或安全认证。

## 项目入口

| 入口 | 内容 |
|---|---|
| [HTML 总教程](index.html) | 岗位画像、系统全景、24 周路线、公开资源和作品集标准 |
| [项目参考](PROJECT_REFERENCE.md) | 路线判断、证据等级、依赖边界、当前缺口和维护日志 |
| [Notebook Track 说明](notebooks/README.md) | 20 个 Notebook 的学习顺序、目标和交付物 |
| [核心依赖](requirements.txt) | NumPy/SciPy + Jupyter，支持基础系统实验 |
| [学习模型依赖](requirements-ml.txt) | 在核心依赖之上加入 PyTorch，支持 03/05 |
| [前沿依赖](requirements-frontier.txt) | 在学习模型依赖之上加入 Hugging Face Transformers/VLM 生态 |
| [Review Protocol](review/README.md) | Reviewer 与 Devil's Advocate 的双角色审查、优先级和合并门禁 |

## Notebook 路线

### A. L4 系统契约与模型基础

| Notebook | 核心问题 |
|---|---|
| [00 · ODD 与系统契约](notebooks/00_odd_and_system_contract.ipynb) | 如何定义运行边界、输入输出、时间预算和降级状态？ |
| [01 · SE(3) / 标定 / 投影](notebooks/01_se3_calibration_projection.ipynb) | 不同传感器如何映射到一致坐标系？ |
| [02 · BEV 融合鲁棒性](notebooks/02_sensor_fusion_robustness.ipynb) | 模态缺失和时间错位如何破坏融合？ |
| [03 · Flow Matching / Action Chunk](notebooks/03_flow_matching_action_chunk.ipynb) | 连续动作如何生成并在有限步数内执行？ |
| [04 · Corner Case Safety Monitor](notebooks/04_corner_case_safety_monitor.ipynb) | TTC、置信度和传感器健康度如何触发 fallback？ |
| [05 · PyTorch Transformer / BEV Query](notebooks/05_training_evaluation_baseline.ipynb) | 如何真正训练 attention 模型并测量鲁棒性与 latency？ |

### B. 3D、时序、定位与规划

| Notebook | 核心问题 |
|---|---|
| [06 · LiDAR / BEV Occupancy](notebooks/06_lidar_bev_occupancy.ipynb) | 点云如何体素化并变成 BEV 表示？ |
| [07 · 时序跟踪与时间对齐](notebooks/07_temporal_tracking_alignment.ipynb) | dropout、outlier 和 timestamp offset 如何影响 tracking？ |
| [16 · Localization 与 Mapping](notebooks/16_localization_and_mapping.ipynb) | 漂移、GNSS outage、地图匹配和 relocalization 如何评估？ |
| [08 · 轨迹预测指标](notebooks/08_trajectory_prediction_metrics.ipynb) | 如何评估多模态、多智能体未来？ |
| [09 · 规划与闭环控制](notebooks/09_planning_control_closed_loop.ipynb) | 轨迹如何变成带约束的可执行控制？ |

### C. 场景、数据闭环与安全验证

| Notebook | 核心问题 |
|---|---|
| [17 · Scenario Runner / Log Replay](notebooks/17_scenario_runner_log_replay.ipynb) | 如何把道路事件变成可重复、可回归的场景？ |
| [10 · 多传感器数据管线](notebooks/10_dataset_pipeline_sensor_bundle.ipynb) | 如何组装可审计的 sensor bundle？ |
| [11 · Closed-loop 评测](notebooks/11_closed_loop_evaluation_metrics.ipynb) | 如何同时报告进度、碰撞、越界和舒适性？ |
| [12 · Corner-case Mining](notebooks/12_corner_case_mining_scenario_slices.ipynb) | 如何从场景属性中找到高风险 slice 和 hard cases？ |
| [18 · Safety State Machine / Degraded Mode](notebooks/18_safety_state_machine_degraded_mode.ipynb) | 故障检测、降级、最小风险动作和 ODD exit 如何组织？ |
| [15 · 部署画像与量化](notebooks/15_deployment_profiling_quantization.ipynb) | 如何报告 p50/p95/p99、MACs、吞吐和量化误差？ |
| [19 · L4 Model Development Capstone](notebooks/19_l4_model_development_capstone.ipynb) | 如何交付一套模型、数据、闭环、安全和运行时证据？ |

### D. 前沿分支：建立在主干之上

| Notebook | 核心问题 |
|---|---|
| [13 · VLM Structured Conditions](notebooks/13_vlm_structured_driving_conditions.ipynb) | VLM 如何输出可验证的驾驶条件，而不是直接控车？ |
| [14 · VLA / WA / π0 Interface](notebooks/14_vla_world_action_interface.ipynb) | action chunk、模仿学习和 world-action 后果预测如何连接？ |

## 依赖边界：为什么核心路线不强制 Hugging Face Transformers？

`transformers` 是加载预训练 Transformer/VLM、processor 和 checkpoint 的生态库；它不是 attention 的数学定义，也不是所有 BEV、tracking 或 planning 模型的必要依赖。许多自动驾驶模型使用自定义 PyTorch 模块、MMDetection3D、timm、稀疏卷积或车端 CUDA kernel。

本项目因此分层：

- `requirements-ml.txt` 包含 PyTorch，因为 Notebook 03 的可选分支和 Notebook 05 需要真正训练 learned model；
- Notebook 05 直接用 `torch.nn.TransformerEncoder`，便于观察 Q/K/V、query、token、mask、训练和 latency；
- `requirements-frontier.txt` 才加入 `transformers`、`accelerate` 等，用于 VLM/VLA/World Model 分支；
- 不为了调用一个 attention layer 而引入 Hugging Face，也不把下载大模型当成学习完成的证明。

## 本地运行

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
jupyter lab
```

运行 Notebook 03 的 PyTorch 分支或 Notebook 05，再安装：

```bash
python -m pip install -r requirements-ml.txt
```

如果要把 `13–14` 从当前的合成接口扩展为真实 VLM/VLA/World Model checkpoint 或 processor，再安装：

```bash
python -m pip install -r requirements-frontier.txt
```

当前 `13–14` 为不依赖大型 checkpoint 的机制教学 notebook，本身不强制安装 `transformers`；`requirements-frontier.txt` 为接入真实预训练模型时的扩展层。

Notebook 00–04、06–12、15–19 默认使用合成数据，保证无需下载大型数据集也能运行。完成机制实验后，再接入 nuScenes、Waymo、nuPlan、NAVSIM、CARLA、Autoware、LeRobot、OpenVLA 或 openpi。

## 推荐学习顺序

1. 先读 HTML 的岗位画像、系统全景、ODD 和依赖边界。
2. 完成 `00–05`，建立系统契约、坐标、融合、Transformer 和安全 monitor 的共同语言。
3. 完成 `06–09` 与 `16`，补齐 BEV、时序、定位、预测和规划。
4. 完成 `17`、`10–12`，建立 scenario runner、数据闭环和 closed-loop evaluation。
5. 完成 `18`、`15`、`19`，形成安全降级、部署和 L4 capstone 证据。
6. 最后完成 `13–14`，把 VLM/VLA/WA/π0 接入已经存在的驾驶接口。

## Notebook 完成标准

每个 Notebook 至少留下：

- 一张可解释的结果图；
- 一个参数或数据扰动实验；
- 一个失败案例及 root cause；
- 一个自己实现并解释的 TODO；
- 主指标以及 p50/p95/p99 或等价的效率指标。

对于 Transformer notebook，还应记录模型参数量、token 数、attention 配置、训练 seed、CPU/GPU、batch size 和 latency 测量方法。

## 作品集标准

每个较大的项目至少记录：

- 数据、坐标约定、时间同步、随机种子和运行环境；
- 主指标与参数/误差分析；
- accuracy/safety/robustness/latency 的联合报告；
- 至少一个 failure replay 及 root cause；
- 至少一个 ablation，而不是只保留最优结果；
- 如果接入真实数据，记录数据许可证、场景划分和版本回归策略。

## 许可证与资料边界

本仓库的教程组织、Notebook 代码和实验设计用于学习。外部课程、论文、数据集、模型和代码仍遵循各自许可证；不要将受限数据、私有 checkpoint 或凭据提交到仓库。
