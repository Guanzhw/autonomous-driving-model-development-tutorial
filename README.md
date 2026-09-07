# Autonomous Driving Model Development Tutorial

面向已有深度学习基础、希望进入智能驾驶模型开发岗位的系统化学习项目。

项目采用一条 T 型路线：

- 主干：3D 感知 → 多传感器融合 → 时序建模 → 预测/规划 → 评测、corner case 与部署。
- 前沿分叉：VLM、VLA、World Model、WA/WAM、Physical Intelligence π0。
- 学习方式：HTML 教程 + 可运行 Jupyter Notebook + 公开数据/代码资源。

## 项目内容

| 路径 | 内容 |
|---|---|
| [在线式教程入口](index.html) | 岗位画像、系统全景、概念辨析、24 周路线、资源和求职交付标准 |
| [Notebook 01](notebooks/01_se3_calibration_projection.ipynb) | SE(3)、标定、坐标变换与相机投影 |
| [Notebook 02](notebooks/02_sensor_fusion_robustness.ipynb) | BEV 特征融合、模态缺失、时序错位和质量感知融合 |
| [Notebook 03](notebooks/03_flow_matching_action_chunk.ipynb) | Flow matching、连续 action chunk 与推理步数/延迟 |
| [Notebook 04](notebooks/04_corner_case_safety_monitor.ipynb) | TTC、置信度、规则 monitor、fallback 与安全评测 |
| [Notebook 说明](notebooks/README.md) | 运行方法、学习顺序和每个习题的完成标准 |

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

然后打开 `notebooks/` 中的文件。Notebook 01、02、04 不依赖大型模型或自动驾驶数据集，适合先运行；Notebook 03 先用 NumPy 版本理解机制，若安装了 PyTorch，可继续运行可选的神经速度场实验。

## 学习顺序

1. 先阅读 `index.html` 中的“岗位画像”和“系统全景”。
2. 完成 Notebook 01，确认坐标系、外参和时间对齐不是黑箱。
3. 完成 Notebook 02，观察融合系统在 modality dropout 和 temporal misalignment 下如何退化。
4. 完成 Notebook 03，把 PI/π0 中的 action chunk、flow matching 映射到车辆轨迹空间。
5. 完成 Notebook 04，把 corner-case 处理写成可测试的安全接口。
6. 再进入 nuScenes、MMDetection3D、nuPlan/NAVSIM、LeRobot、openpi、OpenVLA 和 DriveWAM 等公开项目。

## 作品集标准

每个实验至少记录：

- 数据、坐标约定、随机种子和运行环境；
- 主指标与参数/误差分析；
- 实际运行时间、p50/p95/p99 latency 或吞吐；
- 至少一个失败案例及 root cause；
- 至少一个 ablation，而不是只保留最优结果。

## 许可证与资料边界

本仓库的教程组织、Notebook 代码和实验设计用于学习。外部课程、论文、数据集、模型和代码仍遵循各自许可证；不要将受限数据、私有 checkpoint 或凭据提交到仓库。

