# Interactive Notebook Track

这条路线把 HTML 教程中的关键接口做成可执行学习单元，并在原有机制教程之上补齐 L4 系统闭环。每个 Notebook 都包含概念解释、最小数学接口、可复现数据、参数扰动、失败案例和 TODO 习题。

## 运行方式

```bash
python -m pip install -r requirements.txt
jupyter lab
```

`05_training_evaluation_baseline.ipynb` 使用 PyTorch，需要先安装 `requirements-ml.txt`；`13–14` 的 Hugging Face/VLA 扩展需要额外安装 `requirements-frontier.txt`。GitHub 可以直接渲染并阅读 ipynb；交互控件和训练实验建议在本地 JupyterLab 中运行。

## 推荐顺序与最低产出

| 顺序 | Notebook | 核心问题 | 最低产出 |
|---|---|---|---|
| 00 | ODD 与系统契约 | ODD、接口、延迟预算、降级状态如何定义？ | ODD coverage + contract violation report |
| 01 | SE(3) / 标定 / 投影 | 传感器如何映射到统一坐标系？ | 投影图 + 标定误差曲线 |
| 02 | BEV 融合鲁棒性 | 模态缺失和时间错位如何破坏融合？ | dropout/错位 ablation |
| 03 | Flow Matching / Action Chunk | 连续动作如何生成并执行？ | 轨迹图 + p50/p95 延迟 |
| 04 | Corner Case Safety Monitor | 不确定时如何触发 fallback？ | TTC + recall/误报分析 |
| 05 | PyTorch Transformer / BEV Query | 如何训练多模态 token 模型？ | loss/accuracy + dropout robustness + latency |
| 06 | LiDAR / BEV Occupancy | 点云如何体素化为 BEV？ | 点云图 + 分辨率/dropout ablation |
| 07 | 时序跟踪与时间对齐 | dropout、outlier、timestamp offset 有何影响？ | track 图 + RMSE/gate 曲线 |
| 16 | Localization 与 Mapping | 漂移、GNSS outage 和地图匹配如何评估？ | pose RMSE + outage recovery |
| 08 | 轨迹预测指标 | 如何评估多模态未来？ | ADE/FDE/minADE/miss rate |
| 09 | 规划与闭环控制 | 轨迹如何变成可执行控制？ | closed-loop path + constraint analysis |
| 17 | Scenario Runner / Log Replay | 如何构造可回归场景？ | scenario matrix + failure replay |
| 10 | 多传感器数据管线 | 如何组装可审计 sensor bundle？ | bundle 表 + 数据质量报告 |
| 11 | Closed-loop 评测 | 如何同时报告安全、任务和舒适性？ | episode metrics + weight sensitivity |
| 12 | Corner-case Mining | 如何找到高风险 slice 和 hard cases？ | slice report + replay policy |
| 18 | Safety State Machine | 如何处理 fault、degraded mode 和 ODD exit？ | transition table + detection/recovery latency |
| 15 | 部署画像与量化 | 如何报告真实系统效率？ | MACs + p50/p95/p99 + quant error |
| 19 | L4 Model Development Capstone | 如何交付完整模型开发证据？ | ODD/scenario/report/failure replay |
| 13 | VLM Structured Conditions | VLM 如何输出可验证语义条件？ | schema + precision/recall/coverage |
| 14 | VLA / WA / π0 Interface | action chunk 与 world-action 后果如何连接？ | BC MSE + 闭环 drift |

## 依赖分层

### 基础与系统接口

`00–04`、`06–12`、`15–19` 默认以 NumPy/SciPy/pandas/Matplotlib 讲解几何、数据、评测和系统状态。这样可以在没有大型数据集和 GPU 的情况下验证接口与失败模式。

### PyTorch learned-model track

`05` 用 `torch.nn.TransformerEncoder` 实现一个带 camera/LiDAR token 和 learned BEV query 的小模型。它直接展示 token projection、modality embedding、self-attention、训练循环、模态 dropout 和 latency。学习架构时不需要 `transformers` 包。

### Hugging Face frontier track

`requirements-frontier.txt` 增加 `transformers`、`accelerate`、`safetensors`、`sentencepiece` 和 `einops`，用于加载预训练 VLM/VLA backbone、processor 和公开 checkpoint。它服务于 `13–14`，不应替代定位、融合、规划、闭环评测和安全训练。

## 完成规则

每个 Notebook 都应留下：

1. 至少一张可解释的图；
2. 至少一个参数或数据扰动实验；
3. 至少一个失败案例；
4. 完成并解释所有 TODO，包括为什么你的实现不能直接上车。

Notebook 使用合成数据来保证可运行性；它们用于理解机制，不等价于在 nuScenes、Waymo、nuPlan 或 NAVSIM 上取得真实 benchmark 结果。进入真实数据阶段后，需要补充数据许可、坐标和时间契约、场景划分、闭环 runner、硬件 runtime 和版本回归。
