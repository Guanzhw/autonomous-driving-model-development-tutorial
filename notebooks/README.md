# Interactive Notebook Track

这些 Notebook 把 HTML 教程中的关键接口做成一条可执行学习路线，而不是把概念堆成链接列表。每个 Notebook 都包含：

- 概念解释和最小数学接口；
- 可复现的合成数据或 toy simulator；
- 参数滑块、ablation 或故障注入；
- TODO 习题、失败案例和作品集交付标准。

## 运行方式

    cd autonomous-driving-model-development-tutorial
    python -m pip install -r requirements.txt
    jupyter lab

GitHub 可以直接渲染并阅读 ipynb；要运行交互控件，请在本地 JupyterLab 中打开。

## 路线与产出

| 顺序 | Notebook | 核心问题 | 最低产出 |
|---|---|---|---|
| 01 | SE(3) / 标定 / 投影 | 传感器如何映射到统一坐标系？ | 投影图 + 标定误差曲线 |
| 02 | BEV 融合鲁棒性 | 模态缺失和时间错位如何破坏融合？ | dropout/错位 ablation |
| 03 | Flow Matching / Action Chunk | 连续动作如何生成并执行？ | 轨迹图 + p50/p95 延迟 |
| 04 | Corner Case Safety Monitor | 不确定时如何触发 fallback？ | TTC + recall/误报分析 |
| 05 | 训练与评测基线 | 如何把 loss 变成可解释指标？ | 训练曲线 + F1/ECE/阈值分析 |
| 06 | LiDAR / BEV Occupancy | 点云如何体素化为 BEV？ | 点云图 + 分辨率/dropout ablation |
| 07 | 时序跟踪与时间对齐 | dropout、outlier、timestamp offset 有何影响？ | track 图 + RMSE/gate 曲线 |
| 08 | 轨迹预测指标 | 如何评估多模态未来？ | ADE/FDE/minADE/miss rate |
| 09 | 规划与闭环控制 | 轨迹如何变成可执行控制？ | closed-loop path + lookahead ablation |
| 10 | 多传感器数据管线 | 如何组装可审计 sensor bundle？ | bundle 表 + 数据质量报告 |
| 11 | Closed-loop 评测 | 如何同时报告安全、任务和舒适性？ | episode metrics + 权重敏感性 |
| 12 | Corner-case Mining | 如何找到高风险 slice 和 hard cases？ | slice report + replay policy |
| 13 | VLM Structured Conditions | VLM 如何输出可验证语义条件？ | schema + precision/recall/coverage |
| 14 | VLA / WA / π0 Interface | action chunk 与 world-action 后果预测如何连接？ | BC MSE + 闭环 drift |
| 15 | 部署画像与量化 | 如何报告真实系统效率？ | MACs + p50/p95/p99 + quant error |

## 推荐学习节奏

- 机制基础：01–04。
- 自动驾驶主干：05–12。
- 前沿接口与工程落地：13–15。

01–04 先建立坐标、融合、生成式策略和安全 monitor 的共同语言；05–12 再补训练、3D/BEV、时序、预测、规划、数据管线和闭环评测；13–15 最后进入 VLM/VLA/WA/π0 与部署。

## 完成规则

每个 Notebook 都应留下四类结果：

1. 至少一张可解释的图；
2. 至少一个参数或数据扰动实验；
3. 至少一个失败案例；
4. 完成并解释所有 TODO，包括为什么你的实现不适合直接上车。

Notebook 使用合成数据来保证可运行性；它们用于理解机制，不等价于在 nuScenes、Waymo、nuPlan 或 NAVSIM 上取得真实 benchmark 结果。进入真实数据阶段后，需要补充数据许可、坐标和时间契约、场景划分、闭环 runner 与部署环境。

