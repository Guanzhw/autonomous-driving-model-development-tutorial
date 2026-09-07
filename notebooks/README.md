# Interactive Notebook Track

这些 Notebook 不是独立的 API 练习，而是把教程中的关键接口做成最小可运行实验。

## 运行方式

```bash
cd autonomous-driving-model-development-tutorial
python -m pip install -r requirements.txt
jupyter lab
```

也可以直接在 GitHub 中阅读 `.ipynb`；要运行交互控件，请在本地 JupyterLab 中打开。

## 顺序与产出

| 顺序 | Notebook | 核心问题 | 你的产出 |
|---|---|---|---|
| 01 | SE(3) / 标定 / 投影 | 不同传感器如何映射到一致坐标系？ | 一张投影图 + 一组标定误差曲线 |
| 02 | BEV 融合鲁棒性 | 模态缺失和时间错位如何破坏融合？ | dropout/错位 ablation + 质量感知融合 |
| 03 | Flow Matching / Action Chunk | 连续动作如何建模并在有限步数内生成？ | 不同步数的轨迹和实际延迟统计 |
| 04 | Corner Case Safety Monitor | 模型不确定时如何触发约束和 fallback？ | TTC/召回率/误报分析 + 安全决策图 |

## Notebook 完成规则

每个 Notebook 都应留下四类结果：

1. 至少一张可解释的图；
2. 至少一个参数或数据扰动实验；
3. 至少一个失败案例；
4. 完成并解释所有 `TODO`，包括为什么你的实现可能不适合直接上车。

Notebook 使用合成数据来保证可运行性；它们用于理解机制，不等价于在 nuScenes/Waymo 上取得真实 benchmark 结果。

