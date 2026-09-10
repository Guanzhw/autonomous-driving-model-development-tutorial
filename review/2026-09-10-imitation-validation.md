# 第三阶段验证：行为克隆

## 环境与执行

复用前两阶段的Python 3.11/MetaDrive环境，按requirements-learning.txt安装官方CPU Torch 2.10.0。小型MLP在CPU训练。

```powershell
.venv/Scripts/python.exe scripts/run_imitation.py --output artifacts/imitation_verified
.venv/Scripts/python.exe -m pytest tests/test_imitation.py -q
.venv/Scripts/python.exe scripts/execute_notebooks.py --unit imitation
```

完整默认实验采集14个专家episode、840条样本；本机采集约2.9秒、训练约1.8秒（不含导出图与GIF）。held-out动作MSE为0.000854486；相同四个测试条件下专家0/4失败、未训练模型2/4失败、BC1/4失败。BC平均距离20.630m，专家27.385m。保留BC在0.2秒`not_on_lane`结束的实际案例与回放，不能用漂亮的MSE替代终止分析。

数据manifest保存原始专家trace、完整仿真config、split与hash；特征取当前动作之前的truth和参考。与第二单元的估计输入实验相比，此处回到privileged truth以单独研究策略拟合与状态分布变化。训练统计量仅来自train；validation选择checkpoint；test不参与选择。重载模型实际向MetaDrive发送动作。

## 独立审查与修正

- [技术审查](2026-09-10-imitation-technical.md)：增加每个测试条件单独配对的共同窗口指标，保留完整episode终止/距离；数据边界拒绝同episode跨split。
- [教学反方](2026-09-10-imitation-devils-advocate.md)：明确先修与truth输入，展示实际样本的特征/标准化/标签/预测，提供可编辑的线性基线与分布内对照。
- 训练配置、数据与checkpoint均留下可复核元数据；配对指标的测试覆盖不同测试条件提前结束的情况。

以上数值来自本机固定版本与配置，属于四个测试条件的机制实验；不作为总体成功率或道路泛化结论。最终修正后的复跑与审查结论追加于本记录及独立报告。

## 最终复核

22项累计测试通过（6.37秒）。05/06实际执行，包括新增原始样本审计、线性基线、同checkpoint分布内对照。后者使用原test条件仅改变初始偏移，保留相同seed与horizon；修复了首轮复跑发现的未导入DrivingConfig问题。完整CLI再次得到相同离线MSE与失败计数，并保存训练配置。

四个条件的共同步数为60/54/2/47，平均40.75步。共同窗口平均绝对横向误差：专家0.180894m、BC0.186081m、未训练0.345193m；完整episode的失败/距离仍单列。Ruff、六课结构/链接检查及重复生成通过。独立技术与教学审查已追加修正结论。

上一阶段提交`00beb93`的[GitHub CI](https://github.com/Guanzhw/autonomous-driving-model-development-tutorial/actions/runs/34497849152)已成功；本阶段CI新增官方CPU PyTorch依赖安装并执行六课。
