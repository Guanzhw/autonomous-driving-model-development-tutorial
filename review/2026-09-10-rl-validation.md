# 第四阶段验证：MDP与真实交互策略梯度

## 任务与环境

Python3.11、固定MetaDrive版本、官方CPU Torch2.10.0。纸面MDP用于手算；真实驾驶实验固定几何转向，策略每0.1秒从2/6m/s目标速度中采样，通过油门和4步延迟队列驱动车辆。奖励由真实轨迹的纵向增量、横向误差和失败标志重算。

训练优化有限时域随机策略的折扣目标；评测将相同checkpoint改为argmax，测量确定性部署策略的回报。延迟队列对五维策略观测不可见，此处使用无记忆策略类，观测并非完整Markov状态。

```powershell
.venv/Scripts/python.exe -m pytest tests/test_rl_foundations.py -q
.venv/Scripts/python.exe scripts/run_rl_foundations.py --output artifacts/rl_verified
.venv/Scripts/python.exe scripts/execute_notebooks.py --unit rl_foundations
```

首轮完整默认CLI执行三个独立模型种子7/8/9，各8个训练episode、60步上限；六个开发评测条件，另有四个holdout条件。保存训练前后checkpoint、完整评测轨迹、终止原因、失败GIF与环境配置；训练逐episode保存loss、回报、步数与失败状态。固定慢速、固定快速、未训练与学习策略均有匹配条件。所有种子均保留，没有用holdout挑选模型。

## 实际结果与解释

首轮开发条件确定性部署回报（gamma=.99），均值按相同六个条件计算：

| 模型seed | 训练前折扣回报 | 训练后折扣回报 | 训练前失败 | 训练后失败 |
|---|---|---|---|---|
| 7 | 3.621964 | 3.621964 | 6/6 | 6/6 |
| 8 | 4.244792 | 5.725587 | 1/6 | 1/6 |
| 9 | 5.732705 | 5.977978 | 0/6 | 2/6 |

固定慢速回报5.732705、0/6失败，固定快速3.621964、6/6失败。holdout学习策略失败分别4/4、1/4、2/4。短训练有未改善和回报/失败冲突；不能概括为RL优于固定控制。此处折扣回报是实际部署轨迹的数值，不是对训练随机策略期望J的估计。

## 审查与修正

- [技术审查](2026-09-10-rl-technical.md)：REINFORCE按episode求和且有外层gamma^t；return.detach；回报取实际trace。分别标注训练/评测策略，显式记录控制增益，补充逐条件paired delta与训练配置。
- [教学反方](2026-09-10-rl-devils-advocate.md)：区分纸面MDP与部分可观测驾驶任务；把数值策略梯度、可编辑MDP和单因素延迟实验放入notebook；要求解释全部三个seed与失败取舍。
- 手算回归覆盖一步等概率梯度[-1,+1]与两步折扣权重[-2.3,+2.3]/[+1.8,-1.8]，可发现漏乘外层折扣或按不同episode长度缩放的问题。

最终修正后的复跑、全部主线回归与独立复核结论追加于下方。GitHub实际CI按本阶段提交检查。

## 最终复核（2026-09-11）

32项累计测试通过（7.94秒），八课全部实际执行通过。Lesson08汇总读取也用最终三seed evaluation.json执行；本机将artifacts/rl_final/evaluation.json复制到默认读取位置，JSON中的原始trace路径仍保留。GitHub CI直接执行默认CLI，再执行八课。

最终完整CLI复跑产生80条评测episode及24条训练history记录；部署回报/失败计数与上表一致。独立公式从80份JSON trace重算折扣回报和未折扣奖励，核对每条action数量及记录的控制增益，全部一致。开发条件跨三个模型种子的配对ΔG0均值为+0.575356，失败比例差为+0.111111；与固定慢速相比ΔG0为−0.624196，失败比例差为+0.5。结果保留了未改善与风险取舍。

Ruff、八课及14份归档结构/链接检查、八课重复生成hash检查通过。两位独立审查者已追加最终修正通过记录。[第三阶段GitHub CI](https://github.com/Guanzhw/autonomous-driving-model-development-tutorial/actions/runs/34498327503)已成功；第四阶段推送后检查该提交的独立Linux运行。
