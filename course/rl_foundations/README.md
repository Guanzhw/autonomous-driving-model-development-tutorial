# RL 基础：从手算 MDP 到真实驾驶闭环

本单元面向已经会 Python/PyTorch、但还没有 RL 背景的学习者。先用可手算的有限 MDP 把状态、动作、奖励、回报、价值和策略分开，再把 REINFORCE 接回本仓库的 MetaDrive 闭环。

建议完成前三单元。本课继续使用模拟器特权真值（privileged truth），以隔离策略学习问题；第二单元的噪声和滤波器未接入。即使运动状态是真值，策略仍看不到延迟队列，五维输入不能视为完整Markov状态。

## 学习顺序

1. [07 · MDP、Bellman 方程与回报](07_mdp_bellman_returns.ipynb)：手写状态、转移和奖励，逐轮核对 Bellman 更新、回报和一个 Q-learning 增量更新。
2. [08 · 用 REINFORCE 学一个驾驶速度选择](08_policy_gradient_driving.ipynb)：固定几何转向，每个 5 个物理步的决策周期选择 2 或 6 m/s 目标速度；目标速度产生的连续油门通过真实 `env.step` 执行。

生成 notebook：

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-driving.txt
.venv\Scripts\python.exe -m pip install -r requirements-learning.txt
.venv\Scripts\python.exe scripts\build_rl_foundations.py
```

运行一次小规模 CPU 实验并保存证据：

```powershell
.venv\Scripts\python.exe scripts\run_rl_foundations.py --episodes 8 --output artifacts\rl_foundations
```

CLI 会从 `--seed` 派生 3 个独立的模型/采样 seed；每个 seed 都保存训练前策略、训练后策略和完整训练曲线。输出包含匹配条件下的固定慢速/固定快速/未训练策略/学习策略指标、独立 holdout 条件的同样对照、按 seed/offset 汇总的均值与标准差，以及按 condition 配对的 before/after delta（跨条件和 seed 等权汇总）；所有完整 trace 和失败 GIF 也会保留。训练配置默认使用 `action_delay_steps=4`、`decision_repeat=5`，也就是 0.4 秒的执行延迟和 0.1 秒的策略决策间隔。

## 如何读结果

奖励从 MetaDrive 的真实 trace 重算：实测车道纵向进度减去实测横向误差成本，并在模拟器报告失败时扣分。动作标签不产生奖励，也没有专家动作或 imitation loss。每条 log-prob 与同一行真实 trace reward 对齐，再用有限 horizon 的 reward-to-go 做 score-function 更新；episode 结束使用零 bootstrap。训练使用随机 categorical policy，评测使用保存后 reload 的 `deterministic_argmax_deployment`；评测 deployment return 不能当作训练随机 `J` 的无偏估计。训练曲线和评测同时报告 discounted objective/return 与 undiscounted reward sum，避免把两种指标混为一谈。无限 continuing task 在截断点需要价值估计补尾部，目标不同。

策略只学习纵向 target-speed choice，转向仍是 `GeometricController`；四步执行队列没有放进五维输入，所以驾驶策略是部分可观测过程上的无记忆 reactive policy。任何改善都只能解释为这个窄控制变量在这张固定地图和这些起点上的效果。短 CPU 运行用于学习机制，不足以证明稳定性、道路泛化或完整自动驾驶能力。REINFORCE 的高方差是下一步引入 value baseline、批量轨迹和 PPO 的动机。

推荐先读 Sutton & Barto 的[在线教材](http://incompleteideas.net/book/the-book-2nd.html)，再读 OpenAI Spinning Up 的 [Vanilla Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/vpg.html) 和 [policy optimization 推导](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)。
