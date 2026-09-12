"""Generate the two Chinese RL-foundations notebooks deterministically."""

from pathlib import Path
import textwrap

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "course" / "rl_foundations"


VISUAL = """
<figure class="course-figure">
  <img src="../../assets/visuals/reinforcement-learning.png" width="1536" height="1024" style="max-width:100%;height:auto" alt="原理图：策略在每个决策周期选择 2 或 6 米每秒目标速度，几何转向和隐藏执行队列继续作用于车辆">
  <a href="../../assets/visuals/reinforcement-learning.png">查看原图</a>
  <figcaption><strong>AI 原理图 · 手算/机制示意</strong> · 纸上 MDP（γ=0.9）与驾驶速度选择</figcaption>
</figure>

**因果链**：先用纸上 MDP 理解回报与价值；再在驾驶任务中学习 2/6 m/s 速度选择。驾驶中的几何转向与隐藏执行队列继续作用于车辆。

纸上 MDP 使用 γ=0.9，驾驶 REINFORCE 配置使用 γ=0.99；两者用于不同的教学检查。

**手算检查**：`γ=0.9` 时，`G(short)=2`，`G(long)=1+0.9×4`。哪条路线回报更高？
<details><summary>展开答案</summary><p>长路线回报为 4.6，高于短路线的 2。</p></details>
"""

VISUAL_MDP = """
<figure class="course-figure">
  <img src="../../assets/visuals/reinforcement-learning.png" width="1536" height="1024" style="max-width:100%;height:auto" alt="原理图：纸上 MDP 的短路线与长路线连接到驾驶中的 2 或 6 米每秒速度选择，转向保持几何控制">
  <a href="../../assets/visuals/reinforcement-learning.png">查看原图</a>
  <figcaption><strong>AI 原理图 · 手算/机制示意</strong> · 纸上 MDP（γ=0.9）与驾驶速度选择</figcaption>
</figure>

**因果链**：先用纸上 MDP 理解回报与价值；再在驾驶任务中学习 2/6 m/s 速度选择 → REINFORCE 更新策略。

**手算检查**：`γ=0.9` 时，`G(short)=2`，`G(long)=1+0.9×4`。哪条路线回报更高？
<details><summary>展开答案</summary><p>长路线回报为 4.6，高于短路线的 2。</p></details>
"""


def md(source):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


SETUP = """
from pathlib import Path
import sys
import numpy as np
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
            if (p / "src" / "ad_tutorial").is_dir())
sys.path.insert(0, str(ROOT / "src"))
from ad_tutorial.rl_foundations import *
"""


def lesson07():
    return [
        md("""
        # 07 · MDP、Bellman 方程与回报

        这一课把“驾驶”先压缩成一张纸上的有限 MDP。目标不是把表格游戏叫作自动驾驶，而是建立能手算、能断言、能解释的词汇：即时奖励 `r_t` 是一步发生了什么，回报 `G_t` 是从某时刻往后的折扣奖励之和，价值 `V(s)` 是策略下回报的期望，策略 `π(a|s)` 才是如何选动作。

        先读代码前写下预测：`start` 选择 `short` 会得到多少回报？折扣系数 `γ=0.9` 时，`long` 为什么可能更好？有限 horizon 到达终点时是否需要猜一个未来价值？
        """),
        md(VISUAL_MDP),
        code(SETUP + "\nmdp = teaching_mdp()\nprint(mdp)"),
        md("""
        ## 1. 手算一个确定性 MDP

        `start --short/2--> terminal`，而 `start --long/1--> long --任意动作/4--> terminal`。因此：

        `G(short)=2`，`G(long)=1+0.9×4=4.6`。

        注意 4 是第二步的奖励，不是第一步的 `start` 价值；奖励由转移发生时产生。若把终点误算成一个普通状态，就会凭空多出未来奖励。
        """),
        md("""
        ### 可编辑的奖励练习

        下面先只改第二步奖励 `r`，其余状态和转移保持不动。长路线胜过短路线的阈值满足 `1 + γr = 2`，所以 `r = 1/γ`；当 `γ=0` 时只看当前奖励，short 的 2 大于 long 的 1。
        """),
        code("""
        editable_second_step_reward = 4.0
        gamma = 0.9
        threshold = 1.0 / gamma
        from dataclasses import replace
        edited_mdp = replace(mdp, rewards={**mdp.rewards,
                             "long": {action: editable_second_step_reward for action in mdp.actions}})
        edited_values = {s: 0.0 for s in mdp.states}
        for _ in range(3):
            edited_values = bellman_optimality_update(edited_mdp, edited_values, gamma)
        long_return = 1 + gamma * editable_second_step_reward
        print({"threshold_r": threshold, "chosen_r": editable_second_step_reward,
               "short_return": 2.0, "long_return": long_return, "optimal_start_value": edited_values["start"]})
        assert np.isclose(edited_values["start"], max(2.0, long_return))
        assert np.isclose(1 + gamma * threshold, 2)
        assert 2.0 > 1.0  # gamma=0: short wins immediately
        """),
        code("""
        values = {s: 0.0 for s in mdp.states}
        history = []
        for _ in range(4):
            values = bellman_optimality_update(mdp, values, gamma=0.9)
            history.append(values.copy())
        print(history)
        assert np.isclose(history[-1]["start"], 4.6)
        assert np.isclose(discounted_return([1, 4], 0.9), 4.6)
        """),
        md("""
        ## 2. Bellman 更新和 Q-learning 的关系

        最优 Bellman 更新把所有动作的候选值取最大：`V(s)←max_a [r+γ E V(s')]`。固定策略时把 `max` 换成策略的动作/期望。下面仍然是同步更新：右边只读上一轮的表。

        Q-learning 不需要先知道转移表；它从一次实际样本 `(s,a,r,s')` 做增量更新：`Q←Q+α(target−Q)`，终止样本的 target 就是 `r`。这是学习估计，不是把一次样本冒充精确答案。
        """),
        code("""
        print("one Q update:", q_learning_update(0.0, 1.0, 4.0, learning_rate=0.5, gamma=0.9))
        assert np.isclose(q_learning_update(0, 2, 99, terminal=True), 0.2)
        fixed = bellman_policy_update(mdp, {s: 0 for s in mdp.states}, {"start": "long", "long": "short"})
        print("fixed-policy one-step values:", fixed)
        """),
        md("""
        ## 3. reward、return、value、policy 的检查表

        - reward：一步的结果，例如 `1` 或 `4`。
        - return：一条已采样轨迹的数值，例如 `[1,4]` 的 `G_0=4.6`。
        - value：在策略和环境随机性下，对 return 的期望；它不等于任意一条轨迹的 return。
        - policy：给定状态的动作分布；改变策略会改变后续状态分布，不能只把动作当静态标签。

        `reward_to_go[t]` 是每个动作对应的 `G_t`，也是下一课 REINFORCE 的权重。短有限轨迹结束时使用零 bootstrap；无限 continuing task 则需要价值估计来补上截断之后的尾部，两者目标不同。
        """),
        code("""
        rewards = [1.0, 4.0, 2.0]
        rtg = reward_to_go(rewards, gamma=0.9)
        print("reward-to-go:", rtg)
        assert np.allclose(rtg, [1 + .9*4 + .9**2*2, 4 + .9*2, 2])
        assert np.isclose(finite_horizon_returns([1, 4], .9)[-1], 4)
        assert np.isclose(finite_horizon_returns([1, 4], .9, bootstrap=10)[-1], 13)
        """),
        md("""
        ## 4. 练习与边界

        练习：把 `long` 的第二步奖励从 4 改为 0，找出两条策略的分界；再手算 `γ=0`。为一个随机转移加入两个 next state，比较期望 Bellman 更新和单次 Q-learning 样本。

        这张表没有车辆、延迟、连续油门，也没有证明任何道路性能。它只验证定义和数值关系。下一课把一个很小的动作空间接回真实 MetaDrive 闭环。
        """),
        code("""
        probability_continue = 0.5
        mixed = replace(mdp, transitions={**mdp.transitions, "start": {
            **mdp.transitions["start"], "long": ((probability_continue, "long"),
                                                (1 - probability_continue, "terminal"))}})
        values = {s: 0.0 for s in mixed.states}
        fixed_policy = {"start": "long", "long": "short"}
        for _ in range(3):
            values = bellman_policy_update(mixed, values, fixed_policy, gamma=0.9)
        expected_target = 1 + 0.9 * probability_continue * 4
        rng = np.random.default_rng(7)
        sampled_targets = np.where(rng.random(2000) < probability_continue, 4.6, 1.0)
        print("Bellman expectation:", values["start"], "sample targets:", sampled_targets[:8],
              "sample mean:", sampled_targets.mean())
        assert np.isclose(values["start"], expected_target)
        """),
        md("""
        随机练习答案：默认一半概率继续，另一半立即终止，固定long策略的价值为 `1+0.9×0.5×4=2.8`。
        一次Q-learning样本的target只能是 `4.6` 或 `1`；已知转移表的Bellman期望是 `2.8`。
        样本均值会波动，不能要求单次采样等于期望。现在修改 `probability_continue`，先手算再核对。
        """),
    ]


def lesson08():
    return [
        md("""
        # 08 · 用 REINFORCE 学一个驾驶速度选择

        这里学习的对象很窄：方向盘仍由固定几何控制器计算，策略每个 `decision_repeat=5` 的决策周期从 `{2,6} m/s` 中采样一个目标速度；这个目标只通过真实油门进入 `env.step`。所以结果应称为“固定转向之上的纵向选择”，不是完整自动驾驶策略。

        本课使用模拟器特权真值（privileged truth），没有接入第二单元的带噪测量或滤波器，以单独研究策略更新。五个观测特征是车道横向误差、朝向误差、速度和两个速度误差；奖励从实际 trace 的纵向进度、横向误差和失败标志重算。4步执行队列没有放入这五个特征，因此这是部分可观测过程上的无记忆策略；REINFORCE在这个受限策略类中优化回报。
        """),
        md(VISUAL),
        code(
            SETUP
            + "\nfrom dataclasses import replace\nfrom ad_tutorial.driving import DrivingConfig\nfrom ad_tutorial.rl_foundations import (SpeedChoicePolicy, FixedSpeedPolicy, run_policy_episode, trace_rewards, reinforce_loss, train_reinforce, save_checkpoint, load_checkpoint)"
        ),
        md("""
        ## 1. 先看动作链，不先看分数

        `observation → categorical policy → target speed → geometric throttle → run_episode → MetaDrive state`。`run_episode` 记录 issued command、delayed applied command 和下一状态。延迟 4 步且每步 0.1 秒时，前 0.4 秒执行器还在排队；策略依然每个决策周期根据当时测得状态重新采样。训练用随机 categorical policy，评测用 checkpoint reload 后的 deterministic argmax deployment；评测回报不能当作训练随机 `J` 的无偏估计。

        对 `J=E[Σ_t γ^t r_t]`，REINFORCE 使用 `Σ_t γ^t ∇ log π(a_t|s_t) G_t`。这里采用有限 horizon Monte Carlo，终点 bootstrap=0；每条 episode 的损失做求和，批量时再按 episode 数平均，不按各自长度除。`dist.sample()` 加 `log_prob` 是 score-function estimator；不要把离散速度选择写成 rsample 后又混合路径梯度，那会改变估计器含义。
        """),
        code("""
        config = DrivingConfig(seed=7, horizon=45, action_delay_steps=4, decision_repeat=5)
        policy = SpeedChoicePolicy(seed=7)
        episode = run_policy_episode(config, policy)
        print({"steps": len(episode.rewards), "actions": episode.actions[:8],
               "return": float(episode.rewards.sum()), "metrics": episode.result.metrics})
        assert len(episode.actions) == len(episode.rewards) == len(episode.log_probs)
        assert np.allclose(episode.rewards, trace_rewards(episode.result.trace))
        """),
        md("""
        ## 2. baseline 只减方差，不替换目标

        下面的 running mean 只使用先前轨迹的回报。对当前动作独立的 baseline 满足 `E_a[∇logπ(a|s)b]=b∇Σ_aπ(a|s)=0`，因此不改变期望梯度；合适的 baseline 可以减小方差，任意 baseline 未必更好。用当前轨迹自己的均值/标准差处理 return 会依赖当前动作，单轨迹时可能引入偏差；本课不采用它。
        """),
        md("""
        ### 手算 score-function 梯度

        对 `p=softmax([0,0])=[.5,.5]`，选择 action 0 时
        `∂log softmax_0/∂logits = [1-p0, -p1] = [.5,-.5]`。
        因而单步 loss `-2 log p0` 的梯度是 `[-1,+1]`。这个梯度只来自
        policy 的 log-prob；环境、reward 和 MetaDrive 没有路径梯度。
        """),
        code("""
        import torch
        from torch.distributions import Categorical

        # Analytic two-action check: logits=[0,0], sampled action 0, return 2.
        analytic_logits = torch.zeros(2, requires_grad=True)
        analytic_log_prob = Categorical(logits=analytic_logits).log_prob(torch.tensor(0))
        analytic_loss = reinforce_loss([analytic_log_prob], [2.0], gamma=1.0, baseline="none")
        analytic_loss.backward()
        print("analytic gradient:", analytic_logits.grad.tolist())
        assert torch.allclose(analytic_logits.grad, torch.tensor([-1.0, 1.0]))

        # An SGD step raises p(action=0) for this positive-return sample.
        before_probability = Categorical(logits=analytic_logits.detach()).probs[0].item()
        optimizer_logits = torch.optim.SGD([analytic_logits], lr=0.5)
        optimizer_logits.step()
        after_probability = Categorical(logits=analytic_logits.detach()).probs[0].item()
        print("p(action=0) before/after:", before_probability, after_probability)
        assert after_probability > before_probability

        loss = reinforce_loss(episode.log_probs, episode.rewards, baseline="running_mean", baseline_value=0.0)
        before = [p.detach().clone() for p in policy.parameters()]
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        print("loss", float(loss), "parameter changed", any(not torch.equal(a, b) for a, b in zip(before, policy.parameters())))
        """),
        md("""
        ## 3. 小规模训练与公平评测

        下面的 notebook 示例只跑一个独立模型，便于逐 cell 修改；CLI 会从 `--seed` 派生 3 个独立模型/采样 seed，并为每个保存训练前后 checkpoint。所有模型使用相同固定地图上的匹配初始偏移，最终再用未参与选择的 seed/offset 作为 holdout。报告 discounted objective、undiscounted reward sum、失败、距离和失败原因。一次短 CPU 运行若没有改善，就保留这个结果，解释高方差、延迟和有限样本，而不是挑一条好看的轨迹。
        """),
        code("""
        trained = train_reinforce(config, episodes=3, seeds=(7, 11, 19), checkpoint=ROOT / "artifacts" / "rl_foundations" / "lesson_policy.pt")
        print(trained.history)
        """),
        code("""
        reloaded = load_checkpoint(ROOT / "artifacts" / "rl_foundations" / "lesson_policy.pt")
        a = run_policy_episode(config, reloaded)
        b = run_policy_episode(config, reloaded)
        reloaded_actions = a.actions
        print("reloaded deterministic actions:", reloaded_actions[:10])
        assert reloaded_actions == b.actions
        """),
        md("""
        ## 4. 一个因素实验：只去掉执行延迟

        使用同一个已保存 checkpoint、同一个 seed/offset、同一个 horizon，只把 `action_delay_steps` 从 4 改成 0。运行前预测：队列不再隐藏旧动作，实际 throttle 更快跟随命令，进度和失败风险可能改变，但不能把这个单因素结果解释成泛化或训练胜利。记录 issued/applied command、实际 progress、undiscounted reward、discounted deployment return `G_0` 和 failure，再解释代价。
        """),
        code("""
        one_factor = DrivingConfig(seed=7, horizon=45, action_delay_steps=4, decision_repeat=5)
        delayed = run_policy_episode(one_factor, reloaded)
        no_delay = run_policy_episode(replace(one_factor, action_delay_steps=0), reloaded)
        def one_factor_report(ep):
            return {
                "delay_steps": ep.result.config["action_delay_steps"],
                "longitudinal_progress_m": ep.result.trace[-1]["longitudinal_m"] - ep.result.trace[0]["before_longitudinal_m"],
                "undiscounted_reward_sum": float(ep.rewards.sum()),
                "discounted_deployment_return": float(ep.returns[0]),
                "failure": ep.result.metrics["failure"],
                "first_issued_applied": [(ep.result.trace[0]["command_throttle"], ep.result.trace[0]["applied_throttle"])],
            }
        print("delay4:", one_factor_report(delayed))
        print("delay0:", one_factor_report(no_delay))
        """),
        md("""
        ## 5. 三个独立 seed 的结论表

        CLI 的最终表格必须逐行保留 model seed，并报告 `G_0`、undiscounted reward、distance 和 failure 的 learned−untrained delta。跨条件和 seed 的均值是等权汇总；没有要求每个 seed 都赢。下面的 cell 读取 CLI 产物，避免在 notebook 中挑选最好的一次运行。
        """),
        code("""
        import json
        evaluation_path = ROOT / "artifacts" / "rl_foundations" / "evaluation.json"
        if evaluation_path.exists():
            report = json.loads(evaluation_path.read_text(encoding="utf-8"))
            print("model_seed | mean ΔG0 | mean Δreward | mean Δdistance | mean Δfailure")
            for run in report["runs"]:
                deltas = run["paired_deltas"]["learned_minus_untrained"]
                n = len(deltas)
                mean = lambda key: sum(float(row[key]) for row in deltas) / n
                print(run["model_seed"], mean("discounted_deployment_return_delta"),
                      mean("undiscounted_reward_sum_delta"),
                      mean("distance_traveled_m_delta"), mean("failure_delta"))
        else:
            print("先运行 scripts/run_rl_foundations.py，再读取三 seed 结论表。")
        """),
        md("""
        ## 6. 结果解释和下一问

        这次实验的因果证据是：每个 categorical 选择改变连续 throttle，throttle 经 MetaDrive 动力学改变真实速度、进度和可能的失败标志。它没有学习 steering，也没有视觉、交互交通或路线规划；真值/测量接口和单一路段限制了泛化结论。

        REINFORCE 的优点是公式短、目标直接，代价是回报噪声大、样本效率低、更新对学习率和 seed 敏感。下一步问题是：如何用 value baseline、批量轨迹和 PPO 的 clipped objective 降低更新风险？先确认这里的 on-policy 数据和 reward-to-go 对齐，再进入 PPO。

        选读：[Sutton & Barto 在线教材](http://incompleteideas.net/book/the-book-2nd.html)；[OpenAI Spinning Up 的 Vanilla Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/vpg.html) 与 [policy optimization 推导](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)。
        """),
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for stem, cells in (
        ("07_mdp_bellman_returns", lesson07()),
        ("08_policy_gradient_driving", lesson08()),
    ):
        for index, cell in enumerate(cells):
            cell.id = f"{stem[:2]}-{index:02d}"
        nb = nbf.v4.new_notebook(
            cells=cells,
            metadata={
                "kernelspec": {
                    "name": "python3",
                    "language": "python",
                    "display_name": "Python 3.11",
                },
                "language_info": {"name": "python", "version": "3.11"},
            },
        )
        nbf.write(nb, OUT / f"{stem}.ipynb")
    print("built 2 RL foundations notebooks")


if __name__ == "__main__":
    main()
