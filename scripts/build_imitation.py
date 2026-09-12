"""Generate the two active unit 3 behavioural-cloning notebooks."""

from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "course" / "imitation"


VISUAL = """
<figure class="course-figure">
  <img src="../../assets/visuals/imitation-learning.png" width="1536" height="1024" style="max-width:100%;height:auto" alt="原理图：专家车辆的完整驾驶 episode 先划分为训练验证测试集合，再训练模型并放回闭环">
  <a href="../../assets/visuals/imitation-learning.png">查看原图</a>
  <figcaption><strong>AI 原理图 · 手算/机制示意</strong> · 完整专家 episode 先划分集合，再训练并放回闭环</figcaption>
</figure>

**因果链**：专家完整 episode → 按 episode 划分 train/validation/test → 当前状态到动作的监督训练 → 重载策略 → `env.step` 闭环。

**手算检查**：三条测试 episode 长度为 8、5、7，共同时间窗口长度是多少？
<details><summary>展开答案</summary><p>是 5，即最短 episode 的长度。共同窗口用于公平比较，完整轨迹和各自失败信息仍单独保留。</p></details>
"""


def md(value):
    return nbf.v4.new_markdown_cell(textwrap.dedent(value).strip())


def code(value):
    return nbf.v4.new_code_cell(textwrap.dedent(value).strip())


SETUP = r"""
from pathlib import Path
import sys
from dataclasses import replace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from IPython.display import display
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "src" / "ad_tutorial").is_dir())
sys.path.insert(0, str(ROOT / "src"))
from ad_tutorial.imitation import *
"""


def lesson05():
    return [
        md("""
        # 05 · 示范不是标签表：行为克隆怎样进入驾驶闭环？

        你已经训练过监督学习模型。本课把一个熟悉的训练循环接到真实的 MetaDrive 轨迹上：几何控制器访问状态并发出动作，车辆执行动作后产生下一状态；模型只学习当前状态到专家动作的映射。

        **前置课：Unit 02 状态估计。** 本课暂时使用模拟器 privileged truth 来隔离“监督训练和闭环”这个问题；它不是已经接入传感器估计的自动驾驶系统。上一单元的真值状态、参考车道和 `run_episode` 轨迹契约仍是本课的输入边界。

        目标是能解释三件事：输入从哪里来、episode 为什么先切分再展平、离线 MSE 为什么不能代替闭环评测。所有运行结果保存在 `artifacts/imitation/`，先收集数据，再训练，再重载 checkpoint。
        """),
        md(VISUAL),
        md("""
        ## 1. 先写数据契约

        每个样本的特征是 `(e_y, heading_error, speed, reference_bearing)`：横向误差、相对车道朝向、速度、车辆到前视参考点的方位角。动作是 `(steering, throttle)`，范围为 `[-1,1]`。

        这些字段描述控制时刻已经可见的状态和参考。动作、reward、terminal flag、执行后状态以及未来状态都不进入输入；它们可以用于评测，但不能成为输入泄漏。数据还保留 episode id、seed、初始偏移和完整配置。
        """),
        code(SETUP),
        code("""
        configs = default_split_configs(horizon=40)
        dataset = collect_demonstrations(configs)
        print(dataset.manifest())
        print("hash:", dataset_hash(dataset))
        assert set(FEATURE_NAMES) == {"e_y_m", "heading_error_rad", "speed_mps", "reference_bearing_rad"}
        assert not any("action" in name or "reward" in name or "next" in name for name in FEATURE_NAMES)
        """),
        md("""
        ## 2. 为什么不能按 timestep 随机切分？

        同一条轨迹相邻时刻的状态高度相关。若把一条 episode 的前半放训练、后半放测试，MSE 可能很漂亮，但测试状态仍是训练车辆刚刚访问过的局部邻域。这里先按完整 episode 分成 train、validation、test，再把各组展平。

        默认 train 偏移为 `±0.1、±0.2m`，validation 是独立 episode 的 `±0.25m`，test 是 `±0.4m`。seed 也分开；地图仍是固定的 `block_sequence/S`，所以 seed 变化用于复现实验条件和初始随机性，不能称为道路泛化。
        """),
        code("""
        train_ids = set(dataset.select("train").unique_episode_ids)
        val_ids = set(dataset.select("validation").unique_episode_ids)
        test_ids = set(dataset.select("test").unique_episode_ids)
        assert not train_ids & val_ids and not train_ids & test_ids and not val_ids & test_ids
        print({split: dataset.select(split).unique_episode_ids for split in ("train", "validation", "test")})
        """),
        md("""
        ## 3. 归一化和最小监督梯度

        均值和标准差只能从 train 计算。validation/test 可以使用这些统计量，但不能影响它们。下面的函数仍然是普通的 MSE + Adam；没有 reward，也没有策略梯度。
        """),
        code("""
        train, val, test = (dataset.select(name) for name in ("train", "validation", "test"))
        fit = train_behavioral_cloning(train, val, BCTrainingConfig(epochs=60, patience=12))
        print(fit.metrics)
        curve = pd.DataFrame(fit.history)
        display(curve.tail())
        curve.plot(x="epoch", y=["train_mse", "val_mse"], grid=True)
        plt.show()
        assert np.isfinite(curve[["train_mse", "val_mse"]].to_numpy()).all()
        """),
        md("""
        ## 4. 一条样本的可审计链

        下面只展开一条真实专家 episode：原始 `trace` → 四维 feature → train-only normalization → MLP prediction，并和该时刻的 expert command 并排显示。这样可以检查字段来源和数量级，而不是把整个过程藏在一次黑盒调用里。
        """),
        code("""
        episode = dataset.episodes[0]
        raw_trace = episode["trace"]
        one_features = features_from_trace(raw_trace)
        one_actions = actions_from_trace(raw_trace)
        one_normalized = fit.normalizer.transform(one_features)
        with torch.inference_mode():
            one_predictions = fit.model(torch.from_numpy(one_normalized)).numpy()
        audit = pd.DataFrame({
            "e_y_m": one_features[:, 0],
            "heading_error_rad": one_features[:, 1],
            "speed_mps": one_features[:, 2],
            "reference_bearing_rad": one_features[:, 3],
            "expert_steering": one_actions[:, 0],
            "model_steering": one_predictions[:, 0],
            "expert_throttle": one_actions[:, 1],
            "model_throttle": one_predictions[:, 1],
            "steering_error": one_predictions[:, 0] - one_actions[:, 0],
            "throttle_error": one_predictions[:, 1] - one_actions[:, 1],
        })
        for index, feature_name in enumerate(FEATURE_NAMES):
            audit[f"normalized_{feature_name}"] = one_normalized[:, index]
        print("episode:", episode["episode_id"], "config seed/offset:",
              episode["config"]["seed"], episode["config"]["initial_lateral_offset_m"])
        display(pd.DataFrame(raw_trace)[["before_x_m", "before_y_m", "before_heading_rad",
                                        "before_lateral_error_m", "before_speed_mps",
                                        "reference_x_m", "reference_y_m", "reference_heading_rad"]].head(5))
        display(audit.head(5))
        assert one_features.shape[1] == 4 and one_actions.shape[1] == 2
        """),
        md("""
        ## 5. 一个可运行的线性基线

        用 `e_y` 加常数列对两个 expert action 分量分别做闭式最小二乘。它只作为 validation 对照，不参与 MLP 的 checkpoint 选择；如果四特征 MLP 没有超过它，保留这个结果并先查数据和实验条件。
        """),
        code("""
        design_train = np.c_[np.ones(len(train.features)), train.features[:, 0]]
        design_val = np.c_[np.ones(len(val.features)), val.features[:, 0]]
        linear_coef = np.linalg.lstsq(design_train, train.actions, rcond=None)[0]
        linear_val_prediction = np.clip(design_val @ linear_coef, -1, 1)
        linear_val_mse = float(np.mean((linear_val_prediction - val.actions) ** 2))
        print({"e_y_only_linear_validation_mse": linear_val_mse,
               "four_feature_mlp_validation_mse": fit.metrics["validation_mse"]})
        """),
        md("""
        ## 6. checkpoint 是接口边界

        保存模型时也保存 train-only normalization、特征顺序和动作顺序。重新加载后的 policy 实现 `control(observation, reference)`；`driving.run_episode` 会把它的动作裁剪后传给 `env.step`，所以这一步才是从离线数组回到动力学的连接。
        """),
        code("""
        checkpoint = ROOT / "artifacts" / "imitation" / "lesson05_checkpoint.pt"
        save_checkpoint(checkpoint, fit)
        reloaded = load_checkpoint(checkpoint)
        a = fit.model(torch.from_numpy(fit.normalizer.transform(train.features))).detach().numpy()
        b = reloaded.model(torch.from_numpy(reloaded.normalizer.transform(train.features))).detach().numpy()
        assert np.allclose(a, b)
        print("checkpoint reload action max diff:", np.max(np.abs(a - b)))
        """),
        md("""
        ## 7. 练习与推理答案

        练习：把输入拆成只有 `e_y` 的线性模型；再把它和四特征 MLP 的 validation MSE 对照。预测哪个会更好，并说明理由。然后画出 test 的动作误差直方图。

        答案：只有横向误差无法区分“车身朝向已经偏了”和“车辆正在回正”的状态，也无法表达前视点的当前方位，因此通常会损失信息。四特征模型若 MSE 较小，只能说明在离线 test 状态上动作接近专家；它还没有证明自己在 rollout 中访问到的新状态上能恢复。若你的结果相反，保留结果并检查 seed、数据切分和字段来源。
        """),
        md("""
        选读：阅读 [DAgger](https://arxiv.org/abs/1011.0686) Introduction 中关于 learner-induced distribution shift 的段落。用本课的偏移量写出“专家访问的状态”和“模型自己访问的状态”各一个例子。这里不实现 DAgger；它是下一阶段的研究阅读方向。
        """),
    ]


def lesson06():
    return [
        md("""
        # 06 · 离线分数和闭环轨迹为什么会不一致？

        本课把同一个重载 checkpoint 放入真实 `env.step` 循环，并在完全相同的 held-out 初始条件下运行专家、未训练模型和行为克隆模型。你会同时看到 action MSE、车辆访问过的状态、终止原因、共同时间窗口和真实轨迹。

        **前置课：Unit 02 状态估计。** 本课仍明确标记 privileged truth 的范围：它用于隔离行为克隆的训练与闭环问题，不能代表视觉或状态估计已经解决。
        """),
        md(VISUAL),
        code(SETUP),
        md("""
        ## 1. 先确认你测的是什么

        offline MSE 的输入来自已保存 expert episode；closed-loop 的每个下一输入来自上一动作推动后的车辆状态。后者可能离开专家数据的窄分布。比较条件固定地图、时长、decision repeat、test offset 和 seed；test offset 为 `±0.4m`，未参与训练或模型选择。
        """),
        code("""
        configs = default_split_configs(horizon=60)
        data = collect_demonstrations(configs)
        fit = train_behavioral_cloning(data.select("train"), data.select("validation"), BCTrainingConfig(epochs=70))
        save_checkpoint(ROOT / "artifacts" / "imitation" / "lesson06_checkpoint.pt", fit)
        bc = load_checkpoint(ROOT / "artifacts" / "imitation" / "lesson06_checkpoint.pt")
        untrained = make_untrained_policy(bc.normalizer, seed=7, hidden_dim=32)
        print("offline held-out test:", offline_mse(bc.model, bc.normalizer, data.select("test")))
        """),
        code("""
        results = evaluate_closed_loop(configs["test"], bc, untrained_policy=untrained)
        report = summarize_closed_loop(results)
        display(pd.DataFrame({name: {
            "failure_rate": value["failure_rate"],
            "mean_abs_lateral_error_m": value["mean_abs_lateral_error_m"],
            "mean_distance_traveled_m": value["mean_distance_traveled_m"],
        } for name, value in report.items()}).T)
        """),
        md("""
        ## 2. 看 covariate shift，而不是猜测

        对每个 rollout 重新提取 `before_*` 特征。专家、untrained 和 BC 访问的 feature 均值/范围可能不同；这就是状态分布变化的可见证据。把它和最后的失败标志一起看，不要用一张 loss 曲线推断“模型学会了驾驶”。
        """),
        code("""
        rows = []
        for policy_name, episodes in results.items():
            for index, episode in enumerate(episodes):
                f = features_from_trace(episode.trace)
                rows.append({"policy": policy_name, "test_index": index,
                              "feature_mean_e_y": f[:, 0].mean(),
                              "feature_max_abs_e_y": np.abs(f[:, 0]).max(),
                              "steps": episode.metrics["steps"],
                              "outcome": episode.metrics["outcome"],
                              "failure_reason": episode.metrics["failure_reason"]})
        display(pd.DataFrame(rows))
        """),
        md("""
        ## 3. 共同窗口和失败解释

        如果一个策略提前失败，它没有和跑满 horizon 的策略经历相同长度。先报告终止原因和 distance，再只在 `min(steps)` 的共同窗口比较 lateral error。`horizon` 代表时间截断，不能写成到达；`failure` 需要以 MetaDrive 的 crash/out_of_road 等标志为证据。
        """),
        code("""
        common = paired_common_window(results)
        display(pd.DataFrame([
            {"policy": policy, "test_index": row["test_index"],
             "common_steps": row["common_steps"], **row["policies"][policy]}
            for row in common["per_config"]
            for policy in results
        ]))
        display(pd.DataFrame(common["aggregate"]).T)
        """),
        md("""
        ## 4. 同一模型的 in-distribution 对照

        这是一个可编辑的短实验：沿用刚刚重载的同一个 checkpoint，只把初始偏移改为训练覆盖过的 `±0.1m`，运行真实闭环。它和 held-out `±0.4m` 的结果分开标记；如果 in-distribution 更稳定，只说明状态覆盖不同，不能推出道路泛化。
        """),
        code("""
        in_distribution = [replace(config, initial_lateral_offset_m=float(np.sign(config.initial_lateral_offset_m) * 0.1))
                           for config in configs["test"]]
        in_distribution_results = evaluate_closed_loop(in_distribution, bc)
        display(pd.DataFrame([
            {"policy": name, "offsets": [-0.1, 0.1],
             "failure_rate": float(np.mean([episode.metrics["failure"] for episode in episodes])),
             "mean_steps": float(np.mean([episode.metrics["steps"] for episode in episodes]))}
            for name, episodes in in_distribution_results.items()
        ]))
        """),
        md("""
        ## 5. 结论模板：把实际 CLI 观察写进去

        完成 CLI 后，用下面的句式写结论，替换方括号中的实际值：

        `命令 [完整命令] 在 [collection_time_s] 秒收集 [samples] 个样本，训练 [training_time_s] 秒；dataset hash=[hash]。held-out offline test MSE=[mse]。完整闭环 outcome/failure 为 expert=[...]/[...]、untrained=[...]/[...]、BC=[...]/[...]；paired common-window 的 BC mean lateral error=[...]、mean distance=[...]。最早失败原因为 [...]，对应 trace/GIF 是 [...]。因此本次结果支持 [...]，不支持 [...]；下一个受控实验是 [...]。`

        只把 `horizon` 写成到达当 `arrive_dest=true`；如果是时间截断就写 `horizon`。比较时同时保留 full-outcome 和 paired common-window 两种数字。
        """),
        md("""
        ## 6. 可编辑练习与答案

        练习 A：把 test offset 改成 `±0.1m`，不要重训，比较 offline 与 closed-loop。你的预测是 covariate shift 变小还是仍然存在？

        练习 B：把 `horizon` 加倍，记录哪一个指标最先改变，并写出它对应的轨迹证据。

        答案：A 通常让状态更接近示范分布，但不能保证闭环一致；要以本次 rollout 的 feature 范围和失败标志判断。B 常见变化是 steps、distance 和 outcome；若模型较早漂移，延长 horizon 会暴露失败，而不是把之前的短时存活变成完成。不同机器或 MetaDrive 版本结果有差异，保留实际结果和配置。
        """),
        md("""
        选读 [DAgger](https://arxiv.org/abs/1011.0686)：bounded reading 只看 Introduction 和 algorithm idea。想象下一步实验：让学习者访问自己的状态，再让几何专家为这些状态给标签；写清楚新增数据来自哪里、哪些 test episode 仍必须保持 untouched。此处只讨论设计，没有把它冒充已完成的 DAgger 实验。
        """),
    ]


def write(name, cells):
    for index, cell in enumerate(cells):
        cell.id = f"{name[:2]}-{index:02d}"
    notebook = nbf.v4.new_notebook(
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
    nbf.write(notebook, OUT / f"{name}.ipynb")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write("05_demonstrations_and_bc", lesson05())
    write("06_offline_vs_closed_loop", lesson06())
    print("built 2 imitation notebooks")


if __name__ == "__main__":
    main()
