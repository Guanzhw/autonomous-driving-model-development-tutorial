"""Generate active lessons without importing or executing the simulator."""

from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "course" / "first_loop"


def md(source):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


SETUP = """
from pathlib import Path
from dataclasses import replace
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
            if (p / "src" / "ad_tutorial").is_dir())
sys.path.insert(0, str(ROOT / "src"))
from ad_tutorial.driving import DrivingConfig, run_episode, save_episode
OUTPUT = ROOT / "artifacts" / "first_loop"
"""


def lesson1():
    return [
        md("""
        # 01 · 车为什么能跟住一条线？

        你训练过网络，可能习惯了“输入—预测—loss”。驾驶多了一件事：**本次输出会改变下次输入。**
        向左打方向后，位置和朝向改变；下一次控制必须使用新的状态。这就是反馈闭环。

        本课目标是独立解释一条实际轨迹：参考从哪里来，动作如何计算，车辆为什么偏离又回来。
        假设会 Python、数组和基础三角函数；所需几何在下面解释，先不引入 RL。学习安排见 [单元说明](README.md)。
        """),
        md("""
        ## 1. 把问题拆成五步

        | 步骤 | 本课实际对象 | 应检查的关系 |
        |---|---|---|
        | 观察 | 自车位置、朝向、速度 | 输入状态与参考路线不同 |
        | 选参考 | 固定车道中心线前方的点 | 想去哪里，与现在在哪里不同 |
        | 控制 | 误差 → 转向、油门/制动 | 改参考或增益，命令应改变 |
        | 执行 | `env.step(applied_action)` | MetaDrive 推进仿真动力学 |
        | 再观察 | 执行后的状态 | 反馈到下一步，并写入轨迹 |

        本课直接读取模拟器真值状态。MetaDrive 返回的数值观测向量没有输入这个控制器；
        相机、检测和状态估计在后续单元接入。当前验证的是几何控制机制。

        道路固定为长直道、单车道、无交通。参考车道在 reset 时固定，偏移不会让目标切换到相邻车道。
        这是研究控制与延迟的实验条件，尚未包含路口、变道或交互决策。
        """),
        code(SETUP),
        md("""
        ## 2. 从坐标得到误差

        世界坐标 `(x,y)` 描述位置；车道坐标 `(s,e_y)` 描述“沿中心线走了多远”和“偏向哪一侧”。
        本课直道沿世界 `+x` 延伸。MetaDrive 横向坐标 **右侧为正**，所以这里 `e_y = y_center - y_ego`。
        正转向使车向左转。朝向 theta 用弧度，180°=π rad；车旋转后世界 x 轴不会跟着旋转。

        取车道上前方 L 米的点，车辆看向它的角度为：

        `bearing = atan2(y_ref - y_ego, x_ref - x_ego)`

        `e_heading = wrap(theta_ego - bearing)`

        wrap 把角度差映射到 `[-π,π)`，避免跨过 ±π 时跳变。bearing 是**车到参考点的方向**；
        车道切线方向在这条直道上恒为 0，两者不总相等。

        本课采用容易手算的反馈基线：

        `steering = clip(k_y * e_y - k_heading * e_heading, -1, 1)`

        `throttle = clip(k_speed * (v_target - v), -1, 1)`

        动作无量纲，不是方向盘角度或加速度。k_y 单位 1/m，k_heading 单位 1/rad，k_speed 单位 s/m。
        速度用 m/s，6 m/s=21.6 km/h。正油门加速，负值制动。
        这是几何反馈启发式，不是标准 Pure Pursuit 或 Stanley 的完整实现。
        """),
        md("""
        ## 3. 先预测第一步，再运行

        车辆在中心线右侧 0.5 m，朝向与直道一致，前视距离 8 m：
        bearing=atan2(0.5,8)≈0.06242 rad，heading error≈−0.06242 rad。
        在 k_y=0.32、k_heading=0.85 下，转向约 **+0.21306**；静止起步的油门为 **1**。

        先写下：偏移换成 −0.5 m，第一步转向是什么符号？k_y 翻倍，误差会在所有时刻减半吗？
        区分“第一步命令容易预测”和“整段动力学需要实验”。
        """),
        code("""
        predicted = np.clip(0.32 * 0.5 + 0.85 * np.arctan2(0.5, 8), -1, 1)
        print("手算第一步转向:", predicted)
        config = DrivingConfig()
        result = run_episode(config)
        files = save_episode(result, OUTPUT, "lesson1_baseline")
        print(result.metrics)
        assert np.isclose(result.trace[0]["command_steering"], predicted, atol=1e-5)
        display(pd.DataFrame(result.trace).head(6)[[
            "step", "time_s", "before_lateral_error_m", "command_steering",
            "applied_steering", "lateral_error_m", "speed_mps"]])
        display(Image(filename=str(files["plot"])))
        """),
        md("""
        ## 4. 用轨迹检查因果关系

        每行保存 before_*（控制输入）、reference_*（目标）、command_*（发出）、applied_*（执行）和
        无 before 前缀的状态（执行后）。第 t 行的执行后状态应等于第 t+1 行的输入。
        初始状态、固定车道、车身尺寸、有效采样间隔和依赖版本保存在 JSON 配置中。

        一步包含 5 次、每次 0.02 秒的物理推进，dt=0.1秒。首行输入在 t=0，执行后状态在 t=0.1秒。
        动作经动力学才逐渐改变位置，不会把车辆立即传送到参考点。
        """),
        code("""
        rows = result.trace
        r = rows[0]
        bearing = np.arctan2(r["reference_y_m"] - r["before_y_m"], r["reference_x_m"] - r["before_x_m"])
        heading_error = (r["before_heading_rad"] - bearing + np.pi) % (2*np.pi) - np.pi
        recomputed = np.clip(config.lateral_gain * r["before_lateral_error_m"]
                             - config.heading_gain * heading_error, -1, 1)
        assert np.isclose(recomputed, r["command_steering"], atol=1e-6)
        assert np.allclose([r["x_m"] for r in rows[:-1]], [r["before_x_m"] for r in rows[1:]])
        errors = np.array([r["lateral_error_m"] for r in rows])
        assert np.isclose(np.mean(np.abs(errors)), result.metrics["mean_abs_lateral_error_m"])
        hits = np.flatnonzero(np.abs(errors) < 0.1)
        print("首次 |误差| < 0.1m 的采样时间:", rows[hits[0]]["time_s"] if hits.size else "本次未达到")
        """),
        md("""
        ## 5. 一次只改一个条件

        每次先保存一句预测，再运行：

        1. **方向**：初始偏移改为 −0.5 m，比较第一步转向和曲线方向。
        2. **参考**：保持 +0.5 m，只把前视距离从 8 m 改为 2 m，比较第一步动作、最大误差和终止原因。
        3. **迁移**：不再调参，改为 +0.4 m 或 −0.4 m，能否用同一公式解释？

        下面是可修改的模板；给每组使用不同文件名以保留证据。
        """),
        code("""
        changed_config = replace(config, initial_lateral_offset_m=-0.5)
        changed = run_episode(changed_config)
        save_episode(changed, OUTPUT, "lesson1_negative_offset")
        display(pd.DataFrame([result.metrics, changed.metrics], index=["+0.5m", "-0.5m"])[[
            "steps", "mean_abs_lateral_error_m", "distance_traveled_m", "outcome"]])
        """),
        md("""
        <details><summary>推理与答案：先预测再展开</summary>

        负偏移时车在中心线左侧，转向应变负。第一步有对称关系；后续轮胎动力学、边界和数值误差需运行后判断，
        不要求轨迹逐位完全镜像。L=2 时 bearing≈0.24498 rad，第一步转向≈0.36823，比 L=8 更大。
        更强修正可能更快回正，也可能过冲，不能直接推出“看得近更好”。

        增益翻倍直接改变命令的一部分；饱和、heading 项和反馈都会影响后续轨迹。平均误差不会按比例减半。
        </details>

        ## 6. 连接到更大的系统

        位置和朝向若来自传感器估计，控制器就要面对误差；参考若由规划器给出，就要考虑道路、障碍和车辆约束。
        下一课保留真值输入，只给执行通道增加延迟。

        选读 [TUM 第 08 节 Control](https://github.com/TUMFTM/Lecture_ADSE/tree/master/08_control/practice)，限 60–90 分钟：
        找出目标速度、当前速度、动作和反馈，逐项对应本课速度环。[后续教材](../../reference/README.md)。
        """),
    ]


def lesson2():
    return [
        md("""
        # 02 · 动作晚到之后，怎样定位失败并尝试恢复？

        本课只改变命令到执行之间的时间，观察反馈何时失效，再在**相同延迟和初始状态**下尝试降低目标速度。
        你要给出一个有对照、有适用范围的结论。先完成 [上一课](01_drive_and_observe.ipynb) 的手算与复核。
        """),
        code(SETUP),
        md("""
        ## 1. 延迟不是随机噪声

        第 t 步发出 u[t]，延迟 d 步后实际执行 u[t−d]；控制器仍用当前状态算新命令。
        队列未填满时执行 `[0,0]`。d=4、dt=0.1秒，对应 **0.4秒**：

        | 执行区间开始 | 发出 | 执行 |
        |---|---|---|
        | 0.0s | u[0] | [0,0] |
        | 0.1s | u[1] | [0,0] |
        | 0.3s | u[3] | [0,0] |
        | 0.4s | u[4] | u[0] |

        原本纠正右偏的左转命令，晚到时车辆可能已经接近中心，继续左转就可能过冲。
        本实验同时延迟转向和油门，初始等待也改变速度过程。若要单独研究转向延迟，应另外设计执行器和对照。
        """),
        md("""
        ## 2. 三组条件与预测

        | 组别 | 道路、偏移、种子、时长 | 延迟 | 目标速度 |
        |---|---|---|---|
        | baseline | 相同 | 0 步 | 6 m/s |
        | delay | 相同 | 4 步 | 6 m/s |
        | recovery | 相同 | 4 步 | 2 m/s |

        baseline→delay 只改变延迟，delay→recovery 只改变目标速度。“恢复方案”是重新运行同一失败条件时的减速策略，
        并非检测到碰撞后倒退回安全状态，也不保证能应对任意延迟。

        先写下：延迟可能怎样改变第一次转向、回正时间和失败标志？减速若减少过冲，会付出什么代价？
        """),
        code("""
        base = DrivingConfig()
        configs = {"baseline": base, "delay": replace(base, action_delay_steps=4),
                   "recovery": replace(base, action_delay_steps=4, target_speed_mps=2.0)}
        results, saved = {}, {}
        for name, cfg in configs.items():
            results[name] = run_episode(cfg)
            saved[name] = save_episode(results[name], OUTPUT, name, failure_gif=True)
        columns = ["steps", "elapsed_s", "distance_traveled_m", "mean_abs_lateral_error_m",
                   "max_abs_lateral_error_m", "failure_reason", "outcome", "route_completion"]
        display(pd.DataFrame({n: r.metrics for n, r in results.items()}).T[columns])
        """),
        md("""
        ## 3. 先核对动作链，再解释成绩

        若 delay 第4行实际动作不等于第0行命令，先检查实现，再讨论延迟效果。
        横向误差测量执行后的车辆中心；出界按模拟器车身/路面接触等规则判定，两者测量对象不同。
        """),
        code("""
        delayed = results["delay"]
        d = configs["delay"].action_delay_steps
        commands = np.array([[r["command_steering"], r["command_throttle"]] for r in delayed.trace])
        applied = np.array([[r["applied_steering"], r["applied_throttle"]] for r in delayed.trace])
        assert np.allclose(applied[:d], 0)
        assert np.allclose(applied[d:], commands[:-d])
        display(pd.DataFrame(delayed.trace).head(9)[[
            "step", "time_s", "before_lateral_error_m", "command_steering", "applied_steering", "lateral_error_m"]])
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
        for name, result in results.items():
            frame = pd.DataFrame(result.trace)
            axes[0].plot(frame.time_s, frame.lateral_error_m, label=name)
            axes[1].plot(frame.time_s, frame.speed_mps, label=name)
        axes[0].set(ylabel="Lateral error / m")
        axes[1].set(xlabel="Time / s", ylabel="Speed / m/s")
        for ax in axes:
            ax.legend()
            ax.grid(alpha=0.2)
        plt.show()
        """),
        md("""
        ## 4. 失败现场：车的中心不是整辆车

        矩形车身长度 a、宽度 b、朝向 theta 时，横向包络半宽约为
        `a/2 * abs(sin(theta)) + b/2 * abs(cos(theta))`。
        所以车中心离中心线不远，斜着行驶的前角也可能先碰连续线。这是几何近似；确切终止原因以本次接触标志为证据。

        下方显示失败最后几行，播放基于真实轨迹和车身尺寸的俯视回放。GIF 是轨迹可视化，不是相机视频。
        参数改变后若没有失败，就如实报告该次未出现失败。
        """),
        code("""
        for name, result in results.items():
            if result.metrics["failure"]:
                print(name, result.metrics["failure_reason"])
                display(pd.DataFrame(result.trace).tail(4)[[
                    "time_s", "lateral_error_m", "heading_rad", "on_lane",
                    "on_white_continuous_line", "on_yellow_continuous_line", "crash_sidewalk"]])
                display(Image(filename=str(saved[name]["failure_gif"])))
        """),
        md("""
        ## 5. 一个公平但有限的比较

        baseline 走了更长时间，delay 可能提前失败，recovery 又可能因速度低走得短。先报告终止原因与距离，
        再比较共同时间窗口中的误差。共同窗口避免平均值直接受不同时长影响，但仍不是同路程/同完成度比较。

        outcome=arrived 表示到达，failure 表示触发失败，horizon 表示跑满时长。本课18秒内未到终点很正常。
        跑满时长且未失败，只支持“这段试验内未触发失败”。
        route_completion 是模拟器整条路线的累计位置比例；本课从直段中部起步，因此其中包含起点之前的路线，
        不能把它直接当成本次行驶完成的比例。本次运动量看 distance_traveled_m。
        """),
        code("""
        common_steps = min(len(r.trace) for r in results.values())
        common = {}
        for name, result in results.items():
            rows = result.trace[:common_steps]
            common[name] = {"common_time_s": rows[-1]["time_s"],
                "mean_abs_error_m": np.mean([abs(r["lateral_error_m"]) for r in rows]),
                "distance_m": sum(r["distance_m"] for r in rows)}
        display(pd.DataFrame(common).T)
        """),
        md("""
        ## 6. 独立实验与研究笔记

        固定其余条件，只改延迟为1、2、8、12步；使用独立文件名保存，不只保留好看的曲线。再选未调参的初始偏移（如−0.4m），
        检查减速方案能否迁移。道路固定且无交通，更换 seed 不等于泛化到新道路；偏移变化才是这里的具体分布变化。

        一页笔记按此顺序交付：问题 → 运行前预测 → 唯一改动 → 配置/轨迹位置 → 完整结果 → 失败片段 →
        机制解释 → 结论范围 → 下一项对照。预测不符时，写出仍待排查的因素。

        <details><summary>推理与答案</summary>

        延迟使动作基于过时状态；减速通常缩短相同延迟期间的位移，可能减少过冲，但不是稳定性证明。
        减速降低进度，短时存活不足以证明任务完成。本次固定条件下 delay=12、target_speed=2 仍会失败；改变起点后要重新验证。

        不能用 recovery 没出界单独证明有效，必须保留相同偏移、延迟、道路和时长的原策略对照。
        不能给 recovery 去掉延迟或换容易的起点。平均误差小却提前失败，先看时间与接触标志。
        最后一帧中心误差不大时，检查车身朝向、尺寸和车道边界；保留所有失败标志。
        </details>

        ## 7. 与研究脉络的连接

        后续可用几何控制器的状态—动作示范做行为克隆；但离线动作误差小，不代表策略在自己造成的新状态上能恢复。
        DAgger 讨论这种状态分布变化。RL 再通过交互和奖励优化策略，仍面对执行、延迟和评测问题。

        选读 [DAgger](https://arxiv.org/abs/1011.0686) 的 Introduction，限45分钟：用偏离—反馈例子解释
        “专家访问的状态”和“学习者访问的状态”为何不同。论文结论与本课规则控制实验分别是什么证据？类比不等于论文复现。
        """),
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, cells in [
        ("01_drive_and_observe", lesson1()),
        ("02_delay_and_recovery", lesson2()),
    ]:
        for index, cell in enumerate(cells):
            cell.id = f"{name[:2]}-{index:02d}"
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
        nbf.write(nb, OUT / f"{name}.ipynb")
    print("built 2 active notebooks")


if __name__ == "__main__":
    main()
