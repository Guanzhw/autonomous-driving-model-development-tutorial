"""Generate the deterministic Chinese notebooks for the state-estimation unit."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "course" / "state_estimation"


def md(source: str):
    return nbf.v4.new_markdown_cell(source)


def code(source: str):
    return nbf.v4.new_code_cell(source)


SETUP = """
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path.cwd()
while ROOT != ROOT.parent and not (ROOT / "src" / "ad_tutorial").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
from ad_tutorial.driving import DrivingObservation, DrivingConfig, run_episode
from ad_tutorial.estimation import (
    KalmanObserver, MeasurementConfig, RawMeasurementObserver, ScalarKalman,
    ego_to_world, make_observer, world_to_ego,
)
"""


def frames_measurements():
    return [
        md("""
# 03 · 坐标帧与测量：同一个点到底在哪里？

上一单元的 `before_*` 是模拟器真实状态。现在先建立一个可检查的边界：车辆姿态在 world frame，控制器拿到的是传感器给出的 `DrivingObservation`。本课不训练视觉模型，而是显式生成有偏差、有噪声的测量，方便把误差和因果链一一对上。

每周约 26 小时的学习节奏中，本课建议 2 小时读公式与手算，3 小时改噪声参数并解释图，剩余时间写一页“测量误差如何改变动作”的实验笔记。
"""),
        code(SETUP),
        md("""
## 1. SE(2) 世界帧与自车帧

自车位置为 `t=(x,y)`、朝向为 `θ`。点 `p_w` 到 ego frame 的变换是

`p_e = R(θ)^T (p_w - t)`，其中 `R(θ)=[[cosθ,-sinθ],[sinθ,cosθ]]`。

`x_e` 指车头方向，`y_e` 指左侧。逆变换为 `p_w=R(θ)p_e+t`。先手算：自车在 `(10,2)`、朝向 90°，世界点 `(10,5)` 应在 ego `(3,0)`。
"""),
        code("""
ego_position = (10.0, 2.0)
heading = np.pi / 2
point_world = (10.0, 5.0)
point_ego = world_to_ego(point_world, ego_position, heading)
print("world -> ego:", point_ego)
assert np.allclose(point_ego, (3.0, 0.0), atol=1e-10)
assert np.allclose(ego_to_world(point_ego, ego_position, heading), point_world)
"""),
        md("""
## 2. 测量模型：真值、偏差、随机噪声

我们写成 `z_k = x_k + b + v_k`：`b` 是固定偏差，`v_k ~ N(0,R)` 是每个采样独立的噪声。位置单位是 m，速度是 m/s，朝向是 rad；`MeasurementConfig` 保存各自带单位的 R/Q（Q 为每秒过程方差）、`seed` 和噪声参数。`lane_width_m` 是固定车道几何，因此不凭空加传感器误差。

同一 `seed` 的 raw 与 filter observer 每次 `reset()` 后会按同一顺序抽取噪声。两条闭环轨迹可能因动作不同而分叉，但第 k 行测量噪声仍有可比的 sample-index。进入闭环后，固定 lane 用 noisy world position 重新投影 `s/e_y`，因此不会把同一位置再独立加一份 lane 噪声。
"""),
        code("""
truth = DrivingObservation((10.0, 2.0), 0.1, 4.0, 10.0, 0.2, 0.0, 3.5)
sensor = MeasurementConfig(seed=23, position_bias_m=(0.0, 0.0),
                           speed_bias_mps=0.0)
raw = RawMeasurementObserver(sensor)
raw.reset(None)
samples = [raw.observe(truth, step, 0.1) for step in range(300)]
raw_rmse = np.sqrt(np.mean([(z.position[0] - truth.position[0]) ** 2 for z in samples]))
print(f"x-position raw RMSE = {raw_rmse:.3f} m; declared sigma = {sensor.position_noise_std_m[0]:.3f} m")
"""),
        md("""
固定偏差不会被随机滤波自动标定。保持 stationary truth 和同一个 noise seed，只加入 `+0.20m` 的 world-x bias；比较 raw 与 filter 的 signed mean error。filter 可以压低抖动，最终仍会围绕带偏差的位置。
"""),
        code("""
biased = MeasurementConfig(seed=23, position_bias_m=(0.20, 0.0),
                           heading_bias_rad=0.0, speed_bias_mps=0.0)
raw_biased = RawMeasurementObserver(biased)
filter_biased = KalmanObserver(biased)
raw_biased.reset(None); filter_biased.reset(None)
raw_samples = [raw_biased.observe(truth, step, 0.1) for step in range(300)]
filter_samples = [filter_biased.observe(truth, step, 0.1) for step in range(300)]
raw_signed = np.mean([z.position[0] - truth.position[0] for z in raw_samples[20:]])
filter_signed = np.mean([z.position[0] - truth.position[0] for z in filter_samples[20:]])
print(f"signed x bias: raw={raw_signed:.3f}m, filter={filter_signed:.3f}m")
assert abs(raw_signed - 0.20) < 0.05 and abs(filter_signed - 0.20) < 0.05
"""),
        md("""
## 3. 可编辑练习：先预测再运行

1. 把 `heading` 改为 0，世界点 `(13,2)` 的 ego 坐标是什么？
2. 只把 `position_noise_std_m` 的 x 分量从 0.18 改成 0.40。raw position RMSE 会如何变化？偏差是否会被更多样本平均掉？
3. 用 `world_to_ego` 与 `ego_to_world` 做 100 个随机点的逆变换测试。

答案：第 1 题是 `(3,0)`；第 2 题 RMSE 通常随噪声标准差增大，固定 bias 不会因取平均自动消失；第 3 题误差应在浮点精度内。不要把这些答案当作真实传感器标定结论。
"""),
        code("""
rng = np.random.default_rng(4)
round_trip_error = []
for _ in range(100):
    pose = rng.normal(size=2)
    angle = rng.uniform(-np.pi, np.pi)
    point = rng.normal(size=2)
    round_trip_error.append(np.linalg.norm(ego_to_world(world_to_ego(point, pose, angle), pose, angle) - point))
print("max inverse error:", max(round_trip_error))
"""),
        md("""
朝向是圆变量。实现会把新测量相对上一次估计的 innovation wrap 到 `[-π,π)`；因此 `+π−0.02` 到 `−π+0.02` 是约 0.04 rad 的小变化。CLI 的 heading RMSE 也使用同样的 wrapped error。
"""),
        code("""
boundary = MeasurementConfig(seed=2, position_noise_std_m=(0.0, 0.0),
                             heading_noise_std_rad=0.05, speed_noise_std_mps=0.0,
                             position_bias_m=(0.0, 0.0), heading_process_variance_rad2_per_s=0.0001)
crossing = KalmanObserver(boundary)
crossing.reset(None)
crossing.observe(replace(truth, heading_rad=np.pi - 0.02), 0, 0.1)
estimate = crossing.observe(replace(truth, heading_rad=-np.pi + 0.02), 1, 0.1).heading_rad
wrapped_error = (estimate - (-np.pi + 0.02) + np.pi) % (2*np.pi) - np.pi
print("wrapped heading error:", wrapped_error)
assert abs(wrapped_error) < 0.15
"""),
        md("""
## 4. 研究入口

TUMFTM 的 [Autonomous Driving Software Engineering](https://github.com/TUMFTM/Lecture_ADSE) 在 mapping/localization 章节给出 GNSS 与 Kalman filter 的工程练习；Simo Särkkä 与 Lennart Svensson 的 [Bayesian Filtering and Smoothing](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf) 是系统推导。阅读时只需回答：状态转移模型、测量模型、协方差各自描述什么？本课的标量随机游走模型做了哪些简化？
"""),
    ]


def closed_loop():
    return [
        md("""
# 04 · 闭环状态估计：滤波变好了吗？

本课把第 03 课的测量边界接进真实 MetaDrive 闭环。`run_episode` 仍把 `before_*` 保存为 truth，但 planner 与 controller 只能看到 `input_*`。比较 oracle、raw measurement、causal scalar Kalman 三组：同时报告测量 RMSE、横向误差、距离、终止原因和动作轨迹。

滤波使用一维 random-walk Kalman：`P⁻=P+Q·dt`，`K=P⁻/(P⁻+R)`，`x=x⁻+K(z-x⁻)`。它假定一个决策间隔内状态近似平稳，用带单位的 Q 留出运动余量。它是教学模型，不能直接当作车辆定位方案。
"""),
        code(SETUP),
        md("""
## 1. 两步手算：R、Q·dt、后验 P

例如 x 测量的 `σ=0.18m`，所以 `R=σ²=0.0324m²`；`Q=0.01m²/s`，`dt=0.1s`，一次预测增加 `Q·dt=0.001m²`。第一笔样本初始化状态，第二笔样本执行 `P⁻=P+Q·dt`，再算 `K`、后验状态和 `P=(1-K)P⁻`。每个物理量使用自己单位的 Q/R。
"""),
        code("""
R = 0.18 ** 2
Q_per_s = 0.01
dt = 0.1
kf = ScalarKalman(R, process_variance=Q_per_s)
x1 = kf.update(10.20, dt)
P1 = kf.variance
P_minus = P1 + Q_per_s * dt
K = P_minus / (P_minus + R)
x2 = kf.update(10.00, dt)
print({"R_m2": R, "Qdt_m2": Q_per_s * dt, "P_minus_m2": P_minus,
       "K": K, "x_after_two_m": x2, "P_posterior_m2": kf.variance,
       "P_formula_m2": (1 - K) * P_minus})
assert np.isclose(kf.variance, (1 - K) * P_minus)
"""),
        code("""
base = DrivingConfig(seed=7, horizon=120)
sensor = MeasurementConfig(seed=19)  # same noise, bias and Q as the default CLI
results = {mode: run_episode(base, observer=make_observer(mode, sensor))
           for mode in ("oracle", "raw", "filter")}

def rmse(result, estimate, truth):
    return float(np.sqrt(np.mean([(r[estimate] - r[truth]) ** 2 for r in result.trace])))

def signed_error(result, estimate, truth):
    return float(np.mean([r[estimate] - r[truth] for r in result.trace]))

for mode, result in results.items():
    print(mode, {
        "measurement_x_rmse_m": rmse(result, "measurement_x_m", "before_x_m"),
        "input_x_rmse_m": rmse(result, "input_x_m", "before_x_m"),
        "input_x_signed_error_m": signed_error(result, "input_x_m", "before_x_m"),
        "input_lateral_rmse_m": rmse(result, "input_lateral_error_m", "before_lateral_error_m"),
        "mean_abs_lateral_error_m": result.metrics["mean_abs_lateral_error_m"],
        "distance_m": result.metrics["distance_traveled_m"],
        "outcome": result.metrics["outcome"],
    })
print("required observation: default filter failure=", results["filter"].metrics["failure"],
      "filter outcome=", results["filter"].metrics["outcome"],
      "; random-walk filter lags moving x; Q tuning is a trade-off, not a repair")
"""),
        md("""
## 1. 看两条链：估计误差与真实驾驶结果

上图的真值曲线只用于评估，不能进入控制器；`measurement_*` 是 filter 看到的当前原始测量，`input_*` 才是控制器输入。默认 120 步配置中，filter 的 x-position RMSE 会高于原始 measurement，并在本固定场景提前失败；这正是缺少运动模型的反例。若 filter RMSE 下降但横向误差或距离变差，这是可以发生的：平滑带来滞后，噪声变小不等于闭环一定更好。
"""),
        code("""
fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
colors = {"oracle": "#222222", "raw": "#246b89", "filter": "#c74c3c"}
for mode, result in results.items():
    t = [r["time_s"] - result.metrics["decision_dt_s"] for r in result.trace]
    color = colors[mode]
    axes[0].plot(t, [r["before_longitudinal_m"] for r in result.trace], "--", color=color, alpha=.6, label=f"{mode} truth")
    axes[0].plot(t, [r["measurement_longitudinal_m"] for r in result.trace], ":", color=color, alpha=.8, label=f"{mode} measurement")
    axes[0].plot(t, [r["input_longitudinal_m"] for r in result.trace], color=color, label=f"{mode} estimate")
    axes[1].plot(t, [r["input_x_m"] - r["before_x_m"] for r in result.trace], color=color, label=mode)
    axes[2].plot(t, [r["command_steering"] for r in result.trace], color=color, label=mode)
axes[0].set(xlabel="Time / s", ylabel="longitudinal s / m", title="Truth, measurement and estimate")
axes[1].set(xlabel="Time / s", ylabel="signed x error / m", title="Estimate minus truth")
axes[2].set(xlabel="Time / s", ylabel="steering command")
for ax in axes:
    ax.grid(alpha=.2); ax.legend()
plt.show()
"""),
        md("""
## 2. 因果性与同噪声检查

观测器接口是 `observe(truth, step, dt)`。闭环会在 reset 后提供固定 lane，使 noisy world pose 能投影成 `s/e_y`；它只收到当前行 truth，没有历史轨迹参数，也没有 future truth。filter 的内部状态只由过去的 measurement 更新。下面的 probe 记录每次调用，验证调用顺序、`reset()` 次数和控制输入确实变化。
"""),
        code("""
class ProbeObserver(RawMeasurementObserver):
    def __init__(self, config):
        self.calls = []
        self.reset_calls = 0
        super().__init__(config)
    def reset(self, lane=None):
        self.reset_calls += 1
        super().reset(lane)
    def observe(self, truth, step, dt):
        self.calls.append((step, dt, truth.lane_lateral_m))
        return super().observe(truth, step, dt)

probe = ProbeObserver(sensor)
probe_result = run_episode(DrivingConfig(seed=7, horizon=8), observer=probe)
assert probe.reset_calls == 2  # constructor + exactly one run reset
assert [call[0] for call in probe.calls] == list(range(len(probe_result.trace)))
assert not np.allclose([r["input_lateral_error_m"] for r in probe_result.trace],
                       [r["before_lateral_error_m"] for r in probe_result.trace])
print("reset calls:", probe.reset_calls, "observe calls:", len(probe.calls))
"""),
        md("""
## 3. 可编辑练习：不要追求单一赢家

1. 固定 `seed=19`，把位置噪声和 lateral 噪声分别放大 2 倍；预测 raw/filter 的 RMSE 和驾驶指标如何变化。
2. 固定噪声，把有 m²/s 单位的 `position_process_variance_m2_per_s` 改成 `0.0` 与 `0.10`。哪个更容易滞后？用 input 曲线与 command 曲线解释；朝向和速度的 Q 使用各自带单位字段。
3. 让初始偏移变成 `-0.4m`，不调增益，比较三组的失败原因。

参考答案：更大的 R 通常让 raw 更抖，filter 可能更依赖历史；更大的 Q 提高跟随新测量的速度、减少滞后，但会放进更多噪声。没有哪组在所有指标上必胜；必须保留相同 seed、噪声配置和完整 trace，分别报告 RMSE 与真实驾驶结果。短时跑满 horizon 也不等于到达终点。
"""),
        code("""
changed = replace(sensor, position_process_variance_m2_per_s=0.10)
fast_filter = run_episode(base, observer=KalmanObserver(changed))
print("Q=0.10 filter:", fast_filter.metrics["mean_abs_lateral_error_m"],
      "lateral RMSE:", rmse(fast_filter, "input_lateral_error_m", "before_lateral_error_m"))
"""),
        md("""
## 4. 从课程机制到研究问题

本单元只做仿真机制实验：真值用于合成测量并评估，固定单车道与几何控制器。下一步可研究速度/转弯模型、异步传感器、观测丢失、真实地图标定，再考虑学习型 perception。不要因为一条滤波曲线更平滑就声称定位更可靠；先问状态、坐标、延迟、协方差和失效证据是否匹配。

CLI 会把 `oracle_seed*.json/csv/png`、`raw_*`、`filter_*`、`summary.json` 和比较图保存到 `artifacts/state_estimation/`，便于从轨迹重算指标。
"""),
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, cells in (
        ("03_frames_measurements", frames_measurements()),
        ("04_state_estimation_closed_loop", closed_loop()),
    ):
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
    print("built 2 state-estimation notebooks")


if __name__ == "__main__":
    main()
