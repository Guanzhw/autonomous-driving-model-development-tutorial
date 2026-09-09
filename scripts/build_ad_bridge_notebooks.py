"""Build the AD-domain bridge notebooks.

These notebooks assume the learner already knows Python and deep learning.
They introduce autonomous-driving vocabulary, interfaces, data geometry and
evaluation before the existing mechanism-heavy notebooks.
"""

from __future__ import annotations

import json
import textwrap
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def source(text: str) -> str:
    return textwrap.dedent(text).strip() + "\n"


def cell(kind: str, text: str, cell_id: str) -> dict:
    result = {
        "cell_type": kind,
        "id": cell_id,
        "metadata": {},
        "source": source(text),
    }
    if kind == "code":
        result["execution_count"] = None
        result["outputs"] = []
    return result


def make_notebook(name: str, title: str, cells: list[tuple[str, str]]) -> None:
    notebook = {
        "cells": [
            cell(kind, text, uuid.uuid5(uuid.NAMESPACE_URL, f"ad-bridge:{name}:{index}").hex[:12])
            for index, (kind, text) in enumerate(cells)
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "title": title,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NOTEBOOK_DIR / name).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def build() -> None:
    make_notebook(
        "00a_autonomous_driving_system_overview.ipynb",
        "00A Autonomous Driving System Overview",
        [
            ("markdown", """
            # 00A · 智能驾驶系统全景：你到底在开发什么模型？

            本项目假设你已经具备 Python 和深度学习基础。本 notebook 不讲反向传播或如何调用 PyTorch，而是先建立自动驾驶领域的共同语言：`ego`、`actor`、`scene`、`ODD`、sensor、perception、tracking、prediction、planning、control 和 safety envelope。

            如果没有这张地图，后面的 BEV、tracking 或 Transformer 很容易变成孤立的模型名。本节的目标是：看到一个岗位描述或一段模型代码时，知道它位于哪条系统接口上、输入输出是什么、错误会如何传到下游。

            **完成后你应该能回答：**

            - ego vehicle、other actor、scene、scenario 有什么区别？
            - perception 输出为什么不能直接等同于 planning 输入？
            - open-loop、closed-loop 和 safety monitor 分别观察什么？
            - 为什么 L4 讨论必须同时出现 ODD、fallback 和 scenario regression？
            """),
            ("markdown", """
            ## 1. 用接口而不是模型名理解系统

            | 层 | 典型输入 | 典型输出 | 失败会影响什么 |
            |---|---|---|---|
            | Sensor / calibration | 原始图像、点云、雷达、定位、时间戳 | 对齐后的 sensor bundle | 坐标错、时间错、数据缺失 |
            | Perception | sensor bundle | object、lane、occupancy、map element | 漏检、误检、位置/速度不准 |
            | Tracking / state estimation | 多帧 observation、ego-motion | 带 ID 的 agent state | ID switch、延迟、漂移 |
            | Prediction | agent state、map、traffic context | 多模态 future trajectories | 未来分布漏掉关键模式 |
            | Planning | scene state、route、constraints | ego trajectory / maneuver | 碰撞、越界、不可执行 |
            | Control | planned trajectory、vehicle state | steering、acceleration、brake | 跟踪误差、舒适性、稳定性 |
            | Safety / runtime | health、confidence、TTC、latency | continue、degrade、minimal risk | 需要 fallback 或 ODD exit |

            这些层不一定对应独立的神经网络；现代系统可能把多个层放进一个 learned model，但接口、指标和失效分析仍然存在。
            """),
            ("code", """
            from dataclasses import dataclass

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd

            @dataclass
            class Module:
                name: str
                input_contract: str
                output_contract: str
                evidence: str

            modules = [
                Module("perception", "sensor bundle", "objects / occupancy", "detection + geometry"),
                Module("tracking", "multi-frame objects", "agent state + ID", "association + temporal error"),
                Module("prediction", "agent state + map", "future trajectories", "ADE/FDE + miss rate"),
                Module("planning", "scene + route + constraints", "ego trajectory", "collision + progress"),
                Module("control", "trajectory + vehicle state", "actuation", "tracking + comfort"),
                Module("safety", "health + uncertainty + TTC", "continue / degrade", "recall + response latency"),
            ]
            interface_table = pd.DataFrame([m.__dict__ for m in modules])
            display(interface_table)
            """),
            ("code", """
            # 一个最小 scene：ego 在原点，其他 actor 在 ego 坐标系中表达。
            ego = np.array([0.0, 0.0])
            actors = pd.DataFrame([
                {"actor_id": "lead", "x_m": 18.0, "y_m": 0.2, "speed_mps": 7.5, "kind": "vehicle"},
                {"actor_id": "cut_in", "x_m": 12.0, "y_m": 2.8, "speed_mps": 5.0, "kind": "vehicle"},
                {"actor_id": "pedestrian", "x_m": 9.0, "y_m": -3.2, "speed_mps": 1.4, "kind": "vulnerable"},
            ])

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.scatter(*ego, s=120, marker="^", label="ego")
            for _, actor in actors.iterrows():
                ax.scatter(actor.x_m, actor.y_m, s=80, label=f"{actor.actor_id} ({actor.kind})")
                ax.annotate(actor.actor_id, (actor.x_m, actor.y_m), xytext=(5, 5), textcoords="offset points")
            ax.axhline(0, color="gray", linewidth=0.8)
            ax.set(xlabel="ego-forward x / m", ylabel="ego-left y / m", title="A scene is more than an image")
            ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
            plt.tight_layout()
            """),
            ("code", """
            from ipywidgets import FloatSlider, IntSlider, interact

            def inspect_scene(sensor_dropout=0.0, perception_noise=0.2, planner_delay_ms=120):
                visible = np.random.default_rng(4).random(len(actors)) > sensor_dropout
                measured = actors.loc[visible].copy()
                measured["x_m"] += np.random.default_rng(5).normal(0, perception_noise, len(measured))
                measured["y_m"] += np.random.default_rng(6).normal(0, perception_noise, len(measured))
                end_to_end_ms = 80 + planner_delay_ms + 60
                print(f"visible actors: {len(measured)}/{len(actors)}")
                print(f"toy sensor→perception→planning→control budget: {end_to_end_ms} ms")
                print("system question:", "degrade / minimal risk" if sensor_dropout > 0.35 or end_to_end_ms > 250 else "continue")

            interact(
                inspect_scene,
                sensor_dropout=FloatSlider(min=0, max=0.8, step=0.05, value=0.0, description="dropout"),
                perception_noise=FloatSlider(min=0, max=1.5, step=0.1, value=0.2, description="noise / m"),
                planner_delay_ms=IntSlider(min=20, max=300, step=10, value=120, description="planner ms"),
            )
            """),
            ("markdown", """
            ## 2. 领域检查点

            1. 如果一个模型输出 BEV occupancy，它属于 perception 还是 planning？说明你的接口判断。
            2. 如果同一个行人连续三帧被检测到但 ID 不一致，为什么这不是单纯的 detection accuracy 问题？
            3. 把 `sensor_dropout` 改到高值，应该由哪个模块决定是否继续运行？为什么不能只看模型 softmax confidence？

            **下一步**：`00b` 进入传感器、坐标系和时间契约；然后再进入 `01` 的 SE(3)/标定/投影实现。
            """),
        ],
    )

    make_notebook(
        "00b_sensors_frames_and_time.ipynb",
        "00B Sensors Frames and Time",
        [
            ("markdown", """
            # 00B · 传感器、坐标系与时间：自动驾驶模型的输入为什么容易错？

            深度学习模型通常把输入写成 tensor，但自动驾驶的 tensor 有三个经常被忽略的语义：**它来自哪个传感器、在哪个坐标系、对应哪个时间**。这三件事错一个，模型可能仍然能运行，却学到错误的几何关系。

            本节只做领域引入。你不需要先掌握完整的 SE(3) 群论；先理解 frame、pose、extrinsic、ego-motion、timestamp offset 和 interpolation，下一篇 `01` 再实现矩阵细节。

            | 名词 | 本教程中的工作定义 |
            |---|---|
            | ego frame | 以自车为原点、约定前/左/上方向的坐标系 |
            | sensor frame | 相机、LiDAR、radar 各自的测量坐标系 |
            | world/map frame | 相对稳定的地图或世界参考系 |
            | extrinsic | sensor frame 与 ego frame 之间的安装变换 |
            | ego-motion | 自车在时间上的位姿变化 |
            | timestamp offset | 不同传感器观测的真实时间不一致 |
            """),
            ("code", """
            import matplotlib.pyplot as plt
            import numpy as np

            def yaw_transform(yaw_rad, translation):
                c, s = np.cos(yaw_rad), np.sin(yaw_rad)
                transform = np.array([[c, -s, translation[0]], [s, c, translation[1]], [0, 0, 1.0]])
                return transform

            def apply_transform(points_xy, transform):
                homogeneous = np.c_[points_xy, np.ones(len(points_xy))]
                return (transform @ homogeneous.T).T[:, :2]

            world_points = np.array([[12, 0], [15, 1], [18, -1], [22, 0.5]])
            ego_pose_t0 = yaw_transform(np.deg2rad(0), [0, 0])
            ego_pose_t1 = yaw_transform(np.deg2rad(12), [2.0, 0.5])
            points_in_ego_t0 = apply_transform(world_points, np.linalg.inv(ego_pose_t0))
            points_in_ego_t1 = apply_transform(world_points, np.linalg.inv(ego_pose_t1))

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.scatter(world_points[:, 0], world_points[:, 1], label="world/map")
            ax.scatter(points_in_ego_t0[:, 0], points_in_ego_t0[:, 1], label="ego at t0")
            ax.scatter(points_in_ego_t1[:, 0], points_in_ego_t1[:, 1], label="ego at t1")
            ax.set_aspect("equal")
            ax.set(xlabel="x / m", ylabel="y / m", title="The same world actors have different ego-frame coordinates")
            ax.legend()
            """),
            ("code", """
            # 时间错位：一个横穿道路的 actor 在不同 timestamp 被投影到不同位置。
            def actor_position(t, speed=2.5, start=-4.0):
                return np.array([12.0, start + speed * t])

            camera_time = 1.00
            lidar_time = 1.12
            camera_actor = actor_position(camera_time)
            lidar_actor = actor_position(lidar_time)
            print("camera timestamp:", camera_time, "actor:", camera_actor)
            print("lidar timestamp: ", lidar_time, "actor:", lidar_actor)
            print("naive fusion error / m:", np.linalg.norm(camera_actor - lidar_actor))
            print("domain lesson: synchronization is a model input contract, not a preprocessing footnote")
            """),
            ("code", """
            from ipywidgets import FloatSlider, interact

            def show_alignment(timestamp_offset_ms=0.0, yaw_error_deg=0.0):
                transform = yaw_transform(np.deg2rad(12 + yaw_error_deg), [2.0, 0.5])
                aligned = apply_transform(world_points, np.linalg.inv(transform))
                stale = actor_position(1.0 + timestamp_offset_ms / 1000.0)
                print(f"timestamp offset: {timestamp_offset_ms:.0f} ms")
                print(f"calibration yaw error: {yaw_error_deg:.1f} deg")
                print(f"one moving actor's stale-position shift: {np.linalg.norm(stale - actor_position(1.0)):.3f} m")
                print("next deep dive: 01_se3_calibration_projection.ipynb")

            interact(
                show_alignment,
                timestamp_offset_ms=FloatSlider(min=-200, max=200, step=10, value=0, description="offset / ms"),
                yaw_error_deg=FloatSlider(min=-8, max=8, step=0.5, value=0, description="yaw / deg"),
            )
            """),
            ("markdown", """
            ## 领域检查点

            - 为什么把所有传感器直接 concatenate 成一个 tensor 不能解决坐标不一致？
            - ego-motion compensation 解决的是“传感器在动”，还是“目标在动”？两者如何区分？
            - 如果相机延迟 100 ms，车辆速度 15 m/s，静态目标在 ego frame 中会产生多大的位置偏差？

            **下一步**：打开 `01_se3_calibration_projection.ipynb`，把这里的 2D toy transform 换成 3D 齐次变换、相机投影和标定误差分析。
            """),
        ],
    )

    make_notebook(
        "00c_bev_occupancy_and_scene_representation.ipynb",
        "00C BEV Occupancy and Scene Representation",
        [
            ("markdown", """
            # 00C · BEV、Occupancy 与 Scene Representation：为什么自动驾驶不只做 2D 检测？

            你已经知道卷积、attention、feature map 等深度学习概念。本节补充的是自动驾驶中的表示选择：同一段道路可以表示为 camera image、LiDAR points、3D boxes、BEV occupancy、lane graph、vectorized agents 或 map elements。

            这里的关键问题不是“哪个模型更大”，而是：**下游 prediction/planning 需要什么几何和拓扑信息？** BEV 是把多视角/多传感器信息放到统一的地面坐标参考中，方便进行空间关系、速度和地图约束建模。

            本节用一个小型点云 rasterizer 直观比较 resolution、dropout 和 occupancy IoU；`06` 再深入 LiDAR/BEV occupancy，`02` 处理多传感器 feature fusion。
            """),
            ("code", """
            import matplotlib.pyplot as plt
            import numpy as np

            rng = np.random.default_rng(9)
            vehicle = np.c_[rng.normal(16.0, 0.9, 280), rng.normal(0.0, 0.7, 280)]
            pedestrian = np.c_[rng.normal(9.0, 0.25, 45), rng.normal(-3.2, 0.25, 45)]
            lane_marking = np.c_[np.linspace(0, 30, 150), np.full(150, 3.5)]
            points = np.vstack([vehicle, pedestrian, lane_marking])

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.scatter(points[:, 0], points[:, 1], s=4, alpha=0.45)
            ax.set_aspect("equal")
            ax.set(xlabel="forward x / m", ylabel="left y / m", title="A sparse scene before choosing a representation")
            """),
            ("code", """
            def rasterize(points_xy, resolution=0.5, x_range=(0, 32), y_range=(-8, 8)):
                x_edges = np.arange(x_range[0], x_range[1] + resolution, resolution)
                y_edges = np.arange(y_range[0], y_range[1] + resolution, resolution)
                occupancy, _, _ = np.histogram2d(points_xy[:, 0], points_xy[:, 1], bins=[x_edges, y_edges])
                return (occupancy > 0).astype(np.uint8), x_edges, y_edges

            occupancy, x_edges, y_edges = rasterize(points, resolution=0.5)
            plt.figure(figsize=(8, 4))
            plt.imshow(occupancy.T, origin="lower", aspect="auto", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
            plt.xlabel("forward x / m")
            plt.ylabel("left y / m")
            plt.title(f"BEV occupancy: grid={occupancy.shape}, occupied cells={occupancy.sum()}")
            plt.colorbar(label="occupied")
            """),
            ("code", """
            from ipywidgets import FloatSlider, interact

            def occupancy_experiment(resolution=0.5, dropout=0.0):
                keep = np.random.default_rng(20).random(len(points)) > dropout
                estimate, _, _ = rasterize(points[keep], resolution=resolution)
                target, _, _ = rasterize(points, resolution=resolution)
                intersection = np.logical_and(estimate, target).sum()
                union = np.logical_or(estimate, target).sum()
                iou = intersection / max(union, 1)
                print(f"resolution={resolution:.2f} m, dropout={dropout:.2f}, occupancy IoU={iou:.3f}")
                print("representation question: occupancy preserves space; vector/map representations preserve semantics and topology")

            interact(
                occupancy_experiment,
                resolution=FloatSlider(min=0.2, max=1.5, step=0.1, value=0.5, description="grid / m"),
                dropout=FloatSlider(min=0, max=0.8, step=0.05, value=0.0, description="dropout"),
            )
            """),
            ("markdown", """
            ## 领域检查点

            1. occupancy、3D box、lane vector、agent state 各自保留了什么信息，又丢掉了什么信息？
            2. 为什么 BEV 对 planning 友好，但不能因此认为“所有任务都应该转成 BEV”？
            3. 如果 occupancy IoU 不变，车辆速度或时间戳错位仍可能让 prediction 失败吗？为什么？

            **下一步**：`02` 研究 camera/LiDAR feature fusion；`06` 研究点云体素化和 occupancy；两者都建立在这里的表示选择之上。
            """),
        ],
    )

    make_notebook(
        "00d_temporal_scene_state_and_tracking.ipynb",
        "00D Temporal Scene State and Tracking",
        [
            ("markdown", """
            # 00D · 时序场景状态与 Tracking：模型为什么不能只看当前帧？

            单帧 perception 给的是 observation，不是稳定的 world state。自动驾驶需要知道：这个目标是不是同一个目标？它的速度和加速度是多少？自车运动造成的相对位移是多少？观测丢失时应该保持多久？

            本节不从 Kalman Filter 的公式开始，而从一个工程接口开始：

            ```text
            observation_t + ego_motion_t + previous_state_{t-1}
                → association / state estimation
                → agent_state_t = {id, position, velocity, age, uncertainty}
            ```

            `07` 会把这里的 toy association、dropout、outlier 和 timestamp offset 扩展为可调实验。
            """),
            ("code", """
            import matplotlib.pyplot as plt
            import numpy as np

            rng = np.random.default_rng(12)
            time = np.arange(0, 8, 0.1)
            true_position = 8.0 + 1.8 * time + 0.25 * np.sin(time)
            observation = true_position + rng.normal(0, 0.45, len(time))
            observation[30:40] = np.nan
            observation[58] += 2.2

            def smooth_track(values, alpha=0.25):
                estimate = []
                current = values[0]
                for value in values:
                    if np.isfinite(value):
                        current = alpha * value + (1 - alpha) * current
                    estimate.append(current)
                return np.asarray(estimate)

            estimated = smooth_track(observation)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(time, true_position, label="latent actor state", linewidth=2)
            ax.scatter(time, observation, s=12, alpha=0.55, label="noisy observation")
            ax.plot(time, estimated, label="simple temporal estimate")
            ax.axvspan(time[30], time[39], color="orange", alpha=0.15, label="dropout")
            ax.legend()
            ax.set(xlabel="time / s", ylabel="longitudinal position / m", title="Tracking is state estimation over imperfect observations")
            """),
            ("code", """
            def tracking_report(dropout_start=3.0, dropout_duration=1.0, outlier_size=2.0, alpha=0.25):
                local = true_position + rng.normal(0, 0.45, len(time))
                dropout = (time >= dropout_start) & (time < dropout_start + dropout_duration)
                local[dropout] = np.nan
                local[np.argmin(np.abs(time - 5.8))] += outlier_size
                estimate = smooth_track(local, alpha=alpha)
                valid = ~dropout
                rmse = np.sqrt(np.mean((estimate[valid] - true_position[valid]) ** 2))
                max_gap = int(dropout.sum())
                print(f"dropout frames={max_gap}, track RMSE outside dropout={rmse:.3f} m")
                print("state fields to preserve: track_id, position, velocity, age, covariance/uncertainty")

            from ipywidgets import FloatSlider, interact
            interact(
                tracking_report,
                dropout_start=FloatSlider(min=0, max=6, step=0.5, value=3, description="dropout start"),
                dropout_duration=FloatSlider(min=0, max=2.5, step=0.1, value=1, description="duration / s"),
                outlier_size=FloatSlider(min=0, max=5, step=0.25, value=2, description="outlier / m"),
                alpha=FloatSlider(min=0.05, max=0.8, step=0.05, value=0.25, description="update alpha"),
            )
            """),
            ("markdown", """
            ## 领域检查点

            - tracking 的“记忆”为什么会把错误传播到 prediction？
            - dropout 期间是继续输出预测、降低置信度、冻结轨迹，还是触发降级？谁负责这个决定？
            - ego-motion compensation 和 actor velocity estimation 各自需要什么输入？

            **下一步**：`07` 深入 tracking；`08` 使用 agent state 评估未来轨迹。不要在没有定义 state 和 timestamp 的情况下直接讨论 prediction model。
            """),
        ],
    )

    make_notebook(
        "00e_prediction_planning_control_closed_loop.ipynb",
        "00E Prediction Planning Control Closed Loop",
        [
            ("markdown", """
            # 00E · Prediction → Planning → Control：一条轨迹如何真正影响车辆？

            这三个词在岗位描述里经常连在一起，但它们不是同一个任务：

            - **Prediction**：其他交通参与者接下来可能怎么运动？输出不确定的 future distribution。
            - **Planning**：在 route、规则、障碍物和舒适性约束下，自车应该采取哪条 trajectory/maneuver？
            - **Control**：车辆当前状态如何跟踪这条 trajectory，输出 steering/throttle/brake？

            本节用一个前车急刹场景区分 open-loop 与 closed-loop：open-loop 只比较预测/规划与记录答案，closed-loop 则把动作反馈回环境，下一帧输入会因策略而改变。L4 的风险不能只靠单帧 accuracy 描述。
            """),
            ("code", """
            import matplotlib.pyplot as plt
            import numpy as np

            dt = 0.1
            horizon = 60
            time = np.arange(horizon) * dt
            ego_speed = np.full(horizon, 12.0)
            lead_speed = np.where(time < 2.0, 10.0, np.maximum(4.0, 10.0 - 3.0 * (time - 2.0)))
            gap_open_loop = 25.0 + np.cumsum((lead_speed - ego_speed) * dt)

            def safe_speed(gap, lead_speed, reaction_time=0.8, min_gap=5.0):
                available = np.maximum(gap - min_gap, 0.0)
                return np.minimum(lead_speed + available / max(reaction_time, 0.1), 14.0)

            planned_speed = safe_speed(gap_open_loop, lead_speed)
            closed_loop_gap = 25.0
            gaps = [closed_loop_gap]
            ego_speed_closed = []
            for step in range(horizon - 1):
                command_speed = safe_speed(closed_loop_gap, lead_speed[step])
                ego_speed_closed.append(command_speed)
                closed_loop_gap += (lead_speed[step] - command_speed) * dt
                gaps.append(closed_loop_gap)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(time, ego_speed, label="nominal ego speed")
            axes[0].plot(time, lead_speed, label="lead speed")
            axes[0].plot(time, planned_speed, label="planner speed")
            axes[0].set(title="Planning converts predicted risk to an ego trajectory", xlabel="time / s", ylabel="speed / m/s")
            axes[0].legend()
            axes[1].plot(time, gap_open_loop, label="open-loop replay")
            axes[1].plot(time, gaps, label="closed-loop guarded")
            axes[1].axhline(5.0, color="red", linestyle="--", label="minimum gap")
            axes[1].set(title="Actions change the next observation", xlabel="time / s", ylabel="gap / m")
            axes[1].legend()
            plt.tight_layout()
            """),
            ("code", """
            from ipywidgets import FloatSlider, interact

            def closed_loop_experiment(reaction_time=0.8, observation_noise=0.0, horizon_s=6.0):
                steps = int(horizon_s / dt)
                gap = 25.0
                min_gap = 5.0
                rng = np.random.default_rng(42)
                trace = []
                for step in range(steps):
                    measured_gap = gap + rng.normal(0, observation_noise)
                    command = safe_speed(measured_gap, lead_speed[step], reaction_time=reaction_time, min_gap=min_gap)
                    gap += (lead_speed[step] - command) * dt
                    trace.append(gap)
                print(f"minimum closed-loop gap={min(trace):.2f} m")
                print("planner/control question:", "fallback or emergency braking" if min(trace) < min_gap else "continue")

            interact(
                closed_loop_experiment,
                reaction_time=FloatSlider(min=0.2, max=2.0, step=0.1, value=0.8, description="reaction / s"),
                observation_noise=FloatSlider(min=0, max=2.0, step=0.1, value=0.0, description="gap noise / m"),
                horizon_s=FloatSlider(min=2, max=6, step=0.5, value=6, description="horizon / s"),
            )
            """),
            ("markdown", """
            ## 领域检查点

            1. 为什么一个预测模型的 ADE/FDE 下降，不一定会让 closed-loop collision rate 下降？
            2. planner 输出 trajectory 和 control 输出 steering/brake 的边界在哪里？
            3. 哪些约束应该由 learned policy 学习，哪些约束应该由独立 safety layer 兜底？

            **下一步**：`08` 进入 prediction metrics，`09` 进入 planning/control，`11` 和 `17` 进入闭环/场景回放。
            """),
        ],
    )

    make_notebook(
        "00f_data_safety_evaluation_deployment.ipynb",
        "00F Data Safety Evaluation Deployment",
        [
            ("markdown", """
            # 00F · 数据闭环、Safety、评测与部署：模型开发不是训练结束

            自动驾驶岗位中的“模型开发”通常不是只改网络和 loss。一次线上/仿真失败需要经过：场景定位 → 数据切片 → 重放 → root cause → 修复模型或规则 → 重新评测 → runtime 回归 → 发布门禁。

            本节先建立四种证据的区别：

            - **open-loop**：模型对固定记录的预测/检测是否正确；
            - **closed-loop**：策略动作会不会改变后续场景和风险；
            - **safety evidence**：故障、退化、TTC、fallback、ODD exit 是否可检测且有响应；
            - **deployment evidence**：batch=1 latency、p95/p99、显存、吞吐、量化误差和版本回归。

            这些证据共同决定“模型能否进入下一轮验证”，而不是某一个漂亮的平均分。
            """),
            ("code", """
            import numpy as np
            import pandas as pd

            rng = np.random.default_rng(31)
            scenarios = pd.DataFrame([
                {"scenario_id": "sunny_dense", "weather": "sunny", "density": "dense", "sensor_fault": "none"},
                {"scenario_id": "rain_sparse", "weather": "rain", "density": "sparse", "sensor_fault": "none"},
                {"scenario_id": "night_occlusion", "weather": "night", "density": "dense", "sensor_fault": "camera"},
                {"scenario_id": "gnss_outage", "weather": "sunny", "density": "medium", "sensor_fault": "gnss"},
                {"scenario_id": "cut_in", "weather": "sunny", "density": "medium", "sensor_fault": "none"},
            ])
            scenarios["episodes"] = [80, 70, 45, 35, 50]
            scenarios["collision_rate"] = [0.01, 0.04, 0.15, 0.08, 0.12]
            scenarios["p95_latency_ms"] = [82, 90, 108, 99, 87]
            display(scenarios)
            print("aggregate collision rate:", np.average(scenarios.collision_rate, weights=scenarios.episodes).round(3))
            print("worst slice:", scenarios.loc[scenarios.collision_rate.idxmax(), "scenario_id"])
            """),
            ("code", """
            def slice_report(min_collision_rate=0.08, max_latency_ms=100):
                risky = scenarios[(scenarios.collision_rate >= min_collision_rate) | (scenarios.p95_latency_ms >= max_latency_ms)]
                print(risky[["scenario_id", "weather", "sensor_fault", "collision_rate", "p95_latency_ms"]].to_string(index=False))
                print("data-loop action: replay → label/root-cause → targeted training or rule → regression gate")

            slice_report()
            """),
            ("code", """
            from ipywidgets import FloatSlider, IntSlider, interact

            def gate_experiment(collision_threshold=0.08, latency_budget_ms=100, fallback_recall=0.92):
                collision_pass = scenarios.collision_rate < collision_threshold
                latency_pass = scenarios.p95_latency_ms < latency_budget_ms
                safety_pass = fallback_recall >= 0.95
                report = scenarios[["scenario_id"]].copy()
                report["collision_gate"] = collision_pass
                report["latency_gate"] = latency_pass
                report["safety_gate"] = safety_pass
                display(report)
                print("release decision:", "candidate for next validation stage" if report.iloc[:, 1:].all().all() else "hold / investigate slices")

            interact(
                gate_experiment,
                collision_threshold=FloatSlider(min=0.02, max=0.2, step=0.01, value=0.08, description="collision"),
                latency_budget_ms=IntSlider(min=70, max=140, step=5, value=100, description="p95 / ms"),
                fallback_recall=FloatSlider(min=0.8, max=1.0, step=0.01, value=0.92, description="fallback recall"),
            )
            """),
            ("markdown", """
            ## 领域检查点

            - 为什么平均 collision rate 可能掩盖 night/occlusion 或 GNSS outage 的风险？
            - 一个模型 accuracy 提升但 p99 latency 超预算，是否应该发布？需要谁来决定？
            - corner-case mining 产出的 hard slice 如何回到数据、训练、场景回放和 regression gate？

            **下一步**：`10–12` 进入数据管线、closed-loop metrics 和 corner-case mining；`15` 进入 runtime；`18–19` 进入安全状态机和 capstone。
            """),
        ],
    )


if __name__ == "__main__":
    build()
    print("built AD domain bridge notebooks")
