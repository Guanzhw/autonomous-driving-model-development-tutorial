"""Rebuild archived mechanism notebooks with their known limitations visible."""

from __future__ import annotations

import hashlib
import textwrap
import uuid
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


COMMON = '''
from pathlib import Path
import sys

PROJECT_ROOT = next(path for path in (Path.cwd(), *Path.cwd().parents)
                    if (path / "src" / "ad_tutorial").is_dir())
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ad_tutorial import (
    ARTIFACT_DIR,
    BEVConfig,
    build_bev_dataset,
    build_urban_cut_in_scene,
    ensure_artifact_dir,
    load_json_artifact,
    load_numpy_artifact,
    save_json_artifact,
    save_numpy_artifact,
    scene_to_bev,
)

ensure_artifact_dir()
print("project root:", PROJECT_ROOT)
print("artifact directory:", ARTIFACT_DIR)
'''


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str) -> nbf.NotebookNode:
    # ``COMMON`` is already flush-left.  Dedent the appended triple-quoted
    # block separately; otherwise its function-level indentation survives and
    # the generated notebook fails AST validation.
    if source.startswith(COMMON):
        tail = textwrap.dedent(source[len(COMMON):])
        source = COMMON.rstrip() + "\n" + tail
    else:
        source = textwrap.dedent(source)
    return nbf.v4.new_code_cell(source.strip())


def build_notebook(relative_path: str, cells: list[nbf.NotebookNode]) -> None:
    path = ROOT / "reference" / "legacy" / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    notes = {
        "course/02_bev_and_fusion.ipynb": "这里的 camera_points 是点集近似；LiDAR occupancy 同时进入输入和标签。比较模型时必须先计算复制 LiDAR 输入的 identity baseline。",
        "course/05_learnable_bev_model.ipynb": "occupancy 标签可以直接从输入 LiDAR 通道复制；当前指标不能证明模型学会了图像到 BEV 或有效融合。",
        "course/06_prediction.ipynb": "candidates 表示他车的可能未来，不能作为自车规划轨迹。此处 miss_rate 是跨 mode 的阈值比例，不能当成公开 benchmark 的跨样本 miss rate。",
        "course/07_planning_closed_loop.ipynb": "选中的轨迹来自他车预测；下方 rollout 不消费它。这是保留供辨析的历史实现，学习闭环请使用新的起步单元。",
        "course/08_data_and_evaluation.ipynb": "本节的简化评测不能用来验证第 05 章模型改善了驾驶行为；模型输出尚未进入此处的执行链。",
        "course/10_capstone.ipynb": "本节加载模型后汇总旧 artifact；它没有用模型输出重新驱动规划、控制和评测，因此不构成端到端交付。",
        "labs/vla_world_action_interface.ipynb": "动作、下一状态与 margin 是手工构造的接口例子；没有训练或加载 VLA / world model。",
    }
    caveat = notes.get(relative_path, "保留此材料用于局部机制学习；先修、结论与下游连接需要结合归档索引审查。")
    archive = md("# 归档材料\n\n" + caveat +
                 "\n\n[归档索引](../README.md) · [当前学习入口](../../../course/first_loop/README.md)")
    notebook = nbf.v4.new_notebook(
        cells=[archive, *cells],
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
    )
    for index, cell in enumerate(notebook.cells):
        digest = hashlib.sha1(f"{relative_path}:{index}".encode()).hexdigest()
        cell["id"] = str(uuid.UUID(digest[:32]))[:8] + digest[32:]
    nbf.write(notebook, path)


def course_00() -> list[nbf.NotebookNode]:
    return [
        md('''
        # 00 · System & ODD：一套 L4 系统究竟在解决什么问题？

        你已经会深度学习；本章补的是自动驾驶的系统语言。先不要从模型名字开始，而要从一个可审计的任务开始：在明确的 **Operational Design Domain（ODD）** 内，车辆如何把多传感器观测变成安全动作？

        整门课贯穿同一个 `urban cut-in` 场景：左侧车辆逐渐切入 ego lane，前方还有 lead vehicle。后续章节会复用本章的场景 ID、坐标约定、时间戳和 artifact。

        本章合并了旧版 `00A` 的系统全景、旧版 `00` 的 ODD/系统契约，以及旧版 `00F` 的数据和发布证据入口。
        '''),
        code(COMMON + '''
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from dataclasses import dataclass, asdict

        scene = build_urban_cut_in_scene(seed=7, timestamp_s=0.0, scene_id="urban_cut_in_demo")
        print(scene.as_metadata())
        print("camera points:", scene.camera_points.shape, "lidar points:", scene.lidar_points.shape)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.scatter(scene.lidar_points[:, 0], scene.lidar_points[:, 1], s=5, alpha=0.25, label="LiDAR observation")
        ax.scatter(*scene.cut_in_xy, color="crimson", s=80, label="cut-in actor")
        ax.scatter(*scene.lead_xy, color="darkorange", s=80, label="lead actor")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_aspect("equal")
        ax.set(xlabel="ego x forward / m", ylabel="ego y left / m", title="The shared urban cut-in scene")
        ax.legend()
        plt.show()
        '''),
        md('''
        ## 1. ODD、ego、actor、scene 与模块接口

        - **ego**：自车状态和坐标原点；
        - **actor/agent**：其他交通参与者，至少需要位置、速度、尺寸和不确定性；
        - **scene**：带时间戳的多传感器观测、地图/路线、ego pose 和 actor state；
        - **ODD**：地理区域、道路类型、天气、光照、速度、地图和传感器健康度的约束；
        - **模块接口**：输入字段、坐标系、时间 age、输出语义、latency 和退化动作。

        `perception → tracking → prediction → planning → control` 是功能链，不等于必须使用五个独立神经网络。真正重要的是每个边界能否被测试、回放和降级。
        '''),
        code('''
        @dataclass(frozen=True)
        class ODD:
            geography: str = "urban_mapped"
            max_speed_mps: float = 13.9
            night_allowed: bool = False
            max_rain_mm_h: float = 8.0
            min_sensor_health: float = 0.70
            max_localization_sigma_m: float = 0.80

        odd = ODD()
        scenario_table = pd.DataFrame([
            {"scenario_id": "urban_cut_in_demo", "geography": "urban_mapped", "speed_mps": 8.0,
             "night": False, "rain_mm_h": 0.0, "sensor_health": 0.94, "localization_sigma_m": 0.25},
            {"scenario_id": "night_glare", "geography": "urban_mapped", "speed_mps": 8.0,
             "night": True, "rain_mm_h": 0.0, "sensor_health": 0.74, "localization_sigma_m": 0.42},
            {"scenario_id": "stale_map", "geography": "urban_mapped", "speed_mps": 14.5,
             "night": False, "rain_mm_h": 4.0, "sensor_health": 0.88, "localization_sigma_m": 1.05},
        ])
        checks = pd.DataFrame({
            "geography_ok": scenario_table.geography.eq(odd.geography),
            "speed_ok": scenario_table.speed_mps.le(odd.max_speed_mps),
            "light_ok": scenario_table.night.le(odd.night_allowed),
            "rain_ok": scenario_table.rain_mm_h.le(odd.max_rain_mm_h),
            "sensor_ok": scenario_table.sensor_health.ge(odd.min_sensor_health),
            "localization_ok": scenario_table.localization_sigma_m.le(odd.max_localization_sigma_m),
        })
        scenario_table["in_odd"] = checks.all(axis=1)
        display(scenario_table)
        display(checks.mean().sort_values().rename("constraint pass rate").to_frame())
        '''),
        md('''
        ## 2. 把系统契约落到输入、时间与状态

        一个可用的 sensor bundle 不只是几个 tensor。它要声明 `frame_id`、每个传感器的 timestamp age、缺失处理、最大 latency 和输出可以触发的状态。模型 confidence 高并不能覆盖 stale input。
        '''),
        code('''
        SENSOR_CONTRACT = {
            "required": {"camera", "lidar", "timestamp_s", "frame_id"},
            "frame_id": "base_link",
            "max_age_s": {"camera": 0.15, "lidar": 0.10},
            "max_latency_ms": 100.0,
        }

        def validate_bundle(bundle):
            issues = sorted(SENSOR_CONTRACT["required"] - set(bundle))
            if bundle.get("frame_id") != SENSOR_CONTRACT["frame_id"]:
                issues.append("frame mismatch")
            for sensor, max_age in SENSOR_CONTRACT["max_age_s"].items():
                if bundle.get(f"{sensor}_age_s", np.inf) > max_age:
                    issues.append(f"{sensor} is stale")
            if bundle.get("latency_ms", np.inf) > SENSOR_CONTRACT["max_latency_ms"]:
                issues.append("latency budget exceeded")
            return {"valid": not issues, "issues": issues}

        valid = {"camera": [], "lidar": [], "timestamp_s": 0.0, "frame_id": "base_link",
                 "camera_age_s": 0.04, "lidar_age_s": 0.03, "latency_ms": 62.0}
        stale = {**valid, "lidar_age_s": 0.24, "latency_ms": 128.0}
        print("valid:", validate_bundle(valid))
        print("stale:", validate_bundle(stale))

        artifact = {
            "scenario": scene.as_metadata(),
            "odd": asdict(odd),
            "sensor_contract": SENSOR_CONTRACT,
            "domain_checkpoint": {
                "next": "01_sensors_geometry.ipynb",
                "question": "how do raw camera/LiDAR observations enter a common frame?",
            },
        }
        save_json_artifact("00_system_contract.json", artifact)
        print("saved:", ARTIFACT_DIR / "00_system_contract.json")
        '''),
        md('''
        ### 练习与完成标准

        1. 给 ODD 增加 `route_available` 和 `map_version`，并写一个 `DEGRADED` / `MINIMAL_RISK` 判定；
        2. 写出一个“模型 confidence 很高但输入 stale”的反例；
        3. 用自己的话解释：为什么 ODD、sensor contract 和模型 loss 是三种不同层级的对象？

        完成后你应该能画出一条 `sensor bundle → representation → agent state → prediction → planner → safety` 链路，并为每个箭头说出输入、输出和失败模式。
        '''),
    ]


def course_01() -> list[nbf.NotebookNode]:
    return [
        md('''
        # 01 · Sensors & Geometry：传感器怎样进入同一个世界？

        `camera/LiDAR/radar/GNSS/IMU` 的观测不天然对齐。自动驾驶模型输入之前，需要明确 **frame、SE(3)、intrinsic/extrinsic、timestamp、ego-motion**。本章把旧版 `00B` 和 `01` 合并，并让输出成为下一章 BEV 的输入。

        约定：ego frame 使用 `x forward, y left, z up`；相机 frame 使用 `z forward, x right, y down`。真实项目必须以数据集/车辆平台文档为准。
        '''),
        code(COMMON + '''
        import numpy as np
        import matplotlib.pyplot as plt

        scene = build_urban_cut_in_scene(seed=7, timestamp_s=0.0)
        np.set_printoptions(precision=3, suppress=True)

        def make_T(yaw_deg=0.0, translation=(0.0, 0.0, 0.0)):
            yaw = np.deg2rad(yaw_deg)
            c, s = np.cos(yaw), np.sin(yaw)
            T = np.eye(4)
            T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
            T[:3, 3] = np.asarray(translation, dtype=float)
            return T

        def transform(T, points_xyz):
            points_xyz = np.asarray(points_xyz)
            homogeneous = np.c_[points_xyz, np.ones(len(points_xyz))]
            return (T @ homogeneous.T).T[:, :3]

        def project(K, points_camera):
            z = points_camera[:, 2]
            valid = z > 1e-6
            uv = np.full((len(points_camera), 2), np.nan)
            uv[valid] = (K @ points_camera[valid].T).T[:, :2] / z[valid, None]
            return uv, valid

        R_ce = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
        camera_origin_in_ego = np.array([1.3, 0.0, 1.4])
        T_c_from_e = np.eye(4)
        T_c_from_e[:3, :3] = R_ce
        T_c_from_e[:3, 3] = -R_ce @ camera_origin_in_ego
        K = np.array([[720.0, 0.0, 640.0], [0.0, 720.0, 360.0], [0.0, 0.0, 1.0]])

        camera_points = transform(T_c_from_e, scene.lidar_points)
        uv, valid = project(K, camera_points)
        print("valid projected points:", int(valid.sum()), "of", len(valid))
        '''),
        code('''
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].scatter(scene.lidar_points[:, 0], scene.lidar_points[:, 1], s=4, alpha=0.35)
        axes[0].set_aspect("equal")
        axes[0].set(xlabel="ego x / m", ylabel="ego y / m", title="LiDAR in ego frame")
        axes[1].scatter(uv[valid, 0], uv[valid, 1], s=4, alpha=0.35)
        axes[1].invert_yaxis()
        axes[1].set(xlim=(0, 1280), ylim=(720, 0), xlabel="u / px", ylabel="v / px", title="Projected observation")
        plt.tight_layout()
        plt.show()
        '''),
        md('''
        ## 1. Calibration and time are model errors, not just plumbing

        一个小的 yaw/translation 误差会变成像素偏移，随后又变成 BEV cell 错位。时间错位还会把“同一个 actor 的不同位置”当成空间融合冲突。先预测曲线，再调整下面的扰动。
        '''),
        code('''
        yaw_errors = np.linspace(-2.0, 2.0, 41)
        pixel_shift = []
        uv_ref, ref_valid = project(K, transform(T_c_from_e, scene.lidar_points))
        for error in yaw_errors:
            perturbed = make_T(error) @ T_c_from_e
            uv_error, error_valid = project(K, transform(perturbed, scene.lidar_points))
            both = ref_valid & error_valid
            pixel_shift.append(np.nanmean(np.linalg.norm(uv_error[both] - uv_ref[both], axis=1)))
        plt.plot(yaw_errors, pixel_shift, marker=".")
        plt.xlabel("injected extrinsic yaw error / deg")
        plt.ylabel("mean pixel displacement / px")
        plt.title("Calibration error becomes feature/fusion error")
        plt.show()

        for lag in [0.0, 0.05, 0.15, 0.30]:
            lagged = build_urban_cut_in_scene(seed=7, timestamp_s=0.0, sensor_lag_s=lag)
            dx = np.mean(lagged.lidar_points[:, 0]) - np.mean(lagged.camera_points[:, 0])
            print(f"camera lag={lag:.2f}s -> mean x discrepancy={dx:.3f}m")
        '''),
        md('''
        ## 2. Real-data checkpoint: nuScenes mini

        Toy geometry is useful only if it transfers to a real schema. After downloading **nuScenes mini** under its own license, set `NUSCENES_ROOT` and run the following adapter. It uses the official devkit to fetch one `LIDAR_TOP` sample and one `CAM_FRONT` calibrated sensor, then asks you to compare the projection with the toy convention. The cell is intentionally skipped when the optional package/data are absent.
        '''),
        code('''
        import os

        def nuScenes_projection_checkpoint():
            root = os.environ.get("NUSCENES_ROOT")
            if not root:
                print("Set NUSCENES_ROOT to run the real-data checkpoint; toy experiment remains reproducible.")
                return
            try:
                from nuscenes.nuscenes import NuScenes
                from nuscenes.utils.data_classes import LidarPointCloud
                from pyquaternion import Quaternion
                from PIL import Image
            except ImportError as exc:
                print("Install requirements-real-data.txt first:", exc)
                return

            nusc = NuScenes(version="v1.0-mini", dataroot=root, verbose=False)
            sample = nusc.sample[0]
            lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            camera_sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
            lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
            camera_cs = nusc.get("calibrated_sensor", camera_sd["calibrated_sensor_token"])
            lidar_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
            camera_pose = nusc.get("ego_pose", camera_sd["ego_pose_token"])

            # Official nuScenes frame chain: lidar sensor → lidar ego → global
            # → camera ego → camera sensor.  The order is part of the result.
            cloud = LidarPointCloud.from_file(str(Path(root) / lidar_sd["filename"]))
            cloud.rotate(Quaternion(lidar_cs["rotation"]).rotation_matrix)
            cloud.translate(np.asarray(lidar_cs["translation"]))
            cloud.rotate(Quaternion(lidar_pose["rotation"]).rotation_matrix)
            cloud.translate(np.asarray(lidar_pose["translation"]))
            cloud.translate(-np.asarray(camera_pose["translation"]))
            cloud.rotate(Quaternion(camera_pose["rotation"]).inverse.rotation_matrix)
            cloud.translate(-np.asarray(camera_cs["translation"]))
            cloud.rotate(Quaternion(camera_cs["rotation"]).inverse.rotation_matrix)

            camera_points = cloud.points[:3].T
            uv_real, valid_real = project(np.asarray(camera_cs["camera_intrinsic"]), camera_points)
            image_width, image_height = Image.open(Path(root) / camera_sd["filename"]).size
            in_image = valid_real & (uv_real[:, 0] >= 0) & (uv_real[:, 0] < image_width) & (uv_real[:, 1] >= 0) & (uv_real[:, 1] < image_height)
            result = {"sample": sample["token"], "points": int(len(camera_points)), "projected_in_image": int(in_image.sum()),
                      "image_size": [image_width, image_height], "frame_chain": "lidar→ego→global→camera_ego→camera"}
            print(result)
            return camera_points, uv_real, in_image

        nuScenes_projection_checkpoint()
        '''),
        md('''
        ### 完成标准

        解释 `T_camera←ego` 的旋转和平移；画出一条“标定/时间错误 → BEV 错位 → 模型后果”的链；并完成一次 real-data checkpoint 或记录缺少的数据/依赖。下一章会消费同一个 `scene`，将传感器观测栅格化为 BEV。
        '''),
        code('''
        save_json_artifact("01_geometry.json", {
            "scene_id": scene.scene_id,
            "frame_convention": {"ego": "x forward, y left, z up", "camera": "z forward, x right, y down"},
            "projected_points": int(valid.sum()),
            "calibration_sensitivity_px_per_deg": float(np.polyfit(yaw_errors, pixel_shift, 1)[0]),
            "next": "02_bev_and_fusion.ipynb",
        })
        print("saved geometry artifact")
        '''),
    ]


def course_02() -> list[nbf.NotebookNode]:
    return [
        md('''
        # 02 · BEV & Sensor Fusion：为什么要选择空间表示？

        本章合并旧版 `00C`、`02` 和 `06`。目标不是背 BEV 模型名，而是亲手完成：

        ```text
        camera/LiDAR observations → ego-frame BEV grid → occupancy/risk targets
        ```

        这份 grid 会被 `05 Learnable BEV Model` 直接读取训练；它不再是与前面无关的随机 token 分类题。
        '''),
        code(COMMON + '''
        import numpy as np
        import matplotlib.pyplot as plt

        scene = build_urban_cut_in_scene(seed=7, timestamp_s=0.0)
        config = BEVConfig(resolution=1.0)
        encoded = scene_to_bev(scene, config)
        print("BEV shape:", encoded["camera"].shape, "feature channels:", encoded["features"].shape)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, image, title in zip(axes, [encoded["camera"], encoded["lidar"], encoded["targets"][1]],
                                    ["camera occupancy", "LiDAR occupancy", "cut-in risk target"]):
            ax.imshow(image.T, origin="lower", aspect="auto")
            ax.set_title(title)
            ax.set_xlabel("y cell")
            ax.set_ylabel("x cell")
        plt.tight_layout()
        plt.show()
        '''),
        md('''
        ## 1. Representation and fusion are different choices

        - occupancy preserves “where is space occupied?” but not necessarily class, intent, or topology;
        - object boxes preserve instance semantics but may hide free space;
        - vector/map representations preserve lanes and topology but depend on map quality;
        - agent state preserves identity/velocity and feeds prediction.

        Fusion can happen before rasterization, in a shared BEV feature space, or after separate heads. Whatever the architecture, calibration/time quality and modality health must be visible to the model or safety layer.
        '''),
        code('''
        from ipywidgets import FloatSlider, interact

        def fusion_experiment(resolution=1.0, camera_dropout=0.0, time_lag=0.0):
            local = build_urban_cut_in_scene(seed=7, timestamp_s=0.0, sensor_lag_s=time_lag)
            cfg = BEVConfig(resolution=resolution)
            result = scene_to_bev(local, cfg)
            rng = np.random.default_rng(20)
            camera = result["camera"].copy()
            camera[rng.random(camera.shape) < camera_dropout] = 0.0
            fused = np.maximum(camera, result["lidar"])
            target = result["targets"][0] > 0
            iou = np.logical_and(fused > 0, target).sum() / max(np.logical_or(fused > 0, target).sum(), 1)
            print(f"resolution={resolution:.1f}m, camera dropout={camera_dropout:.2f}, lag={time_lag:.2f}s, fused IoU={iou:.3f}")
            print("failure interpretation: resolution changes quantization; dropout removes evidence; lag shifts evidence")

        interact(
            fusion_experiment,
            resolution=FloatSlider(min=0.5, max=2.0, step=0.5, value=1.0),
            camera_dropout=FloatSlider(min=0.0, max=0.9, step=0.1, value=0.0),
            time_lag=FloatSlider(min=0.0, max=0.4, step=0.05, value=0.0),
        )
        '''),
        md('''
        ## 2. Real-data checkpoint: construct one nuScenes BEV

        The public checkpoint is deliberately small: use the official devkit to read one `LIDAR_TOP` sample from nuScenes mini, transform points into ego coordinates, and pass the resulting `N×3` array through `rasterize_points`. Compare its point count, range and empty-cell pattern with the toy scene. Do not call the toy occupancy grid a benchmark result.
        '''),
        code('''
        from ad_tutorial.scene import rasterize_points
        import os

        def real_data_bev_checkpoint(points_xyz=None, sample_token=None):
            if points_xyz is None:
                root = os.environ.get("NUSCENES_ROOT")
                if not root:
                    print("Set NUSCENES_ROOT or pass ego-frame points to run the real-data checkpoint.")
                    return None
                try:
                    from nuscenes.nuscenes import NuScenes
                    from nuscenes.utils.data_classes import LidarPointCloud
                    from pyquaternion import Quaternion
                except ImportError as exc:
                    print("Install requirements-real-data.txt first:", exc)
                    return None
                nusc = NuScenes(version="v1.0-mini", dataroot=root, verbose=False)
                sample = nusc.sample[0] if sample_token is None else nusc.get("sample", sample_token)
                lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
                lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
                ego_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
                cloud = LidarPointCloud.from_file(str(Path(root) / lidar_sd["filename"]))
                cloud.rotate(Quaternion(lidar_cs["rotation"]).rotation_matrix)
                cloud.translate(np.asarray(lidar_cs["translation"]))
                cloud.rotate(Quaternion(ego_pose["rotation"]).rotation_matrix)
                cloud.translate(np.asarray(ego_pose["translation"]))
                # Move global points back to the ego frame at the sample time.
                cloud.translate(-np.asarray(ego_pose["translation"]))
                cloud.rotate(Quaternion(ego_pose["rotation"]).inverse.rotation_matrix)
                points_xyz = cloud.points[:3].T
            real_grid = rasterize_points(np.asarray(points_xyz), BEVConfig(resolution=1.0))
            print("real-data BEV shape:", real_grid.shape, "occupied cells:", int(real_grid.sum()))
            return real_grid

        real_data_bev_checkpoint()
        '''),
        md('''
        ## 3. Produce the training artifact for Chapter 05

        Use a coarser `2m` grid for a CPU-friendly but spatially meaningful Transformer. Every sample is a variant of the same cut-in task; labels are occupancy, cut-in risk, and x velocity. The exact array shape and coordinate convention are part of the artifact contract.
        '''),
        code('''
        train_config = BEVConfig(resolution=2.0)
        dataset = build_bev_dataset(n=72, seed=101, config=train_config)
        save_numpy_artifact("02_bev_dataset.npz", features=dataset["features"], targets=dataset["targets"])
        save_json_artifact("02_bev_dataset_meta.json", {
            "scenario": dataset["scenario"],
            "config": dataset["config"],
            "features": ["camera_occupancy", "lidar_occupancy", "camera_age"],
            "targets": ["occupancy", "cut_in_risk", "velocity_x"],
            "next": "05_learnable_bev_model.ipynb",
        })
        print("saved training features:", dataset["features"].shape)
        print("saved training targets:", dataset["targets"].shape)
        '''),
        md('''
        ### 完成标准

        你应能回答：为什么 BEV 对 planning 友好但不是所有任务的唯一表示？错位和 dropout 如何传播？`02_bev_dataset.npz` 中每个 channel/target 的含义是什么？下一章会将 temporal state 与 tracking 接到同一场景，随后 Chapter 05 会真的训练这些 BEV targets。
        '''),
    ]


def course_03() -> list[nbf.NotebookNode]:
    return [
        md('''
        # 03 · Temporal State：从 observation 到稳定的 agent state

        本章合并旧版 `00D` 和 `07`。单帧 perception 是 observation，不是 world state。prediction 需要的是带有 `track_id、位置、速度、年龄、不确定性` 的时序状态，还需要考虑 ego-motion、dropout、outlier 和 timestamp offset。
        '''),
        code(COMMON + '''
        import numpy as np
        import matplotlib.pyplot as plt

        times = np.arange(0.0, 8.0, 0.1)
        truth = np.array([build_urban_cut_in_scene(seed=7, timestamp_s=float(t)).cut_in_xy for t in times])
        rng = np.random.default_rng(12)
        observation = truth + rng.normal(0.0, 0.35, truth.shape)
        dropout = (times >= 3.0) & (times < 4.2)
        observation[dropout] = np.nan
        observation[58] += np.array([2.2, -1.5])

        def smooth_track(values, alpha=0.25):
            estimate = []
            current = values[0].copy()
            for value in values:
                if np.isfinite(value).all():
                    current = alpha * value + (1.0 - alpha) * current
                estimate.append(current.copy())
            return np.asarray(estimate)

        estimate = smooth_track(observation)
        velocity = np.gradient(estimate, times, axis=0)
        uncertainty = np.where(np.isfinite(observation).all(axis=1), 0.35, 0.9)
        '''),
        code('''
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(truth[:, 0], truth[:, 1], label="latent cut-in")
        axes[0].scatter(observation[:, 0], observation[:, 1], s=8, alpha=0.35, label="observation")
        axes[0].plot(estimate[:, 0], estimate[:, 1], label="tracked state")
        axes[0].set_aspect("equal")
        axes[0].legend()
        axes[0].set(title="Association/state estimation in the ego frame", xlabel="x / m", ylabel="y / m")
        axes[1].plot(times, velocity[:, 1], label="estimated lateral velocity")
        axes[1].fill_between(times, -uncertainty, uncertainty, alpha=0.2, label="uncertainty proxy")
        axes[1].axvspan(3.0, 4.2, color="orange", alpha=0.15, label="observation dropout")
        axes[1].legend()
        axes[1].set(title="State memory changes prediction input", xlabel="time / s", ylabel="m/s")
        plt.tight_layout()
        plt.show()

        valid = ~dropout
        rmse = float(np.sqrt(np.mean((estimate[valid] - truth[valid]) ** 2)))
        print({"track_rmse_m": round(rmse, 3), "max_dropout_frames": int(dropout.sum()), "last_velocity": velocity[-1].round(3).tolist()})
        '''),
        md('''
        ## Tracking is an interface, not a magic filter

        A Kalman filter, learned tracker, or transformer memory can implement the update. The system still needs to decide what to do when association is ambiguous: keep a stale track, reduce confidence, create a new ID, or trigger a safety/degraded mode. Tracking mistakes can be amplified by prediction and planning.
        '''),
        code('''
        from ipywidgets import FloatSlider, interact

        def tracking_ablation(alpha=0.25, dropout_duration=1.0, outlier_m=2.0):
            local_truth = np.array([build_urban_cut_in_scene(seed=7, timestamp_s=float(t)).cut_in_xy for t in times])
            local_obs = local_truth + np.random.default_rng(18).normal(0, 0.35, local_truth.shape)
            local_dropout = (times >= 3.0) & (times < 3.0 + dropout_duration)
            local_obs[local_dropout] = np.nan
            local_obs[np.argmin(abs(times - 5.8))] += np.array([outlier_m, -0.5 * outlier_m])
            local_est = smooth_track(local_obs, alpha=alpha)
            score = np.sqrt(np.mean((local_est[~local_dropout] - local_truth[~local_dropout]) ** 2))
            print(f"alpha={alpha:.2f}, dropout={dropout_duration:.1f}s, outlier={outlier_m:.1f}m -> RMSE={score:.3f}m")

        interact(
            tracking_ablation,
            alpha=FloatSlider(min=0.05, max=0.8, step=0.05, value=0.25),
            dropout_duration=FloatSlider(min=0.0, max=2.5, step=0.1, value=1.0),
            outlier_m=FloatSlider(min=0.0, max=5.0, step=0.25, value=2.0),
        )
        '''),
        code('''
        save_numpy_artifact(
            "03_temporal_state.npz",
            time_s=times,
            truth_xy=truth,
            observation_xy=np.nan_to_num(observation, nan=-999.0),
            estimate_xy=estimate,
            velocity_xy=velocity,
            uncertainty=uncertainty,
        )
        save_json_artifact("03_temporal_state_meta.json", {
            "track_id": "cut_in_0",
            "fields": ["position_xy", "velocity_xy", "age", "uncertainty"],
            "dropout_frames": int(dropout.sum()),
            "next": "04_localization_mapping.ipynb",
        })
        print("saved temporal state artifact")
        '''),
        md('''
        ### 完成标准

        解释 ego-motion compensation 与 actor velocity estimation 的区别；说明 dropout 期间你会怎样设置 track age/uncertainty；并说出 tracking error 为什么会改变 prediction 的 miss rate。下一章单独处理 ego pose、漂移和地图匹配，避免把 actor state 与 localization 混为一谈。
        '''),
    ]


def course_04() -> list[nbf.NotebookNode]:
    return [
        md('''
        # 04 · Localization & Mapping：自车到底在哪里？

        perception/tracking 处理“别人在哪里”；localization 处理“我在哪里”。本章使用同一个 urban cut-in road frame，模拟 wheel-odometry 漂移、GNSS outage 和 map matching。定位误差会污染所有后续 BEV、tracking、planning 和 closed-loop 指标。
        '''),
        code(COMMON + '''
        import numpy as np
        import matplotlib.pyplot as plt

        times = np.arange(0.0, 20.0, 0.1)
        true_pose = np.c_[0.8 * times, 0.4 * np.sin(times / 3.0)]
        rng = np.random.default_rng(33)
        odometry = true_pose + np.cumsum(rng.normal(0, [0.015, 0.012], true_pose.shape), axis=0)
        gnss = true_pose + rng.normal(0, 0.25, true_pose.shape)
        outage = (times >= 8.0) & (times < 13.0)
        gnss[outage] = np.nan
        '''),
        code('''
        def fuse_pose(odometry_xy, gnss_xy, map_xy=None):
            estimate = odometry_xy.copy()
            for index in range(len(estimate)):
                if np.isfinite(gnss_xy[index]).all():
                    estimate[index] = 0.25 * estimate[index] + 0.75 * gnss_xy[index]
            if map_xy is not None:
                estimate = 0.85 * estimate + 0.15 * map_xy
            return estimate

        map_centerline = np.c_[0.8 * times, np.zeros_like(times)]
        estimate = fuse_pose(odometry, gnss, map_centerline)
        error = np.linalg.norm(estimate - true_pose, axis=1)
        print({"pose_rmse_m": float(np.sqrt(np.mean(error ** 2))), "outage_seconds": float(outage.sum() * 0.1),
               "max_error_m": float(error.max())})

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(true_pose[:, 0], true_pose[:, 1], label="ground truth")
        axes[0].plot(odometry[:, 0], odometry[:, 1], alpha=0.7, label="odometry")
        axes[0].plot(estimate[:, 0], estimate[:, 1], label="fused + map")
        axes[0].set_aspect("equal")
        axes[0].legend()
        axes[0].set(title="Pose and map alignment", xlabel="x / m", ylabel="y / m")
        axes[1].plot(times, error)
        axes[1].axvspan(8, 13, color="orange", alpha=0.2, label="GNSS outage")
        axes[1].legend()
        axes[1].set(title="Localization error over time", xlabel="time / s", ylabel="error / m")
        plt.tight_layout()
        plt.show()
        '''),
        md('''
        ## 练习：漂移、地图匹配和 ODD

        1. 增大 odometry drift，观察 outage 后的恢复时间；
        2. 把地图中心线故意平移 1m，说明 map matching 何时会把系统“拉向错误答案”；
        3. 把 `max_localization_sigma_m` 写成 Chapter 00 的 ODD gate，并说明 planner 应该减速、冻结，还是退出 ODD。
        '''),
        code('''
        drift_scale = np.linspace(0.5, 3.0, 8)
        outage_rmse = []
        for scale in drift_scale:
            local_odometry = true_pose + np.cumsum(rng.normal(0, [0.015, 0.012], true_pose.shape) * scale, axis=0)
            local = fuse_pose(local_odometry, gnss, map_centerline)
            outage_rmse.append(np.sqrt(np.mean(np.linalg.norm(local[outage] - true_pose[outage], axis=1) ** 2)))
        plt.plot(drift_scale, outage_rmse, marker="o")
        plt.xlabel("odometry drift multiplier")
        plt.ylabel("outage RMSE / m")
        plt.title("Localization robustness is an ODD/system concern")
        plt.show()

        save_numpy_artifact("04_localization.npz", time_s=times, true_pose=true_pose, estimate=estimate, error=error)
        save_json_artifact("04_localization_meta.json", {
            "pose_frame": "map",
            "outage": [8.0, 13.0],
            "pose_rmse_m": float(np.sqrt(np.mean(error ** 2))),
            "next": "05_learnable_bev_model.ipynb",
        })
        print("saved localization artifact")
        '''),
        md('''
        ### 完成标准

        你要能区分 pose drift、sensor outage、map error 和 actor tracking error 的观测症状，并把每一种症状映射到一个可测指标和一个系统动作。下一章开始真正训练有 AD 语义的 BEV 模型。
        '''),
    ]


def course_05() -> list[nbf.NotebookNode]:
    return [
        md('''
        # 05 · Learnable BEV Model：真正训练一个空间查询模型

        这是全课程最重要的重写。旧版随机生成 camera/LiDAR token，再根据人工统计量做三分类；现在模型直接读取 Chapter 02 生成的 **urban cut-in BEV feature grid**，用每个空间 cell 的 learned BEV query 预测：

        \\[
        \\hat y = \\{\\hat O,\\hat R,\\hat v_x\\}
        \\]

        其中 `occupancy`、`cut-in risk` 和 `velocity_x` 都对应共享场景中的空间语义。模型仍然使用原生 PyTorch `TransformerEncoder` / `MultiheadAttention`，因此可以看清 attention；不需要 Hugging Face `transformers`。

        运行本章前：`python -m pip install -r requirements-ml.txt`。
        '''),
        code(COMMON + '''
        import time
        import numpy as np
        import matplotlib.pyplot as plt
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
        from ad_tutorial.bev_model import LearnableBEVModel, occupancy_iou, risk_f1

        torch.set_num_threads(1)
        torch.manual_seed(23)
        data = load_numpy_artifact("02_bev_dataset.npz")
        features = torch.tensor(data["features"], dtype=torch.float32)
        targets = torch.tensor(data["targets"], dtype=torch.float32)
        split = int(0.8 * len(features))
        train_x, val_x = features[:split], features[split:]
        train_y, val_y = targets[:split], targets[split:]
        train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=8, shuffle=True)
        height, width = features.shape[-2:]
        print("features:", tuple(features.shape), "targets:", tuple(targets.shape), "BEV cells:", height * width)
        '''),
        md('''
        ## 1. Architecture: sensor memory and spatial queries

        The encoder mixes per-cell sensor features. The learned query table has one query per BEV cell, so output `[:, :, i, j]` still has a spatial meaning. This is intentionally smaller than a production BEVFormer-style stack, but the interface is the same: feature space, query space, head, target, metric, robustness and runtime.
        '''),
        code('''
        model = LearnableBEVModel(height, width, in_channels=features.shape[1], d_model=48, nhead=4, layers=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

        def loss_and_metrics(logits, truth):
            occupancy_loss = F.binary_cross_entropy_with_logits(logits[:, 0], truth[:, 0])
            risk_loss = F.binary_cross_entropy_with_logits(logits[:, 1], truth[:, 1])
            dynamic = truth[:, 1] > 0
            velocity_loss = F.mse_loss(logits[:, 2][dynamic], truth[:, 2][dynamic]) if dynamic.any() else logits[:, 2].mean() * 0.0
            loss = occupancy_loss + risk_loss + 0.25 * velocity_loss
            with torch.no_grad():
                metrics = {
                    "loss": float(loss),
                    "occupancy_iou": occupancy_iou(logits[:, 0], truth[:, 0]),
                    "risk_f1": risk_f1(logits[:, 1], truth[:, 1]),
                    "velocity_mae": float(torch.abs(logits[:, 2][dynamic] - truth[:, 2][dynamic]).mean()) if dynamic.any() else 0.0,
                }
            return loss, metrics

        history = []
        for epoch in range(8):
            model.train()
            train_losses = []
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss, _ = loss_and_metrics(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss))
            model.eval()
            with torch.no_grad():
                _, val_metrics = loss_and_metrics(model(val_x), val_y)
            history.append({"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)), **val_metrics})
            print(history[-1])
        '''),
        code('''
        model.eval()
        with torch.no_grad():
            logits = model(val_x)
        print("final validation:", loss_and_metrics(logits, val_y)[1])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot([row["train_loss"] for row in history], label="train loss")
        axes[0].set(title="training curve", xlabel="epoch")
        axes[1].plot([row["occupancy_iou"] for row in history], label="occupancy IoU")
        axes[1].plot([row["risk_f1"] for row in history], label="risk F1")
        axes[1].legend()
        axes[1].set(title="spatial task metrics", xlabel="epoch")
        plt.tight_layout()
        plt.show()
        '''),
        md('''
        ## 2. Robustness and runtime are part of the model result

        Drop LiDAR evidence and perturb the camera-age channel. A model that wins only on nominal inputs is not yet a useful AD artifact.
        '''),
        code('''
        def evaluate_perturbation(x, y, lidar_dropout=0.0, age_bias=0.0):
            perturbed = x.clone()
            if lidar_dropout:
                perturbed[:, 1] *= (1.0 - lidar_dropout)
            perturbed[:, 2] = torch.clamp(perturbed[:, 2] + age_bias, 0.0, 1.0)
            with torch.no_grad():
                out = model(perturbed)
            return loss_and_metrics(out, y)[1]

        for dropout in [0.0, 0.5, 1.0]:
            print("LiDAR dropout", dropout, evaluate_perturbation(val_x, val_y, lidar_dropout=dropout))

        sample = val_x[:1]
        times_ms = []
        with torch.no_grad():
            for _ in range(5):
                model(sample)
            for _ in range(30):
                start = time.perf_counter()
                model(sample)
                times_ms.append((time.perf_counter() - start) * 1000)
        runtime = {"p50_ms": float(np.percentile(times_ms, 50)), "p95_ms": float(np.percentile(times_ms, 95)),
                   "p99_ms": float(np.percentile(times_ms, 99)), "parameters": int(sum(p.numel() for p in model.parameters()))}
        print("runtime:", runtime)
        '''),
        md('''
        ## 3. Save the artifact used later by the capstone

        This checkpoint is intentionally a small teaching artifact, not a public benchmark result. Chapter 10 will load it and verify the model output shape before running the planner/safety chain.
        '''),
        code('''
        checkpoint_path = ARTIFACT_DIR / "05_bev_model.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "model_config": {"height": height, "width": width, "in_channels": int(features.shape[1]), "d_model": 48, "nhead": 4, "layers": 2},
            "history": history,
            "runtime": runtime,
        }, checkpoint_path)
        save_json_artifact("05_bev_metrics.json", {"validation": history[-1], "runtime": runtime,
                                                   "artifact": str(checkpoint_path.relative_to(PROJECT_ROOT))})
        print("saved:", checkpoint_path)
        '''),
        md('''
        ### 练习与完成标准

        1. 把 head 扩展为 `class/height/velocity` 或 occupancy + multiple risk heads；
        2. 比较不同 BEV resolution 对 occupancy IoU 和 latency 的影响；
        3. 给 camera/LiDAR 分别加入 calibration perturbation，解释它为什么不是普通 iid noise；
        4. 记录 seed、data artifact、训练配置、参数量、p50/p95/p99 和至少一个 failure slice。

        你应该能指出：哪一部分是 Transformer 机制，哪一部分是自动驾驶语义，哪一部分仍然只是合成教学近似。
        '''),
    ]


def course_06() -> list[nbf.NotebookNode]:
    return [
        md('''
        # 06 · Prediction：agent state 到未来分布

        本章把 Chapter 03 的 tracked state 变成多模态 future trajectories，并合并旧版 `00E` 的 prediction 入口和 `08` 的 ADE/FDE/Miss Rate。Prediction 不是“猜一条最像 label 的线”：cut-in/keep-lane/brake 等多个 mode 都可能合理，planning 需要看分布和风险。
        '''),
        code(COMMON + '''
        import numpy as np
        import matplotlib.pyplot as plt

        state = load_numpy_artifact("03_temporal_state.npz")
        time_s = state["time_s"]
        last_position = state["estimate_xy"][-1]
        last_velocity = state["velocity_xy"][-1]
        horizon_s = np.arange(0.1, 3.1, 0.1)
        true_future = np.array([build_urban_cut_in_scene(seed=7, timestamp_s=float(time_s[-1] + dt)).cut_in_xy for dt in horizon_s])

        keep_lane = last_position + np.c_[last_velocity[0] * horizon_s, np.zeros_like(horizon_s)]
        cut_in = last_position + np.c_[last_velocity[0] * horizon_s, last_velocity[1] * horizon_s]
        brake = last_position + np.c_[0.5 * last_velocity[0] * horizon_s, 0.4 * last_velocity[1] * horizon_s]
        candidates = np.stack([keep_lane, cut_in, brake])
        probabilities = np.array([0.25, 0.55, 0.20])
        '''),
        code('''
        def ade(prediction, truth):
            return float(np.mean(np.linalg.norm(prediction - truth[None, ...], axis=-1)))

        def fde(prediction, truth):
            return float(np.mean(np.linalg.norm(prediction[:, -1] - truth[-1], axis=-1)))

        per_mode_ade = np.mean(np.linalg.norm(candidates - true_future[None, ...], axis=-1), axis=1)
        per_mode_fde = np.linalg.norm(candidates[:, -1] - true_future[-1], axis=-1)
        min_ade = float(per_mode_ade.min())
        min_fde = float(per_mode_fde.min())
        miss_rate = float(np.mean(per_mode_fde > 2.0))
        print({"per_mode_ADE": per_mode_ade.round(3).tolist(), "per_mode_FDE": per_mode_fde.round(3).tolist(),
               "minADE": round(min_ade, 3), "minFDE": round(min_fde, 3), "miss_rate@2m": round(miss_rate, 3)})

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(true_future[:, 0], true_future[:, 1], "k-", linewidth=3, label="future")
        for name, path, prob in zip(["keep lane", "cut in", "brake"], candidates, probabilities):
            ax.plot(path[:, 0], path[:, 1], label=f"{name} p={prob:.2f}")
        ax.scatter(last_position[0], last_position[1], color="red", label="current state")
        ax.set_aspect("equal")
        ax.legend()
        ax.set(title="Multimodal prediction for the tracked cut-in actor", xlabel="x / m", ylabel="y / m")
        plt.show()
        '''),
        md('''
        ## Metrics are not interchangeable

        `ADE/FDE` summarize geometric distance; `minADE/minFDE` reward covering one plausible mode; `miss rate` exposes tail failure. A planner may care more about a low-probability cut-in mode than a mean displacement. This is why prediction metrics alone cannot substitute closed-loop evaluation.
        '''),
        code('''
        mode_weights = np.linspace(0.0, 1.0, 11)
        risk_aware_cost = []
        for cut_in_weight in mode_weights:
            costs = per_mode_ade + cut_in_weight * np.array([0.0, 2.5, 0.5])
            risk_aware_cost.append(int(np.argmin(costs)))
        print("planner-selected mode as cut-in risk weight changes:", list(zip(mode_weights.round(2), risk_aware_cost)))

        save_numpy_artifact("06_prediction.npz", horizon_s=horizon_s, truth_future=true_future,
                            candidates=candidates, probabilities=probabilities, per_mode_ade=per_mode_ade,
                            per_mode_fde=per_mode_fde)
        save_json_artifact("06_prediction_metrics.json", {
            "minADE": min_ade, "minFDE": min_fde, "miss_rate_at_2m": miss_rate,
            "modes": ["keep_lane", "cut_in", "brake"], "next": "07_planning_closed_loop.ipynb",
        })
        print("saved prediction artifact")
        '''),
        md('''
        ### 完成标准

        解释为什么“更低 ADE”不一定意味着更低 closed-loop collision rate；为 cut-in 设计一个 mode coverage 或 risk-weighted metric；并说明 prediction 的输入为什么应该是 tracked state，而不是未经时间语义处理的单帧 boxes。
        '''),
    ]


def course_07() -> list[nbf.NotebookNode]:
    return [
        md('''
        # 07 · Planning & Closed Loop：预测怎样改变驾驶行为？

        本章合并旧版 `00E` 的 planning/control 入口和 `09`。Prediction 输出的是可能的 agent futures；planner 需要在 route、碰撞风险、舒适性和车辆可行性之间做决策；controller 再把 trajectory 变成动作。最关键的转折是：**动作会改变下一帧输入**，因此 open-loop 好看不代表 closed-loop 安全。
        '''),
        code(COMMON + '''
        import numpy as np
        import matplotlib.pyplot as plt

        prediction = load_numpy_artifact("06_prediction.npz")
        candidate = prediction["candidates"]
        probabilities = prediction["probabilities"]
        horizon = prediction["horizon_s"]

        def choose_plan(candidates, probabilities, risk_weight=1.0):
            # Cut-in mode is the second mode in the shared artifact.
            progress = -candidates[:, -1, 1]
            risk_penalty = np.array([0.0, 3.0, 0.7]) * risk_weight
            cost = -progress + risk_penalty - np.log(probabilities + 1e-6)
            return int(np.argmin(cost)), cost

        selected_mode, costs = choose_plan(candidate, probabilities, risk_weight=1.0)
        selected = candidate[selected_mode]
        print("selected mode:", selected_mode, "costs:", costs.round(3))
        '''),
        code('''
        def rollout(policy="guarded", dt=0.1, horizon_s=6.0, seed=4):
            rng = np.random.default_rng(seed)
            steps = int(horizon_s / dt)
            ego_x, ego_speed = 0.0, 8.0
            obstacle_x, obstacle_y = 18.0, 0.0
            rows = []
            for step in range(steps):
                time = step * dt
                obstacle_speed = 4.5 if time < 2.0 else 2.0
                gap = obstacle_x - ego_x
                lateral_risk = abs(obstacle_y) < 1.3
                risk = gap < 12.0 and lateral_risk
                if policy == "naive":
                    target_speed = 8.0
                    action = "nominal"
                elif risk:
                    target_speed = max(0.0, obstacle_speed - 0.5)
                    action = "degraded" if gap > 6.0 else "minimal_risk"
                else:
                    target_speed = 8.0
                    action = "learned_plan"
                acceleration = np.clip((target_speed - ego_speed) * 1.5, -4.0, 2.0)
                ego_speed = max(0.0, ego_speed + acceleration * dt)
                ego_x += ego_speed * dt
                obstacle_x += obstacle_speed * dt
                obstacle_y += rng.normal(0.0, 0.01)
                rows.append({"time_s": time, "gap_m": obstacle_x - ego_x, "ego_speed": ego_speed,
                             "acceleration": acceleration, "action": action})
            return rows

        traces = {policy: rollout(policy) for policy in ["naive", "guarded"]}
        for policy, rows in traces.items():
            frame = np.array([[row["gap_m"], row["ego_speed"], row["acceleration"]] for row in rows])
            jerk = np.diff(frame[:, 2], prepend=frame[0, 2]) / 0.1
            print(policy, {"min_gap_m": round(float(frame[:, 0].min()), 3),
                            "progress_m": round(float(frame[:, 1].sum() * 0.1), 3),
                            "max_jerk": round(float(np.abs(jerk).max()), 3),
                            "fallback_s": round(sum(r["action"] != "learned_plan" for r in rows) * 0.1, 2)})

        for policy, rows in traces.items():
            plt.plot([r["time_s"] for r in rows], [r["gap_m"] for r in rows], label=policy)
        plt.axhline(2.0, color="red", linestyle="--", label="collision threshold")
        plt.legend()
        plt.xlabel("time / s")
        plt.ylabel("gap / m")
        plt.title("Actions change the next observation")
        plt.show()
        '''),
        md('''
        ## Planner/control contract and exercises

        - planner: trajectory/maneuver with time, frame, feasibility and confidence;
        - controller: current ego state + trajectory → steering/throttle/brake;
        - safety layer: independent constraints and fallback, not a second name for the planner.

        练习：改变 reaction time、observation noise 和 risk weight；比较 open-loop trajectory error 与 closed-loop minimum gap；说明为什么一个 planner 只优化 geometric distance 可能把 cut-in 直接撞上。
        '''),
        code('''
        risk_weights = np.linspace(0.0, 3.0, 13)
        chosen = [choose_plan(candidate, probabilities, weight)[0] for weight in risk_weights]
        print("risk weight → selected mode:", list(zip(risk_weights.round(2), chosen)))
        save_numpy_artifact("07_planner.npz", selected_mode=np.array([selected_mode]), selected_trajectory=selected,
                            risk_weights=risk_weights, selected_modes=np.asarray(chosen))
        save_json_artifact("07_planner_metrics.json", {
            "selected_mode": int(selected_mode), "mode_names": ["keep_lane", "cut_in", "brake"],
            "closed_loop_policies": ["naive", "guarded"], "next": "08_data_and_evaluation.ipynb",
        })
        print("saved planner artifact")
        '''),
        md('''
        ### 完成标准

        你应能解释 prediction→planning→control 的边界，以及为什么 closed-loop 会改变评估分布。下一章会把同一条链放进 scenario bundle、log replay、metrics 和 corner-case slice，而不是继续增加孤立 demo。
        '''),
    ]


def course_08() -> list[nbf.NotebookNode]:
    return [
        md('''
        # 08 · Data & Evaluation：从 sensor bundle 到 failure slice

        本章合并旧版 `10`、`17`、`11` 和 `12`。它把“一个 demo”变成可回归的开发流程：

        ```text
        sensor bundle → scenario ID → replay → open/closed-loop metrics → slice → mining
        ```

        评测至少要同时报告安全、任务、舒适性、数据质量和 runtime；平均分不能掩盖 `night/occlusion`、`cut-in` 或 `GNSS outage`。
        '''),
        code(COMMON + '''
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import zlib

        prediction = load_numpy_artifact("06_prediction.npz")
        planner = load_numpy_artifact("07_planner.npz")

        def make_bundle(seed, lidar_age_s=0.03, camera_age_s=0.04, sensor_fault="none"):
            scene = build_urban_cut_in_scene(seed=seed, timestamp_s=0.0,
                                             sensor_lag_s=max(camera_age_s - lidar_age_s, 0.0),
                                             scene_id=f"cut_in_{seed:03d}")
            return {
                "scene_id": scene.scene_id,
                "scenario": "urban_cut_in",
                "camera": scene.camera_points,
                "lidar": scene.lidar_points,
                "camera_age_s": camera_age_s,
                "lidar_age_s": lidar_age_s,
                "sensor_fault": sensor_fault,
            }

        bundles = [make_bundle(seed) for seed in range(20, 28)]
        print("bundle schema:", sorted(k for k in bundles[0] if k not in {"camera", "lidar"}))
        '''),
        code('''
        def replay_bundle(bundle, guarded=True):
            rng = np.random.default_rng(zlib.crc32(bundle["scene_id"].encode("utf-8")) % 10000)
            gap = 18.0
            ego_speed = 8.0
            rows = []
            for step in range(60):
                stale = bundle["camera_age_s"] > 0.15 or bundle["lidar_age_s"] > 0.10
                faulty = bundle["sensor_fault"] != "none"
                risk = gap < 10.0 or stale or faulty
                target_speed = 8.0 if (not guarded or not risk) else 2.5
                acceleration = np.clip((target_speed - ego_speed) * 1.5, -4.0, 2.0)
                ego_speed = max(0.0, ego_speed + acceleration * 0.1)
                gap += (3.0 - ego_speed) * 0.1
                rows.append({"gap_m": gap, "speed": ego_speed, "acceleration": acceleration,
                             "latency_ms": 65 + rng.lognormal(1.0, 0.15), "risk": risk})
            frame = pd.DataFrame(rows)
            jerk = np.diff(frame.acceleration, prepend=frame.acceleration.iloc[0]) / 0.1
            return {
                "scene_id": bundle["scene_id"], "collision": bool((frame.gap_m < 2.0).any()),
                "min_gap_m": float(frame.gap_m.min()), "progress_m": float((frame.speed * 0.1).sum()),
                "max_jerk_mps3": float(np.abs(jerk).max()),
                "p95_latency_ms": float(frame.latency_ms.quantile(0.95)),
                "sensor_fault": bundle["sensor_fault"], "stale": bool(frame.risk.iloc[0]),
            }

        reports = []
        for bundle in bundles:
            reports.append(replay_bundle(bundle, guarded=True))
            if bundle["scene_id"].endswith("022"):
                bundle["sensor_fault"] = "lidar_dropout"
        report = pd.DataFrame(reports)
        report["slice"] = np.where(report.sensor_fault != "none", "sensor_fault", "nominal")
        display(report.head())
        display(report.groupby("slice").agg(episodes=("scene_id", "count"), collision_rate=("collision", "mean"),
                                             min_gap_m=("min_gap_m", "mean"), p95_latency_ms=("p95_latency_ms", "max")))
        '''),
        md('''
        ## Real-data checkpoint: replay a fixed nuScenes mini sample

        The course's default execution stays offline and synthetic. For a credible portfolio, run the same bundle schema on a fixed nuScenes mini sample and record dataset version, sample token, coordinate convention, time policy and the exact command. The important lesson is the **format of evidence**, not a fabricated benchmark number.
        '''),
        code('''
        import os
        real_checkpoint = {
            "dataset": "nuScenes mini (optional)",
            "sample_token": os.environ.get("NUSCENES_SAMPLE_TOKEN", "not supplied"),
            "status": "not run" if not os.environ.get("NUSCENES_ROOT") else "ready for fixed-sample replay",
            "required_record": ["dataset version", "sample token", "frame chain", "timestamp policy", "split"],
        }
        if os.environ.get("NUSCENES_ROOT"):
            try:
                from nuscenes.nuscenes import NuScenes
                nusc = NuScenes(version="v1.0-mini", dataroot=os.environ["NUSCENES_ROOT"], verbose=False)
                sample = nusc.sample[0] if real_checkpoint["sample_token"] == "not supplied" else nusc.get("sample", real_checkpoint["sample_token"])
                real_checkpoint.update({
                    "sample_token": sample["token"],
                    "lidar_file": nusc.get("sample_data", sample["data"]["LIDAR_TOP"])["filename"],
                    "camera_file": nusc.get("sample_data", sample["data"]["CAM_FRONT"])["filename"],
                    "status": "sample metadata loaded; run fixed replay and save output",
                })
            except ImportError as exc:
                real_checkpoint.update({"status": "dependency missing", "error": str(exc)})
        print(real_checkpoint)
        '''),
        code('''
        worst = report.sort_values(["collision", "p95_latency_ms"], ascending=[False, False]).head(3)
        print("corner-case candidates:")
        display(worst[["scene_id", "collision", "min_gap_m", "p95_latency_ms", "sensor_fault"]])
        save_json_artifact("08_eval_report.json", {
            "episodes": int(len(report)),
            "collision_rate": float(report.collision.mean()),
            "mean_min_gap_m": float(report.min_gap_m.mean()),
            "max_p95_latency_ms": float(report.p95_latency_ms.max()),
            "slice_counts": report["slice"].value_counts().to_dict(),
            "real_data_checkpoint": real_checkpoint,
            "next": "09_safety_runtime.ipynb",
        })
        print("saved evaluation artifact")
        '''),
        md('''
        ### 完成标准

        提交一个固定 scenario matrix、bundle schema、open/closed-loop 指标、slice breakdown 和一次 failure replay。至少做两个 ablation（例如 sensor age、LiDAR dropout、planner guard），并记录为什么平均 composite score 不能代替安全门禁。
        '''),
    ]


def course_09() -> list[nbf.NotebookNode]:
    return [
        md('''
        # 09 · Safety & Runtime：accuracy 之外的发布门禁

        本章把旧版 `04`、`18` 和 `15` 合并。一个 learned policy 可能置信度高但输入 stale、定位漂移、TTC 很小或 p99 超时；L4 系统需要独立的 health/safety state machine 和 degraded mode，并把 runtime 证据加入 gate。
        '''),
        code(COMMON + '''
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        eval_report = load_json_artifact("08_eval_report.json")
        model_metrics = load_json_artifact("05_bev_metrics.json") if (ARTIFACT_DIR / "05_bev_metrics.json").exists() else {"runtime": {}}
        '''),
        code('''
        def safety_state(sensor_health, localization_sigma_m, latency_ms, ttc_s, odd_ok=True):
            if not odd_ok:
                return "ODD_EXIT"
            if sensor_health < 0.45 or localization_sigma_m > 1.5 or latency_ms > 150 or ttc_s < 1.0:
                return "MINIMAL_RISK"
            if sensor_health < 0.70 or localization_sigma_m > 0.8 or latency_ms > 100 or ttc_s < 2.0:
                return "DEGRADED"
            return "NOMINAL"

        test_rows = []
        for sensor_health in [0.95, 0.62, 0.35]:
            for ttc in [3.5, 1.6, 0.7]:
                test_rows.append({"sensor_health": sensor_health, "ttc_s": ttc,
                                  "state": safety_state(sensor_health, 0.4, 85, ttc)})
        state_table = pd.DataFrame(test_rows)
        display(state_table)
        print("state counts:", state_table.state.value_counts().to_dict())
        '''),
        md('''
        ## Runtime evidence: p50/p95/p99 and watchdog violations

        Runtime is not an afterthought. Report batch size, hardware, warm-up policy, precision, model version and latency distribution. A p50 that passes while p99 violates the watchdog still produces a system failure.
        '''),
        code('''
        rng = np.random.default_rng(51)
        latency = 60.0 + rng.lognormal(mean=2.0, sigma=0.32, size=2000)
        runtime_table = {
            "p50_ms": float(np.percentile(latency, 50)),
            "p95_ms": float(np.percentile(latency, 95)),
            "p99_ms": float(np.percentile(latency, 99)),
            "watchdog_budget_ms": 100.0,
            "watchdog_violation_rate": float(np.mean(latency > 100.0)),
        }
        runtime_table.update({f"model_{key}": value for key, value in model_metrics.get("runtime", {}).items()})
        print(runtime_table)
        plt.hist(latency, bins=40, alpha=0.8)
        plt.axvline(100, color="red", linestyle="--", label="watchdog")
        plt.legend()
        plt.xlabel("latency / ms")
        plt.title("Runtime distributions, not just averages")
        plt.show()
        '''),
        code('''
        def release_gate(collision_rate, fallback_recall, p95_ms, p99_ms, max_collision=0.05):
            checks = {
                "collision_rate": collision_rate < max_collision,
                "fallback_recall": fallback_recall >= 0.95,
                "p95_latency": p95_ms < 100.0,
                "p99_latency": p99_ms < 130.0,
            }
            return checks, all(checks.values())

        checks, passed = release_gate(eval_report["collision_rate"], 0.97,
                                      runtime_table["p95_ms"], runtime_table["p99_ms"])
        print("release gate:", checks, "PASS" if passed else "HOLD / INVESTIGATE")
        save_json_artifact("09_safety_runtime.json", {
            "state_table": state_table.to_dict(orient="records"),
            "runtime": runtime_table,
            "release_checks": checks,
            "next": "10_capstone.ipynb",
        })
        '''),
        md('''
        ### 完成标准

        构造至少三个 adversarial cases：高 confidence + stale input、低 TTC + nominal latency、p99 超时 + 看似高 accuracy。每个 case 都要有检测信号、系统状态、fallback/degraded action 和恢复条件。不要把 safety gate 当成“把模型阈值调低”。
        '''),
    ]


def course_10() -> list[nbf.NotebookNode]:
    return [
        md('''
        # 10 · 归档审计：模型与报告之间还缺什么连接？

        本节检查旧章节生成的文件，并运行一次 BEV 模型推理。下图区分实际执行和原设计中尚未执行的连接：

        ```text
        实际执行：05_bev_model.pt → BEV risk/occupancy inference
        实际执行：读取 03/06/07/08/09 的旧文件 → 汇总表

        原设计设想（本节未执行）：
        BEV inference --未连接--> prediction --未连接--> ego planner
        ego planner --未连接--> env.step --未连接--> 新的闭环评测
        ```

        审计练习：找出实际推理调用和汇总表的输入。若要判断换模型后车是否开得更好，还需要在哪个函数中把模型输出送入规划，再调用 `env.step`？现有执行结果只支持文件可读取和推理可运行。
        '''),
        code(COMMON + '''
        import json
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        required = [
            "00_system_contract.json", "01_geometry.json", "02_bev_dataset.npz",
            "03_temporal_state.npz", "04_localization.npz", "05_bev_model.pt",
            "06_prediction.npz", "07_planner.npz", "08_eval_report.json", "09_safety_runtime.json",
        ]
        artifact_status = {name: (ARTIFACT_DIR / name).exists() for name in required}
        print(artifact_status)
        missing = [name for name, present in artifact_status.items() if not present]
        if missing:
            raise FileNotFoundError("Run Chapters 00–09 in order; missing artifacts: " + ", ".join(missing))
        '''),
        md('''
        ## 1. Load the learned BEV artifact and run one inference

        The model class is shared with Chapter 05 so this is not a prose-only reference to a checkpoint. We load the exact state dict, run one sample from the same BEV artifact, and expose a risk summary to the downstream report. The planner remains a separate component; it should not silently become a learned model just because a checkpoint exists.
        '''),
        code('''
        import torch
        from ad_tutorial.bev_model import LearnableBEVModel

        bev_data = load_numpy_artifact("02_bev_dataset.npz")
        checkpoint = torch.load(ARTIFACT_DIR / "05_bev_model.pt", map_location="cpu", weights_only=False)
        config = checkpoint["model_config"]
        bev_model = LearnableBEVModel(**config)
        bev_model.load_state_dict(checkpoint["state_dict"])
        bev_model.eval()
        with torch.no_grad():
            logits = bev_model(torch.tensor(bev_data["features"][:1], dtype=torch.float32))
            risk_probability = torch.sigmoid(logits[0, 1]).numpy()
        print("loaded BEV checkpoint:", config)
        print("predicted risk cells:", int((risk_probability > 0.5).sum()), "max risk:", float(risk_probability.max()))
        '''),
        code('''
        prediction = load_numpy_artifact("06_prediction.npz")
        planner = load_numpy_artifact("07_planner.npz")
        eval_report = load_json_artifact("08_eval_report.json")
        safety = load_json_artifact("09_safety_runtime.json")
        temporal = load_numpy_artifact("03_temporal_state.npz")

        summary = pd.DataFrame([
            {"stage": "BEV model", "artifact": "05_bev_model.pt", "evidence": f"risk cells={int((risk_probability > 0.5).sum())}"},
            {"stage": "Prediction", "artifact": "06_prediction.npz", "evidence": f"modes={prediction['candidates'].shape[0]}, minFDE={prediction['per_mode_fde'].min():.2f}m"},
            {"stage": "Planner", "artifact": "07_planner.npz", "evidence": f"selected_mode={int(planner['selected_mode'][0])}"},
            {"stage": "Evaluation", "artifact": "08_eval_report.json", "evidence": f"collision_rate={eval_report['collision_rate']:.3f}"},
            {"stage": "Safety/runtime", "artifact": "09_safety_runtime.json", "evidence": str(safety['release_checks'])},
        ])
        display(summary)
        '''),
        code('''
        # One compact failure replay: stale sensor input + cut-in risk.
        failure = {
            "scenario_id": "urban_cut_in_failure_replay",
            "root_cause_hypothesis": "sensor age and cut-in mode were not reflected in nominal policy",
            "observed_signals": {"risk_cells": int((risk_probability > 0.5).sum()),
                                 "min_prediction_fde_m": float(prediction["per_mode_fde"].min()),
                                 "eval_collision_rate": float(eval_report["collision_rate"])},
            "safety_action": "DEGRADED or MINIMAL_RISK depending on TTC/latency",
            "next_experiment": "fixed nuScenes mini replay with the same bundle/slice contract",
        }
        print(json.dumps(failure, indent=2))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(risk_probability.T, origin="lower", aspect="auto", cmap="magma")
        ax.set(title="Loaded BEV risk map from the trained artifact", xlabel="x cell", ylabel="y cell")
        plt.show()
        '''),
        md('''
        ## Portfolio handoff

        Commit these items together:

        - exact environment/requirements and command;
        - data/scene version, coordinate and timestamp policy;
        - model checkpoint metadata, seed and training split;
        - open-loop metrics, closed-loop metrics, safety gate, p50/p95/p99 latency;
        - at least two ablations and one failure replay;
        - explicit boundary: synthetic mechanism tutorial ≠ public benchmark ≠ vehicle safety case.

        Optional labs now sit after this core: Flow Matching/Action Chunk, VLM structured conditions, and VLA/WA/π0. They extend interfaces already built here; they do not replace the core perception–state–planning–safety chain.
        '''),
    ]


def lab_flow() -> list[nbf.NotebookNode]:
    return [
        md('''
        # Advanced Lab · Flow Matching / Action Chunk

        Optional lab. Complete Chapters 06–07 first. Here an action model predicts a short trajectory chunk, and the planner decides how many steps to execute before re-planning. The point is to connect a generative mechanism to action-space, horizon, latency and closed-loop drift—not to move it into the prerequisite path.
        '''),
        code(COMMON + '''
        import numpy as np
        import matplotlib.pyplot as plt

        prediction = load_numpy_artifact("06_prediction.npz")
        target = prediction["truth_future"]
        rng = np.random.default_rng(3)
        noise = rng.normal(0, 1.0, target.shape)
        times = np.linspace(0.0, 1.0, len(target))[:, None]
        samples = noise * (1.0 - times) + target * times
        '''),
        code('''
        def integrate_flow(initial, target, steps=8):
            state = initial.copy()
            for t in np.linspace(0.0, 1.0, steps):
                velocity = target - state
                state = state + velocity / steps
            return state

        initial = np.zeros_like(target)
        for steps in [2, 4, 8, 16]:
            generated = integrate_flow(initial, target, steps=steps)
            print("steps", steps, "terminal error", np.linalg.norm(generated[-1] - target[-1]).round(3))
        plt.plot(target[:, 0], target[:, 1], linewidth=3, label="target action chunk")
        plt.plot(samples[:, 0], samples[:, 1], "--", label="flow interpolation")
        plt.legend(); plt.axis("equal"); plt.title("Action chunk as a trajectory object"); plt.show()
        '''),
        md('''
        练习：改变 action chunk 长度、replan frequency 和 initial noise；报告 terminal error、closed-loop minimum gap 和 p95 latency。解释为什么更长 chunk 可能减少 compute，却增加 model-mismatch exposure。
        '''),
    ]


def lab_vlm() -> list[nbf.NotebookNode]:
    return [
        md('''
        # Advanced Lab · VLM Structured Driving Conditions

        Optional lab. VLM 的安全接口不是“让语言模型直接控制方向盘”，而是把图像/场景描述转成 schema-constrained、可验证、可拒答的 driving conditions，交给既有 BEV/planning/safety contract。

        当前实验不下载大型 checkpoint；`requirements-frontier.txt` 只在你把 `predict_conditions` 替换为真实 Hugging Face processor/model 时安装。
        '''),
        code(COMMON + '''
        import json
        import numpy as np

        SCHEMA = {"road_work": bool, "occluded_actor": bool, "cut_in_risk": float,
                  "confidence": float, "evidence": list}

        def validate_conditions(payload):
            required = set(SCHEMA)
            missing = required - set(payload)
            valid = not missing and 0.0 <= payload.get("cut_in_risk", -1) <= 1.0 and 0.0 <= payload.get("confidence", -1) <= 1.0
            return {"valid": valid, "missing": sorted(missing)}

        observation = {
            "road_work": False, "occluded_actor": True, "cut_in_risk": 0.82,
            "confidence": 0.71, "evidence": ["left actor crosses lane boundary", "partial occlusion"],
        }
        print(validate_conditions(observation))
        print(json.dumps(observation, indent=2))
        '''),
        md('''
        练习：加入 abstain/unknown 状态，构造 schema-valid but semantically-wrong output，并把低 confidence 或 contradiction 送入 Chapter 09 safety gate。评估 precision/recall/coverage，而不是只展示一段漂亮文字。
        '''),
    ]


def lab_vla() -> list[nbf.NotebookNode]:
    return [
        md('''
        # Advanced Lab · VLA / World-Action Interface

        Optional lab. VLA/WA/π0 类工作把视觉状态映射到动作 chunk，或学习 action/world consequences。这里把它限制在已有 planner/safety interface：action proposal 必须带时间窗、frame、uncertainty 和可拒绝条件。
        '''),
        code(COMMON + '''
        import numpy as np
        import matplotlib.pyplot as plt

        state = load_numpy_artifact("03_temporal_state.npz")
        rng = np.random.default_rng(9)
        current = state["estimate_xy"][-1]
        actions = np.array([[0.0, 0.0], [0.5, -0.3], [1.0, -0.8], [1.5, -1.2]])
        predicted_next = current[None, :] + actions + rng.normal(0, 0.08, actions.shape)
        safety_margin = np.array([4.0, 3.2, 1.7, 0.6])
        for action, next_state, margin in zip(actions, predicted_next, safety_margin):
            accepted = margin > 1.0
            print({"action": action.round(2).tolist(), "predicted_next": next_state.round(2).tolist(),
                   "safety_margin": float(margin), "accepted": bool(accepted)})
        '''),
        code('''
        plt.scatter(predicted_next[:, 0], predicted_next[:, 1], c=safety_margin, cmap="viridis", s=80)
        plt.scatter(current[0], current[1], color="red", label="current state")
        plt.colorbar(label="safety margin")
        plt.legend(); plt.axis("equal"); plt.title("World-action proposals need a safety interface"); plt.show()
        '''),
        md('''
        练习：加入 action chunk horizon、model uncertainty 和 latency budget；比较 behavior-cloning MSE 与 closed-loop safety；解释为什么 action proposals 不能绕过 Chapter 09 的 independent safety layer。
        '''),
    ]


def build() -> None:
    notebooks = {
        "course/00_system_and_odd.ipynb": course_00(),
        "course/01_sensors_geometry.ipynb": course_01(),
        "course/02_bev_and_fusion.ipynb": course_02(),
        "course/03_temporal_state.ipynb": course_03(),
        "course/04_localization_mapping.ipynb": course_04(),
        "course/05_learnable_bev_model.ipynb": course_05(),
        "course/06_prediction.ipynb": course_06(),
        "course/07_planning_closed_loop.ipynb": course_07(),
        "course/08_data_and_evaluation.ipynb": course_08(),
        "course/09_safety_runtime.ipynb": course_09(),
        "course/10_capstone.ipynb": course_10(),
        "labs/flow_matching_action_chunk.ipynb": lab_flow(),
        "labs/vlm_structured_driving_conditions.ipynb": lab_vlm(),
        "labs/vla_world_action_interface.ipynb": lab_vla(),
    }
    for relative_path, cells in notebooks.items():
        build_notebook(relative_path, cells)
    print(f"built {len(notebooks)} archived notebooks")


if __name__ == "__main__":
    build()
