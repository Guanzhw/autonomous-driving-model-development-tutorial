"""A deterministic, minimal urban cut-in scene shared by all core lessons.

This is not a simulator and it is not a vehicle model.  It is a teaching
fixture with explicit coordinate/time semantics so that the course can carry
one small piece of evidence from geometry to BEV, state estimation, planning,
and safety evaluation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "urban_cut_in"


@dataclass(frozen=True)
class BEVConfig:
    """Ego-frame BEV window and resolution.

    The convention is x forward, y left, z up.  Array rows follow x and array
    columns follow y; this is stated explicitly because silently transposing a
    BEV tensor is a common source of training/evaluation bugs.
    """

    x_min: float = 0.0
    x_max: float = 32.0
    y_min: float = -12.0
    y_max: float = 12.0
    resolution: float = 1.0

    @property
    def height(self) -> int:
        return int(round((self.x_max - self.x_min) / self.resolution))

    @property
    def width(self) -> int:
        return int(round((self.y_max - self.y_min) / self.resolution))

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        x = self.x_min + (np.arange(self.height) + 0.5) * self.resolution
        y = self.y_min + (np.arange(self.width) + 0.5) * self.resolution
        return np.meshgrid(x, y, indexing="ij")


@dataclass
class UrbanCutInScene:
    """One synchronized scene sample in the ego frame."""

    scene_id: str
    timestamp_s: float
    ego_pose_xy: np.ndarray
    cut_in_xy: np.ndarray
    lead_xy: np.ndarray
    camera_points: np.ndarray
    lidar_points: np.ndarray
    camera_timestamp_s: float
    lidar_timestamp_s: float
    metadata: dict[str, Any]

    def as_metadata(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "timestamp_s": self.timestamp_s,
            "camera_timestamp_s": self.camera_timestamp_s,
            "lidar_timestamp_s": self.lidar_timestamp_s,
            "metadata": self.metadata,
        }


def ensure_artifact_dir() -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIR


def _make_actor_points(center: np.ndarray, size: tuple[float, float, float],
                       rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample sparse surface points for a box-like actor."""

    half = np.asarray(size, dtype=float) / 2.0
    points = rng.uniform(-1.0, 1.0, size=(n, 3)) * half
    # Concentrate a few roof/side points so the cloud is visually identifiable.
    points[:, 2] = np.abs(points[:, 2])
    return points + center


def build_urban_cut_in_scene(seed: int = 7, timestamp_s: float = 0.0,
                             sensor_lag_s: float = 0.0,
                             scene_id: str | None = None) -> UrbanCutInScene:
    """Create one deterministic urban cut-in scene.

    The cut-in actor is to the left of the ego vehicle and moving toward the
    ego lane.  The lead actor is in-lane.  Camera points are a noisy, partial
    observation while LiDAR points are sparser but metrically useful.
    """

    rng = np.random.default_rng(seed)
    t = float(timestamp_s)
    ego_pose = np.array([0.45 * t, 0.0], dtype=float)
    cut_in = np.array([15.5 - 0.8 * t, 3.8 - 0.75 * t], dtype=float)
    lead = np.array([22.0 - 0.35 * t, 0.2], dtype=float)
    cut_points = _make_actor_points(
        np.r_[cut_in, 0.8], (4.2, 1.9, 1.6), rng, n=170
    )
    lead_points = _make_actor_points(
        np.r_[lead, 0.8], (4.5, 1.9, 1.6), rng, n=150
    )
    road_points = np.column_stack([
        rng.uniform(2.0, 31.0, 80),
        rng.uniform(-10.0, 10.0, 80),
        rng.normal(0.0, 0.025, 80),
    ])
    lidar = np.vstack([cut_points, lead_points, road_points])
    camera_keep = rng.random(len(lidar)) > 0.27
    camera_points = lidar[camera_keep].copy()
    camera_points += rng.normal(0.0, [0.08, 0.08, 0.04], camera_points.shape)
    # A lagged camera observation includes a small ego-motion-induced shift.
    camera_points[:, 0] -= 0.45 * float(sensor_lag_s)

    return UrbanCutInScene(
        scene_id=scene_id or f"urban_cut_in_{seed:03d}_{int(t * 10):04d}",
        timestamp_s=t,
        ego_pose_xy=ego_pose,
        cut_in_xy=cut_in,
        lead_xy=lead,
        camera_points=camera_points,
        lidar_points=lidar,
        camera_timestamp_s=t - float(sensor_lag_s),
        lidar_timestamp_s=t,
        metadata={
            "scenario": "urban_cut_in",
            "ego_frame": {"x": "forward_m", "y": "left_m", "z": "up_m"},
            "actors": ["cut_in", "lead"],
            "cut_in_velocity_mps": [-0.8, -0.75],
            "sensor_lag_s": float(sensor_lag_s),
        },
    )


def rasterize_points(points_xyz: np.ndarray, config: BEVConfig) -> np.ndarray:
    """Rasterize xyz points into a binary occupancy grid."""

    points = np.asarray(points_xyz)
    x_index = np.floor((points[:, 0] - config.x_min) / config.resolution).astype(int)
    y_index = np.floor((points[:, 1] - config.y_min) / config.resolution).astype(int)
    valid = (
        (x_index >= 0) & (x_index < config.height)
        & (y_index >= 0) & (y_index < config.width)
    )
    grid = np.zeros((config.height, config.width), dtype=np.float32)
    grid[x_index[valid], y_index[valid]] = 1.0
    return grid


def actor_cell(center_xy: np.ndarray, config: BEVConfig) -> tuple[int, int] | None:
    x, y = np.asarray(center_xy, dtype=float)
    i = int(np.floor((x - config.x_min) / config.resolution))
    j = int(np.floor((y - config.y_min) / config.resolution))
    if 0 <= i < config.height and 0 <= j < config.width:
        return i, j
    return None


def scene_to_bev(scene: UrbanCutInScene, config: BEVConfig | None = None) -> dict[str, np.ndarray]:
    """Build sensor features and semantic targets for one scene."""

    config = config or BEVConfig()
    camera = rasterize_points(scene.camera_points, config)
    lidar = rasterize_points(scene.lidar_points, config)
    camera_age = np.full_like(camera, scene.timestamp_s - scene.camera_timestamp_s)
    # Target occupancy uses the metrically cleaner LiDAR view; target risk is
    # deliberately semantic and highlights the cut-in actor rather than all
    # occupied cells.
    occupancy = (lidar > 0).astype(np.float32)
    risk = np.zeros_like(occupancy)
    velocity_x = np.zeros_like(occupancy)
    cut_cell = actor_cell(scene.cut_in_xy, config)
    if cut_cell:
        i, j = cut_cell
        risk[max(0, i - 1): min(config.height, i + 2), max(0, j - 1): min(config.width, j + 2)] = 1.0
        velocity_x[i, j] = -0.8
    lead_cell = actor_cell(scene.lead_xy, config)
    if lead_cell:
        i, j = lead_cell
        velocity_x[i, j] = -0.35
    features = np.stack([camera, lidar, np.clip(camera_age, 0.0, 1.0)], axis=0)
    targets = np.stack([occupancy, risk, velocity_x], axis=0)
    return {"features": features, "targets": targets, "camera": camera, "lidar": lidar}


def build_bev_dataset(n: int = 96, seed: int = 101,
                      config: BEVConfig | None = None) -> dict[str, Any]:
    """Generate variants of the same cut-in task for the learnable BEV model."""

    config = config or BEVConfig()
    features, targets, metadata = [], [], []
    for index in range(n):
        local_seed = seed + index
        lag = 0.0 if index % 4 else 0.15
        scene = build_urban_cut_in_scene(
            seed=local_seed, timestamp_s=0.1 * (index % 8),
            sensor_lag_s=lag, scene_id=f"cut_in_train_{index:04d}"
        )
        encoded = scene_to_bev(scene, config)
        features.append(encoded["features"])
        targets.append(encoded["targets"])
        metadata.append(scene.as_metadata())
    return {
        "features": np.stack(features).astype(np.float32),
        "targets": np.stack(targets).astype(np.float32),
        "metadata": metadata,
        "config": asdict(config),
        "scenario": "urban_cut_in",
    }


def save_numpy_artifact(name: str, **arrays: np.ndarray) -> Path:
    path = ensure_artifact_dir() / name
    np.savez_compressed(path, **arrays)
    return path


def load_numpy_artifact(name: str) -> dict[str, np.ndarray]:
    path = ARTIFACT_DIR / name
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def save_json_artifact(name: str, payload: dict[str, Any]) -> Path:
    path = ensure_artifact_dir() / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def load_json_artifact(name: str) -> dict[str, Any]:
    return json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))
