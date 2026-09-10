"""Small, inspectable behavioural cloning experiment for the driving lessons.

The module deliberately keeps the learning boundary narrow: a geometric MetaDrive
controller supplies state/action demonstrations, a CPU MLP fits those actions, and
the fitted policy is sent through ``driving.run_episode`` for closed-loop evaluation.
No action, reward, terminal flag, or future state is part of the feature vector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .driving import (
    DrivingConfig,
    DrivingObservation,
    EpisodeResult,
    LaneReference,
    run_episode,
)


FEATURE_NAMES = (
    "e_y_m",
    "heading_error_rad",
    "speed_mps",
    "reference_bearing_rad",
)
ACTION_NAMES = ("steering", "throttle")


def _wrap_angle(value: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(value) + np.pi) % (2 * np.pi) - np.pi


def extract_features(
    observation: DrivingObservation, reference: LaneReference
) -> np.ndarray:
    """Return the four privileged, present-state features used by the policy.

    ``reference_bearing_rad`` is the bearing from the vehicle to the planned
    look-ahead point.  ``heading_error_rad`` is vehicle heading relative to the
    reference lane heading.  Keeping both makes the distinction between lane
    alignment and where the vehicle is currently looking explicit.
    """

    delta = np.asarray(reference.position, dtype=np.float32) - np.asarray(
        observation.position, dtype=np.float32
    )
    bearing = float(np.arctan2(delta[1], delta[0]))
    return np.asarray(
        [
            observation.lane_lateral_m,
            float(_wrap_angle(observation.heading_rad - reference.heading_rad)),
            observation.speed_mps,
            bearing,
        ],
        dtype=np.float32,
    )


def features_from_trace(trace: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Reconstruct features from trace fields without reading action fields."""

    values = []
    for row in trace:
        delta_x = float(row["reference_x_m"]) - float(row["before_x_m"])
        delta_y = float(row["reference_y_m"]) - float(row["before_y_m"])
        values.append(
            [
                float(row["before_lateral_error_m"]),
                float(
                    _wrap_angle(
                        float(row["before_heading_rad"])
                        - float(row["reference_heading_rad"])
                    )
                ),
                float(row["before_speed_mps"]),
                float(np.arctan2(delta_y, delta_x)),
            ]
        )
    result = np.asarray(values, dtype=np.float32)
    return result.reshape((-1, len(FEATURE_NAMES)))


def actions_from_trace(trace: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Read only the expert commands that correspond to each present state."""

    result = np.asarray(
        [[row["command_steering"], row["command_throttle"]] for row in trace],
        dtype=np.float32,
    )
    return result.reshape((-1, len(ACTION_NAMES)))


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    split: str
    config: dict[str, Any]
    features: np.ndarray
    actions: np.ndarray
    expert_metrics: dict[str, Any]
    trace: tuple[dict[str, Any], ...] = ()


@dataclass
class DemonstrationDataset:
    """Flattened samples plus episode provenance used for split auditing."""

    features: np.ndarray
    actions: np.ndarray
    episode_ids: tuple[str, ...]
    splits: tuple[str, ...]
    episodes: tuple[dict[str, Any], ...] = ()
    feature_names: tuple[str, ...] = FEATURE_NAMES
    action_names: tuple[str, ...] = ACTION_NAMES

    def __post_init__(self) -> None:
        self.features = np.asarray(self.features, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        if self.features.ndim != 2 or self.features.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"features must have shape (N, {len(FEATURE_NAMES)})")
        if self.actions.shape != (len(self.features), len(ACTION_NAMES)):
            raise ValueError("actions must have shape (N, 2), matching features")
        if not np.all(np.isfinite(self.features)) or not np.all(
            np.isfinite(self.actions)
        ):
            raise ValueError("features and actions must be finite")
        if len(self.episode_ids) != len(self.features) or len(self.splits) != len(
            self.features
        ):
            raise ValueError("episode_ids and splits must have one entry per sample")
        episode_splits: dict[str, str] = {}
        for episode_id, split in zip(self.episode_ids, self.splits):
            previous = episode_splits.setdefault(episode_id, split)
            if previous != split:
                raise ValueError(
                    f"episode_id {episode_id!r} occurs in multiple dataset splits"
                )

    @property
    def size(self) -> int:
        return int(len(self.features))

    @property
    def unique_episode_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.episode_ids))

    def select(self, split: str) -> "DemonstrationDataset":
        mask = np.asarray([part == split for part in self.splits], dtype=bool)
        return self.select_mask(mask)

    def select_mask(self, mask: np.ndarray) -> "DemonstrationDataset":
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.size,):
            raise ValueError("mask must have one value per sample")
        selected_ids = tuple(np.asarray(self.episode_ids, dtype=object)[mask].tolist())
        selected_splits = tuple(np.asarray(self.splits, dtype=object)[mask].tolist())
        selected_id_set = set(selected_ids)
        selected_episodes = tuple(
            episode
            for episode in self.episodes
            if episode.get("episode_id") in selected_id_set
        )
        return DemonstrationDataset(
            self.features[mask],
            self.actions[mask],
            selected_ids,
            selected_splits,
            selected_episodes,
            self.feature_names,
            self.action_names,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "samples": self.size,
            "feature_names": list(self.feature_names),
            "action_names": list(self.action_names),
            "episode_ids": list(self.unique_episode_ids),
            "splits": {
                split: {
                    "samples": int(sum(part == split for part in self.splits)),
                    "episode_ids": list(
                        dict.fromkeys(
                            episode
                            for episode, part in zip(self.episode_ids, self.splits)
                            if part == split
                        )
                    ),
                }
                for split in sorted(set(self.splits))
            },
        }


def _config_dict(config: DrivingConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, DrivingConfig):
        return asdict(config)
    return asdict(DrivingConfig.from_mapping(dict(config)))


def collect_demonstrations(
    split_configs: Mapping[str, Iterable[DrivingConfig | Mapping[str, Any]]],
    *,
    max_steps_per_episode: int | None = None,
) -> DemonstrationDataset:
    """Collect geometric-controller samples from real MetaDrive episodes.

    Configurations are grouped before flattening.  Every sample keeps its full
    episode id, so callers cannot accidentally create a timestep-level split.
    """

    if max_steps_per_episode is not None and max_steps_per_episode < 1:
        raise ValueError("max_steps_per_episode must be positive")
    records: list[EpisodeRecord] = []
    scenario_splits: dict[str, str] = {}
    for split, configs in split_configs.items():
        for index, raw_config in enumerate(configs):
            config = (
                DrivingConfig.from_mapping(dict(raw_config))
                if isinstance(raw_config, Mapping)
                else raw_config
            )
            if not isinstance(config, DrivingConfig):
                raise TypeError("episode configs must be DrivingConfig or mappings")
            result = run_episode(config)
            trace = (
                result.trace[:max_steps_per_episode]
                if max_steps_per_episode
                else result.trace
            )
            if not trace:
                raise ValueError(f"episode {split}-{index} produced no trace rows")
            scenario_key = json.dumps(_config_dict(config), sort_keys=True)
            previous_split = scenario_splits.setdefault(scenario_key, split)
            if previous_split != split:
                raise ValueError(
                    "the same initial condition/configuration cannot occur in multiple splits"
                )
            records.append(
                EpisodeRecord(
                    episode_id=f"{split}-{index:03d}-seed{config.seed}-offset{config.initial_lateral_offset_m:g}",
                    split=split,
                    config=dict(result.config),
                    features=features_from_trace(trace),
                    actions=actions_from_trace(trace),
                    expert_metrics=dict(result.metrics),
                    trace=tuple(dict(row) for row in trace),
                )
            )

    if not records:
        raise ValueError("at least one episode configuration is required")
    return dataset_from_records(records)


def dataset_from_records(records: Sequence[EpisodeRecord]) -> DemonstrationDataset:
    features = np.concatenate([record.features for record in records], axis=0)
    actions = np.concatenate([record.actions for record in records], axis=0)
    ids = tuple(
        value
        for record in records
        for value in [record.episode_id] * len(record.features)
    )
    splits = tuple(
        value for record in records for value in [record.split] * len(record.features)
    )
    episodes = tuple(
        {
            "episode_id": record.episode_id,
            "split": record.split,
            "config": record.config,
            "steps": len(record.features),
            "expert_metrics": record.expert_metrics,
            "trace": list(record.trace),
        }
        for record in records
    )
    return DemonstrationDataset(features, actions, ids, splits, episodes)


def dataset_hash(dataset: DemonstrationDataset) -> str:
    """Hash ordered arrays and provenance, making dataset changes visible."""

    digest = hashlib.sha256()
    digest.update(dataset.features.tobytes())
    digest.update(dataset.actions.tobytes())
    digest.update(json.dumps(dataset.manifest(), sort_keys=True).encode("utf-8"))
    digest.update(
        json.dumps(dataset.episodes, sort_keys=True, default=str).encode("utf-8")
    )
    return digest.hexdigest()


@dataclass(frozen=True)
class FeatureNormalizer:
    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        mean, std = (
            np.asarray(self.mean, dtype=np.float32),
            np.asarray(self.std, dtype=np.float32),
        )
        if mean.shape != (len(FEATURE_NAMES),) or std.shape != mean.shape:
            raise ValueError("normalizer statistics must have one value per feature")
        if (
            not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(std))
            or np.any(std <= 0)
        ):
            raise ValueError("normalizer statistics must be finite with positive std")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.shape[-1] != len(FEATURE_NAMES):
            raise ValueError("features have the wrong width")
        return (values - self.mean) / self.std


def fit_normalizer(train_features: np.ndarray) -> FeatureNormalizer:
    values = np.asarray(train_features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES) or not len(values):
        raise ValueError("train_features must be a non-empty (N, 4) array")
    std = values.std(axis=0)
    std[std < 1e-6] = 1.0
    return FeatureNormalizer(values.mean(axis=0), std)


class BehavioralCloningModel(nn.Module):
    """Small tanh MLP; its final range matches MetaDrive's normalized actions."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, len(ACTION_NAMES)),
            nn.Tanh(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


@dataclass(frozen=True)
class BCTrainingConfig:
    epochs: int = 80
    batch_size: int = 64
    learning_rate: float = 0.01
    weight_decay: float = 1e-5
    patience: int = 15
    seed: int = 7
    hidden_dim: int = 32

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("epochs, batch_size and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError(
                "learning_rate must be positive and weight_decay nonnegative"
            )


@dataclass
class TrainingResult:
    model: BehavioralCloningModel
    normalizer: FeatureNormalizer
    history: list[dict[str, float]]
    metrics: dict[str, Any]
    training_config: BCTrainingConfig = field(default_factory=BCTrainingConfig)


def _mse(
    model: BehavioralCloningModel,
    normalizer: FeatureNormalizer,
    dataset: DemonstrationDataset,
) -> float:
    if not dataset.size:
        return float("nan")
    model.eval()
    with torch.inference_mode():
        predictions = model(torch.from_numpy(normalizer.transform(dataset.features)))
        targets = torch.from_numpy(dataset.actions)
        return float(torch.mean((predictions - targets) ** 2).item())


def train_behavioral_cloning(
    train: DemonstrationDataset,
    validation: DemonstrationDataset | None = None,
    config: BCTrainingConfig | None = None,
) -> TrainingResult:
    """Fit on train episodes and select the checkpoint using validation episodes."""

    if not train.size:
        raise ValueError("training dataset is empty")
    config = config or BCTrainingConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    normalizer = fit_normalizer(train.features)
    model = BehavioralCloningModel(config.hidden_dim)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(normalizer.transform(train.features)),
            torch.from_numpy(train.actions),
        ),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(config.seed),
    )
    validation = validation if validation is not None else train
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        losses = []
        for batch_features, batch_actions in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_features), batch_actions)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        train_mse = _mse(model, normalizer, train)
        val_mse = _mse(model, normalizer, validation)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_mse": train_mse,
                "val_mse": val_mse,
                "batch_loss": float(np.mean(losses)),
            }
        )
        if val_mse < best_val - 1e-10:
            best_val, best_epoch, stale = val_mse, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best_state)
    metrics = {
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "train_mse": _mse(model, normalizer, train),
        "validation_mse": _mse(model, normalizer, validation),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "feature_names": list(FEATURE_NAMES),
        "action_names": list(ACTION_NAMES),
        "training_config": asdict(config),
    }
    return TrainingResult(model, normalizer, history, metrics, config)


def offline_mse(
    model: BehavioralCloningModel,
    normalizer: FeatureNormalizer,
    dataset: DemonstrationDataset,
) -> dict[str, float]:
    """Report offline action error; this says nothing about closed-loop recovery."""

    if not dataset.size:
        raise ValueError("offline evaluation dataset is empty")
    model.eval()
    with torch.inference_mode():
        prediction = model(
            torch.from_numpy(normalizer.transform(dataset.features))
        ).numpy()
    errors = (prediction - dataset.actions) ** 2
    return {
        "mse": float(errors.mean()),
        "steering_mse": float(errors[:, 0].mean()),
        "throttle_mse": float(errors[:, 1].mean()),
        "samples": dataset.size,
    }


class BehavioralCloningPolicy:
    """Adapter implementing the ``policy.control`` protocol used by driving.py."""

    def __init__(self, model: BehavioralCloningModel, normalizer: FeatureNormalizer):
        self.model = model.eval()
        self.normalizer = normalizer

    def control(
        self, observation: DrivingObservation, reference: LaneReference
    ) -> np.ndarray:
        features = extract_features(observation, reference).reshape(1, -1)
        with torch.inference_mode():
            action = self.model(
                torch.from_numpy(self.normalizer.transform(features))
            ).numpy()[0]
        return np.clip(action, -1, 1).astype(np.float32)


def make_untrained_policy(
    normalizer: FeatureNormalizer, *, seed: int = 7, hidden_dim: int = 32
) -> BehavioralCloningPolicy:
    torch.manual_seed(seed)
    return BehavioralCloningPolicy(BehavioralCloningModel(hidden_dim), normalizer)


def save_checkpoint(path: str | Path, result: TrainingResult) -> Path:
    """Save everything needed to reload the exact policy and train-only scaling."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "ad_tutorial.behavioral_cloning.v1",
            "model": result.model.state_dict(),
            "hidden_dim": result.model.hidden_dim,
            "normalizer_mean": result.normalizer.mean,
            "normalizer_std": result.normalizer.std,
            "feature_names": list(FEATURE_NAMES),
            "action_names": list(ACTION_NAMES),
            "training_metrics": result.metrics,
            "training_config": asdict(result.training_config),
            "history": result.history,
        },
        path,
    )
    return path


def load_checkpoint(path: str | Path) -> BehavioralCloningPolicy:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format") != "ad_tutorial.behavioral_cloning.v1":
        raise ValueError("unsupported behavioral cloning checkpoint")
    if (
        tuple(payload.get("feature_names", ())) != FEATURE_NAMES
        or tuple(payload.get("action_names", ())) != ACTION_NAMES
    ):
        raise ValueError(
            "checkpoint feature/action contract does not match this lesson"
        )
    model = BehavioralCloningModel(int(payload["hidden_dim"]))
    model.load_state_dict(payload["model"])
    normalizer = FeatureNormalizer(
        payload["normalizer_mean"], payload["normalizer_std"]
    )
    return BehavioralCloningPolicy(model, normalizer)


def default_split_configs(
    *, horizon: int = 60, decision_repeat: int = 5
) -> dict[str, list[DrivingConfig]]:
    """Create a small split: train offsets ±0.1/0.2, held-out test ±0.4."""

    def configs(
        split: str, offsets: Sequence[float], seeds: Sequence[int]
    ) -> list[DrivingConfig]:
        return [
            DrivingConfig(
                seed=seed,
                horizon=horizon,
                decision_repeat=decision_repeat,
                initial_lateral_offset_m=offset,
            )
            for seed in seeds
            for offset in offsets
        ]

    return {
        "train": configs("train", (-0.2, -0.1, 0.1, 0.2), (7, 11)),
        "validation": configs("validation", (-0.25, 0.25), (13,)),
        "test": configs("test", (-0.4, 0.4), (17, 19)),
    }


def evaluate_closed_loop(
    configs: Sequence[DrivingConfig],
    bc_policy: BehavioralCloningPolicy,
    *,
    untrained_policy: BehavioralCloningPolicy | None = None,
) -> dict[str, list[EpisodeResult]]:
    """Run expert, untrained, and BC policies on exactly the same configs."""

    if untrained_policy is None:
        untrained_policy = make_untrained_policy(bc_policy.normalizer)
    return {
        "expert": [run_episode(config) for config in configs],
        "untrained": [
            run_episode(config, policy=untrained_policy) for config in configs
        ],
        "bc": [run_episode(config, policy=bc_policy) for config in configs],
    }


def summarize_closed_loop(
    results: Mapping[str, Sequence[EpisodeResult]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, episodes in results.items():
        if not episodes:
            continue
        metrics = [episode.metrics for episode in episodes]
        summary[name] = {
            "episodes": len(episodes),
            "outcomes": {
                outcome: sum(metric["outcome"] == outcome for metric in metrics)
                for outcome in sorted({metric["outcome"] for metric in metrics})
            },
            "failure_rate": float(np.mean([metric["failure"] for metric in metrics])),
            "mean_abs_lateral_error_m": float(
                np.mean([metric["mean_abs_lateral_error_m"] for metric in metrics])
            ),
            "mean_distance_traveled_m": float(
                np.mean([metric["distance_traveled_m"] for metric in metrics])
            ),
            "mean_steps": float(np.mean([metric["steps"] for metric in metrics])),
            "episodes_detail": [dict(metric) for metric in metrics],
        }
    return summary


def paired_common_window(
    results: Mapping[str, Sequence[EpisodeResult]],
) -> dict[str, Any]:
    """Compare policy prefixes at each test index using one paired window.

    The common length is the minimum trace length across all policies for that
    exact test configuration. Aggregate values are an unweighted mean of the
    per-configuration values.
    """

    if not results:
        return {"per_config": [], "aggregate": {}}
    names = list(results)
    count = len(results[names[0]])
    if any(len(results[name]) != count for name in names):
        raise ValueError("all policies must have one episode per test configuration")
    per_config: list[dict[str, Any]] = []
    for index in range(count):
        episodes = {name: results[name][index] for name in names}
        common_steps = min(len(episode.trace) for episode in episodes.values())
        if common_steps < 1:
            raise ValueError(f"test configuration {index} has an empty trace")
        first_config = episodes[names[0]].config
        config = {
            key: first_config.get(key)
            for key in (
                "seed",
                "initial_lateral_offset_m",
                "horizon",
                "decision_repeat",
            )
        }
        policy_metrics = {}
        for name, episode in episodes.items():
            prefix = episode.trace[:common_steps]
            policy_metrics[name] = {
                "mean_abs_lateral_error_m": float(
                    np.mean([abs(row["lateral_error_m"]) for row in prefix])
                ),
                "distance_m": float(sum(row["distance_m"] for row in prefix)),
            }
        per_config.append(
            {
                "test_index": index,
                "config": config,
                "common_steps": common_steps,
                "policies": policy_metrics,
            }
        )
    aggregate = {
        name: {
            "configs": len(per_config),
            "mean_common_steps": float(
                np.mean([row["common_steps"] for row in per_config])
            ),
            "mean_abs_lateral_error_m": float(
                np.mean(
                    [
                        row["policies"][name]["mean_abs_lateral_error_m"]
                        for row in per_config
                    ]
                )
            ),
            "mean_distance_m": float(
                np.mean([row["policies"][name]["distance_m"] for row in per_config])
            ),
        }
        for name in names
    }
    return {"per_config": per_config, "aggregate": aggregate}


def write_dataset_manifest(path: str | Path, dataset: DemonstrationDataset) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = dataset.manifest()
    manifest.update(
        {"dataset_hash": dataset_hash(dataset), "episodes": list(dataset.episodes)}
    )
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def write_training_curve(
    path: str | Path, history: Sequence[Mapping[str, float]]
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in history]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["epoch", "train_mse", "val_mse", "batch_loss"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


__all__ = [
    "ACTION_NAMES",
    "FEATURE_NAMES",
    "BCTrainingConfig",
    "BehavioralCloningModel",
    "BehavioralCloningPolicy",
    "DemonstrationDataset",
    "EpisodeRecord",
    "FeatureNormalizer",
    "TrainingResult",
    "actions_from_trace",
    "collect_demonstrations",
    "dataset_from_records",
    "dataset_hash",
    "default_split_configs",
    "evaluate_closed_loop",
    "extract_features",
    "features_from_trace",
    "fit_normalizer",
    "load_checkpoint",
    "make_untrained_policy",
    "offline_mse",
    "paired_common_window",
    "save_checkpoint",
    "summarize_closed_loop",
    "train_behavioral_cloning",
    "write_dataset_manifest",
    "write_training_curve",
]
