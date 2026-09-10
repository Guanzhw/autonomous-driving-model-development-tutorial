from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ad_tutorial.driving import DrivingConfig, EpisodeResult, run_episode  # noqa: E402
from ad_tutorial.imitation import (  # noqa: E402
    BCTrainingConfig,
    DemonstrationDataset,
    collect_demonstrations,
    dataset_hash,
    features_from_trace,
    fit_normalizer,
    load_checkpoint,
    offline_mse,
    paired_common_window,
    save_checkpoint,
    train_behavioral_cloning,
)


def _synthetic_dataset(split="train", offset=0.0):
    features = np.asarray(
        [
            [-0.2 + offset, 0.1, 2.0, 0.1],
            [0.1 + offset, -0.1, 3.0, -0.1],
            [0.0 + offset, 0.0, 4.0, 0.0],
        ],
        dtype=np.float32,
    )
    actions = np.tanh(features[:, :2]).astype(np.float32)
    return DemonstrationDataset(
        features,
        actions,
        tuple(f"{split}-episode" for _ in features),
        tuple(split for _ in features),
    )


def test_episode_split_and_train_only_normalization():
    train = _synthetic_dataset()
    test = _synthetic_dataset("test", offset=100.0)
    normalizer = fit_normalizer(train.features)
    assert np.allclose(normalizer.mean, train.features.mean(axis=0))
    assert not np.allclose(
        normalizer.mean, np.concatenate([train.features, test.features]).mean(axis=0)
    )
    dataset = DemonstrationDataset(
        np.concatenate([train.features, test.features]),
        np.concatenate([train.actions, test.actions]),
        train.episode_ids + test.episode_ids,
        train.splits + test.splits,
    )
    assert set(dataset.select("train").unique_episode_ids).isdisjoint(
        dataset.select("test").unique_episode_ids
    )
    assert dataset_hash(dataset) == dataset_hash(dataset)


def test_dataset_rejects_episode_id_in_multiple_splits():
    with pytest.raises(ValueError, match="multiple dataset splits"):
        DemonstrationDataset(
            np.zeros((2, 4), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            ("episode-1", "episode-1"),
            ("train", "test"),
        )


def test_training_changes_parameters_and_checkpoint_reloads_actions(tmp_path):
    dataset = _synthetic_dataset()
    from ad_tutorial.imitation import BehavioralCloningModel

    torch.manual_seed(7)
    before = [
        parameter.detach().clone()
        for parameter in BehavioralCloningModel().parameters()
    ]
    result = train_behavioral_cloning(
        dataset, config=BCTrainingConfig(epochs=4, patience=4)
    )
    after = list(result.model.parameters())
    assert np.isfinite([row["train_mse"] for row in result.history]).all()
    assert any(not torch.allclose(old, new) for old, new in zip(before, after))
    path = save_checkpoint(tmp_path / "bc.pt", result)
    reloaded = load_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["training_config"]["epochs"] == 4
    assert payload["training_config"]["seed"] == 7
    with torch.inference_mode():
        expected = result.model(
            torch.from_numpy(result.normalizer.transform(dataset.features))
        ).numpy()
        actual = reloaded.model(
            torch.from_numpy(reloaded.normalizer.transform(dataset.features))
        ).numpy()
    assert np.allclose(expected, actual)
    assert offline_mse(reloaded.model, reloaded.normalizer, dataset)["mse"] < 1.0


def test_real_learned_policy_changes_actions_and_trajectory():
    config = DrivingConfig(seed=7, horizon=24, initial_lateral_offset_m=0.2)
    demonstrations = collect_demonstrations({"train": [config]})
    result = train_behavioral_cloning(
        demonstrations.select("train"),
        config=BCTrainingConfig(epochs=20, patience=20, seed=9),
    )
    from ad_tutorial.imitation import BehavioralCloningPolicy

    expert = run_episode(config)
    learned = run_episode(
        config, policy=BehavioralCloningPolicy(result.model, result.normalizer)
    )
    expert_actions = np.asarray(
        [[row["command_steering"], row["command_throttle"]] for row in expert.trace]
    )
    learned_actions = np.asarray(
        [[row["command_steering"], row["command_throttle"]] for row in learned.trace]
    )
    assert not np.allclose(expert_actions, learned_actions, atol=1e-8)
    n = min(len(expert.trace), len(learned.trace))
    assert not np.allclose(
        [row["y_m"] for row in expert.trace[:n]],
        [row["y_m"] for row in learned.trace[:n]],
        atol=1e-8,
    )


def test_features_are_reconstructed_from_present_state_trace():
    row = {
        "before_lateral_error_m": 0.2,
        "before_heading_rad": 0.1,
        "before_speed_mps": 3.0,
        "reference_heading_rad": 0.0,
        "before_x_m": 1.0,
        "before_y_m": 2.0,
        "reference_x_m": 5.0,
        "reference_y_m": 2.0,
        "command_steering": 0.3,
        "command_throttle": 0.1,
        "reward": 999,
        "x_m": 999,
    }
    values = features_from_trace([row])
    assert np.allclose(values[0], [0.2, 0.1, 3.0, 0.0])


def test_common_window_is_paired_per_test_configuration():
    def episode(steps, seed):
        trace = [
            {"lateral_error_m": float(step), "distance_m": 1.0} for step in range(steps)
        ]
        return EpisodeResult(
            {
                "seed": seed,
                "initial_lateral_offset_m": 0.4,
                "horizon": 5,
                "decision_repeat": 5,
            },
            {
                "steps": steps,
                "outcome": "failure" if steps < 5 else "horizon",
                "failure": steps < 5,
                "mean_abs_lateral_error_m": 0.0,
                "distance_traveled_m": float(steps),
            },
            trace,
        )

    paired = paired_common_window(
        {
            "expert": [episode(5, 1), episode(2, 2)],
            "bc": [episode(3, 1), episode(4, 2)],
        }
    )
    assert [row["common_steps"] for row in paired["per_config"]] == [3, 2]
    assert paired["aggregate"]["expert"]["mean_distance_m"] == 2.5
    assert paired["aggregate"]["bc"]["mean_distance_m"] == 2.5
