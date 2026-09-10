import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from ad_tutorial.driving import DrivingConfig, DrivingObservation, run_episode  # noqa: E402
from ad_tutorial.estimation import (  # noqa: E402
    KalmanObserver,
    MeasurementConfig,
    RawMeasurementObserver,
    ScalarKalman,
    ego_to_world,
    world_to_ego,
)


def test_se2_round_trip_and_numeric_example():
    assert np.allclose(world_to_ego((10, 5), (10, 2), np.pi / 2), (3, 0))
    rng = np.random.default_rng(3)
    for _ in range(20):
        pose = rng.normal(size=2)
        angle = rng.uniform(-np.pi, np.pi)
        point = rng.normal(size=2)
        assert np.allclose(
            ego_to_world(world_to_ego(point, pose, angle), pose, angle), point
        )


def test_scalar_kalman_reduces_stationary_measurement_noise():
    rng = np.random.default_rng(4)
    truth = 2.0
    measurements = truth + rng.normal(0, 0.5, size=200)
    filter_ = ScalarKalman(0.5**2, process_variance=0.005)
    estimates = np.asarray([filter_.update(value, 0.1) for value in measurements])
    assert np.sqrt(np.mean((estimates[10:] - truth) ** 2)) < np.sqrt(
        np.mean((measurements[10:] - truth) ** 2)
    )


def test_filter_has_no_future_truth_dependency():
    config = MeasurementConfig(
        seed=13,
        position_bias_m=(0.0, 0.0),
        heading_bias_rad=0.0,
        speed_bias_mps=0.0,
    )
    first = DrivingObservation((1, 2), 0.1, 3, 1, 0.2, 0, 3.5)
    future_a = DrivingObservation((2, 2), 0.1, 3, 2, 0.2, 0, 3.5)
    future_b = DrivingObservation((200, -80), -2, 30, 200, -8, 1, 3.5)
    left = KalmanObserver(config)
    right = KalmanObserver(config)
    assert left.observe(first, 0, 0.1) == right.observe(first, 0, 0.1)
    assert left.observe(future_a, 1, 0.1) != right.observe(future_b, 1, 0.1)


def test_heading_filter_uses_circular_innovation_at_pi_boundary():
    config = MeasurementConfig(
        seed=2,
        position_noise_std_m=(0.0, 0.0),
        heading_noise_std_rad=0.05,
        speed_noise_std_mps=0.0,
        position_bias_m=(0.0, 0.0),
        heading_process_variance_rad2_per_s=0.0001,
    )
    observer = KalmanObserver(config)
    first = DrivingObservation((0, 0), np.pi - 0.02, 0, 0, 0, 0, 3.5)
    second = replace(first, heading_rad=-np.pi + 0.02)
    observer.reset(None)
    observer.observe(first, 0, 0.1)
    estimate = observer.observe(second, 1, 0.1).heading_rad
    error = (estimate - second.heading_rad + np.pi) % (2 * np.pi) - np.pi
    assert abs(error) < 0.15


def test_observer_reset_once_per_run_and_baseline_hook_regression():
    class CountingObserver(RawMeasurementObserver):
        def __init__(self, config):
            self.reset_count = 0
            super().__init__(config)

        def reset(self, lane=None):
            self.reset_count += 1
            super().reset(lane)

    observer = CountingObserver(MeasurementConfig(position_bias_m=(0, 0)))
    result = run_episode(DrivingConfig(horizon=8), observer=observer)
    assert observer.reset_count == 2  # construction plus exactly one run reset
    assert result.config["observer"]["method"] == "synthetic_noisy_measurement"
    baseline = run_episode(DrivingConfig(horizon=8))
    assert baseline.config["observer"] == {"method": "truth", "config": None}
    for row in baseline.trace:
        assert np.allclose(
            [row["input_x_m"], row["input_y_m"], row["input_speed_mps"]],
            [row["before_x_m"], row["before_y_m"], row["before_speed_mps"]],
        )


def test_real_controller_input_changes_trajectory():
    config = DrivingConfig(seed=7, horizon=25)

    class ShiftedObserver:
        def reset(self, lane=None):
            pass

        def observe(self, truth, step, dt):
            self.last_measurement = replace(
                truth, lane_lateral_m=truth.lane_lateral_m + 0.45
            )
            return self.last_measurement

        def manifest(self):
            return {"method": "test_shift", "config": {"lateral_bias_m": 0.45}}

    oracle = run_episode(config)
    shifted = run_episode(config, observer=ShiftedObserver())
    assert any(
        not np.isclose(a["command_steering"], b["command_steering"])
        for a, b in zip(oracle.trace, shifted.trace)
    )
    n = min(len(oracle.trace), len(shifted.trace))
    assert not np.allclose(
        [row["y_m"] for row in oracle.trace[:n]],
        [row["y_m"] for row in shifted.trace[:n]],
    )


def test_raw_and_filter_share_indexed_measurement_noise():
    config = DrivingConfig(seed=7, horizon=8)
    sensor = MeasurementConfig(seed=31, position_bias_m=(0.0, 0.0))
    raw = run_episode(config, observer=RawMeasurementObserver(sensor))
    filtered = run_episode(config, observer=KalmanObserver(sensor))
    for raw_row, filter_row in zip(raw.trace, filtered.trace):
        assert np.isclose(
            raw_row["measurement_x_m"] - raw_row["before_x_m"],
            filter_row["measurement_x_m"] - filter_row["before_x_m"],
        )
        assert np.isclose(
            raw_row["measurement_y_m"] - raw_row["before_y_m"],
            filter_row["measurement_y_m"] - filter_row["before_y_m"],
        )
