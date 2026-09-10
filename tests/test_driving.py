from dataclasses import asdict, replace
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from ad_tutorial.driving import (  # noqa: E402
    DrivingConfig,
    DrivingObservation,
    LaneReference,
    GeometricController,
    DelayedActuator,
    run_episode,
    save_episode,
)
from run_first_unit import experiment_configs  # noqa: E402


def test_hand_calculated_action_and_reference_effect():
    state = DrivingObservation((0, -0.5), 0, 0, 0, 0.5, 0, 3.5)
    ref = LaneReference((8, 0), 0, 8, 3.5)
    controller = GeometricController()
    assert np.allclose(controller.control(state, ref), [0.21305599, 1], atol=1e-6)
    near = replace(ref, position=(2, 0), longitudinal_m=2)
    assert np.isclose(controller.control(state, near)[0], 0.36823186, atol=1e-6)


def test_delay_queue_exact_commands():
    actuator = DelayedActuator(2)
    issued = [[0.1, 0.2], [0.3, 0.4], [-0.5, 0.6], [0.7, -0.8]]
    assert np.allclose(
        [actuator.push(c) for c in issued], [[0, 0], [0, 0], issued[0], issued[1]]
    )
    assert np.allclose(DelayedActuator(0).push(issued[0]), issued[0])


def test_invalid_experiment_config_is_explicit():
    with pytest.raises(TypeError):
        DrivingConfig.from_mapping({"unknown": 1})
    with pytest.raises(ValueError):
        DrivingConfig(horizon=0)
    with pytest.raises(ValueError):
        DrivingConfig(lookahead_m=float("nan"))


@pytest.fixture(scope="module")
def experiments():
    return {
        name: run_episode(config)
        for name, config in experiment_configs(DrivingConfig()).items()
    }


def test_matched_real_delay_and_speed_intervention(experiments):
    configs = {n: asdict(c) for n, c in experiment_configs(DrivingConfig()).items()}
    assert {
        k for k in configs["baseline"] if configs["baseline"][k] != configs["delay"][k]
    } == {"action_delay_steps"}
    assert {
        k for k in configs["delay"] if configs["delay"][k] != configs["recovery"][k]
    } == {"target_speed_mps"}
    base, delay, recovery = (experiments[n] for n in ("baseline", "delay", "recovery"))
    assert base.metrics["outcome"] == recovery.metrics["outcome"] == "horizon"
    assert delay.metrics["failure"] and delay.metrics["steps"] < base.metrics["steps"]
    assert recovery.metrics["distance_traveled_m"] < base.metrics["distance_traveled_m"]
    applied = np.array(
        [[r["applied_steering"], r["applied_throttle"]] for r in delay.trace]
    )
    command = np.array(
        [[r["command_steering"], r["command_throttle"]] for r in delay.trace]
    )
    assert np.allclose(applied[:4], 0)
    assert np.allclose(applied[4:], command[:-4])


def test_fixed_lane_and_before_after_chain(experiments):
    result = experiments["baseline"]
    assert result.config["reference_lane"]["index"] == [">>", ">>>", 0]
    assert result.config["reference_lane"]["length_m"] == 290
    assert result.config["simulator"]["map_config"]["config"] == "S"
    assert result.config["simulator"]["continuous_line_done"] is True
    rows = result.trace
    assert np.allclose(
        [rows[0]["before_x_m"], rows[0]["before_y_m"]],
        result.config["initial_state"]["position"],
    )
    for after, before in (
        ("x_m", "before_x_m"),
        ("y_m", "before_y_m"),
        ("heading_rad", "before_heading_rad"),
        ("speed_mps", "before_speed_mps"),
    ):
        assert np.allclose([r[after] for r in rows[:-1]], [r[before] for r in rows[1:]])
    assert (
        min(r["reference_longitudinal_m"] - r["before_longitudinal_m"] for r in rows)
        > 7.99
    )


def test_trace_metrics_and_export(experiments, tmp_path):
    import csv
    import json

    result = experiments["delay"]
    files = save_episode(result, tmp_path, "failed")
    data = json.loads(files["trace"].read_text())
    with files["trace_csv"].open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == len(data["trace"]) == result.metrics["steps"]
    rows = data["trace"]
    movement = np.array(
        [[r["x_m"] - r["before_x_m"], r["y_m"] - r["before_y_m"]] for r in rows]
    )
    assert np.isclose(
        np.linalg.norm(movement, axis=1).sum(), data["metrics"]["distance_traveled_m"]
    )
    assert np.isclose(
        np.mean([abs(r["lateral_error_m"]) for r in rows]),
        data["metrics"]["mean_abs_lateral_error_m"],
    )
    assert rows[-1]["out_of_road"] and data["metrics"]["failure_reason"]


def test_same_seed_repeats_measured_trajectory(experiments):
    repeat = run_episode(DrivingConfig())
    for field in ("x_m", "y_m", "heading_rad", "speed_mps", "applied_steering"):
        assert np.allclose(
            [r[field] for r in repeat.trace],
            [r[field] for r in experiments["baseline"].trace],
            atol=1e-5,
        )


def test_reference_changes_real_environment_trajectory(experiments):
    near = run_episode(DrivingConfig(lookahead_m=2, horizon=30))
    n = len(near.trace)
    assert not np.allclose(
        [r["y_m"] for r in near.trace],
        [r["y_m"] for r in experiments["baseline"].trace[:n]],
    )


def test_changed_physics_cadence_is_effective(experiments):
    result = run_episode(DrivingConfig(decision_repeat=2, horizon=10))
    assert np.isclose(result.metrics["elapsed_s"], 0.4)
    assert result.config["simulator"]["decision_dt_s"] == 0.04
    assert (
        result.trace[-1]["distance_m"] != experiments["baseline"].trace[9]["distance_m"]
    )
