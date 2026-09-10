"""Run oracle, raw-measurement and causal-filter driving comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ad_tutorial.driving import DrivingConfig, run_episode, save_episode  # noqa: E402
from ad_tutorial.estimation import MeasurementConfig, make_observer  # noqa: E402


def _wrapped_angle(error):
    return (error + np.pi) % (2 * np.pi) - np.pi


def _rmse(result) -> dict[str, float]:
    pairs = {
        "position_x_rmse_m": ("input_x_m", "before_x_m"),
        "position_y_rmse_m": ("input_y_m", "before_y_m"),
        "heading_rmse_rad": ("input_heading_rad", "before_heading_rad"),
        "speed_rmse_mps": ("input_speed_mps", "before_speed_mps"),
        "lateral_rmse_m": ("input_lateral_error_m", "before_lateral_error_m"),
        "longitudinal_rmse_m": ("input_longitudinal_m", "before_longitudinal_m"),
    }
    values = {}
    for prefix, source in (("input", "input"), ("measurement", "measurement")):
        for name, (estimate, truth) in pairs.items():
            field = estimate.replace("input_", f"{source}_")
            errors = np.asarray([row[field] - row[truth] for row in result.trace])
            if "heading" in name:
                errors = _wrapped_angle(errors)
            values[f"{prefix}_{name}"] = float(np.sqrt(np.mean(errors * errors)))
    values["state_rmse"] = float(
        np.sqrt(
            np.mean(
                np.asarray(
                    [
                        (row["input_x_m"] - row["before_x_m"]) ** 2
                        + (row["input_y_m"] - row["before_y_m"]) ** 2
                        for row in result.trace
                    ]
                )
            )
        )
    )
    values["measurement_state_rmse"] = float(
        np.sqrt(
            np.mean(
                np.asarray(
                    [
                        (row["measurement_x_m"] - row["before_x_m"]) ** 2
                        + (row["measurement_y_m"] - row["before_y_m"]) ** 2
                        for row in result.trace
                    ]
                )
            )
        )
    )
    return values


def save_comparison_plot(results: dict, output: Path) -> Path:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for name, result in results.items():
        times = [
            row["time_s"] - result.metrics["decision_dt_s"] for row in result.trace
        ]
        axes[0].plot(
            times,
            [row["before_lateral_error_m"] for row in result.trace],
            linestyle="--",
            alpha=0.45,
        )
        axes[0].plot(
            times,
            [row["input_lateral_error_m"] for row in result.trace],
            label=f"{name} input",
        )
        axes[1].plot(
            times, [row["command_steering"] for row in result.trace], label=name
        )
    axes[0].set(
        xlabel="Time / s",
        ylabel="Lateral state / m",
        title="Truth and controller input",
    )
    axes[1].set(xlabel="Time / s", ylabel="Steering", title="Actual controller command")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
    fig.savefig(output, dpi=130)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise-seed", type=int, default=19)
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts" / "state_estimation"
    )
    args = parser.parse_args()

    config = DrivingConfig(seed=args.seed, horizon=args.horizon)
    sensor = MeasurementConfig(seed=args.noise_seed)
    results = {}
    summaries = []
    for mode in ("oracle", "raw", "filter"):
        result = run_episode(config, observer=make_observer(mode, sensor))
        results[mode] = result
        files = save_episode(
            result, args.output, f"{mode}_seed{args.seed}", failure_gif=True
        )
        summary = {
            "unit": mode,
            "measurement": _rmse(result),
            "driving": result.metrics,
            "files": {key: str(value) for key, value in files.items()},
        }
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    plot = save_comparison_plot(
        results, args.output / f"comparison_seed{args.seed}.png"
    )
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "config": {**config.__dict__, "measurement": sensor.manifest()},
                "runs": summaries,
                "plot": str(plot),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
