"""Run matched MetaDrive baseline, delay and slower-policy experiments."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ad_tutorial.driving import DrivingConfig, run_episode, save_episode  # noqa: E402


def experiment_configs(base, delay_steps=4, recovery_speed=2.0):
    delayed = replace(base, action_delay_steps=delay_steps)
    return {
        "baseline": base,
        "delay": delayed,
        "recovery": replace(delayed, target_speed_mps=recovery_speed),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unit", choices=("baseline", "delay", "recovery", "all"), default="all"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=180)
    parser.add_argument("--delay-steps", type=int, default=4)
    parser.add_argument("--initial-offset", type=float, default=0.5)
    parser.add_argument("--recovery-speed", type=float, default=2.0)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts" / "first_loop"
    )
    parser.add_argument("--failure-gif", action="store_true")
    args = parser.parse_args()
    configs = experiment_configs(
        DrivingConfig(
            seed=args.seed,
            horizon=args.horizon,
            initial_lateral_offset_m=args.initial_offset,
        ),
        args.delay_steps,
        args.recovery_speed,
    )
    names = configs if args.unit == "all" else [args.unit]
    summaries = []
    for name in names:
        result = run_episode(configs[name])
        files = save_episode(
            result, args.output, f"{name}_seed{args.seed}", args.failure_gif
        )
        summary = {
            "unit": name,
            **result.metrics,
            **{k: str(v) for k, v in files.items()},
        }
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    (args.output / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
