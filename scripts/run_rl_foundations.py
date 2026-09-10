"""Run the bounded RL foundations experiment and save auditable evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from statistics import fmean, pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ad_tutorial.driving import DrivingConfig, save_episode  # noqa: E402
from ad_tutorial.rl_foundations import (  # noqa: E402
    FixedSpeedPolicy,
    SpeedChoicePolicy,
    load_checkpoint,
    run_policy_episode,
    save_checkpoint,
    save_evaluation,
    train_reinforce,
)


def conditions(base: DrivingConfig, seeds=(7, 11), offsets=(0.5, -0.4)):
    """Build matched conditions; callers may reserve one list as holdout."""

    return [
        replace(base, seed=int(seed), initial_lateral_offset_m=float(offset))
        for seed in seeds
        for offset in offsets
    ]


def collect(policy, configs, output: Path, stem: str):
    policy.eval()
    rows = []
    for index, config in enumerate(configs):
        episode = run_policy_episode(config, policy)
        files = save_episode(
            episode.result,
            output / "traces",
            f"{stem}_{index}",
            failure_gif=episode.result.metrics["failure"],
        )
        rows.append(
            {
                "condition_id": f"seed{config.seed}_offset{config.initial_lateral_offset_m:+.3f}",
                "config": asdict(config),
                "metrics": episode.result.metrics,
                "discounted_deployment_return": float(episode.returns[0]),
                "undiscounted_reward_sum": float(episode.rewards.sum()),
                "actions": episode.actions,
                "policy_mode": episode.result.config.get("policy", {}).get(
                    "policy_mode", "unknown"
                ),
                "files": {name: str(path) for name, path in files.items()},
            }
        )
    return rows


def variability(rows):
    """Report seed/offset variability without selecting a best rollout."""

    objectives = [float(row["discounted_deployment_return"]) for row in rows]
    rewards = [float(row["undiscounted_reward_sum"]) for row in rows]
    distances = [float(row["metrics"]["distance_traveled_m"]) for row in rows]
    failures = [bool(row["metrics"]["failure"]) for row in rows]
    return {
        "n": len(rows),
        "discounted_deployment_return_mean": fmean(objectives),
        "discounted_deployment_return_std": pstdev(objectives)
        if len(objectives) > 1
        else 0.0,
        "undiscounted_reward_sum_mean": fmean(rewards),
        "undiscounted_reward_sum_std": pstdev(rewards) if len(rewards) > 1 else 0.0,
        "distance_mean_m": fmean(distances),
        "distance_std_m": pstdev(distances) if len(distances) > 1 else 0.0,
        "failure_rate": fmean(failures),
    }


def paired_deltas(after_rows, before_rows):
    """Pair rows by condition and retain all per-condition effects."""

    before_by_id = {row["condition_id"]: row for row in before_rows}
    deltas = []
    for after in after_rows:
        before = before_by_id[after["condition_id"]]
        deltas.append(
            {
                "condition_id": after["condition_id"],
                "discounted_deployment_return_delta": after[
                    "discounted_deployment_return"
                ]
                - before["discounted_deployment_return"],
                "undiscounted_reward_sum_delta": after["undiscounted_reward_sum"]
                - before["undiscounted_reward_sum"],
                "distance_traveled_m_delta": after["metrics"]["distance_traveled_m"]
                - before["metrics"]["distance_traveled_m"],
                "failure_delta": int(bool(after["metrics"]["failure"]))
                - int(bool(before["metrics"]["failure"])),
            }
        )
    return deltas


def delta_summary(rows):
    fields = (
        "discounted_deployment_return_delta",
        "undiscounted_reward_sum_delta",
        "distance_traveled_m_delta",
        "failure_delta",
    )
    return {
        "n": len(rows),
        **{
            f"{field}_mean": fmean([float(row[field]) for row in rows])
            for field in fields
        },
        **{
            f"{field}_std": pstdev([float(row[field]) for row in rows])
            if len(rows) > 1
            else 0.0
            for field in fields
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts" / "rl_foundations"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Four-step delay = 0.4 s at the fixed 5 x 0.02 s decision interval.
    base = DrivingConfig(
        seed=args.seed,
        horizon=args.horizon,
        action_delay_steps=4,
        decision_repeat=5,
        target_speed_mps=6.0,
    )
    model_seeds = tuple(args.seed + index for index in range(3))
    train_seeds = tuple(args.seed + 100 + 10 * index for index in range(3))
    selection_conditions = conditions(base, seeds=train_seeds, offsets=(0.5, -0.4))
    # These conditions are held out until the final report and are not used to
    # choose episodes, learning rate, or a checkpoint.
    holdout_conditions = conditions(
        base, seeds=(args.seed + 200, args.seed + 210), offsets=(0.3, -0.6)
    )
    before = {
        "fixed_slow_2mps": collect(
            FixedSpeedPolicy(2.0), selection_conditions, args.output, "before_slow"
        ),
        "fixed_fast_6mps": collect(
            FixedSpeedPolicy(6.0), selection_conditions, args.output, "before_fast"
        ),
    }
    holdout = {
        "fixed_slow_2mps": collect(
            FixedSpeedPolicy(2.0), holdout_conditions, args.output, "holdout_slow"
        ),
        "fixed_fast_6mps": collect(
            FixedSpeedPolicy(6.0), holdout_conditions, args.output, "holdout_fast"
        ),
    }
    runs = []
    for model_seed in model_seeds:
        untrained = SpeedChoicePolicy(seed=model_seed)
        before_path = (
            args.output / "checkpoints" / f"speed_choice_seed{model_seed}_before.pt"
        )
        after_path = (
            args.output / "checkpoints" / f"speed_choice_seed{model_seed}_after.pt"
        )
        save_checkpoint(
            untrained,
            before_path,
            config=base,
            history=[],
            model_seed=model_seed,
            training_hyperparameters={
                "episodes": args.episodes,
                "gamma": 0.99,
                "learning_rate": 0.01,
                "baseline": "running_mean",
                "hidden_size": 16,
                "model_seed": model_seed,
                "environment_seeds": train_seeds,
            },
        )
        trained = train_reinforce(
            base,
            episodes=args.episodes,
            seeds=train_seeds,
            checkpoint=after_path,
            learning_rate=0.01,
            model_seed=model_seed,
        )
        # Reload the artifact for evaluation so actions test the saved model,
        # not an in-memory object with accidental extra state.
        untrained_loaded = load_checkpoint(before_path)
        learned_loaded = load_checkpoint(after_path)
        before_untrained = collect(
            untrained_loaded,
            selection_conditions,
            args.output,
            f"before_untrained_seed{model_seed}",
        )
        after_learned = collect(
            learned_loaded,
            selection_conditions,
            args.output,
            f"after_learned_seed{model_seed}",
        )
        holdout_untrained = collect(
            untrained_loaded,
            holdout_conditions,
            args.output,
            f"holdout_untrained_seed{model_seed}",
        )
        holdout_learned = collect(
            learned_loaded,
            holdout_conditions,
            args.output,
            f"holdout_learned_seed{model_seed}",
        )
        runs.append(
            {
                "model_seed": model_seed,
                "untrained_checkpoint": str(before_path),
                "trained_checkpoint": str(after_path),
                "training_hyperparameters": {
                    "episodes": args.episodes,
                    "gamma": 0.99,
                    "learning_rate": 0.01,
                    "baseline": "running_mean",
                    "hidden_size": 16,
                    "model_seed": model_seed,
                    "environment_seeds": train_seeds,
                },
                "training_history": trained.history,
                "before_untrained": before_untrained,
                "after_learned": after_learned,
                "holdout_untrained": holdout_untrained,
                "holdout_learned": holdout_learned,
                "paired_deltas": {
                    "learned_minus_untrained": paired_deltas(
                        after_learned, before_untrained
                    ),
                    "learned_minus_fixed_slow": paired_deltas(
                        after_learned, before["fixed_slow_2mps"]
                    ),
                    "learned_minus_fixed_fast": paired_deltas(
                        after_learned, before["fixed_fast_6mps"]
                    ),
                },
                "holdout_paired_deltas": {
                    "learned_minus_untrained": paired_deltas(
                        holdout_learned, holdout_untrained
                    ),
                    "learned_minus_fixed_slow": paired_deltas(
                        holdout_learned, holdout["fixed_slow_2mps"]
                    ),
                    "learned_minus_fixed_fast": paired_deltas(
                        holdout_learned, holdout["fixed_fast_6mps"]
                    ),
                },
            }
        )
    after = {
        "learned_speed_choice": {
            str(run["model_seed"]): run["after_learned"] for run in runs
        }
    }
    before["untrained_speed_choice"] = {
        str(run["model_seed"]): run["before_untrained"] for run in runs
    }
    holdout["untrained_speed_choice"] = {
        str(run["model_seed"]): run["holdout_untrained"] for run in runs
    }
    holdout["learned_speed_choice"] = {
        str(run["model_seed"]): run["holdout_learned"] for run in runs
    }
    payload = {
        "experiment": "REINFORCE target-speed choice over fixed geometric steering",
        "scope": "learned longitudinal choice; steering remains fixed geometric control",
        "action_semantics": "categorical 2 or 6 m/s target, throttle sent via env.step; decision_repeat=5",
        "training_policy_mode": "stochastic_categorical_sampling",
        "evaluation_policy_mode": "deterministic_argmax_deployment",
        "evaluation_note": "deployment returns are deterministic argmax rollouts and are not estimates of the stochastic training J",
        "observation_model": {
            "features": [
                "lateral_error_m",
                "heading_error_rad",
                "speed_mps",
                "speed_error_to_slow_mps",
                "speed_error_to_fast_mps",
            ],
            "partial_observation": True,
            "hidden_delayed_action_queue_steps": base.action_delay_steps,
            "policy_class": "memoryless reactive policy; valid restricted class under partial observation",
        },
        "reward_source": "measured trace progress - lateral error cost - simulator failure penalty",
        "finite_horizon": {"horizon": base.horizon, "bootstrap": 0.0, "gamma": 0.99},
        "config": asdict(base),
        "model_seeds": model_seeds,
        "training_environment_seeds": train_seeds,
        "selection_conditions": [asdict(c) for c in selection_conditions],
        "holdout_conditions": [asdict(c) for c in holdout_conditions],
        "runs": runs,
        "before": before,
        "after": after,
        "holdout": holdout,
        "seed_variability": {
            "fixed_slow_2mps": variability(before["fixed_slow_2mps"]),
            "fixed_fast_6mps": variability(before["fixed_fast_6mps"]),
            **{
                f"model_seed_{run['model_seed']}_before_untrained": variability(
                    run["before_untrained"]
                )
                for run in runs
            },
            **{
                f"model_seed_{run['model_seed']}_after_learned": variability(
                    run["after_learned"]
                )
                for run in runs
            },
            "holdout_fixed_slow_2mps": variability(holdout["fixed_slow_2mps"]),
            "holdout_fixed_fast_6mps": variability(holdout["fixed_fast_6mps"]),
            **{
                f"holdout_model_seed_{run['model_seed']}_untrained": variability(
                    run["holdout_untrained"]
                )
                for run in runs
            },
            **{
                f"holdout_model_seed_{run['model_seed']}_learned": variability(
                    run["holdout_learned"]
                )
                for run in runs
            },
        },
        "paired_delta_summary": {
            "selection_learned_minus_untrained": delta_summary(
                [
                    delta
                    for run in runs
                    for delta in run["paired_deltas"]["learned_minus_untrained"]
                ]
            ),
            "selection_learned_minus_fixed_slow": delta_summary(
                [
                    delta
                    for run in runs
                    for delta in run["paired_deltas"]["learned_minus_fixed_slow"]
                ]
            ),
            "selection_learned_minus_fixed_fast": delta_summary(
                [
                    delta
                    for run in runs
                    for delta in run["paired_deltas"]["learned_minus_fixed_fast"]
                ]
            ),
            "holdout_learned_minus_untrained": delta_summary(
                [
                    delta
                    for run in runs
                    for delta in run["holdout_paired_deltas"]["learned_minus_untrained"]
                ]
            ),
            "holdout_learned_minus_fixed_slow": delta_summary(
                [
                    delta
                    for run in runs
                    for delta in run["holdout_paired_deltas"][
                        "learned_minus_fixed_slow"
                    ]
                ]
            ),
            "holdout_learned_minus_fixed_fast": delta_summary(
                [
                    delta
                    for run in runs
                    for delta in run["holdout_paired_deltas"][
                        "learned_minus_fixed_fast"
                    ]
                ]
            ),
        },
        "limitations": [
            "The policy only chooses longitudinal target speed; it is not a full autonomous-driving policy.",
            "REINFORCE has high variance and this short CPU run is a mechanism lesson, not a performance claim.",
            "The fixed map, privileged measured state, and chosen offsets do not establish road generalization.",
        ],
    }
    save_evaluation(args.output / "evaluation.json", payload)
    print(
        json.dumps(
            {
                "checkpoints": [run["trained_checkpoint"] for run in runs],
                "evaluation": str(args.output / "evaluation.json"),
                "episodes": args.episodes,
                "model_seeds": model_seeds,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
