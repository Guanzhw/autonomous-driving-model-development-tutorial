"""Run the unit 3 behavioural-cloning experiment and export auditable artifacts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ad_tutorial.imitation import (  # noqa: E402
    BCTrainingConfig,
    collect_demonstrations,
    dataset_hash,
    default_split_configs,
    evaluate_closed_loop,
    load_checkpoint,
    make_untrained_policy,
    offline_mse,
    paired_common_window,
    save_checkpoint,
    summarize_closed_loop,
    train_behavioral_cloning,
    write_dataset_manifest,
    write_training_curve,
)


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot encode {type(value).__name__}")


def _save_training_plot(path: Path, history):
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [row["train_mse"] for row in history], label="train")
    ax.plot(epochs, [row["val_mse"] for row in history], label="validation")
    ax.set(xlabel="epoch", ylabel="action MSE", title="Behavioural cloning training")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_results(output: Path, results, *, failure_gif: bool):
    trace_dir = output / "closed_loop"
    trace_dir.mkdir(parents=True, exist_ok=True)
    for condition, episodes in results.items():
        for index, episode in enumerate(episodes):
            seed = episode.config.get("seed", "unknown")
            stem = f"{condition}_test{index:02d}_seed{seed}"
            from ad_tutorial.driving import save_episode

            files = save_episode(episode, trace_dir, stem, failure_gif=failure_gif)
            # Keep the relative paths in the summary easy to inspect from JSON.
            episode_files = {
                name: str(path.relative_to(output)) for name, path in files.items()
            }
            episode_files["condition"] = condition
            episode_files["test_index"] = index
            yield episode_files


def _save_expert_traces(output: Path, dataset):
    trace_dir = output / "expert_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for episode in dataset.episodes:
        path = trace_dir / f"{episode['episode_id']}.json"
        path.write_text(
            json.dumps(episode, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        files.append(str(path.relative_to(output)))
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "imitation")
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument("--no-gif", action="store_true", help="skip failure GIFs")
    args = parser.parse_args(argv)
    if args.horizon < 1 or args.epochs < 1:
        parser.error("--horizon and --epochs must be positive")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    configs = default_split_configs(horizon=args.horizon)
    serial_configs = {
        split: [asdict(config) for config in values]
        for split, values in configs.items()
    }
    started = time.perf_counter()
    dataset = collect_demonstrations(
        configs, max_steps_per_episode=args.max_steps_per_episode
    )
    collection_s = time.perf_counter() - started
    np.savez_compressed(
        output / "demonstrations.npz",
        features=dataset.features,
        actions=dataset.actions,
        episode_ids=np.asarray(dataset.episode_ids),
        splits=np.asarray(dataset.splits),
    )
    write_dataset_manifest(output / "dataset_manifest.json", dataset)
    expert_trace_files = _save_expert_traces(output, dataset)

    train = dataset.select("train")
    validation = dataset.select("validation")
    test = dataset.select("test")
    training_config = BCTrainingConfig(epochs=args.epochs)
    started = time.perf_counter()
    trained = train_behavioral_cloning(train, validation, training_config)
    training_s = time.perf_counter() - started
    save_checkpoint(output / "bc_checkpoint.pt", trained)
    write_training_curve(output / "training_curve.csv", trained.history)
    _save_training_plot(output / "training_curve.png", trained.history)

    reloaded = load_checkpoint(output / "bc_checkpoint.pt")
    untrained = make_untrained_policy(
        reloaded.normalizer,
        seed=training_config.seed,
        hidden_dim=training_config.hidden_dim,
    )
    closed_loop = evaluate_closed_loop(
        configs["test"], reloaded, untrained_policy=untrained
    )
    # Offline test MSE uses the final held-out episodes once, after selection.
    offline = offline_mse(reloaded.model, reloaded.normalizer, test)
    files = list(_save_results(output, closed_loop, failure_gif=not args.no_gif))
    manifest = json.loads(
        (output / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    summary = {
        "unit": "03_imitation",
        "dataset_hash": dataset_hash(dataset),
        "dataset": manifest,
        "configs": serial_configs,
        "collection_time_s": collection_s,
        "training_time_s": training_s,
        "training": trained.metrics,
        "training_config": asdict(training_config),
        "expert_trace_files": expert_trace_files,
        "offline_test": offline,
        "closed_loop": summarize_closed_loop(closed_loop),
        "closed_loop_common_window": paired_common_window(closed_loop),
        "closed_loop_files": files,
        "checkpoint": "bc_checkpoint.pt",
        "notes": [
            "Expert is the geometric controller in driving.py and every policy action is sent through MetaDrive env.step.",
            "Train normalization and model selection use train/validation episodes; test offsets and seeds are held out.",
            "The fixed block-sequence map is a simulation mechanism test; changing seed is not road generalization.",
        ],
    }
    (output / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "dataset_hash": summary["dataset_hash"],
                "samples": dataset.size,
                "collection_time_s": collection_s,
                "training_time_s": training_s,
                "offline_test_mse": offline["mse"],
                "closed_loop": summary["closed_loop"],
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
