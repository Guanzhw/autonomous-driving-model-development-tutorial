"""State -> fixed lane reference -> feedback -> delayed action -> MetaDrive."""

from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import csv
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np

METADRIVE_COMMIT = "85e5dadc6c7436d324348f6e3d8f8e680c06b4db"


@dataclass(frozen=True)
class DrivingConfig:
    seed: int = 7
    horizon: int = 180
    target_speed_mps: float = 6.0
    lookahead_m: float = 8.0
    lateral_gain: float = 0.32
    heading_gain: float = 0.85
    speed_gain: float = 0.25
    action_delay_steps: int = 0
    initial_lateral_offset_m: float = 0.5
    decision_repeat: int = 5
    physics_step_s: float = 0.02

    def __post_init__(self):
        for name in ("seed", "horizon", "action_delay_steps", "decision_repeat"):
            if not isinstance(getattr(self, name), int):
                raise TypeError(f"{name} must be an integer")
        if self.horizon < 1 or self.decision_repeat < 1 or self.action_delay_steps < 0:
            raise ValueError(
                "horizon/repeat must be positive; delay must be nonnegative"
            )
        for name in (
            "target_speed_mps",
            "lookahead_m",
            "lateral_gain",
            "heading_gain",
            "speed_gain",
            "initial_lateral_offset_m",
            "physics_step_s",
        ):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if (
            self.target_speed_mps < 0
            or self.lookahead_m <= 0
            or self.physics_step_s <= 0
        ):
            raise ValueError(
                "speed must be nonnegative; lookahead and physics step positive"
            )
        if abs(self.initial_lateral_offset_m) >= 1.7:
            raise ValueError("initial centre must fit inside the 3.5m lane")

    @classmethod
    def from_mapping(cls, values):
        return cls(**values)


@dataclass(frozen=True)
class DrivingObservation:
    position: tuple[float, float]
    heading_rad: float
    speed_mps: float
    lane_longitudinal_m: float
    lane_lateral_m: float
    lane_heading_rad: float
    lane_width_m: float


@dataclass(frozen=True)
class LaneReference:
    position: tuple[float, float]
    heading_rad: float
    longitudinal_m: float
    lane_width_m: float


def observe_agent(agent, lane):
    """Read privileged simulator state relative to the fixed reference lane."""
    s, lateral = lane.local_coordinates(agent.position)
    return DrivingObservation(
        tuple(map(float, agent.position)),
        float(agent.heading_theta),
        float(agent.speed),
        float(s),
        float(lateral),
        float(lane.heading_theta_at(s)),
        float(lane.width_at(s)),
    )


class ReferencePlanner:
    """Select a forward point on the lane fixed at reset; no nearest-lane switching."""

    def __init__(self, lookahead_m=8.0):
        self.lookahead_m = lookahead_m

    def plan(self, observation, lane):
        s = min(observation.lane_longitudinal_m + self.lookahead_m, lane.length)
        return LaneReference(
            tuple(map(float, lane.position(s, 0))),
            float(lane.heading_theta_at(s)),
            float(s),
            float(lane.width_at(s)),
        )


class GeometricController:
    """Lateral + target-bearing feedback; normalized actions, not a full Pure Pursuit."""

    def __init__(
        self,
        target_speed_mps=6.0,
        lateral_gain=0.32,
        heading_gain=0.85,
        speed_gain=0.25,
    ):
        self.target_speed_mps = target_speed_mps
        self.lateral_gain = lateral_gain
        self.heading_gain = heading_gain
        self.speed_gain = speed_gain

    def control(self, observation, reference):
        delta = np.asarray(reference.position) - observation.position
        bearing = np.arctan2(delta[1], delta[0])
        error = (observation.heading_rad - bearing + np.pi) % (2 * np.pi) - np.pi
        steering = (
            self.lateral_gain * observation.lane_lateral_m - self.heading_gain * error
        )
        throttle = self.speed_gain * (self.target_speed_mps - observation.speed_mps)
        return np.clip([steering, throttle], -1, 1).astype(np.float32)


class DelayedActuator:
    def __init__(self, delay_steps):
        self.delay_steps = delay_steps
        self.queue = deque()

    def push(self, command):
        self.queue.append(np.asarray(command, dtype=np.float32).copy())
        return (
            np.zeros(2, dtype=np.float32)
            if len(self.queue) <= self.delay_steps
            else self.queue.popleft()
        )


@dataclass
class EpisodeResult:
    config: dict[str, Any]
    metrics: dict[str, Any]
    trace: list[dict[str, Any]]


def build_metadrive_env(config):
    from metadrive import MetaDriveEnv

    return MetaDriveEnv(
        {
            "use_render": False,
            "image_observation": False,
            "map_config": {
                "type": "block_sequence",
                "config": "S",
                "lane_num": 1,
                "lane_width": 3.5,
                "exit_length": 300,
            },
            "agent_configs": {
                "default_agent": {
                    "spawn_lane_index": (">>", ">>>", 0),
                    "spawn_longitude": 145.0,
                    "spawn_lateral": config.initial_lateral_offset_m,
                }
            },
            "traffic_density": 0.0,
            "num_scenarios": 1,
            "start_seed": config.seed,
            "horizon": config.horizon,
            "decision_repeat": config.decision_repeat,
            "physics_world_step_size": config.physics_step_s,
        }
    )


def run_episode(config=None):
    if config is None:
        config = DrivingConfig()
    elif isinstance(config, Mapping):
        config = DrivingConfig.from_mapping(config)
    env = build_metadrive_env(config)
    planner = ReferencePlanner(config.lookahead_m)
    controller = GeometricController(
        config.target_speed_mps,
        config.lateral_gain,
        config.heading_gain,
        config.speed_gain,
    )
    actuator = DelayedActuator(config.action_delay_steps)
    trace = []
    try:
        raw, _ = env.reset(seed=config.seed)
        lane = env.agent.lane
        initial = observe_agent(env.agent, lane)
        dt = float(
            env.config["decision_repeat"] * env.config["physics_world_step_size"]
        )
        dist = importlib.metadata.distribution("metadrive-simulator")
        direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
        manifest = {
            **asdict(config),
            "simulator": {
                "version": dist.version,
                "commit": direct_url.get("vcs_info", {}).get("commit_id"),
                "panda3d": importlib.metadata.version("panda3d"),
                "numpy": np.__version__,
                "decision_dt_s": dt,
                "map_config": env.config["map_config"].get_dict(),
                "continuous_line_done": env.config["on_continuous_line_done"],
                "observation_dim": int(np.asarray(raw).size),
            },
            "initial_state": asdict(initial),
            "reference_lane": {
                "index": list(lane.index),
                "length_m": float(lane.length),
                "width_m": float(lane.width_at(0)),
                "start": list(map(float, lane.position(0, 0))),
                "end": list(map(float, lane.position(lane.length, 0))),
            },
            "vehicle": {
                "length_m": float(env.agent.LENGTH),
                "width_m": float(env.agent.WIDTH),
            },
        }
        for step in range(config.horizon):
            before = observe_agent(env.agent, lane)
            reference = planner.plan(before, lane)
            command = controller.control(before, reference)
            applied = actuator.push(command)
            _, reward, terminated, truncated, info = env.step(applied)
            after = observe_agent(env.agent, lane)
            trace.append(
                {
                    "step": step,
                    "time_s": (step + 1) * dt,
                    "before_x_m": before.position[0],
                    "before_y_m": before.position[1],
                    "before_heading_rad": before.heading_rad,
                    "before_speed_mps": before.speed_mps,
                    "before_lateral_error_m": before.lane_lateral_m,
                    "before_longitudinal_m": before.lane_longitudinal_m,
                    "reference_x_m": reference.position[0],
                    "reference_y_m": reference.position[1],
                    "reference_heading_rad": reference.heading_rad,
                    "reference_longitudinal_m": reference.longitudinal_m,
                    "command_steering": float(command[0]),
                    "command_throttle": float(command[1]),
                    "applied_steering": float(applied[0]),
                    "applied_throttle": float(applied[1]),
                    "x_m": after.position[0],
                    "y_m": after.position[1],
                    "heading_rad": after.heading_rad,
                    "speed_mps": after.speed_mps,
                    "lateral_error_m": after.lane_lateral_m,
                    "longitudinal_m": after.lane_longitudinal_m,
                    "distance_m": float(
                        np.linalg.norm(np.array(after.position) - before.position)
                    ),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "arrive_dest": bool(info["arrive_dest"]),
                    "route_completion": float(info["route_completion"]),
                    "crash": bool(info["crash"]),
                    "out_of_road": bool(info["out_of_road"]),
                    "on_lane": bool(env.agent.on_lane),
                    "on_white_continuous_line": bool(
                        env.agent.on_white_continuous_line
                    ),
                    "on_yellow_continuous_line": bool(
                        env.agent.on_yellow_continuous_line
                    ),
                    "crash_sidewalk": bool(env.agent.crash_sidewalk),
                    "crash_vehicle": bool(env.agent.crash_vehicle),
                    "crash_object": bool(env.agent.crash_object),
                }
            )
            if terminated or truncated:
                break
    finally:
        env.close()
    last = trace[-1]
    failure = any(r["crash"] or r["out_of_road"] for r in trace)
    # Preserve every terminal flag; the text label can contain multiple causes.
    causes = [
        name
        for name in (
            "crash_vehicle",
            "crash_object",
            "crash_sidewalk",
            "on_white_continuous_line",
            "on_yellow_continuous_line",
        )
        if last[name]
    ]
    if not last["on_lane"]:
        causes.append("not_on_lane")
    if failure and not causes:
        causes.append("out_of_road" if last["out_of_road"] else "crash")
    metrics = {
        "steps": len(trace),
        "elapsed_s": last["time_s"],
        "decision_dt_s": dt,
        "distance_traveled_m": sum(r["distance_m"] for r in trace),
        "mean_abs_lateral_error_m": float(
            np.mean([abs(r["lateral_error_m"]) for r in trace])
        ),
        "max_abs_lateral_error_m": max(abs(r["lateral_error_m"]) for r in trace),
        "mean_speed_mps": float(np.mean([r["speed_mps"] for r in trace])),
        "command_applied_difference": float(
            np.mean([abs(r["command_steering"] - r["applied_steering"]) for r in trace])
        ),
        "failure": failure,
        "failure_reason": "+".join(causes) if failure else None,
        "outcome": "failure"
        if failure
        else "arrived"
        if last["arrive_dest"]
        else "horizon"
        if last["truncated"]
        else "terminated",
        **{
            k: last[k]
            for k in (
                "crash",
                "out_of_road",
                "arrive_dest",
                "terminated",
                "truncated",
                "route_completion",
            )
        },
    }
    return EpisodeResult(manifest, metrics, trace)


def _road(ax, result, xmin, xmax):
    lane = result.config["reference_lane"]
    y = lane["start"][1]
    half = lane["width_m"] / 2
    ax.axhspan(y - half, y + half, color="#edf1f3")
    ax.axhline(y, linestyle="--", color="#607080", label="lane centre")
    for edge in (y - half, y + half):
        ax.axhline(edge, color="#777777", linewidth=1)
    ax.set(
        xlim=(xmin, xmax),
        ylim=(y - 3, y + 3),
        xlabel="World x / m",
        ylabel="World y / m",
    )
    ax.set_aspect("equal", adjustable="box")


def _body(ax, result, row):
    from matplotlib.patches import Polygon

    length, width = (
        result.config["vehicle"]["length_m"],
        result.config["vehicle"]["width_m"],
    )
    corners = np.array(
        [
            [length / 2, width / 2],
            [length / 2, -width / 2],
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
        ]
    )
    angle = row["heading_rad"]
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    ax.add_patch(
        Polygon(
            corners @ rotation.T + [row["x_m"], row["y_m"]], color="#c74c3c", alpha=0.5
        )
    )


def save_replay_plot(result, output_path):
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = result.trace
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    xs = [r["x_m"] for r in rows]
    _road(axes[0], result, min(xs) - 5, max(xs) + 5)
    axes[0].plot(xs, [r["y_m"] for r in rows], color="#246b89", label="measured ego")
    _body(axes[0], result, rows[-1])
    axes[0].set_title(f"MetaDrive trajectory: {result.metrics['outcome']}")
    times = [r["time_s"] for r in rows]
    axes[1].plot(times, [r["lateral_error_m"] for r in rows], label="post-step error")
    axes[1].set(xlabel="Time / s", ylabel="Lateral error / m")
    for key, label in (("command_steering", "issued"), ("applied_steering", "applied")):
        axes[2].step(
            [t - result.metrics["decision_dt_s"] for t in times],
            [r[key] for r in rows],
            where="post",
            label=label,
        )
    axes[2].set(xlabel="Command interval start / s", ylabel="Steering")
    for ax in axes:
        ax.legend(loc="best")
    fig.savefig(output, dpi=130)
    plt.close(fig)
    return output


def save_failure_gif(result, output_path, stride=2):
    import matplotlib.pyplot as plt
    from PIL import Image

    if not result.metrics["failure"]:
        raise ValueError("failure replay requires a failed episode")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    indices = sorted(set(range(0, len(result.trace), stride)) | {len(result.trace) - 1})
    frames = []
    for index in indices:
        row = result.trace[index]
        fig, ax = plt.subplots(figsize=(8, 3))
        _road(ax, result, row["x_m"] - 10, row["x_m"] + 10)
        prior = result.trace[: index + 1]
        ax.plot([r["x_m"] for r in prior], [r["y_m"] for r in prior], color="#246b89")
        _body(ax, result, row)
        ax.set_title(
            f"Measured replay | t={row['time_s']:.1f}s | final={index == len(result.trace) - 1}"
        )
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(
            Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        )
        plt.close(fig)
    durations = [
        int(round((b - a) * result.metrics["decision_dt_s"] * 1000))
        for a, b in zip(indices, indices[1:])
    ]
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=durations + [1000],
        loop=0,
    )
    return output


def save_episode(result, directory, stem, failure_gif=False):
    """Use one export path for CLI and notebooks, including failed runs."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "trace": directory / f"{stem}.json",
        "trace_csv": directory / f"{stem}.csv",
    }
    files["trace"].write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    with files["trace_csv"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result.trace[0]))
        writer.writeheader()
        writer.writerows(result.trace)
    files["plot"] = save_replay_plot(result, directory / f"{stem}.png")
    if failure_gif and result.metrics["failure"]:
        files["failure_gif"] = save_failure_gif(
            result, directory / f"{stem}_failure.gif"
        )
    return files
