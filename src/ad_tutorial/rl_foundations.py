"""Small, executable RL foundations built on the real driving loop.

The first half of this module is a tiny tabular MDP used for hand checking
Bellman equations.  The second half is deliberately a narrow control problem:
geometric steering stays fixed and a PyTorch policy chooses a target speed of
2 or 6 m/s before each five-physics-step decision interval.  The resulting
throttle is sent through :func:`ad_tutorial.driving.run_episode`; rewards are
reconstructed from its measured trace rather than from the action label.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .driving import (
    DrivingConfig,
    DrivingObservation,
    GeometricController,
    LaneReference,
    run_episode,
)


# ---------------------------------------------------------------------------
# Lesson 07: finite MDP and Bellman arithmetic


@dataclass(frozen=True)
class FiniteMDP:
    """A finite, tabular, episodic MDP.

    ``transitions[s][a]`` is a list of ``(probability, next_state)`` pairs.
    A transition to ``terminal_state`` ends the episode and has no continuation
    value.  Rewards are attached to state-action pairs, so the example keeps
    reward, return, and value visibly distinct.
    """

    states: tuple[str, ...]
    actions: tuple[str, ...]
    transitions: Mapping[str, Mapping[str, tuple[tuple[float, str], ...]]]
    rewards: Mapping[str, Mapping[str, float]]
    terminal_state: str = "terminal"

    def __post_init__(self):
        if self.terminal_state not in self.states:
            raise ValueError("terminal_state must be present in states")
        for state in self.states:
            if state == self.terminal_state:
                continue
            if state not in self.transitions or state not in self.rewards:
                raise ValueError(f"missing transitions/rewards for {state!r}")
            for action in self.actions:
                outcomes = self.transitions[state][action]
                if not outcomes:
                    raise ValueError("each state-action needs at least one outcome")
                if not np.isclose(sum(p for p, _ in outcomes), 1.0):
                    raise ValueError("transition probabilities must sum to one")
                if any(p < 0 or nxt not in self.states for p, nxt in outcomes):
                    raise ValueError("invalid transition outcome")


def teaching_mdp() -> FiniteMDP:
    """Return the small deterministic MDP used in notebook 07.

    ``start`` can take a short route (reward 2 then terminal) or a longer
    route (reward 1, then 4).  With gamma=0.9 the longer route is optimal:
    V(start)=1+0.9*4=4.6 versus 2 for the short action.
    """

    return FiniteMDP(
        states=("start", "long", "terminal"),
        actions=("short", "long"),
        transitions={
            "start": {"short": ((1.0, "terminal"),), "long": ((1.0, "long"),)},
            "long": {"short": ((1.0, "terminal"),), "long": ((1.0, "terminal"),)},
        },
        rewards={
            "start": {"short": 2.0, "long": 1.0},
            "long": {"short": 4.0, "long": 4.0},
        },
    )


def bellman_optimality_update(
    mdp: FiniteMDP, values: Mapping[str, float], gamma: float = 0.9
) -> dict[str, float]:
    """Perform one synchronous optimality Bellman update."""

    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    updated = {state: 0.0 for state in mdp.states}
    for state in mdp.states:
        if state == mdp.terminal_state:
            continue
        candidates = []
        for action in mdp.actions:
            continuation = sum(
                probability * (0.0 if nxt == mdp.terminal_state else values[nxt])
                for probability, nxt in mdp.transitions[state][action]
            )
            candidates.append(mdp.rewards[state][action] + gamma * continuation)
        updated[state] = max(candidates)
    return updated


def bellman_policy_update(
    mdp: FiniteMDP,
    values: Mapping[str, float],
    policy: Mapping[str, str],
    gamma: float = 0.9,
) -> dict[str, float]:
    """Perform one synchronous Bellman expectation update for a fixed policy."""

    updated = {state: 0.0 for state in mdp.states}
    for state in mdp.states:
        if state == mdp.terminal_state:
            continue
        action = policy[state]
        continuation = sum(
            probability * (0.0 if nxt == mdp.terminal_state else values[nxt])
            for probability, nxt in mdp.transitions[state][action]
        )
        updated[state] = mdp.rewards[state][action] + gamma * continuation
    return updated


def discounted_return(rewards: Sequence[float], gamma: float = 0.9) -> float:
    """Compute the scalar return from time zero, without bootstrapping."""

    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    total = 0.0
    for reward in reversed(rewards):
        total = float(reward) + gamma * total
    return total


def reward_to_go(rewards: Sequence[float], gamma: float = 0.99) -> np.ndarray:
    """Return ``G_t = r_t + gamma*r_{t+1}+...`` for every time step."""

    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    result = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        result[index] = running
    return result


def q_learning_update(
    q_value: float,
    reward: float,
    next_max_q: float,
    learning_rate: float = 0.1,
    gamma: float = 0.9,
    terminal: bool = False,
) -> float:
    """One tabular Q-learning update, with an explicit terminal target."""

    if not 0 < learning_rate <= 1 or not 0 <= gamma <= 1:
        raise ValueError("learning_rate must be in (0,1] and gamma in [0,1]")
    target = float(reward) if terminal else float(reward) + gamma * float(next_max_q)
    return float(q_value) + learning_rate * (target - float(q_value))


# ---------------------------------------------------------------------------
# Lesson 08: score-function policy gradient on MetaDrive


SPEED_CHOICES_MPS = (2.0, 6.0)
FEATURE_NAMES = (
    "lateral_error_m",
    "heading_error_rad",
    "speed_mps",
    "speed_error_to_slow_mps",
    "speed_error_to_fast_mps",
)


def observation_features(
    observation: DrivingObservation, reference: LaneReference
) -> np.ndarray:
    """Convert the measured controller observation into five policy features."""

    delta = np.asarray(reference.position, dtype=np.float64) - np.asarray(
        observation.position, dtype=np.float64
    )
    bearing = float(np.arctan2(delta[1], delta[0]))
    heading_error = (observation.heading_rad - bearing + np.pi) % (2 * np.pi) - np.pi
    return np.asarray(
        [
            observation.lane_lateral_m,
            heading_error,
            observation.speed_mps,
            observation.speed_mps - SPEED_CHOICES_MPS[0],
            observation.speed_mps - SPEED_CHOICES_MPS[1],
        ],
        dtype=np.float32,
    )


def trace_rewards(
    trace: Sequence[Mapping[str, object]],
    *,
    lateral_cost: float = 0.08,
    failure_penalty: float = 5.0,
    arrival_bonus: float = 0.0,
) -> np.ndarray:
    """Compute learning rewards from measured post-step trace fields.

    Progress is the change in measured lane-longitudinal position.  The
    lateral cost uses measured post-step error, while failure/arrival bonuses
    use simulator terminal flags.  The chosen speed label never appears in
    this calculation.
    """

    if lateral_cost < 0 or failure_penalty < 0:
        raise ValueError("cost and penalty must be nonnegative")
    rewards = []
    for row in trace:
        progress = float(row["longitudinal_m"]) - float(row["before_longitudinal_m"])
        value = progress - lateral_cost * abs(float(row["lateral_error_m"]))
        if bool(row.get("crash", False) or row.get("out_of_road", False)):
            value -= failure_penalty
        if bool(row.get("arrive_dest", False)):
            value += arrival_bonus
        rewards.append(value)
    return np.asarray(rewards, dtype=np.float64)


def finite_horizon_returns(
    rewards: Sequence[float], gamma: float = 0.99, bootstrap: float | None = None
) -> np.ndarray:
    """Finite-horizon reward-to-go, optionally exposing an explicit bootstrap.

    For this lesson the episode end is an objective boundary, so callers leave
    ``bootstrap`` as ``None`` (equivalent to zero).  Infinite continuing tasks
    instead estimate a value after the rollout and use it in the final target.
    """

    if bootstrap is None:
        bootstrap = 0.0
    if not np.isfinite(bootstrap):
        raise ValueError("bootstrap must be finite")
    result = reward_to_go(rewards, gamma)
    if len(result):
        result[-1] = float(rewards[-1]) + gamma * float(bootstrap)
        for index in range(len(result) - 2, -1, -1):
            result[index] = float(rewards[index]) + gamma * result[index + 1]
    return result


class RunningMeanBaseline:
    """Causal scalar baseline updated after each episode."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0

    def predict(self, size: int) -> np.ndarray:
        return np.full(size, self.mean, dtype=np.float64)

    def update(self, returns: Sequence[float]) -> None:
        for value in returns:
            self.count += 1
            self.mean += (float(value) - self.mean) / self.count


def reinforce_loss(
    log_probs,
    rewards: Sequence[float],
    gamma: float = 0.99,
    *,
    baseline: str = "running_mean",
    baseline_value: float = 0.0,
):
    """Build one finite-horizon REINFORCE episode loss.

    The objective is ``E[sum_t gamma**t r_t]``.  Consequently each score term
    is weighted by ``gamma**t`` and the episode loss is a sum; a caller that
    batches episodes should average those episode losses, not divide each
    trajectory by its own number of steps. The baseline uses earlier episodes
    and is independent of the current sampled action.
    """

    import torch

    log_probs = list(log_probs)
    returns = finite_horizon_returns(rewards, gamma)
    if not log_probs:
        raise ValueError("at least one log probability is required")
    if len(log_probs) != len(returns):
        raise ValueError("one log probability is required for every trace reward")
    return_tensor = torch.as_tensor(
        returns, dtype=torch.float32, device=log_probs[0].device
    )
    if baseline == "none":
        advantages = return_tensor
    elif baseline == "running_mean":
        advantages = return_tensor - float(baseline_value)
    else:
        raise ValueError("baseline must be none or running_mean")
    stacked = torch.stack(list(log_probs))
    discounts = torch.as_tensor(
        [gamma**index for index in range(len(returns))],
        dtype=stacked.dtype,
        device=stacked.device,
    )
    # The sampled return is a score-function weight; it must not backpropagate
    # through the policy network or accidentally become a pathwise estimator.
    return -(stacked * advantages.detach() * discounts).sum()


@dataclass
class PolicyEpisode:
    result: object
    rewards: np.ndarray
    returns: np.ndarray
    actions: list[int]
    log_probs: list[object]


class SpeedChoicePolicy:
    """Categorical 2-vs-6 m/s policy with fixed geometric steering."""

    def __init__(self, seed: int = 0, hidden_size: int = 16):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        self.network = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, len(SPEED_CHOICES_MPS)),
        )
        self.training = True
        self._log_probs: list[object] = []
        self._actions: list[int] = []
        self._controller_config = {
            "lateral_gain": 0.32,
            "heading_gain": 0.85,
            "speed_gain": 0.25,
        }
        self.deployment_mode = "deterministic_argmax_deployment"

    def configure_controller(self, config: DrivingConfig) -> None:
        self._controller_config = {
            "lateral_gain": float(config.lateral_gain),
            "heading_gain": float(config.heading_gain),
            "speed_gain": float(config.speed_gain),
        }

    def controller_manifest(self) -> dict[str, float]:
        return dict(self._controller_config)

    def policy_mode(self) -> str:
        return (
            "stochastic_categorical_sampling" if self.training else self.deployment_mode
        )

    def parameters(self):
        return self.network.parameters()

    def train(self, mode: bool = True):
        self.training = bool(mode)
        self.network.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def reset_episode(self) -> None:
        self._log_probs = []
        self._actions = []

    @property
    def actions(self) -> list[int]:
        return list(self._actions)

    @property
    def log_probs(self) -> list[object]:
        return list(self._log_probs)

    def logits(self, observation: DrivingObservation, reference: LaneReference):
        import torch

        features = torch.as_tensor(
            observation_features(observation, reference), dtype=torch.float32
        )
        return self.network(features)

    def action_distribution(self, observation, reference):
        from torch.distributions import Categorical

        return Categorical(logits=self.logits(observation, reference))

    def control(
        self, observation: DrivingObservation, reference: LaneReference
    ) -> np.ndarray:
        """Choose speed, then return a real continuous throttle command."""

        import torch

        distribution = self.action_distribution(observation, reference)
        if self.training:
            action = distribution.sample()
            self._log_probs.append(distribution.log_prob(action))
        else:
            action = torch.argmax(distribution.logits, dim=-1)
        index = int(action.item())
        self._actions.append(index)
        target = SPEED_CHOICES_MPS[index]
        steering_controller = GeometricController(
            target_speed_mps=target, **self._controller_config
        )
        command = steering_controller.control(observation, reference)
        return np.asarray(command, dtype=np.float32)


class FixedSpeedPolicy:
    """Evaluation control with the same geometric steering and one fixed speed."""

    def __init__(self, target_speed_mps: float):
        if target_speed_mps < 0 or not np.isfinite(target_speed_mps):
            raise ValueError("target_speed_mps must be finite and nonnegative")
        self.target_speed_mps = float(target_speed_mps)
        self._actions: list[int] = []
        self._controller_config = {
            "lateral_gain": 0.32,
            "heading_gain": 0.85,
            "speed_gain": 0.25,
        }
        self.deployment_mode = "deterministic_fixed_speed"

    def configure_controller(self, config: DrivingConfig) -> None:
        self._controller_config = {
            "lateral_gain": float(config.lateral_gain),
            "heading_gain": float(config.heading_gain),
            "speed_gain": float(config.speed_gain),
        }

    def controller_manifest(self) -> dict[str, float]:
        return dict(self._controller_config)

    def policy_mode(self) -> str:
        return self.deployment_mode

    def eval(self):
        return self

    def train(self, mode: bool = True):
        return self

    def reset_episode(self):
        self._actions = []

    @property
    def actions(self):
        return list(self._actions)

    @property
    def log_probs(self):
        return []

    def control(self, observation, reference):
        self._actions.append(0 if self.target_speed_mps <= 2.0 else 1)
        return GeometricController(
            target_speed_mps=self.target_speed_mps, **self._controller_config
        ).control(observation, reference)


def run_policy_episode(
    config: DrivingConfig, policy, *, gamma: float = 0.99
) -> PolicyEpisode:
    """Run one sequential policy episode and align actions/log-probs/rewards."""

    policy.configure_controller(config)
    policy.reset_episode()
    result = run_episode(config, policy=policy)
    mode = policy.policy_mode()
    result.config["policy"] = {
        "policy_mode": mode,
        "deployment_mode": policy.deployment_mode,
        "controller_gains": policy.controller_manifest(),
    }
    rewards = trace_rewards(result.trace)
    if len(policy.actions) != len(rewards):
        raise RuntimeError("policy decisions and measured rewards are misaligned")
    returns = finite_horizon_returns(rewards, gamma)
    return PolicyEpisode(result, rewards, returns, policy.actions, policy.log_probs)


@dataclass
class TrainingResult:
    policy: SpeedChoicePolicy
    history: list[dict[str, float]]
    checkpoint: Path | None = None


def train_reinforce(
    config: DrivingConfig,
    *,
    episodes: int = 12,
    seeds: Sequence[int] = (7, 11, 19),
    gamma: float = 0.99,
    learning_rate: float = 0.01,
    hidden_size: int = 16,
    baseline: str = "running_mean",
    checkpoint: Path | None = None,
    model_seed: int | None = None,
) -> TrainingResult:
    """Train a small policy for a bounded number of real MetaDrive episodes."""

    import torch

    if episodes < 1 or not seeds:
        raise ValueError("episodes and seeds must be positive")
    torch.set_num_threads(1)
    if model_seed is None:
        model_seed = int(seeds[0])
    torch.manual_seed(int(model_seed))
    policy = SpeedChoicePolicy(seed=int(model_seed), hidden_size=hidden_size)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    running = RunningMeanBaseline()
    history: list[dict[str, float]] = []
    for episode in range(episodes):
        episode_config = DrivingConfig(
            **{**asdict(config), "seed": int(seeds[episode % len(seeds)])}
        )
        policy.train().reset_episode()
        result = run_policy_episode(episode_config, policy, gamma=gamma)
        loss = reinforce_loss(
            result.log_probs,
            result.rewards,
            gamma,
            baseline=baseline,
            baseline_value=running.mean,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(policy.parameters()), 5.0)
        optimizer.step()
        running.update(result.returns)
        history.append(
            {
                "episode": float(episode),
                "loss": float(loss.detach().cpu()),
                "discounted_objective": float(result.returns[0]),
                "undiscounted_reward_sum": float(result.rewards.sum()),
                "steps": float(len(result.rewards)),
                "failure": float(result.result.metrics["failure"]),
            }
        )
    if checkpoint is not None:
        save_checkpoint(
            policy,
            checkpoint,
            config=config,
            history=history,
            model_seed=model_seed,
            training_hyperparameters={
                "episodes": episodes,
                "gamma": gamma,
                "learning_rate": learning_rate,
                "baseline": baseline,
                "hidden_size": hidden_size,
                "model_seed": model_seed,
                "environment_seeds": [int(seed) for seed in seeds],
            },
        )
    return TrainingResult(policy, history, checkpoint)


def save_checkpoint(
    policy,
    path: Path,
    *,
    config: DrivingConfig,
    history: Sequence[Mapping[str, float]],
    model_seed: int | None = None,
    training_hyperparameters: Mapping[str, object] | None = None,
) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_config = {
        "input_features": list(FEATURE_NAMES),
        "hidden_size": int(policy.network[0].out_features),
        "output_actions": list(SPEED_CHOICES_MPS),
        "steering": "fixed GeometricController",
        "controller_gains": {
            "lateral_gain": config.lateral_gain,
            "heading_gain": config.heading_gain,
            "speed_gain": config.speed_gain,
        },
        "partial_observation": True,
        "hidden_delay_queue_steps": config.action_delay_steps,
    }
    config_hash = hashlib.sha256(
        json.dumps(asdict(config), sort_keys=True).encode("utf-8")
    ).hexdigest()
    torch.save(
        {
            "policy_kind": "speed_choice_over_fixed_geometric_steering",
            "model_seed": model_seed,
            "training_hyperparameters": dict(training_hyperparameters or {}),
            "model_config": model_config,
            "config_hash": config_hash,
            "state_dict": policy.network.state_dict(),
            "hidden_size": policy.network[0].out_features,
            "speed_choices_mps": SPEED_CHOICES_MPS,
            "feature_names": FEATURE_NAMES,
            "config": asdict(config),
            "history": list(history),
        },
        path,
    )
    return path


def load_checkpoint(path: Path) -> SpeedChoicePolicy:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    policy = SpeedChoicePolicy(seed=0, hidden_size=int(payload["hidden_size"]))
    policy.network.load_state_dict(payload["state_dict"])
    return policy.eval()


def evaluate_policy(
    policy, configs: Iterable[DrivingConfig]
) -> list[dict[str, object]]:
    """Evaluate by replaying saved policy weights on supplied conditions."""

    rows = []
    for config in configs:
        policy.eval()
        episode = run_policy_episode(config, policy)
        metrics = dict(episode.result.metrics)
        metrics.update(
            {
                "seed": config.seed,
                "initial_lateral_offset_m": config.initial_lateral_offset_m,
                "discounted_deployment_return": float(episode.returns[0]),
                "undiscounted_reward_sum": float(episode.rewards.sum()),
                "policy_mode": getattr(policy, "deployment_mode", "unknown"),
                "actions": episode.actions,
            }
        )
        rows.append(metrics)
    return rows


def save_evaluation(path: Path, payload: Mapping[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


__all__ = [
    "FEATURE_NAMES",
    "FiniteMDP",
    "PolicyEpisode",
    "RunningMeanBaseline",
    "SPEED_CHOICES_MPS",
    "SpeedChoicePolicy",
    "FixedSpeedPolicy",
    "TrainingResult",
    "bellman_optimality_update",
    "bellman_policy_update",
    "discounted_return",
    "evaluate_policy",
    "finite_horizon_returns",
    "load_checkpoint",
    "observation_features",
    "q_learning_update",
    "reinforce_loss",
    "reward_to_go",
    "run_policy_episode",
    "save_checkpoint",
    "save_evaluation",
    "teaching_mdp",
    "trace_rewards",
    "train_reinforce",
]
