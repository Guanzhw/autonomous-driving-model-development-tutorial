from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ad_tutorial.driving import (  # noqa: E402
    DrivingConfig,
    DrivingObservation,
    GeometricController,
    LaneReference,
)
from ad_tutorial.rl_foundations import (  # noqa: E402
    SpeedChoicePolicy,
    bellman_optimality_update,
    discounted_return,
    finite_horizon_returns,
    load_checkpoint,
    reinforce_loss,
    reward_to_go,
    save_checkpoint,
    teaching_mdp,
    trace_rewards,
)


def test_known_mdp_bellman_values_and_returns():
    mdp = teaching_mdp()
    values = {state: 0.0 for state in mdp.states}
    for _ in range(4):
        values = bellman_optimality_update(mdp, values, gamma=0.9)
    assert values["start"] == pytest.approx(4.6)
    assert discounted_return([1, 4], 0.9) == pytest.approx(4.6)
    assert reward_to_go([1, 4, 2], 0.9) == pytest.approx([6.22, 5.8, 2])


def test_finite_horizon_zero_and_explicit_bootstrap():
    assert finite_horizon_returns([1, 4], 0.9) == pytest.approx([4.6, 4])
    assert finite_horizon_returns([1, 4], 0.9, bootstrap=10) == pytest.approx(
        [12.7, 13]
    )


def test_trace_reward_uses_measured_fields_not_action_labels():
    rows = [
        {
            "longitudinal_m": 1.5,
            "before_longitudinal_m": 0.0,
            "lateral_error_m": 0.5,
            "crash": False,
            "out_of_road": False,
            "arrive_dest": False,
            "chosen_speed": 2,
        },
        {
            "longitudinal_m": 2.0,
            "before_longitudinal_m": 1.5,
            "lateral_error_m": 0.0,
            "crash": True,
            "out_of_road": False,
            "arrive_dest": False,
            "chosen_speed": 6,
        },
    ]
    rewards = trace_rewards(rows, lateral_cost=0.1, failure_penalty=5)
    assert rewards == pytest.approx([1.45, -4.5])
    rows[0]["chosen_speed"] = 6
    assert trace_rewards(rows, lateral_cost=0.1, failure_penalty=5)[0] == pytest.approx(
        1.45
    )


def test_score_function_detaches_return_and_updates_parameters():
    import torch

    policy = SpeedChoicePolicy(seed=3)
    observation = DrivingObservation((0.0, -0.5), 0.0, 1.0, 0.0, 0.5, 0.0, 3.5)
    reference = LaneReference((8.0, 0.0), 0.0, 8.0, 3.5)
    policy.reset_episode()
    policy.control(observation, reference)
    policy.control(observation, reference)
    loss = reinforce_loss(policy.log_probs, [1.0, -0.5], gamma=0.9, baseline="none")
    assert torch.isfinite(loss)
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.05)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, policy.parameters()))
    assert all(torch.isfinite(parameter).all() for parameter in policy.parameters())


def test_handworked_two_action_score_gradient():
    import torch
    from torch.distributions import Categorical

    logits = torch.zeros(2, requires_grad=True)
    log_prob = Categorical(logits=logits).log_prob(torch.tensor(0))
    reinforce_loss([log_prob], [2.0], gamma=1.0, baseline="none").backward()
    # At equal logits, d log p(action=0)/d logits = [1/2, -1/2].
    # The negative policy-gradient loss therefore has gradient [-1, +1].
    assert logits.grad == pytest.approx(torch.tensor([-1.0, 1.0]))


def test_checkpoint_preserves_deterministic_actions(tmp_path):
    policy = SpeedChoicePolicy(seed=4)
    observation = DrivingObservation((0.0, -0.5), 0.0, 1.0, 0.0, 0.5, 0.0, 3.5)
    reference = LaneReference((8.0, 0.0), 0.0, 8.0, 3.5)
    path = save_checkpoint(
        policy, tmp_path / "policy.pt", config=DrivingConfig(), history=[]
    )
    reloaded = load_checkpoint(path)
    policy.eval()
    reloaded.eval()
    assert [policy.control(observation, reference).tolist() for _ in range(4)] == [
        reloaded.control(observation, reference).tolist() for _ in range(4)
    ]


def test_two_step_discounted_objective_has_correct_score_weights():
    import torch
    from torch.distributions import Categorical

    logits = torch.zeros((2, 2), requires_grad=True)
    scores = Categorical(logits=logits).log_prob(torch.tensor([0, 1]))
    reinforce_loss(scores, [1.0, 4.0], gamma=0.9, baseline="none").backward()
    # G0=4.6; gamma*G1=3.6. Each equal-probability score has magnitude 1/2.
    # Per-step averaging or omission of the outer discount changes this result.
    assert torch.allclose(logits.grad, torch.tensor([[-2.3, 2.3], [1.8, -1.8]]))


def test_model_seeds_create_independent_initial_policies(tmp_path):
    import torch

    first = SpeedChoicePolicy(seed=7)
    second = SpeedChoicePolicy(seed=8)
    assert any(
        not torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters())
    )


def test_policy_uses_configured_controller_gains():
    policy = SpeedChoicePolicy(seed=3)
    config = DrivingConfig(lateral_gain=0.7, heading_gain=0.2, speed_gain=0.1)
    observation = DrivingObservation((0.0, -0.5), 0.0, 1.0, 0.0, 0.5, 0.0, 3.5)
    reference = LaneReference((8.0, 0.0), 0.0, 8.0, 3.5)
    policy.configure_controller(config)
    policy.eval()
    command = policy.control(observation, reference)
    expected = GeometricController(
        target_speed_mps=2.0 if policy.actions[-1] == 0 else 6.0,
        lateral_gain=0.7,
        heading_gain=0.2,
        speed_gain=0.1,
    ).control(observation, reference)
    assert np.allclose(command, expected)


def test_live_policy_speed_choice_changes_commands_and_trajectory():
    # Import torch before MetaDrive's package initializer; MetaDrive imports a
    # legacy PPO example that otherwise races the Windows DLL loader.
    import torch

    pytest.importorskip("metadrive")
    from ad_tutorial.driving import run_episode

    config = DrivingConfig(seed=7, horizon=12, action_delay_steps=0, decision_repeat=5)
    slow = SpeedChoicePolicy(seed=1)
    fast = SpeedChoicePolicy(seed=1)
    # Force the same measured state to choose distinct categories while the
    # geometric steering formula remains identical.
    with torch.no_grad():
        slow.network[-1].bias[:] = torch.tensor([4.0, -4.0])
        fast.network[-1].bias[:] = torch.tensor([-4.0, 4.0])
    slow.eval()
    fast.eval()
    first = run_episode(config, policy=slow)
    second = run_episode(config, policy=fast)
    assert first.trace[0]["command_throttle"] < second.trace[0]["command_throttle"]
    assert not np.allclose(
        [row["speed_mps"] for row in first.trace],
        [row["speed_mps"] for row in second.trace],
    )
