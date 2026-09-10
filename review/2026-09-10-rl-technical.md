# RL foundations technical review — 2026-09-10

## Scope and evidence

Reviewed `src/ad_tutorial/rl_foundations.py`, `scripts/run_rl_foundations.py`,
`scripts/build_rl_foundations.py`, `tests/test_rl_foundations.py`, and
`course/rl_foundations/` as a read-only technical pass.

The finite-MDP arithmetic is internally consistent. In the driving lesson, the
reward is reconstructed from post-step measured progress, lateral error, and
terminal flags rather than from the selected speed label
([`rl_foundations.py:209-235`](../src/ad_tutorial/rl_foundations.py#L209-L235)).
Each sampled categorical speed choice becomes a geometric throttle command and
then enters `run_episode`/MetaDrive ([`rl_foundations.py:383-401`](../src/ad_tutorial/rl_foundations.py#L383-L401)).

The REINFORCE loss now represents the documented
`J = E[sum_t gamma^t r_t]`: it applies both reward-to-go and the outer
`gamma^t` score weight, and sums a trajectory loss
([`rl_foundations.py:276-318`](../src/ad_tutorial/rl_foundations.py#L276-L318)).
The focused two-step gradient test checks precisely those weights
([`test_rl_foundations.py:118-127`](../tests/test_rl_foundations.py#L118-L127)).

Command actually run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_rl_foundations.py -q
```

Result: `9 passed in 2.78s`. I did not rerun the full CLI experiment.

## Findings

### P2 — evaluation `J` belongs to a different policy from the trained objective

Training samples from the categorical distribution and records score terms
([`rl_foundations.py:390-394`](../src/ad_tutorial/rl_foundations.py#L390-L394)),
so its REINFORCE objective is the stochastic categorical policy's expected
discounted return. Every CLI collection first calls `policy.eval()`
([`run_rl_foundations.py:37-41`](../scripts/run_rl_foundations.py#L37-L41));
evaluation then takes `argmax` instead of sampling
([`rl_foundations.py:390-397`](../src/ad_tutorial/rl_foundations.py#L390-L397)).
The exported `discounted_objective` is consequently one rollout return for a
different, deterministic deployment policy, despite the same name used for
training history ([`rl_foundations.py:502-510`](../src/ad_tutorial/rl_foundations.py#L502-L510)).

Argmax deployment is valid, but before/after evaluation cannot estimate the
stochastic `J` optimized by REINFORCE. Label the metrics separately and state
the policy mode in the manifest. If the lesson wants an estimate of optimized
`J`, add matched stochastic evaluation rollouts with explicit action-sampling
seeds; keep deterministic argmax evaluation as its own deployment comparison.

### P2 — delayed actuation makes the learned input an observation, not a Markov state

The default experiment has a four-decision action queue
([`run_rl_foundations.py:91-98`](../scripts/run_rl_foundations.py#L91-L98)).
The next simulator state therefore depends on issued actions still in
`DelayedActuator.queue`, while the policy input consists only of five current
vehicle/reference features ([`rl_foundations.py:177-206`](../src/ad_tutorial/rl_foundations.py#L177-L206)).
The queue contents are absent.

The tabular lesson is a finite MDP, but the driving policy's input defines a
partially observable process under the configured delay. A memoryless REINFORCE
policy remains a valid restricted policy class; the MDP/state terminology needs
to distinguish it from the policy observation. Say so in the lesson/manifest,
or include delay state when presenting the driving input as Markov.

### P2 — policy wrappers silently replace non-default configured controller gains

The policy adapters create a new `GeometricController` with only
`target_speed_mps` ([`rl_foundations.py:398-401`](../src/ad_tutorial/rl_foundations.py#L398-L401),
[`rl_foundations.py:430-434`](../src/ad_tutorial/rl_foundations.py#L430-L434)).
They therefore use `GeometricController` defaults, while the supplied
`DrivingConfig` can specify different lateral, heading, and speed gains.
`run_episode` does not inject its configured controller into a supplied policy.

The current CLI base uses those same defaults, so the completed default artifact
is unaffected. A caller that changes a gain will train/evaluate a different
fixed controller from the configuration recorded in its evidence. Pass the
relevant gains into policy construction, or constrain and label this stage as
using the default fixed geometric controller only.

### P2 — matched conditions are exported as raw rows but not paired effects

The runner correctly creates the same ordered condition lists for each baseline,
untrained checkpoint, and learned checkpoint
([`run_rl_foundations.py:99-177`](../scripts/run_rl_foundations.py#L99-L177)).
Its variability report pools each policy/configuration's rows into separate
means and standard deviations ([`run_rl_foundations.py:61-77`](../scripts/run_rl_foundations.py#L61-L77)).
The payload has enough raw provenance to reconstruct pairs, but no explicit
condition ID or learned-minus-untrained / learned-minus-fixed delta summary.

For a small, heterogeneous set of seeds and offsets, separate averages do not
show whether a gain is consistent within matched conditions. Add a stable
condition identifier and per-condition paired deltas for discounted return,
undiscounted reward, distance, and failure indicator, with an unweighted
mean/std across conditions. Preserve the existing raw rollouts and baseline
summaries.

## Implementation response (worker; pending independent verification)

The runner now labels every evaluation row
`deterministic_argmax_deployment`; training history and checkpoint manifests
retain the stochastic categorical training mode and full hyperparameters. The
deployment return is reported separately from the training discounted
objective and is not described as an estimate of stochastic training `J`.

The lesson and checkpoint manifest describe the four-step actuator queue as
hidden from the five reactive measured features. This makes the driving input
explicitly partially observable; the memoryless REINFORCE policy is presented
as a valid restricted policy class rather than as a Markov state claim.

Policy wrappers now receive the complete `DrivingConfig` controller gains at
the `run_policy_episode` boundary. The actual gains are recorded under the
result policy manifest and checkpoint model configuration. Evaluation rows
carry stable seed/offset condition IDs, and each model seed exports paired
learned-minus-untrained/fixed deltas plus equal-weight mean/std summaries for
deployment return, undiscounted reward, distance, and failure changes.

The implementation preserves three independent model/sampling seeds derived
from `--seed`, with matched untrained checkpoints and holdout fixed-speed and
untrained controls. A short CPU run remains a mechanism lesson; retained
failure and distance tradeoffs are evidence even when learning does not
improve every condition.

## Completion assessment

No error was found in the revised discounted REINFORCE gradient, reward source,
or policy-to-`env.step` causal chain. The stage is suitable as a bounded
mechanism lesson after it labels deterministic evaluation separately from the
stochastic training objective, accurately describes delayed driving input as
partial observation, and reports paired effects before treating compact
baseline aggregates as evidence of an RL gain.

## Closure recheck — 2026-09-10

The corrective pass resolves the four findings above.

- `run_policy_episode()` now configures the supplied policy from the active
  `DrivingConfig`, and both speed-policy adapters construct their geometric
  controller with those stored gains
  ([`rl_foundations.py:422-425`](../src/ad_tutorial/rl_foundations.py#L422-L425),
  [`rl_foundations.py:474-488`](../src/ad_tutorial/rl_foundations.py#L474-L488)).
  The gain-propagation regression covers a non-default configuration
  ([`test_rl_foundations.py:145-159`](../tests/test_rl_foundations.py#L145-L159)).
- Policy mode is recorded per episode. The CLI explicitly labels stochastic
  categorical training and deterministic-argmax deployment, and says the latter
  is not an estimate of the optimized stochastic `J`
  ([`run_rl_foundations.py:274-292`](../scripts/run_rl_foundations.py#L274-L292)).
- The same manifest and course text identify the hidden four-step actuator queue
  and the resulting memoryless reactive policy as partially observable
  ([`run_rl_foundations.py:281-292`](../scripts/run_rl_foundations.py#L281-L292),
  [`course/rl_foundations/README.md:28-30`](../course/rl_foundations/README.md#L28-L30)).
- Evaluation rows now contain stable seed/offset condition IDs. The runner
  exports per-condition learned-minus-untrained/fixed deltas and equal-weight
  summary statistics across all matched conditions
  ([`run_rl_foundations.py:84-122`](../scripts/run_rl_foundations.py#L84-L122),
  [`run_rl_foundations.py:334-377`](../scripts/run_rl_foundations.py#L334-L377)).
  Training hyperparameters, model seed, and environment seeds are written into
  both checkpoint and run provenance
  ([`rl_foundations.py:563-579`](../src/ad_tutorial/rl_foundations.py#L563-L579),
  [`run_rl_foundations.py:229-258`](../scripts/run_rl_foundations.py#L229-L258)).

I did not rerun the long CLI in this closure review. The corrected source
addresses the original technical findings, with no new substantive defect
observed. The RL stage is technically approved conditional on the stage owner's
final CLI, test, and notebook evidence.
