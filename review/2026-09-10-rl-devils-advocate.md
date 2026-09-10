# Devil's-advocate review — RL-foundations stage (2026-09-10)

## Scope and evidence

This read-only review covers the RL README, generators for Lessons 07/08,
`src/ad_tutorial/rl_foundations.py`, `scripts/run_rl_foundations.py`, and its
focused tests. The lead reports that nine focused tests and both notebooks
passed. The completed default CLI trained three independent model/sampling
seeds for eight episodes and evaluated six matched fixed-map conditions each:
argmax-policy discounted J changed `3.622 → 3.622` (seed 7), `4.245 → 5.726`
(seed 8), and `5.733 → 5.978` (seed 9), while seed 9 failures changed `0/6 →
2/6`. I did not repeat the full training run.

The scope claim is properly narrow. REINFORCE selects only 2 or 6 m/s target
speed; `GeometricController` retains steering; the continuous throttle reaches
real `env.step`; reward is reconstructed from measured trace fields, not action
labels; and the README says the short CPU run does not establish stability,
road generalization, or complete driving. The code also correctly uses a
previous-episode running-mean baseline before updating it, so it is independent
of the current sampled actions; it does not use the biased single-trajectory
standardization path.

## Findings

### P1 — The real delayed-speed task is partially observed, but the MDP lesson never marks the boundary

- **Location:** `course/rl_foundations/README.md`, learning order/CLI section;
  `scripts/build_rl_foundations.py`, Lesson 08 opening and sections 1–4;
  `src/ad_tutorial/rl_foundations.py:observation_features`; and
  `src/ad_tutorial/driving.py:DelayedActuator`.
- **Finding:** The policy receives privileged current simulator-derived vehicle
  features, not an estimator or sensor observation. More importantly, with the
  declared four-step action delay, the next physical state also depends on the
  queued issued throttle/steering commands. That queue is not in the five policy
  features. The simple tabular Lesson 07 is an MDP, but the policy input in
  Lesson 08 is therefore a partial observation of the delayed actuator state.
- **Impact:** A beginner can transfer the finite-MDP Markov assumption to a
  policy that cannot condition on an important part of the real simulator
  state. Calling this merely a “measured state” hides both the privileged-truth
  scope and why delay/history makes policy learning harder.
- **Minimal fix / acceptance:** Add one direct paragraph and a trace table:
  list the five supplied privileged features, name the hidden delay queue, and
  call the Lesson 08 input a POMDP-style observation under delay. State that
  augmenting observation/history with queue contents, or removing delay for a
  separate MDP-style control exercise, is a future controlled comparison.
  Require this distinction in the learner conclusion; do not imply a
  perception or state-estimation integration.
- **Status:** open.

### P1 — The score-function update is stated but not made inspectable enough for an RL beginner

- **Location:** `scripts/build_rl_foundations.py`, Lesson 08 section 1/2 and
  its optimizer cell; `tests/test_rl_foundations.py`,
  `test_handworked_two_action_score_gradient`.
- **Finding:** The notebook shows the REINFORCE formula and calls
  `reinforce_loss`, then asserts that some parameter changed. It does not show
  a two-action logit example in which `log_prob`, reward-to-go, the loss
  gradient, and one gradient-descent update are printed. The repository already
  has precisely this hand-worked equal-logit test, but it is only in tests and
  is not part of the learner sequence.
- **Impact:** The learner can memorize “sample + log_prob” without answering
  why a larger positive return raises the probability of the sampled target
  speed, why a negative return lowers it, or why no gradient passes through
  MetaDrive. That makes the most unfamiliar part of the transition from
  supervised learning opaque.
- **Minimal fix / acceptance:** Bring a small version of the existing
  two-action test into Lesson 08: start with equal logits, sample/fix action 0
  with a positive return, print the score gradient before and logits/probability
  after a gradient-descent step; repeat or state the sign reversal for a
  negative return. Put the detached trace return and the non-differentiable
  simulator explicitly on the causal side of the diagram. Keep the real
  `env.step` episode cell afterward as the evidence that supplies those
  rewards.
- **Status:** open.

### P1 — Lessons 07 and 08 do not supply the promised runnable experiment-to-answer scaffolding

- **Location:** `scripts/build_rl_foundations.py`, Lesson 07 section 4 and all
  of Lesson 08 after the training/reload cells.
- **Finding:** Lesson 07 ends with three useful exercises, but has neither a
  starter cell nor answers for the reward-boundary, `gamma=0`, and stochastic
  transition cases. Lesson 08 has no labelled editable experiment, prediction
  prompt, or answer: its only small training call is a fixed three-episode run,
  and the reload cell only prints deterministic action indices. A learner must
  infer how to alter one variable, how to retain matched conditions, and which
  outcome would confirm or refute their prediction.
- **Impact:** The first notebook teaches definitions but does not verify that a
  learner can apply them. The second risks becoming an opaque “parameters
  changed” demo, rather than the requested prediction → one controlled change
  → trace/return/failure explanation loop.
- **Minimal fix / acceptance:** Add a Lesson-07 cell that changes the second
  long-route reward and gamma, prints both candidate returns/Bellman values,
  and gives the numeric answers (`reward=0`: short wins; `gamma=0`: immediate
  reward selects short). Include a two-outcome expectation versus a sampled
  Q-update check. In Lesson 08, add one named, runnable experiment that keeps
  model seed, training/holdout conditions, horizon, reward, and fixed steering
  unchanged while varying exactly one stated factor (for example the
  four-step delay versus zero). Ask for a prior prediction and require
  discounted J, undiscounted reward, failure count, and trace evidence in the
  answer. Label it a mechanism comparison, not a performance sweep.
- **Status:** open.

### P2 — The learner-facing result path does not make the actual seed trade-off a required conclusion

- **Location:** `scripts/build_rl_foundations.py`, Lesson 08 sections 3–4;
  `scripts/run_rl_foundations.py`.
- **Finding:** The CLI properly saves all three model seeds, before/after
  checkpoints, matched conditions, and holdout conditions. The notebook says a
  short run may not improve, but it runs one three-episode model and has no
  result template/table that makes a learner reconcile all seed outcomes. The
  completed default run has one unchanged J, two increased J values, and seed
  9 increases failures from 0/6 to 2/6.
- **Impact:** A learner can inspect only a favorable seed or call the average
  J movement a win while omitting the concrete safety/robustness regression.
  The current source itself is honest; the missing exercise leaves that
  honesty optional in the learning workflow.
- **Minimal fix:** Add a CLI-summary reading cell/template that lists every
  model seed's before/after J, reward sum, distance, and failures alongside the
  holdout result. Its expected conclusion must say that no single seed or
  short run guarantees improvement, and that seed 9's failure increase limits
  any claim about the observed J increase.
- **Status:** open.

## Judgment

The stage is a technically restrained and useful bridge from a hand-checkable
MDP to a real, narrow on-policy interaction loop. It becomes pedagogically
ready when the delayed queue and privileged-truth observation boundary are
explicit, the score-function step is made numerical, and both notebooks lead
the learner through a concrete prediction and answer rather than treating
tests/CLI artifacts as implicit coursework.

## Final corrective recheck

The corrective source closes all findings in this review. The README and
Lesson 08 now state the privileged-simulator-truth boundary, the absence of
Unit 2 noise/filtering, and the hidden delayed-actuator queue as a partial
observation boundary. Lesson 08 includes the analytic two-action score
gradient/probability step, a same-checkpoint delay 4 versus delay 0 experiment
with matched seed/offset/horizon, and a per-seed CLI delta reader.

Lesson 07 now makes its editable second-step reward drive a new MDP/Bellman
calculation and checks `max(2, 1 + gamma*r)`. Its stochastic starter has an
editable continuation probability, a fixed-policy Bellman expectation of 2.8
at probability 0.5, and sampled targets of 4.6 or 1 with the explanatory
answer. These are meaningful hand-checks of the distinction between expected
value and an individual Q-learning target.

The G0 deployment-return label and progress/distance fields used in the
one-factor report are consistent with their printed computations. `git diff
--check` passed. I did not repeat the main agent's final tests, CLI, or all
notebook executions.
