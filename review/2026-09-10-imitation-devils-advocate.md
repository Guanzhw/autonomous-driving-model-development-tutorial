# Devil's-advocate review — imitation-learning stage (2026-09-10)

## Scope and evidence

This is a read-only review of `course/imitation/README.md`, the generators for
Lessons 05/06, `src/ad_tutorial/imitation.py`, `scripts/run_imitation.py`,
focused tests, and the current course/catalog entry points.  The lead completed
the default CLI and both notebooks.  Its reported run used 840 demonstration
samples, offline held-out action MSE `0.000854`, and four matched fixed-map test
episodes: expert failed 0/4, BC 1/4, and the untrained MLP 2/4.  I did not
repeat the training/simulation run.

The stage gets several essential boundaries right.  Features are reconstructed
from present `before_*` state/reference fields and exclude action, reward,
terminal, post-step, and future fields; normalization is fit on train episodes;
the checkpoint carries the feature/action contract; and the reloaded policy
drives real MetaDrive rollouts.  The README correctly limits fixed-map seeds,
small split counts, and survival.  The findings below are about whether a
Chinese learner with basic DL experience can follow and falsify that claim.

## Findings

### P0 — Unit 3 is implemented locally but remains unreachable and marked planned in the published route

- **Location:** `scripts/course_catalog.py`; `README.md:29`;
  `course/README.md:17-25`.
- **Finding:** The stage supplies a README, two generated notebooks, runner,
  source, and tests, but it is absent from `ACTIVE_UNITS`, the root currently
  says only the first four weeks are executable, and weeks 5–6 describe BC as
  a plan without a lesson link.
- **Impact:** After completing Unit 2, the learner has no supported route to
  this stage.  Generation/execution/structure checks also cannot treat the
  purported stage as one delivered item.
- **Minimal fix / acceptance:** After the pending artifact/manifest fixes and
  full validation land, add the two notebooks to the shared catalogue; list
  Unit 3 in the root/course implemented sequence; and add Unit 2 → Unit 3 and
  course-route links.  Keep later BC extensions such as DAgger explicitly
  planned.  Verify the entry links and active-unit count with the generator and
  structural validator.
- **Status:** open; this is a release-route finding, not a request to block the
  pending technical fixes.

### P1 — The transition from state estimation silently returns to privileged truth

- **Location:** `course/imitation/README.md`, opening/data-boundary sections;
  `scripts/build_imitation.py`, Lesson 05 opening and section 1;
  `src/ad_tutorial/imitation.py:features_from_trace` and
  `collect_demonstrations`.
- **Finding:** Unit 2 established a measurement/estimate input boundary, but
  this unit collects `before_*` simulator truth and evaluates the BC policy on
  truth observations by calling `run_episode(config)` without an observer.  It
  says “MetaDrive 真值状态”, but does not name this as a deliberate controlled
  return to privileged state, state Unit 2 as a prerequisite, or explain why
  filtering is excluded from this BC experiment.
- **Impact:** The learner can reasonably infer that the sequence has made a
  learned policy consume estimated sensor state.  That would overstate what
  the stage demonstrates and obscures the useful design choice: isolate
  demonstration/distribution shift before composing estimation error and
  learning error.
- **Minimal fix / acceptance:** Link Unit 2 in the prerequisites and say,
  before the first collection cell, that Lesson 05 intentionally uses
  privileged simulator state to isolate BC.  Show the exact mapping from
  `before_*` trace fields to the four features and state that raw/filter input
  is a deferred composition experiment.  Require the learner to state this
  boundary in the final conclusion.
- **Status:** open.

### P1 — Lesson 05 calls the data and training path inspectable, but never exposes one state-to-action example

- **Location:** `scripts/build_imitation.py`, Lesson 05 cells 1–4;
  `course/imitation/README.md`, “Notebook 的 cell 可以逐个修改”.
- **Finding:** The notebook calls `collect_demonstrations`,
  `train_behavioral_cloning`, and `save_checkpoint`, then prints a manifest,
  hash, and loss curve.  It never displays one trace row alongside its four
  reconstructed features, the expert action, normalized features, model
  prediction, and action error.  The feature semantics are described in prose,
  but no numeric check ties `e_y`, heading error, bearing, and command to an
  actual controller decision.
- **Impact:** A learner can reproduce a low MSE as a black-box result without
  being able to locate the implementation boundary or diagnose a mistaken
  feature order/unit.  That defeats the stage's stated reason for using a
  small MLP after the learner has already seen routine supervised training.
- **Minimal fix / acceptance:** Add a short, editable audit cell after data
  collection that selects a named episode/step; prints the trace fields,
  `features_from_trace` output and `actions_from_trace` target; recomputes
  heading error/bearing from the row; and, after fitting, prints normalized
  features, prediction, and per-action squared error.  Include a numerical
  expected relation (for example, `heading_error = wrap(before_heading -
  reference_heading)`) and ask the learner to change one named state value
  only in a copy to predict the action direction.
- **Status:** open.

### P1 — Both editable exercises require unintroduced source rewrites, and the in-distribution experiment is mislabeled as a test change

- **Location:** `scripts/build_imitation.py`, Lesson 05 section 5 and Lesson
  06 section 4; `src/ad_tutorial/imitation.py`, fixed `FEATURE_NAMES`,
  `BehavioralCloningModel`, and `default_split_configs`.
- **Finding:** Lesson 05 asks for an `e_y`-only linear model, but every public
  helper fixes four feature names, a four-wide normalizer, and a two-hidden-
  layer MLP.  It supplies no starter data slice, linear model, or evaluator, so
  the requested comparison entails redesigning the source contract rather than
  modifying a taught cell.  Lesson 06 similarly asks learners to change test
  offset to `±0.1 m`, but `default_split_configs()` hard-codes offsets and the
  notebook gives no paired config/collection cell.  At `±0.1 m` the evaluation
  lies inside the train offset range; it is a new unseen *episode* but no
  longer the advertised held-out-offset test.
- **Impact:** The novice either cannot perform the designated experiments or
  changes only closed-loop configs while leaving offline MSE computed on the
  old `±0.4 m` data.  That makes the key offline-versus-closed-loop comparison
  ambiguous.  Calling the `±0.1 m` run a test also weakens the lesson's careful
  distinction between train/validation/test and distribution shift.
- **Minimal fix / acceptance:** Provide a bounded notebook helper for the
  linear baseline (explicit feature index, `nn.Linear`, train-only scaling,
  validation-MSE table) without changing the production policy contract.  For
  Lesson 06 provide a named function/cell that creates a fresh evaluation-only
  expert dataset and matched rollout configs for a requested offset, reports
  both offline and closed-loop evidence from that same setting, and labels
  `±0.1 m` “in-distribution evaluation”.  State that neither alternate
  evaluation may select a checkpoint or alter training.
- **Status:** open.

### P2 — The lesson's expected conclusion does not force the learner to reconcile the actual default result

- **Location:** `scripts/build_imitation.py`, Lesson 06 sections 2–4;
  `scripts/run_imitation.py` default experiment.
- **Finding:** The prose correctly says that offline MSE does not prove
  rollout recovery, but the answer remains generic (“通常…不能保证”).  The
  executed default gives a specific counterexample: MSE `0.000854`, BC failure
  1/4, expert failure 0/4, untrained failure 2/4.  The notebook's final answer
  does not require a learner to report that result, identify the failed test
  episode/reason, or compare BC to both baselines on a common window.
- **Impact:** A learner may correctly repeat the caveat yet never confront the
  observed fact that a small offline error coexists with a closed-loop failure.
- **Minimal fix:** Put a compact default-results table in Lesson 06 and make
  the conclusion template name offline MSE, all three failure counts, one
  failure reason, a common-window metric, and one fixed-map limitation.  Keep
  a blank/result-variable path so alternative machine/version outcomes are
  recorded rather than replaced by the default numbers.
- **Status:** open; the pending pairwise common-window export is the right
  implementation support for this exercise and is intentionally not evaluated
  as a blocker here.

## Judgment

The implemented comparison is a credible narrow BC experiment: it shows a
reloaded learned policy in the simulator and preserves a failure rather than
claiming that low offline MSE learned driving.  It will be a coherent third
unit once its prerequisite boundary is explicit, its experiments are runnable
from the notebook without source archaeology, and the published route advances
to it only after the pending validation/export work is complete.

## Follow-up after the BC rewrite

The source now resolves the three P1 lesson findings. Both notebooks identify
Unit 02 as prerequisite and name the deliberately privileged-truth boundary;
Lesson 05 exposes raw `before_*`/reference rows, all four reconstructed
features, train-only normalized values, both expert/model actions, and their
per-action errors. It supplies a runnable `e_y` least-squares baseline.
Lesson 06 labels the `±0.1 m` experiment in-distribution, creates matched
configs in a cell, and provides a conclusion template that requires full and
paired-common-window evidence. The pending evaluator/manifest work is now
present in the CLI path as well.

The P0 route/catalog item remains a release-integration condition: it must be
updated only when the main sequence publishes Unit 3. This review did not rerun
the reported regenerated notebooks or CLI after the rewrite.
