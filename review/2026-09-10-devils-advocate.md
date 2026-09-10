# Devil's-advocate preliminary review — 2026-09-10

## Scope and evidence

This is a preliminary review of the completed learner navigation, route framing,
reference claims, and archived-material labeling.  It deliberately does **not**
review the first-loop lessons yet: at inspection time, `course/first_loop/`,
`scripts/build_first_unit.py`, and `scripts/run_first_unit.py` were not present.
`src/ad_tutorial/driving.py` and `requirements-driving.txt` had appeared, but
the learner-facing unit and its generated notebooks had not.

I read `AGENTS.md`, `PROJECT_REFERENCE.md`, this role brief, `README.md`,
`course/README.md`, `reference/*.md`, `reference/legacy/README.md`,
`index.html`, the structural/CI scripts, and the archived 07 and 10 notebooks.
I also inspected the first two cells of every archived course notebook.  A
live check of the referenced Hugging Face Robotics Course still shows its RL,
imitation-learning, and foundation-model units as “Coming Soon”; the dated
reference claim is supported at this review date.  The local structural check
could not be executed in the current interpreter because `nbformat` is not
installed there; its source also requires the two active notebooks that were
not yet generated.

## Findings

### P0 — The promised learner entry is not deliverable until the first unit lands

- **Location:** `README.md`, `index.html`, `course/README.md`,
  `reference/README.md`, `reference/legacy/README.md`,
  `PROJECT_REFERENCE.md`, `.github/workflows/quality.yml`, and
  `scripts/validate_project.py`.
- **Evidence:** Every primary “start” link resolves to
  `course/first_loop/README.md`.  At inspection it did not exist; neither did
  the first-unit generator/runner.  The validator requires exactly two
  notebooks in that directory and the CI workflow invokes the missing
  generator.  Thus a learner who follows the new landing page has no lesson,
  while maintenance commands cannot yet establish the claimed 2+14 structure.
- **Impact:** This blocks the central promise that a zero-RL learner can begin
  the two-week unit.  The archive is correctly demoted, but it cannot be the
  fallback because the navigation explicitly says not to follow it as a course.
- **Minimal fix / acceptance:** Add the unit index, exactly two generated
  notebooks, generator, runner, and pinned requirement file as one coherent
  change.  From a fresh first-unit environment, run generation, the structural
  validator, the documented runner, and notebook execution.  Re-run local-link
  validation after those files exist.  Reassess this finding in the final
  lesson review; it is a current delivery gap, not a criticism of the planned
  curriculum.

### P1 — The archived capstone's diagram still reads as an executed closed loop

- **Location:** `reference/legacy/course/10_capstone.ipynb`, first through
  third markdown cells; `reference/legacy/README.md`, Capstone row.
- **Evidence:** The opening archive notice and index correctly state that the
  model does not re-drive planning, execution, or evaluation.  Immediately
  afterward, however, the title says it will connect “model, scenario,
  closed loop and runtime,” and a downward-arrow diagram presents
  `05_bev_model.pt → ... → 07 planner → 08 evaluation → 09 gate`.  For a
  learner scanning the lesson, that conventional causal diagram contradicts
  the actual code boundary rather than teaching the boundary.
- **Impact:** A learner can leave with the exact false inference the archive is
  meant to expose: checkpoint loading plus a report means a model-development
  closed loop.
- **Minimal fix:** Retitle this archived section as an *artifact-lineage audit*
  and label each arrow “claimed handoff / not executed.”  Put one short,
  concrete prompt next to the diagram: identify the missing function call that
  would have to feed model output into planning and `env.step`.  Keep the
  archive notice and index wording; their existing warning is a strength.

### P2 — The optional TUM reading is useful but not yet a time-bounded learner task

- **Location:** `reference/README.md`, “第一单元的选读”; `course/README.md`,
  weeks 1–2 row.
- **Evidence:** Both files recommend “Introduction and Control” through the
  top-level TUM ADSE repository.  The repository contains many numbered
  sessions and separate practice dependencies, so the learner must discover
  which concrete material maps to the first-loop experiment.  This is an
  optional reference, not a reason to delay the unit.
- **Impact:** The 26-hour budget may be spent navigating a larger external
  course instead of predicting and explaining the local control experiment.
- **Minimal fix:** In the first-unit README, give this reading a small budget
  and a single question to answer (state, reference, action, and feedback),
  then link the exact current TUM control material or name its numbered
  session.  Do not add a broader future curriculum for this finding.

## Archive assessment

The archive separation is materially better than the former 11+3 presentation.
The root, course route, HTML landing page, notebooks notice, labs notice, and
archive index consistently direct learners to the first unit and identify the
remaining 24-week map as a plan.  The known historical flaws are explicitly
labeled rather than silently repaired or represented as current capability:

- 02/05 label the LiDAR-copy / point-set limitation;
- 06/07 label the other-agent prediction versus ego-trajectory error and the
  non-consuming rollout;
- 08 and 10 label the absence of a model-to-execution causal chain;
- the three labs state their hand-written/local-interface boundaries.

Those are historical defects intentionally retained for diagnosis, not new P0
or P1 findings.  The P1 above concerns the remaining contradictory *current
explanation* around the capstone diagram.

## Final-review gate

Once the first unit is generated, review the actual README and both notebooks
before accepting the course.  Check that a learner can predict one numeric
control change before running it; identify state, route, commands, applied
actions, units, and delay queue; run baseline/delay/recovery through
`env.step`; recompute metrics from the saved trace; and distinguish horizon
survival, route completion, simulator evidence, and real-road capability.

## Final review of the generated first unit

### Final evidence and scope

The unit is now present: `course/first_loop/README.md`, two generated
notebooks, `scripts/build_first_unit.py`, `scripts/run_first_unit.py`,
`src/ad_tutorial/driving.py`, and `requirements-driving.txt`.  I read all
learner-facing Markdown and every cell in both notebooks, inspected the CLI,
the current trace JSON/CSV/summary artifacts, and the implementation paths
that create them.  I did not run a fresh full MetaDrive simulation in this
pass, as requested; that is the lead's remaining runtime verification.

Static/reproducibility checks passed with the first-unit virtual environment:

```powershell
.venv\Scripts\python.exe scripts\build_first_unit.py
git diff --exit-code -- course/first_loop
.venv\Scripts\python.exe -m pytest tests/test_driving.py -q -k "not real_metadrive and not delay_changes"
.venv\Scripts\python.exe scripts\validate_project.py
git diff --check
```

The focused pure tests reported `3 passed, 2 deselected`; the structural check
reported `PASS: 2 active + 14 archived notebooks, Python syntax, local links
and indexes`.  CLI help exposes the documented unit choices.  These checks do
not establish the new lesson's full simulator results.

The preliminary delivery P0 is therefore resolved.  The archive-capstone P1
is resolved too: the regenerated Chapter 10 is now titled “归档审计” and
visually separates executed file/inference work from the missing
prediction/planner/`env.step` links.  The P2 reading finding is resolved by
the precise TUM 08 control-practice link and 60–90 minute scope.

### P0 — “Recovery” is not a controlled experiment and is absent from the notebook

- **Location:** `course/first_loop/02_delay_and_recovery.ipynb`, cells 3–6;
  `scripts/build_first_unit.py`, `build_unit2`; `scripts/run_first_unit.py`,
  the `recovery` branch; `course/first_loop/README.md`, learning order and
  week-2 schedule.
- **Evidence:** The notebook executes only baseline and four-step-delay
  episodes, both at `initial_lateral_offset_m=0.8`.  It describes the
  recovery threshold but has no recovery result, trace, plot, or exercise
  that compares recovery enabled with recovery disabled.  The CLI's `all`
  mode calls the same controller with a different initial offset (`+1.3 m`)
  for `recovery`, versus `+0.8 m` for baseline/delay.  The current controller
  has no `recovery_enabled` switch or equivalent normal-gain control.  The
  checked `artifacts/first_loop_verify/summary.json` consequently compares
  different starting conditions: baseline/delay at about `0.8 m`, recovery at
  about `1.3 m`.
- **Impact:** A learner cannot attribute any difference to the recovery
  controller.  This fails the stated two-week question “改善了什么、付出了什么”
  and turns the second unit into a delay demo with a prose-only recovery
  branch.
- **Minimal fix / acceptance:** Add a bounded config switch that keeps the
  normal gains active, then run paired recovery-off/recovery-on episodes with
  the same seed, road, horizon, offset, delay, and all other settings.  Put
  that pair, its command/applied traces, terminal cause, and a prediction
  prompt in Unit 2 and the CLI.  A separate `+1.3 m` threshold-trigger
  demonstration may remain, but must be labeled as such rather than compared
  with the `+0.8 m` baseline.  The learner should be able to change precisely
  one recovery control parameter and explain the resulting trace.

### P1 — The two notebooks are runnable demonstrations, not yet a substantive 52-hour guided unit

- **Location:** `course/first_loop/01_drive_and_observe.ipynb` and
  `02_delay_and_recovery.ipynb`; `course/first_loop/README.md`, two-week
  schedule.
- **Evidence:** Each notebook has six cells.  Unit 1 runs one fixed baseline
  and saves one plot; Unit 2 runs one baseline/delay pair and plots only
  lateral error.  The exercises ask learners to change lookahead/offset,
  sweep delays, compare command versus applied steering, locate a failure,
  and run recovery, but they supply no editable experiment cells, trace-table
  display, action comparison plot, sweep loop, or recovery execution.  Most
  proposed work is therefore either prose-only or requires the novice to
  design the essential analysis before the lesson has taught it.
- **Impact:** A learner with shallow deep-learning experience can reproduce a
  black-box run but is not reliably led through prediction → one change →
  observed action/state → explanation.  Calling this a two-week / 52-hour
  unit is not supported by the present guided work.
- **Minimal fix:** Keep the two-notebook boundary, but add a small sequence of
  editable cells: expose one first-row state/reference/command/applied-action
  table, recreate the stated numeric check from that row, modify one named
  config value, compute the requested first-return and shared-window metrics,
  and save a short learner result table.  Unit 2 should include the delay
  sweep, command-versus-applied plot, paired recovery experiment from P0, and
  a fill-in conclusion template that requires a prediction, observed evidence,
  and a limitation.  This is focused scaffolding, not a request for a broader
  future curriculum.

### P1 — The learner comparison omits time and task-outcome evidence needed to interpret delay and “success”

- **Location:** `course/first_loop/02_delay_and_recovery.ipynb`, cells 3–6;
  `course/first_loop/README.md`, numerical example and schedule.
- **Evidence:** The implementation and current traces correctly contain
  `time_s`, `decision_dt_s=0.1`, `route_completion`, `terminated`,
  `truncated`, and failure flags.  The generated notebook instead plots only
  `step` versus lateral error and prints only steps, distance, mean/max error,
  recovery count, out-of-road, and crash.  It does not connect four delay
  steps to `0.4 s`, display `command_steering` beside `applied_steering`, or
  show arrival, route completion, truncation, and failure reason together.
- **Impact:** The learner cannot use the supplied comparison to distinguish a
  0.4-second delayed action from a four-row index difference, or a short
  failed run from a route-completing run.  This contradicts the project's own
  completion criterion and makes mean error especially easy to misread when
  a delayed episode ends early.
- **Minimal fix:** Add `time_s` to the plots/table and state the pinned
  `0.1 s` decision interval in Unit 2.  Present command and applied steering
  on the same time axis.  Put `arrive_dest`, `route_completion`,
  `terminated`, `truncated`, and `failure_reason` next to error metrics, and
  require a shared-time-window comparison when episode lengths differ.

### P2 — The supplied Unit-1 answer is not directly runnable

- **Location:** `course/first_loop/01_drive_and_observe.ipynb`, final
  Markdown cell.
- **Evidence:** The answer proposes
  `np.flatnonzero(np.abs(error) < 0.1)[0]`, but the notebook imports no NumPy,
  creates no `error` array, and gives no safe branch for the no-hit case it
  describes.  It is an explanatory fragment rather than the promised
  runnable check.
- **Impact:** The first attempt to turn the answer into evidence can fail at
  an undefined name, or make a learner mistake an absent return for a coding
  failure.
- **Minimal fix:** Add the four-line starter cell that derives `error` from
  `result.trace`, imports NumPy, and prints either the first hit and `time_s`
  or “not reached within horizon.”

### P2 — Existing first-loop evidence needs replacement after the final source is run

- **Location:** `artifacts/first_loop_verify/summary.json` and
  `review/2026-09-10-validation.md`.
- **Evidence:** The checked `first_loop_verify` summary has
  `route_completion: null` and `termination_reason: null`, while the current
  source emits a numeric `route_completion` and `failure_reason`; the newer
  metadata verification artifact demonstrates the latter fields.  The
  validation record appropriately leaves “第一单元最终验证” pending.
- **Impact:** The old verify directory cannot be used as final evidence for
  the current lesson or its newly explained outcome fields.
- **Minimal fix:** After P0/P1 fixes, run the documented CLI into a clean
  evidence directory, retain the resulting trace/CSV/PNG/GIF as applicable,
  and update the validation record with the exact command and reported
  baseline/delay/recovery comparison.  Do not replace the source-level
  limitation with an unsupported performance claim.

## Final judgment

The navigation, scope boundary, reference route, archive warnings, real
`env.step` chain, trace provenance, and source-level delay mechanics are now
credible.  The active unit is not ready to be called a complete two-week
learning unit until the P0 paired recovery experiment and P1 learner-facing
analysis scaffolding are added and then executed.  The outstanding issues are
in the current active teaching flow, not in the intentionally preserved
historical archive defects.

## Closing re-review after the lesson rewrite

### Result

The active teaching-flow P0 and P1 findings above are **resolved** in the
regenerated lesson.  I found no remaining pedagogical blocker.  The lesson is
now a focused two-week unit whose 52 hours are explicitly a learner time budget
for guided notebooks plus independent modification, reading, debugging, and
analysis; it does not claim that the default cells alone consume 52 hours.

### Evidence inspected

- `01_drive_and_observe.ipynb` now has 12 cells.  It distinguishes privileged
  simulator state from MetaDrive's returned observation vector, defines world
  and lane coordinates, carries units through the feedback equation, hand
  computes the `+0.5 m` first command (`+0.21306`), shows pre-step and
  post-step rows, recomputes the command and mean error from the trace, and
  gives a directly editable offset experiment.
- `02_delay_and_recovery.ipynb` now has 14 cells.  It defines four delayed
  steps as `0.4 s`, verifies the command queue against saved actions, runs the
  three actual conditions, uses a time-axis plot, inspects contact/footprint
  failure evidence, computes a common-time-window metric, and requires a
  bounded research note.
- The three-condition code and CLI now use the same road, initial offset,
  seed, horizon, and delay for the recovery comparison.  `baseline → delay`
  changes only `action_delay_steps`; `delay → recovery` changes only
  `target_speed_mps` from 6 to 2 m/s.  The text accurately calls this a
  slower-policy intervention, retains the delay, and says it is not a
  post-crash rescue or a safety guarantee.
- Current `artifacts/first_loop/summary.json` matches that contract: baseline
  lasts 18.0 s without failure, delay fails at 3.7 s on the white continuous
  line, and the lower-speed recovery lasts 18.0 s without failure but travels
  less distance.  `artifacts/delay12/summary.json` gives the needed
  counterexample: at 12 delayed steps, the 2 m/s policy still fails at 3.9 s.
  These are pinned-simulator, fixed-scene results, not a claim of route
  completion, general driving competence, or real-road safety.
- The regenerated notebooks and source stayed in sync, local links and the
  2-active/14-archive structure passed validation, and the current
  `tests/test_driving.py` run reported `9 passed`.  This provides software
  evidence for the listed configuration and trace relationships.

### Evidence record closed

The final verification section of `review/2026-09-10-validation.md` has now
been filled with exact commands, the 9-test result, two notebook execution
times, matched three-condition CLI table, the 1.2-second-delay counterexample,
and the inspected GIF/trace boundary.  It also preserves the required
distinction: successful software tests and a simulator experiment do not
establish learner understanding.  The latter still needs actual learner trial,
independent prediction, and explanation; it is not an unresolved software or
lesson-design defect.
