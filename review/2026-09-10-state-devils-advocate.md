# Devil's-advocate review — state-estimation stage (2026-09-10)

## Scope and evidence

This review covers only the new `state_estimation` stage: its learner README,
both notebook generators, `src/ad_tutorial/estimation.py`,
`scripts/run_state_estimation.py`, the observer hook in `driving.py`, and its
focused tests.  I also read the active-unit catalogue and course entry points.
The lead's completed fixed-scene run supplies the stated numerical result
below; I did not run a second MetaDrive episode in this review.

The stage does several important things correctly.  It keeps simulator truth
on the synthetic-measurement/evaluation side of the boundary, records both the
measurement and controller input, describes the fixed-lane projection as known
geometry, and explicitly says that a short straight-road result is neither
real-sensor evidence nor a localization claim.  The following findings concern
the learner path and the claims the exercises can actually support.

## Findings

### P0 — The published course still tells the learner that this delivered stage does not exist

- **Location:** `README.md:28`, `course/README.md:13-21`, and `index.html:85`.
- **Finding:** `scripts/course_catalog.py` registers `state_estimation` as an
  active two-notebook unit, but every public entry point says that the first
  unit is the only executable mainline and labels weeks 3–4 as unbuilt.
  Neither the root nor course route links the learner to this stage after
  Unit 1.
- **Impact:** A learner following the intended Chinese sequence cannot begin
  the new stage, and the course makes a false delivery-status statement.  The
  well-scoped state-estimation README cannot repair a broken entry path.
- **Minimal fix / acceptance:** Once the notebook and CLI verification is
  recorded, list Unit 2 under implemented units, link it as the next step from
  Unit 1/course route, and change only the remaining future units to planned.
  Confirm root → Unit 1 → Unit 2 local links and the active catalogue agree.
- **Status:** open.

### P1 — The Kalman notation and units omit the covariance relation the learner must use

- **Location:** `scripts/build_state_estimation.py`, `frames_measurements()`
  section 2 and `closed_loop()` opening; `src/ad_tutorial/estimation.py`,
  `MeasurementConfig` and `ScalarKalman.update()`.
- **Finding:** The lesson writes `v_k ~ N(0,R)` and says
  `MeasurementConfig` stores unit-bearing “R/Q”, while the config actually
  stores measurement *standard deviations* and the implementation constructs
  `R = sigma**2`.  It gives `P-`, `K`, and the state update, but omits the
  posterior covariance update `P = (1-K)P-` that the code performs.  The
  process fields have names ending in `m2`/`rad2`/`mps2`, while their declared
  meaning is variance **per second** and the code uses `Q * dt`.
- **Evidence:** With the default x measurement standard deviation of `0.18 m`,
  the source passes `R = 0.0324 m²`; with `Q = 0.01 m²/s` and a `0.1 s`
  decision interval, one prediction adds `0.001 m²`.  Those conversions are
  required to reproduce the implemented gain, but the notebook asks the
  learner to tune Q without showing them.
- **Impact:** A learner who has shallow DL experience is likely to treat a
  standard deviation as a variance or choose Q by a dimensionless number.  The
  resulting curve can be changed, but its gain and lag cannot be explained;
  that misses the unit's stated goal of connecting measurement, state, and
  action.
- **Minimal fix / acceptance:** State `R=sigma²`, `Q` and `Q*dt` with their
  units, add the posterior-P equation, and include a two-sample hand/computer
  calculation that prints `P-`, K, estimate, and P.  Rename manifest labels or
  add explicit `*_per_s` labels so exported artifacts carry the same unit
  contract.  The exercise should require a learner to predict how doubling
  sigma changes R and K before running it.
- **Status:** open.

### P1 — The default closed-loop result is a motion-model mismatch, but the lesson does not make that the required observation

- **Location:** `scripts/build_state_estimation.py`, `closed_loop()` title,
  formula, sections 1 and 3; `src/ad_tutorial/estimation.py:ScalarKalman`.
- **Finding:** Four independent random-walk filters estimate world x, world y,
  heading, and speed.  The position transition is locally constant, so it has
  no velocity or vehicle motion model.  In the completed default fixed-scene
  comparison, raw position RMSE is `0.233 m`, filter position RMSE is
  `2.448 m`, and the filter leaves the yellow line at about 8 s while raw and
  oracle reach the 12 s horizon.  The notebook mentions that smoothing *can*
  add lag, but frames the practical exercise as a generic Q-tuning question;
  it does not make the observed moving-state failure, signed along-lane lag,
  or failed termination a required explanation.
- **Impact:** “滤波变好了吗？” invites a learner to look for an RMSE winner,
  although this deliberately underspecified filter should expose why a state
  transition model matters.  Without a prescribed counterexample, the learner
  can tune Q until a prettier curve appears and miss that it is feeding stale
  world position to the planner.
- **Minimal fix / acceptance:** Put the default result table and its terminal
  outcome directly in Notebook 04.  Require a plot/table of measurement,
  estimate, and truth versus time for longitudinal position (including signed
  estimate-minus-truth error), then ask why the stationary-signal demo in
  Notebook 03 and the moving-car result differ.  Present a higher-Q run as a
  lag/noise trade-off, not a repair; make a constant-velocity or lane-coordinate
  model an explicitly deferred next experiment.  The expected observation must
  include the actual raw/filter RMSE and outcome, not only “可能”.
- **Status:** open.

### P1 — Bias is explained, then removed from every guided comparison

- **Location:** `scripts/build_state_estimation.py`,
  `frames_measurements()` section 2/3 and `closed_loop()` setup.
- **Finding:** The prose introduces `z=x+b+v`, correctly says that a fixed bias
  does not average away, and the default CLI config contains nonzero biases.
  Yet Notebook 03 sets all shown biases to zero, and Notebook 04 also sets all
  biases to zero before comparing oracle/raw/filter.  No supplied cell
  computes signed mean error or demonstrates the filtered estimate's biased
  steady state.
- **Impact:** The only runnable evidence teaches zero-mean noise.  The answer
  about bias remains an assertion, leaving the learner unable to distinguish
  “lower RMSE” from “calibrated sensor” or to connect a persistent offset to a
  persistent steering error.
- **Minimal fix / acceptance:** Retain a zero-bias noise experiment, then add
  a paired nonzero lateral/world-y bias run with the same seed.  Have the
  learner report signed mean error, RMSE, command shift, and whether either
  raw or scalar filtering removes the bias.  State the expected result:
  filtering suppresses random variation but does not estimate an unmodelled
  constant bias.
- **Status:** open.

### P1 — Heading is treated as an ordinary scalar across the wrap boundary

- **Location:** `src/ad_tutorial/estimation.py`, `_measurement()` and
  `NoisyObserver.observe()`; `scripts/run_state_estimation.py:_rmse`.
- **Finding:** The filter updates heading with the unwrapped noisy scalar, then
  wraps only its returned estimate.  Its innovation and RMSE use ordinary
  subtraction.  A measurement near `+pi` followed by one near `-pi` therefore
  looks like an almost `2pi` jump, despite representing a small physical
  rotation.  The current straight road does not exercise this case.
- **Impact:** The lesson teaches headings in rad and invites noise edits, but
  its reusable example yields a wrong estimate/metric at the very coordinate
  discontinuity students need to learn about.  A learner may misdiagnose the
  discontinuity as sensor noise or a Kalman limitation.
- **Minimal fix / acceptance:** Either bound this unit explicitly to headings
  away from the wrap and remove heading RMSE from its claimed comparison, or
  implement wrapped innovations and angular RMSE (`wrap(a-b)`) with a small
  `+pi/-pi` probe in Notebook 03.  The latter is the more useful progression
  before adding turns.
- **Status:** open.

### P2 — The linked 2023 filtering book names the wrong coauthor

- **Location:** `course/state_estimation/README.md` and both generated
  notebooks, “研究入口”.
- **Finding:** The local text attributes the 2023 *Bayesian Filtering and
  Smoothing* link to “Simo Särkkä 与 Arno Solin”.  The official Aalto author
  page identifies the 2023 second edition and the linked PDF as Simo Särkkä
  and Lennart Svensson; Särkkä and Arno Solin wrote *Applied Stochastic
  Differential Equations* (2019).
- **Impact:** The reading link works, but the incorrect attribution weakens a
  stage that asks learners to use authoritative sources for covariance models.
- **Minimal fix:** Change the displayed authors to Simo Särkkä and Lennart
  Svensson everywhere generated by this stage.
- **Status:** open.

## Judgment

The synthetic measurement boundary and simulator-capability wording are
appropriately restrained.  The stage becomes a sound next step after the P0
route correction and the P1 scaffolding makes its actual counterexample
visible: the present filter can smooth a stationary scalar, yet can harm a
moving closed loop because it lacks the needed transition model and cannot
remove bias.  That is a valuable lesson for this learner profile, provided it
is taught as the expected result rather than an incidental bad score.

## Follow-up after the state-estimation rewrite

### Resolved findings

The rewrite closes the original P0 route finding: root/course navigation now
publishes Unit 2, and Unit 1 links to it. It also closes the original teaching
findings in the source reviewed here: Lesson 04 now gives a two-sample R/Q/dt/K
and posterior-P calculation; the process-field names carry their per-second
units; Lesson 03 runs a stationary nonzero-bias pair and reports signed error;
and heading innovation is wrapped with a `+pi/-pi` probe plus angular RMSE.
Lesson 04 now requires the random-walk moving-state counterexample, shows
signed position lag and actual failure/outcome rather than implying that Q
tuning repairs it. The cited 2023 book now correctly names Lennart Svensson.

### Remaining P1 — Notebook 04 and the documented CLI use different sensor defaults

- **Location:** `scripts/build_state_estimation.py`, Lesson 04 setup;
  `scripts/run_state_estimation.py:main`; `course/state_estimation/README.md`,
  documented CLI/default-result wording.
- **Evidence:** Notebook 04 constructs `MeasurementConfig(seed=19,
  position_bias_m=(0,0), heading_bias_rad=0, speed_bias_mps=0,
  position_process_variance_m2_per_s=0.01)`. The documented CLI constructs
  `MeasurementConfig(seed=19)`, whose source defaults retain position bias
  `(0.10,-0.06)`, heading bias `0.008 rad`, and speed bias `0.05 m/s`.
  Both use the same seed/Q, but they are different measurement experiments.
- **Impact:** The notebook's 120-step RMSE/failure observation cannot be
  reproduced by the documented CLI and should not be described as its default
  result. The difference also hides the default CLI's persistent bias during
  the central moving-state comparison.
- **Minimal fix / acceptance:** Use `MeasurementConfig(seed=19)` in the
  notebook's principal oracle/raw/filter comparison, or explicitly call the
  zero-bias setup a separate controlled variant and report it separately from
  CLI-default evidence. The notebook, CLI, and saved manifest must agree for
  whichever result is used in learner instructions.
- **Status:** open at follow-up inspection.

The README still says that the manifest writes unit-bearing “Q/R”, while the
current manifest exports measurement standard deviations and per-second Q
fields rather than computed R variances. The new notebook's `R=sigma²`
calculation makes the teaching math clear; change the README phrase to “noise
standard deviations and Q” or export computed R values if literal manifest-R
provenance is intended.

### Closure

The follow-up repair aligns the primary Notebook 04 comparison to
`MeasurementConfig(seed=19)`, exactly the CLI default sensor (including bias
and Q), and labels that fact in the cell. The README now accurately records
measurement standard deviations, per-second Q, and `R=标准差²` instead of
claiming serialized R fields. The remaining P1 and manifest-wording note above
are therefore resolved. The reported regenerated-notebook, CLI, and focused
test results were not rerun in this review pass.
