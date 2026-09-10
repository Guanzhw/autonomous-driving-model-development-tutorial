# State-estimation technical review — 2026-09-10

## Scope and evidence

Reviewed the state-estimation stage only: `src/ad_tutorial/estimation.py`, its
`driving.py` integration, `scripts/run_state_estimation.py`, the deterministic
notebook generator and generated notebooks, `tests/test_estimation.py`, and
`course/state_estimation/README.md`.

The static control path is sound: each decision reads current truth, obtains one
observer output, feeds that output to both planner and controller, pushes the
resulting command through the delay queue, and then calls `env.step`.
`before_*`, `measurement_*`, and `input_*` make the intended causal boundary
auditable for the provided observers. The SE(2) transforms use the documented
`x forward, y left` convention and are mutual inverses.

Commands actually run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_estimation.py -q
```

Result: `6 passed in 2.45s`.

I also ran a source-level, no-simulator boundary probe with the stage's default
heading covariance (`R=0.018^2`, `Q=0.0002`, `dt=0.1`).  Feeding headings
`pi - 0.01` then `-pi + 0.01` produced estimates `3.131593` then `-0.093760`.
Those two measurements differ by about 0.02 rad on the circle, so the second
estimate should remain near either representation of pi rather than point near
zero.

The full `run_state_estimation.py` experiment was deliberately not run in this
review.

## Findings

### P1 — heading is filtered and scored as a line, not an angle

`NoisyObserver._measurement()` forms an unwrapped noisy heading
([`estimation.py:180`](../src/ad_tutorial/estimation.py#L180)), and the filter
passes it to the generic scalar update ([`estimation.py:204`](../src/ad_tutorial/estimation.py#L204)).
`ScalarKalman.update()` computes the innovation with ordinary subtraction
([`estimation.py:127-129`](../src/ad_tutorial/estimation.py#L127-L129)); only
the already-corrupted estimate is wrapped afterward ([`estimation.py:209`](../src/ad_tutorial/estimation.py#L209)).

At the `-pi`/`pi` representation boundary, physically adjacent headings become
a nearly `2*pi` innovation. The CLI repeats the same error in
`heading_rmse_rad`, subtracting heading fields without an angular residual
([`run_state_estimation.py:20-33`](../scripts/run_state_estimation.py#L20-L33)).
It can therefore report a multi-radian heading error and push the controller
toward the wrong bearing when the vehicle crosses that boundary.

Use a wrapped innovation for heading and calculate heading RMSE from wrapped
residuals. Add a boundary regression that covers both the filtered estimate and
the exported RMSE. The current six tests do not cover this case.

### P2 — default position filter has no motion model for a vehicle moving 6 m/s

The lesson's default control interval is 0.1 s (`decision_repeat=5`,
`physics_step_s=0.02`) and the target speed is 6 m/s
([`driving.py:20-31`](../src/ad_tutorial/driving.py#L20-L31)). A typical
longitudinal position change is consequently about 0.6 m per estimate.
However, the state transition is identity: `P` is merely enlarged by `Q*dt`
and the predicted state is the preceding position
([`estimation.py:116-129`](../src/ad_tutorial/estimation.py#L116-L129)). The
default position process standard deviation over one interval is only
`sqrt(0.01 * 0.1) = 0.0316 m` ([`estimation.py:62`](../src/ad_tutorial/estimation.py#L62)).

This is a causal filter, but it is not a dynamically credible position
estimator for the configured vehicle. It will systematically lag global
position and longitudinal coordinate even with zero sensor bias. The course
does mention lag, but it also presents the three-condition comparison as an
evaluation of state estimation ([`build_state_estimation.py:116-145`](../scripts/build_state_estimation.py#L116-L145)); the default model makes an apparent
filter-versus-raw conclusion largely an unmodelled-motion artifact.

Either estimate the position with a velocity/control-aware prediction model, or
narrow this activity to stationary/near-stationary scalar signals and make the
moving-position result an explicit failure demonstration. Add a moving-state
test that reports the expected lag and prevents a claim that default filter
position RMSE is an improvement.

### P2 — custom observers can export a fabricated `measurement_*` trace

The documented observer protocol only requires `reset(lane)` and
`observe(truth, step, dt) -> DrivingObservation`
([`course/state_estimation/README.md:28`](../course/state_estimation/README.md#L28)).
`run_episode()` then obtains a raw measurement from an undeclared optional
`last_measurement` attribute; if absent, it records `controller_input` as the
measurement ([`driving.py:283-287`](../src/ad_tutorial/driving.py#L283-L287)).
The in-scope `ShiftedObserver` test has no such attribute
([`test_estimation.py:86-94`](../tests/test_estimation.py#L86-L94)), so its
export labels the shifted controller input as a sensor measurement.

That silently breaks the stated `measurement_*` meaning and makes the CLI's
measurement RMSE unreliable for a valid protocol implementation. Require a
typed raw-measurement field or return separate raw and consumed observations;
otherwise record the raw measurement as unavailable rather than substituting
the estimate.

## Completion assessment

The state-estimation stage is not technically ready to claim correct closed-loop
heading estimation or heading RMSE until the P1 boundary error is fixed and
covered. The actual causal command path, coordinate transform, and focused
tests provide a sound base. The P2 items should be resolved before presenting
the default moving-position comparison as evidence that filtering improves
state estimation.

## Closure recheck — 2026-09-10

The requested corrective pass resolves the P1 and trace-semantics findings.

- `KalmanObserver` now unwraps each heading measurement relative to its prior
  estimate before the scalar update and wraps the resulting estimate afterward
  ([`estimation.py:239-253`](../src/ad_tutorial/estimation.py#L239-L253)). The
  CLI now applies the same wrapped residual to heading RMSE
  ([`run_state_estimation.py:19-39`](../scripts/run_state_estimation.py#L19-L39)).
  `test_heading_filter_uses_circular_innovation_at_pi_boundary` covers the
  estimator boundary ([`test_estimation.py:60-76`](../tests/test_estimation.py#L60-L76)).
- Process-variance fields now state their per-second units, and the generated
  lesson works through `R`, `Q*dt`, and posterior covariance while explicitly
  presenting moving-position lag as a limitation rather than filter-improvement
  evidence ([`build_state_estimation.py:151-200`](../scripts/build_state_estimation.py#L151-L200)).
- The observer contract now requires `last_measurement`, and `run_episode`
  rejects a custom observer that does not supply it instead of fabricating a
  `measurement_*` row ([`course/state_estimation/README.md:28`](../course/state_estimation/README.md#L28),
  [`driving.py:283-293`](../src/ad_tutorial/driving.py#L283-L293)).

I did not rerun the simulator in this closure recheck. The corrected source and
new regression address the original release-blocking P1; the stage is
technically approved conditional on the stage owner's reported final execution
and notebook validation.
