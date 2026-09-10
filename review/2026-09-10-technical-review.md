# Technical review — 2026-09-10

## Result

The integrated first-loop unit now meets the technical review gate. No P0 or P1 finding remains in the final `driving.py`, CLI, tests, or current `artifacts/first_loop` evidence.

The unit is deliberately a state-observation control experiment: it does not claim camera perception, a learned planner, a recovery controller that acts after a collision, route completion, or real-road performance. Those limits are explicit in the generated lessons and unit README.

## Evidence inspected

I read `AGENTS.md`, `PROJECT_REFERENCE.md`, the Reviewer brief, final `src/ad_tutorial/driving.py`, `scripts/run_first_unit.py`, `scripts/build_first_unit.py`, `tests/test_driving.py`, the active-unit README, requirements, CI, and the installed MetaDrive 0.4.3 source. I inspected final JSON/CSV/PNG/GIF artifacts for baseline, delay, and recovery in `artifacts/first_loop`, including the manifest, transition trace, and failure frame.

I also ran:

```powershell
git diff --check
.venv\Scripts\python.exe -m py_compile src/ad_tutorial/driving.py scripts/run_first_unit.py scripts/build_first_unit.py tests/test_driving.py
```

Both passed. I did not start another MetaDrive run while the lead completed final runtime/notebook validation under the available-memory constraint. The separate validation record now reports nine tests passed, the two final notebooks executed successfully (2.4 s and 4.0 s), three matched CLI results, and the delay-12 counterexample.

## Resolved review findings

### T1 — Time, action semantics, and effective configuration are source-backed

- **Evidence:** `DrivingConfig` passes `decision_repeat` and `physics_step_s` into MetaDrive. `run_episode` reads the effective values from `env.config` and records `decision_dt_s`, per-transition `time_s`, and the installed simulator manifest. Installed MetaDrive 0.4.3 documents its physical step as `0.02 s` multiplied by `decision_repeat`.
- **Result:** The final manifest records a `0.1 s` decision interval, action shape, observation dimension, package version, Git commit, fixed map configuration, vehicle dimensions, reference lane, and actual initial state.

### T2 — Every trace row now represents one explicit state transition

- **Evidence:** Each row stores `before_*` state, the lane reference and command calculated from that state, the action applied after queue delay, and post-step state fields. The test suite checks that one row's post-state is the next row's pre-state and that the reference stays more than 7.99 m ahead of the control-time longitudinal coordinate.
- **Result:** A reader can recompute controller inputs and measured distance from saved JSON/CSV instead of relating a pre-step command to an unlabeled post-step observation.

### T3 — Baseline, delay, and mitigation are controlled comparisons

- **Evidence:** `experiment_configs` changes only `action_delay_steps` from baseline to delay, then only `target_speed_mps` from delay to recovery. The tests assert both configuration differences. All runs retain the fixed 290 m lane, same seed, initial offset, horizon, and controller gains.
- **Result:** In `artifacts/first_loop/summary.json`, baseline runs 180 steps for 98.6386 m and ends by horizon; delay=4 fails after 37 steps / 12.1678 m on a white continuous line; the slower 2 m/s mitigation survives the same 180-step horizon for 33.1224 m. The `delay12` artifact preserves the useful counterexample: 2 m/s still fails at 39 steps on a yellow continuous line.

### T4 — Success, horizon, and failure are distinct outcomes

- **Evidence:** The trace retains MetaDrive `arrive_dest`, `route_completion`, `terminated`, `truncated`, and physical contact flags. Metrics derive the meaningful `outcome` values `arrived`, `failure`, `horizon`, or `terminated`; the notebook comparison displays outcome, failure reason, and route completion.
- **Result:** The observed baseline and mitigation outcomes are `horizon`, not arrival. The teaching text correctly interprets this as no failure during the measured window, not task completion.

### T5 — Failure preservation now identifies the simulator cause

- **Evidence:** The fixed route uses one 3.5 m lane from x=10 to x=300, with spawn longitudinal 145 m and a 290 m reference lane; this keeps the 18 s experiment away from the old 10 m segment boundary. Continuous-line termination remains enabled and is persisted in the manifest. The delayed artifact has `on_white_continuous_line=true`, `out_of_road=true`, and `failure_reason="on_white_continuous_line"`.
- **Result:** The failure GIF plots the measured trajectory and actual rectangular vehicle dimensions through the final frame. It makes the centre-versus-footprint distinction inspectable without describing a replay drawing as a camera video.

## Material limits

- This is one fixed, traffic-free, single-lane MetaDrive condition. Changing seed does not create a new road distribution; the unit asks learners to use independent offset and delay experiments for its bounded transfer check.
- The intervention called recovery is a same-condition lower-target-speed mitigation. It is not an automatic recovery branch or a general stability guarantee; delay=12 provides an observed counterexample.
- The controller consumes privileged simulator position, heading, speed, and lane projection. The 259-dimensional MetaDrive observation is recorded for provenance, not represented as the controller input.
- No final artifact reaches the destination in the 18 s horizon, so the unit teaches route tracking and failure mechanisms rather than complete-route success.

## Acceptance follow-up

Preserve the current JSON/CSV/PNG/GIF outputs as the reference evidence for the documented command. Future changes to map geometry, simulator version, action cadence, controller inputs, or comparison conditions should rerun the real-environment tests and update this evidence.
