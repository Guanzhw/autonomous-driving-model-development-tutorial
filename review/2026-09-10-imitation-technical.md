# Imitation-stage technical review — 2026-09-10

## Scope and evidence

Reviewed `src/ad_tutorial/imitation.py`, `scripts/run_imitation.py`,
`scripts/build_imitation.py`, `tests/test_imitation.py`, and
`course/imitation/` as a read-only technical pass.

The main causal path is correct. Demonstrations reconstruct features from the
pre-action `before_*` state and the reference created at that same decision
([`imitation.py:73-89`](../src/ad_tutorial/imitation.py#L73-L89)); labels are
the geometric expert's contemporaneous commands
([`imitation.py:92-99`](../src/ad_tutorial/imitation.py#L92-L99)). In closed
loop, the reloaded MLP extracts those same features from its current
observation/reference ([`imitation.py:502-510`](../src/ad_tutorial/imitation.py#L502-L510)),
and `driving.run_episode` pushes the resulting action into MetaDrive.

Train-only normalization and validation checkpoint selection are also correctly
implemented ([`imitation.py:346-408`](../src/ad_tutorial/imitation.py#L346-L408));
the test data is not passed to that path. The default split uses disjoint seeds
and lateral offsets ([`imitation.py:561-584`](../src/ad_tutorial/imitation.py#L561-L584)).

Command actually run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_imitation.py -q
```

Result: `4 passed in 3.29s`. I did not rerun the full CLI experiment; its live
result is recorded by the stage owner.

## Findings

### P2 — the exported closed-loop summary is not a paired common-window evaluation

`evaluate_closed_loop()` does run expert, untrained, and BC under the same
ordered test configs ([`imitation.py:587-603`](../src/ad_tutorial/imitation.py#L587-L603)).
But `summarize_closed_loop()` independently averages each policy's full episode
metrics ([`imitation.py:606-630`](../src/ad_tutorial/imitation.py#L606-L630)).
When one policy fails earlier, its lateral error and distance cover a shorter
time window than the other conditions, so those aggregates cannot answer a
same-condition comparison.

The notebook computes per-`test_index` common rows with the correct
`min(len(...))` idea ([`build_imitation.py:172-181`](../scripts/build_imitation.py#L172-L181)),
but this calculation is neither a reusable evaluator nor part of the CLI
`metrics.json`; its rows are easy to accidentally flatten across conditions or
to compare full-episode distance against common-window lateral error. The course
README promises common-window trajectory evidence
([`course/imitation/README.md:25`](../course/imitation/README.md#L25)).

Export one paired record per test configuration. It should identify the config,
use one common step count across all conditions, calculate lateral error and
distance from that exact prefix for each condition, then average those
per-configuration values without weighting longer rollouts more heavily. Keep
the existing full-episode failure/outcome and distance summaries alongside it;
they answer a different question. Add a regression where one condition ends
early, proving no pair borrows the common-window length from another test case.

### P2 — the dataset abstraction does not itself enforce episode-disjoint splits

`DemonstrationDataset.__post_init__()` validates array widths and per-sample
metadata lengths, but never checks that each `episode_id` is associated with
exactly one split ([`imitation.py:125-135`](../src/ad_tutorial/imitation.py#L125-L135)).
`select()` will therefore produce apparently separate train/test datasets from
the same episode if a caller labels different rows from that episode differently
([`imitation.py:145-162`](../src/ad_tutorial/imitation.py#L145-L162)). This is
precisely the timestep-level leakage the lesson says it avoids.

`collect_demonstrations()` prevents an exact duplicate `DrivingConfig` across
splits ([`imitation.py:202-217`](../src/ad_tutorial/imitation.py#L202-L217)),
so the default CLI is protected. The public `DemonstrationDataset` and
`dataset_from_records` boundary remains permissive, and the current test only
asserts disjointness for a manually well-formed fixture
([`test_imitation.py:38-53`](../tests/test_imitation.py#L38-L53)).

Validate that an episode id maps to one split when constructing the dataset,
and add a rejecting test. If callers need multiple partitions of one trajectory
for a different purpose, give that workflow a different explicit type rather
than weakening the demonstration split contract.

## Completion assessment

There is no observed feature/action leakage in the default collection, training,
or reloaded-policy control path, and the focused tests pass. The stage should
not present the current full-episode aggregate as a fair cross-policy comparison
after variable-length failures. Resolve the paired-evaluation finding before
using the closed-loop summary as comparative evidence; enforce episode-disjoint
splits at the dataset boundary to make the advertised anti-leakage guarantee
true for every supported caller.

## Closure recheck — 2026-09-10

The requested fixes resolve both P2 findings.

- `DemonstrationDataset` now rejects any `episode_id` mapped to more than one
  split ([`imitation.py:140-150`](../src/ad_tutorial/imitation.py#L140-L150)),
  with a direct rejecting regression
  ([`test_imitation.py:65-74`](../tests/test_imitation.py#L65-L74)).
- `paired_common_window()` computes a minimum prefix separately for each test
  index, reports prefix lateral error and distance for every policy, and forms
  unweighted per-configuration aggregates
  ([`imitation.py:643-714`](../src/ad_tutorial/imitation.py#L643-L714)). The CLI
  now exports it in `metrics.json`, and the notebook displays both per-config
  and aggregate results ([`run_imitation.py:159-164`](../scripts/run_imitation.py#L159-L164),
  [`build_imitation.py:216-223`](../scripts/build_imitation.py#L216-L223)). The
  early-failure regression verifies that distinct test configurations retain
  distinct common lengths ([`test_imitation.py:153-172`](../tests/test_imitation.py#L153-L172)).

I did not rerun the full CLI in this closure check. The current source resolves
the original leakage and variable-window comparison findings; the imitation
stage is technically approved conditional on the stage owner's final execution
and notebook validation.
