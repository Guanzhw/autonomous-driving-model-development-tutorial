# Initial Dual Review

- **Date**: 2026-09-09
- **Scope**: L4 route update, Transformer dependency split, project reference, review workflow
- **Decision**: merge the maintenance changes; keep the P1 roadmap items explicitly visible

## Review method

The maintainer ran two separate passes using the role definitions in this directory. The L4 update had previously passed all-20-notebook nbformat, AST and CPU execution. In this maintenance pass, the local evidence checks were:

- all 20 notebooks passed structural nbformat validation and Python AST parsing;
- 18 non-PyTorch notebooks passed a fresh CPU code-cell execution;
- the fresh rerun of `03` and `05` was blocked by the local `torch==2.5.1+cpu` wheel exiting with `SIGBUS` during import, so this environment result is not counted as a new learned-model execution pass;
- README, Notebook Track and HTML contained links to all 20 notebooks;
- `scripts/validate_project.py` passed;
- `git diff --check` and both maintenance scripts passed Python compilation.

Two parallel read-only agent reviews were also scheduled for this pass, but both exceeded the tool execution window before returning a report and were stopped. No finding below is attributed to an agent that did not return; the report is a transparent maintainer fallback and should be rerun by the configured roles on the next content change.

## Reviewer pass

### R-001 — Frontier dependency wording could imply a false requirement

- **Priority**: P1
- **Location**: previous wording in `README.md` and `notebooks/README.md`
- **Finding**: the current `13–14` notebooks use synthetic NumPy interfaces and do not import Hugging Face `transformers`, while the old prose could be read as if the frontier dependency were required to run them.
- **Evidence**: imports in `notebooks/13_vlm_structured_driving_conditions.ipynb` and `notebooks/14_vla_world_action_interface.ipynb`; `requirements-frontier.txt`.
- **Minimal fix**: state that the current notebooks are mechanism teaching and that Frontier dependencies are only for a future real-checkpoint/processor extension.
- **Status**: **fixed** in this change.

### R-002 — Notebook Track should expose direct file links

- **Priority**: P1
- **Location**: `notebooks/README.md`
- **Finding**: a learning track that names 20 notebooks but does not link each filename makes it harder to verify coverage and navigate from GitHub.
- **Evidence**: the route table previously showed labels without `.ipynb` links.
- **Minimal fix**: link every route row to its notebook and let the structural validator check the set.
- **Status**: **fixed** in this change.

### R-003 — Maintenance checks need a repository-level gate

- **Priority**: P1
- **Location**: repository root and `.github/workflows/quality.yml`
- **Finding**: local validation alone is easy to forget when a notebook or route changes.
- **Evidence**: prior validation was run manually; no workflow checked notebook count, syntax, links or review artifacts.
- **Minimal fix**: add `scripts/validate_project.py` and a lightweight GitHub Actions structural check. Keep full learned-model execution as a local/release check because Core CI does not install PyTorch.
- **Status**: **fixed** in this change.

### R-004 — Environment resolution is not fully reproducible

- **Priority**: P1
- **Location**: `requirements-ml.txt`, `requirements-frontier.txt`
- **Finding**: open-ended ranges are appropriate for a landing tutorial but can resolve different PyTorch/CUDA/Transformers wheels over time; default PyTorch installation can also be much heavier than a CPU learner expects.
- **Evidence**: `torch>=2.2,<3` and similarly broad frontier ranges.
- **Minimal fix**: add a tested environment lock or a dated `requirements-lock-cpu.txt`/`requirements-lock-cuda.txt`, plus a platform-specific PyTorch install note, after choosing the supported Python/PyTorch matrix.
- **Status**: **record-and-park**; do not pretend the current files are a frozen reproduction environment.

### R-005 — Notebook cell IDs produce a future-maintenance warning

- **Priority**: P2
- **Location**: all existing `.ipynb` files; `scripts/build_l4_update.py`
- **Finding**: nbformat currently emits `MissingIDFieldWarning`. Future nbformat versions may treat missing cell IDs as a hard validation error.
- **Evidence**: validation warning during local checks.
- **Minimal fix**: normalize notebooks and add deterministic IDs in the generator, then make the migration a separate low-risk commit so the notebook diffs remain reviewable.
- **Status**: **record-and-park**.

## Devil's Advocate pass

### DA-001 — “L4-oriented” can still be mistaken for “L4-capable”

- **Priority**: P1
- **Location**: `README.md`, `index.html`, `notebooks/19_l4_model_development_capstone.ipynb`
- **Challenge**: a recruiter may see “L4 capstone”, collision rate and safety state machine and infer a real autonomous-driving system.
- **Counterexample**: the current capstone uses synthetic episodes and a toy policy; it has no real vehicle dynamics, hardware redundancy, safety case or public runner result.
- **Minimal fix**: keep the explicit “interface/validation skeleton, not certification evidence” language beside every L4 claim and use `toy-mechanism` in the portfolio handoff.
- **Status**: **mitigated** in the current docs; keep as a permanent review question.

### DA-002 — Public resource links are not integrations

- **Priority**: P1
- **Location**: `index.html`, `PROJECT_REFERENCE.md`
- **Challenge**: listing nuScenes, nuPlan, NAVSIM, CARLA, Autoware or SafeBench can look like the project already runs them.
- **Counterexample**: the current repository contains teaching stubs and synthetic data, not adapters, data licenses, fixed splits and result artifacts for all those systems.
- **Minimal fix**: label resources as “next integration target” until an adapter, command, version and result file land in the repository.
- **Status**: **record-and-park**; the gap is now explicit in `PROJECT_REFERENCE.md`.

### DA-003 — Runtime and safety evidence is still illustrative

- **Priority**: P1
- **Location**: `05`, `15`, `18`, `19`
- **Challenge**: p95 latency, degraded mode and collision/fallback metrics on CPU/synthetic data are useful learning artifacts but not vehicle runtime or safety evidence.
- **Counterexample**: there is no fixed hardware profile, warm-up/synchronization policy, C++/CUDA/TensorRT path, watchdog trace, redundancy argument or real scenario regression set.
- **Minimal fix**: add a separate runtime/safety evidence milestone with hardware metadata, batch=1 methodology, failure replay and explicit non-claims.
- **Status**: **record-and-park**; listed as a current project gap.

### DA-004 — Frontier keywords must not crowd out the hiring signal

- **Priority**: P2
- **Location**: `13–14`, `requirements-frontier.txt`, HTML frontier section
- **Challenge**: VLM/VLA/WA/π0 terminology is attractive but can make the project look like a model-name tour rather than model development.
- **Minimal fix**: keep the frontier branch after the L4 mainline, require a structured output contract and closed-loop comparison, and do not count checkpoint loading as completion.
- **Status**: **mitigated** by the T-shaped route and the dependency boundary; revisit whenever frontier content expands.

### DA-005 — The portfolio still needs role-specific proof

- **Priority**: P2
- **Location**: `PROJECT_REFERENCE.md` current gaps
- **Challenge**: one 20-notebook route may demonstrate breadth without showing which job family the learner can actually contribute to.
- **Minimal fix**: create one focused portfolio variant each for perception/fusion, prediction/planning, data/evaluation, and runtime/safety, each with a real public benchmark or runner result.
- **Status**: **record-and-park**.

## Maintainer decision

There is no unresolved P0. The maintenance changes can be merged because they improve the truthfulness and repeatability of the project without pretending that the known P1 gaps are solved. The next milestone should prioritize:

1. a real public-data or public-runner adapter;
2. a reproducible CPU/CUDA environment matrix;
3. runtime/safety evidence with fixed hardware and failure replay;
4. role-specific portfolio outputs.

## Feedback follow-up: learning gradient

### R-006 — The route was topic-complete but not sufficiently graduated

- **Priority**: P1
- **Feedback**: the project assumed deep-learning ability correctly, but still assumed too much autonomous-driving context between its topic notebooks. Compared with HF Course and fast.ai, it introduced domain abstractions without enough orientation, motivation and bridge experiments.
- **Evidence**: the previous route entered ODD/SE(3)/BEV/Transformer as if `ego`, `actor`, `scene`, `sensor frame`, `timestamp`, `agent state`, `trajectory` and `closed-loop` were already familiar.
- **Minimal fix**: integrate the domain primers into the chapters that consume them, reduce the route to 11 core chapters + 3 Advanced Labs, and make each chapter produce/consume a shared urban cut-in artifact.
- **Status**: **fixed in this follow-up**. The learner is still assumed to know deep learning, while the missing autonomous-driving context is taught in-place rather than as six extra gates.

## Structural follow-up: compression and cumulative evidence

### R-007 — Canonical route was over-expanded

- **Priority**: P1
- **Location**: previous `notebooks/00a–00f`, README, Notebook Track and HTML
- **Finding**: adding six domain bridge notebooks solved missing context but created a 26-notebook route with duplicated syllabus maintenance and weak cumulative experiments.
- **Evidence**: previous route had separate toy actors, cubes and random token tasks; `notebooks/README.md`, README and HTML all carried route detail.
- **Minimal fix**: canonicalize `course/` (11) and `labs/` (3), make README the only learner entry, slim HTML, retain `notebooks/README.md` only as a compatibility notice.
- **Status**: **fixed in this follow-up**.

### R-008 — The learned model and capstone needed actual semantic inheritance

- **Priority**: P1
- **Location**: `course/05_learnable_bev_model.ipynb`, `course/10_capstone.ipynb`
- **Finding**: a Transformer demo is not an AD model exercise if its labels are arbitrary token statistics; a capstone is weak if it only compares hand-written policies.
- **Evidence**: Chapter 05 now reads `02_bev_dataset.npz`, predicts occupancy/risk/velocity per BEV cell, saves `05_bev_model.pt`; Chapter 10 loads that checkpoint and runs an inference before summarizing prediction/planning/safety artifacts.
- **Minimal fix**: preserve the shared model class in `src/ad_tutorial/bev_model.py`, keep checkpoint config/metrics, and fail loudly when chain artifacts are missing.
- **Status**: **fixed in this follow-up**, subject to the PyTorch smoke test.

### DA-006 — Real-data checkpoint is still a boundary, not a benchmark result

- **Priority**: P1
- **Location**: Chapters 01, 02 and 08; `requirements-real-data.txt`
- **Challenge**: a nuScenes adapter/checkpoint cell can be mistaken for a completed public-data integration.
- **Evidence**: the cells explicitly skip without `NUSCENES_ROOT`, record required metadata and state that no result is claimed.
- **Minimal fix**: run one fixed nuScenes mini sample, commit the command/output metadata, then upgrade only that artifact to `open-benchmark-result`.
- **Status**: **record-and-park**.

## Implementation follow-up evidence

- `scripts/validate_project.py` passes: 14 canonical notebooks under `course/` + `labs/` have valid nbformat and Python AST; README, course/labs maps and thin HTML contain the expected links.
- A deterministic direct code-cell smoke run passed for 12 notebooks in dependency order, including the shared artifact chain through Chapters 00–04 and 06–09 plus all three labs. This run uses the same Python namespace semantics needed by the cells, with `MPLBACKEND=Agg`.
- Chapter 05 and Chapter 10 were not executed in this environment because PyTorch is not installed here; they remain a required local/release smoke test under `requirements-ml.txt`. No learned-model result is claimed by this report.
- nuScenes checkpoints were exercised only in their “data/dependency absent” branch. No `NUSCENES_ROOT` or public-data result artifact was available, so the project keeps them at `open-benchmark-ready` scaffolding.
- `git diff --check`, generator determinism, Python compilation and the structural validator pass. The Jupyter kernel transport itself was not used for the smoke run because this managed environment terminates the kernel during startup; this is an environment limitation, not evidence that the optional PyTorch path passed.
