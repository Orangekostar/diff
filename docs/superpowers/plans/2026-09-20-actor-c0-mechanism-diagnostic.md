# Actor C0 Mechanism Diagnostic Implementation Plan

> **Task:** `CAI_ACTOR_C0_MECHANISM_VIS_R2_9e765b04`
>
> **Scope:** Frozen VALID-only diagnostics. No training, Qwen/CNN/OOF forwards,
> TEST access, bootstrap, paper edits, or checkpoint selection.

**Goal:** Build and run a resumable diagnostic package that separates external
C0 gating, fixed-state VLM prior sensitivity, direct versus predictor-mediated
surface sensitivity, and Actor attention for six deterministically selected
VALID specimens, then publish auditable raw arrays, figures, offline HTML, and
Git handoff artifacts.

**Architecture:** A self-locating CLI drives six idempotent stages. `prepare`
validates immutable bindings and selects cases. `replay` reproduces C/N native
episodes and the bounded C-no-C0 intervention while caching physical states.
`diagnose` makes all fixed-state score, prior-zero, gradient, perturbation, and
attention queries once. `render` consumes only cached files. `verify` enforces
Q1-Q8 and resource limits. `publish` commits/pushes existing outputs without
running models.

**Stack:** Python 3, PyTorch, NumPy, Matplotlib/Pillow, stdlib CSV/JSON/HTML,
pytest. All plotting uses the Nature figure Python backend contract.

---

## Task 1: Scope, Context, and Deterministic Preparation

**Files:**
- Create: `docs/cai/actor_c0_diagnostic/ACTOR_C0_DIAGNOSTIC_SCOPE.json`
- Create: `scripts/cai_actor_c0_diagnostic/__init__.py`
- Create: `scripts/cai_actor_c0_diagnostic/context.py`
- Create: `scripts/cai_actor_c0_diagnostic/prepare.py`
- Create: `tests/test_cai_actor_c0_diagnostic_context.py`
- Create: `tests/test_cai_actor_c0_diagnostic_prepare.py`

1. Add failing tests for exact scope identity, safe relative roots, source
   ancestry, forbidden TEST access, atomic stage state, global counters, and
   salted hash-min selection of one VALID case in each remaining domain.
2. Implement the minimal context, hashing, atomic I/O, resource accounting,
   and stage state needed to pass those tests.
3. Implement binding checks for branch/base ancestry, C/N/P_all hashes and
   formats, 161/50/65 split counts without reading TEST specimens, feature/VLM
   inputs, historical episode rows, clean-image source paths, and package hash.
4. Write `diagnostic_lock.json`, `selected_cases.csv`, `model_bindings.json`,
   `resource_usage.json`, and `SOURCE_AND_TASK_BINDINGS.md` before computation.
5. Verify with the two new test files and `prepare --config ...`.

## Task 2: Native Replay, State Identity, and C0 Intervention

**Files:**
- Create: `scripts/cai_actor_c0_diagnostic/models.py`
- Create: `scripts/cai_actor_c0_diagnostic/replay.py`
- Create: `tests/test_cai_actor_c0_diagnostic_replay.py`

1. Add failing tests for C schema-1 and N schema-3 loaders, exact physical
   state hashing (surface, observed content, measured mask, ordered history,
   float64/float32 costs, remaining budget, prediction, legal mask; excluding
   policy prior), snapshot timing, original argmax ties, and only-external-C0
   intervention semantics.
2. Implement frozen C/N/P_all loaders with state-dict immutability snapshots.
3. Implement one authoritative rollout using the repository cost, legality,
   history, predictor, and `vlm_first_action_mask` contracts. Cache t=0,1,2,8,
   T-1 pre-action states and T display-only terminal state.
4. Reproduce six C and six N published episodes, requiring exact actions,
   cost tolerance 1e-12, and prediction tolerance 1e-4 MPa.
5. Run C-no-C0 only when the same-score unrestricted first action differs;
   otherwise reuse the native C path with an explicit deterministic reason.
6. Deduplicate the union of native physical states per case, cap at 54, and
   write trajectories plus `state_manifest.csv` before diagnostics.
7. Verify replay tests and the `replay` stage; audit counters are 12 native and
   no more than 6 intervention episodes.

## Task 3: Scores, Fixed-State Prior, Attribution, Perturbation, Attention

**Files:**
- Create: `scripts/cai_actor_c0_diagnostic/diagnose.py`
- Create: `tests/test_cai_actor_c0_diagnostic_diagnose.py`
- Create: `tests/test_cai_actor_c0_diagnostic_attention.py`

1. Add failing synthetic tests for p_env/p_policy conditional-softmax identity,
   blocked mass, top-k ranks, prior-zero TVD, direct/total/via gradient algebra,
   10% feature perturbation, 65-token 2-layer/4-head attention capture, row
   sums, query self-mass retention, and instrumented/native logit equality.
2. Query both frozen Actors on each common physical state over the same
   environment-legal set; write exactly 64 score rows per model/state.
3. Query C once per fixed state with all four VLM channels zero and the full
   legal mask; compute TVD, top-1 changes, and rank/top-5 overlap without rollout.
4. Compute TRAIN cellwise surface mean once. For own action and differing common
   C action, compute local Gradient x (S-S_ref) for direct and total paths;
   derive `via = total - direct`, save full 64x512 gradients and cell aggregates,
   and validate identical forward logits/predictions at the actual state.
5. At t0 only, perturb top-3 and bottom-3 attribution-magnitude cells 10% toward
   S_ref with prediction fixed; use deterministic cell-id tie breaks.
6. Capture actual post-norm MHA input and reconstruct/save all per-head weights.
   Validate 65x65 rows, row sums, and logits against the uninstrumented Actor;
   derive separately labelled query/action rows and approximate residual rollout.
7. Persist the required audit/result CSV, JSON, and NPZ files while incrementing
   total-task counters before every forward/autograd query.
8. Verify targeted diagnostic/attention tests and the `diagnose` stage.

## Task 4: Cached Rendering and Offline Report

**Files:**
- Create: `scripts/cai_actor_c0_diagnostic/render.py`
- Create: `tests/test_cai_actor_c0_diagnostic_render.py`

1. Add failing tests proving render cannot import/call model loaders, every HTML
   link is repository-relative and exists, probability maps use [0,1], 8x8 maps
   are nearest-neighbour, and panel metadata binds state/array/image hashes.
2. Re-render each source through `render_c_inputs`, require the frozen clean hash,
   and create independent PNGs for groups A-F plus panel composites using shared
   comparison scales and explicit provenance/type labels.
3. Build a static Chinese `index.html` with case/state tabs and client-side
   toggles for heatmap, VLM candidates, selection, and numbering. Use no CDN.
4. Save figure-source CSV/NPZ and render-time panel-alignment records. Produce
   publication-grade PDF/SVG companions for composite QA where useful.
5. Verify render tests, source preflight, panel alignment, PDF glyph/collision
   checks, and manual panel-by-panel PNG inspection.

## Task 5: CLI, Q1-Q8 Verification, and Findings

**Files:**
- Create: `scripts/cai_actor_c0_diagnostic/cli.py`
- Create: `scripts/cai_actor_c0_diagnostic/validate.py`
- Create: `tests/test_cai_actor_c0_diagnostic_cli.py`
- Create: `tests/test_cai_actor_c0_diagnostic_validate.py`

1. Add failing tests for `prepare/replay/diagnose/render/verify/publish/status/all`,
   root self-location, resume signatures, render cache-only behavior, publish
   compute prohibition, required delivery files, immutable roots, and truthful
   `DIAGNOSTICS_COMPLETE` versus `PARTIAL_DIAGNOSTICS` status.
2. Implement the CLI and idempotent stage orchestration. Cap CPU threads at four
   and one CUDA device; record actual interpreters and elapsed compute sessions.
3. Implement Q1-Q8 validation, parameter state hashes before/after, forbidden
   counter zeros, all numeric tolerances, array-to-CSV/figure consistency,
   manifest hashes, and required-file inventory.
4. Generate `diagnostic_manifest.json`, `FINDINGS_ZH.md`, and `VERIFICATION.md`
   using only observed results and bounded classification language.
5. Run all new targeted tests plus the relevant existing W3/C-retrain tests.

## Task 6: Execute, Review, Commit, and Publish

**Files:**
- Create/update only task code, config, plan, task output root, and task artifact
  root named by the scope.

1. Run `all` with the locked config under the validated actor environment. Do
   not rerun completed cases/states; inspect resource counters after each stage.
2. Personally review the complete diff, output inventory, six case selections,
   all state identities, raw arrays, result tables, figures, HTML, conclusions,
   and every prompt requirement. Confirm old results/papers and TEST are untouched.
3. Run final targeted pytest, CLI `verify`, static HTML link audit, figure QA,
   `git diff --check`, and repository status checks.
4. Commit task implementation and computed results as the results commit, push
   normally to the same branch, and verify upstream/remote SHA.
5. Write `CODEX_HANDOFF_ACTOR_C0_DIAGNOSTIC.md` and `GIT_DELIVERY.json` with the
   verified results commit/remote state. Make at most one handoff commit, push,
   and confirm local/upstream/`ls-remote` equality plus tracked samples for the
   handoff, first-action table, HTML, one NPZ, and one PNG.

## Required Commands

```bash
PYTHONPATH=src pytest -q \
  tests/test_cai_actor_c0_diagnostic_context.py \
  tests/test_cai_actor_c0_diagnostic_prepare.py \
  tests/test_cai_actor_c0_diagnostic_replay.py \
  tests/test_cai_actor_c0_diagnostic_diagnose.py \
  tests/test_cai_actor_c0_diagnostic_attention.py \
  tests/test_cai_actor_c0_diagnostic_render.py \
  tests/test_cai_actor_c0_diagnostic_cli.py \
  tests/test_cai_actor_c0_diagnostic_validate.py

python scripts/cai_actor_c0_diagnostic/cli.py all \
  --config docs/cai/actor_c0_diagnostic/ACTOR_C0_DIAGNOSTIC_SCOPE.json

python scripts/cai_actor_c0_diagnostic/cli.py verify \
  --config docs/cai/actor_c0_diagnostic/ACTOR_C0_DIAGNOSTIC_SCOPE.json

git diff --check
git status --short
```
