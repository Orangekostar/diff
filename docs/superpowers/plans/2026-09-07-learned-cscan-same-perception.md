# Learned C-scan Same-Perception Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate a compact learned C-scan action policy against strong rule controls under one frozen, observable perception and reporting interface.

**Architecture:** A new `cmc_bbdm.learned_cscan` package adapts the frozen retrospective world without modifying it. A task-free Qwen surface percept and Reader v2 feed an immutable `ObservationPacket`; all rules and learned actors consume that packet. Training uses behavior-cloning and queried deterministic suffix cost-to-go targets, followed by at most one learned-state aggregation round. VALID locks all choices before a proxy-scoped TEST comparison.

**Tech stack:** Python 3.11, NumPy, pandas, PyTorch, Pillow, PyYAML, pyarrow, pytest, Ruff, existing `cmc_bbdm` runtime and Qwen backend.

**Spec:** `artifacts/learned_cscan_same_perception/DESIGN_AND_EVIDENCE.md`

**Execution constraints:** Work only on `research/learned-cscan-same-perception` at base `59a67511c0ee6395b68220c6a1644f40c383dfa2`. Run tests before implementation for every production contract. Do not modify frozen historical source/result roots. TEST may be read only by the final evaluation command. No VLM training, old-model training, architecture search, PR, merge, or force push.

## Task 1: Freeze runtime configuration, roster, and split permissions

**Files:**

- Create: `src/cmc_bbdm/learned_cscan/__init__.py`
- Create: `src/cmc_bbdm/learned_cscan/contracts.py`
- Create: `src/cmc_bbdm/learned_cscan/runtime.py`
- Create: `tests/test_learned_cscan_runtime.py`
- Create: `paper_v3/configs/learned_cscan_same_perception.yaml`

- [x] Write three failing tests:
  - deterministic 24/12/24 split with 4/2/4 specimens per domain and no specimen overlap, plus split permissions that reject fitting on VALID or TEST;
  - `ReferenceStatus.ALGORITHM_DERIVED_NOT_REVIEWED` yields formal success `None` and proxy scope `SAME_READER_SELF_CONSISTENCY`;
  - real surface input is not 1x1 and its declared clockwise rotation occurs exactly once.
- [x] Run `PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_learned_cscan_runtime.py` and confirm import/contract failures.
- [x] Define string enums `Split`, `Task`, `ReferenceStatus`, and frozen dataclasses for specimen identity and reference provenance. Keep identity outside actor features.
- [x] Implement `hash_split(records, seed)` using SHA-256 of `<seed>|<specimen_key>` within each domain. Implement `require_fit_split(split)` and explicit formal/proxy result fields.
- [x] Implement a thin runtime adapter over the frozen VLM inventory/load/render functions. It must preserve the prior 60-specimen roster and expose a one-time rotation marker.
- [x] Add one YAML configuration with base SHA, source root, prior VLM config path, seed, split counts, model revision, threshold 0.18, compute caps, and output root.
- [x] Re-run the test file and Ruff on the new files until green.

## Task 2: Define the action-free surface percept and versioned cache

**Files:**

- Create: `src/cmc_bbdm/learned_cscan/perception.py`
- Create: `tests/test_learned_cscan_perception_readout.py`

- [x] Write two failing perception tests:
  - prompt/schema/parser accept only bounded cell sets, cue type, alternative explanation, ordinal confidence, and `no_reliable_cue`; they reject action fields and contain no filled cell 27, indentation answer, or `m1` token;
  - cache key changes with model revision, image digest, render version, prompt, or schema version and stays task-independent.
- [x] Run only those two tests and confirm missing-module failures.
- [x] Define frozen `SurfaceRegion` and `SurfacePercept` dataclasses, JSON schema, strict parser, neutral prompt, deterministic display-label permutation, and inverse mapping.
- [x] Implement cache read/write around `QwenVLBackend.infer` without modifying the backend. Count new calls, cache hits, parse failures, and single repairs.
- [x] Make the parser return an explicit parse error rather than fabricating a percept. Permit at most one formatting repair per main image in orchestration.
- [x] Re-run tests and Ruff until green.

## Task 3: Implement Reader v2 and the immutable common packet

**Files:**

- Create: `src/cmc_bbdm/learned_cscan/readout.py`
- Create: `src/cmc_bbdm/learned_cscan/observation.py`
- Modify: `tests/test_learned_cscan_perception_readout.py`

- [x] Add two failing tests:
  - an entirely unmeasured cell stays UNKNOWN and cannot be a candidate; measured points remain exact, and singleton/line interpolation stays inside valid in-cell support;
  - identical visible history and percept produce byte-equivalent packet arrays, while changing an unmeasured hidden pixel does not change the packet.
- [x] Run the file and confirm failures identify missing Reader v2/packet behavior.
- [x] Implement TRAIN-only `BackgroundPrior`, observed-only RGB distance, `CellReadout`, `TaskReport`, and 0.18 proxy threshold. Keep candidate, signal estimate, support, validity, spacing, and unverified boundary separate.
- [x] Implement per-cell and 4x4 sub-block observed-only aggregates with explicit counts and missing flags. Preserve exact measured values and masks.
- [x] Implement frozen `ObservationPacket` with tensor conversion. Include recent four actions and public geometry/cost fields; assert that forbidden identity/reference/full-scan keys cannot enter tensor features.
- [x] Re-run tests and Ruff until green.

## Task 4: Implement fair rule planners and relevant-update stopping

**Files:**

- Create: `src/cmc_bbdm/learned_cscan/policies.py`
- Create: `src/cmc_bbdm/learned_cscan/stopping.py`
- Create: `tests/test_learned_cscan_controls.py`

- [x] Write two failing tests:
  - every rule emits a progressive legal primitive with correct unique cost, and `R_BALANCED(period=4)` emits coverage no later than every fourth primitive;
  - two unrelated updates do not increase `S_rule` stability, while two relevant stable updates satisfy the stability component without equating signal confidence to completion probability.
- [x] Run the file and confirm expected failures.
- [x] Implement deterministic `R_GEOM`, `R_CENTER`, `R_VLM_OPEN`, `R_LEGACY`, and `R_BALANCED`, with public-cost tie breaking. Implement `mu` as frozen `R_BALANCED(period=4)`.
- [x] Implement neighboring-ring and candidate-support calculations plus task-specific LOCATE/CHARACTERIZE stop requirements from the design authority.
- [x] Keep STOP separate from action selection. Return structured stop reason and unmet conditions.
- [x] Re-run tests and Ruff until green.

## Task 5: Implement the compact masked actor and real mini-batch learning

**Files:**

- Modify: `src/cmc_bbdm/learned_cscan/policies.py`
- Create: `src/cmc_bbdm/learned_cscan/training.py`
- Create: `tests/test_learned_cscan_training.py`

- [ ] Write two failing tests:
  - masked logits never select an illegal cell and the model has fewer than one million trainable parameters with width 128, at most two encoder layers, and four heads;
  - a fixed separable set of at most 32 states performs one optimizer step per mini-batch, changes parameters, and lowers loss.
- [ ] Run the file and confirm failures.
- [ ] Implement `LearnedCellActor` as a shared 64-cell scorer. Derive each legal cell's next level from the packet/state; do not create 192 action slots.
- [ ] Implement BC cross-entropy and sparse queried-preference cross-entropy. The latter indexes only queried legal actions and assigns no target to unqueried actions.
- [ ] Implement balanced specimen/task mini-batches, AdamW defaults, gradient clipping, step-count logging, 250-step validation cadence, 4-check patience, and 4,000-step cap.
- [ ] Re-run tests and Ruff until green.

## Task 6: Implement exact metrics and counterfactual cost-to-go bank

**Files:**

- Create: `src/cmc_bbdm/learned_cscan/metrics.py`
- Create: `src/cmc_bbdm/learned_cscan/rollouts.py`
- Create: `tests/test_learned_cscan_rollouts_metrics.py`

- [ ] Write three failing tests:
  - exact right-continuous step integral includes the final interval to cost 1 and handles repeated costs and non-monotonic report success;
  - candidate cost-to-go includes all future suffix steps, normalizes by remaining cost, softmaxes only queried candidates, and isolates counterfactual branch state;
  - terminal STOP reveals no later positions and cannot mutate the frozen terminal report; no-stop and stop runners share the same prefix before termination.
- [ ] Run the file and confirm failures.
- [ ] Implement exact `step_integral`, success-at-cost, failure-penalized cost, and specimen/domain paired bootstrap with one common 5,000-resample index table.
- [ ] Implement visible-only candidate proposal: `mu`, geometry, boundary, low-cost refinement, and two fixed-random candidates, deduplicated and capped at six.
- [ ] Implement clean-world suffix rollout, main cost integral, separate 0.05 auxiliary bounded task-loss integral, remaining-cost normalization, temperature 0.10 preferences, and transition counting.
- [ ] Implement at most four base states per TRAIN specimen/task near costs 0, 0.025, 0.10, 0.40, plus at most two learned-visited aggregation states. Assert the global 400,000 transition cap.
- [ ] Implement true-break autonomous execution and separate planner-only full-prefix execution.
- [ ] Re-run tests and Ruff until green.

## Task 7: Add learned-stop calibration and provenance/statistical guards

**Files:**

- Modify: `src/cmc_bbdm/learned_cscan/stopping.py`
- Modify: `src/cmc_bbdm/learned_cscan/training.py`
- Modify: `tests/test_learned_cscan_training.py`
- Modify: `tests/test_learned_cscan_rollouts_metrics.py`

- [ ] Add two failing tests:
  - absent credible VALID support returns `LEARNED_STOP_NOT_AUTHORIZED` and the resource-endpoint fallback is not counted as successful STOP;
  - two tasks and multiple policy seeds for one specimen still contribute one physical specimen to paired bootstrap, and proxy provenance leaves formal effects null.
- [ ] Run the affected tests and confirm failures.
- [ ] Implement a separate visible-input stop head and TRAIN-only fitting. Evaluate thresholds 0.90, 0.95, and 0.99 on VALID with support and false-stop counts.
- [ ] Implement authorization result, one shared threshold across planners, fallback semantics, and formal-null/proxy-only summary guards.
- [ ] Re-run tests and Ruff until green.

## Task 8: Build orchestration, artifact writers, and recoverable CLI

**Files:**

- Create: `src/cmc_bbdm/learned_cscan/benchmark.py`
- Create: `src/cmc_bbdm/learned_cscan/artifacts.py`
- Create: `scripts/run_learned_cscan.py`
- Create: `tests/test_learned_cscan_cli.py`

- [ ] Write one failing CLI integration test that runs a tiny synthetic `prepare -> build-train-bank -> train -> validate -> evaluate -> summarize` flow, verifies TEST cannot be used during fit, checks an actual model state dict, and checks all required artifact names.
- [ ] Run the CLI test and confirm missing command/orchestration failures.
- [ ] Implement idempotent subcommands `prepare`, `perception`, `build-train-bank`, `train`, `validate`, `evaluate --split test`, and `summarize`.
- [ ] Implement compact JSON/JSONL/CSV/Parquet/model writers with atomic replacement. Do not store source images, full foundation weights, or per-step full rasters.
- [ ] Record configuration/model/prompt digests, call counts, cache counts, optimizer steps, state/candidate/transition counts, wall time, GPU memory, selected checkpoint/rule/stop threshold, and explicit non-run reasons.
- [ ] Build `CHECKSUMS.sha256` last, excluding itself, with sorted relative paths.
- [ ] Re-run CLI test, all new tests, and Ruff until green. The suite must contain 12–18 meaningful tests total.

## Task 9: Commit the shared-perception and control implementation

- [ ] Run:

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  tests/test_learned_cscan_runtime.py \
  tests/test_learned_cscan_perception_readout.py \
  tests/test_learned_cscan_controls.py
python -m ruff check src/cmc_bbdm/learned_cscan tests/test_learned_cscan_*.py scripts/run_learned_cscan.py
git diff --check
```

- [ ] Inspect the full diff for forbidden inputs, accidental identity features, legacy source edits, and action-bearing VLM outputs.
- [ ] Commit the applicable runtime/perception/readout/control files as `feat: add common C-scan perception and rule controls`.

## Task 10: Commit the learned planner, cost supervision, and CLI

- [ ] Run all new tests and confirm the exact pass count.
- [ ] Inspect optimizer-loop placement, sparse queried-action targets, branch isolation, terminal break, split guards, parameter count, and transition ceilings.
- [ ] Commit learned actor, training, rollouts, metrics, artifacts, CLI, config, and tests as `feat: train cost-sensitive learned C-scan planner`.

## Task 11: Execute W1-W4 on real data and lock VALID choices

- [ ] Run `prepare` against `/home/ww/paper3/cmc_damage_inference`; inspect 60 rows, split balance, source files, reference provenance, and TRAIN-only prior metadata.
- [ ] Run the six original/permuted TRAIN functional checks plus bounded blank/placeholder checks. Inspect mapped locations and raw/parsed outputs. Use at most one prompt/schema repair only if the recorded check demonstrates collapse.
- [ ] Generate the 60 cached surface percepts once. Confirm calls/cache/repair totals and that outputs contain no actions.
- [ ] Run full-input Reader v2 validation and record whether proxy reports have non-degenerate candidates; set `READOUT_LIMITED` if they do not.
- [ ] Smoke-run all five rules from zero on one specimen per domain; inspect legality, masks, acquisition cost, and report evolution.
- [ ] Build the TRAIN base bank within state/candidate/transition caps and run the at-most-32-state overfit check.
- [ ] Train `L_BC` and `L_CTG_0` with seed 1. Generate at most one learned-state aggregation bank and train `L_CTG_1`.
- [ ] Run VALID to select `L_CTG_0` or `L_CTG_1`, rule coverage period 4 or 8, checkpoint, and learned-stop authorization. Only if action-only VALID signal is positive, run seeds 2 and 3 and `L_NO_VLM`; otherwise record them as not run by preregistered gate.
- [ ] Freeze `validation_selection.json` and its digest before any TEST command.

## Task 12: Execute untouched TEST evaluation and summarize results

- [ ] Run `evaluate --split test` exactly once for the locked rule matrix, `L_BC`, and selected `L_CTG`. Run true `S_rule`; run learned stop only if VALID authorized it.
- [ ] Confirm 24 physical specimens per task, no fit access to TEST, no post-stop reveal, and consistent VLM/perception digests across methods.
- [ ] Run `summarize`; independently recompute exact AUSC and paired bootstrap from saved compact trajectories.
- [ ] Inspect per-domain directions, failures, fixed-cost success, endpoint quality, route diagnostics, stop outcomes, and proxy/formal separation.
- [ ] Replay one frozen representative TEST trajectory per domain without retraining or re-running the full VLM set.
- [ ] Ensure every required result file exists and checksums verify.

## Task 13: Write results, limitations, and handoff

**Files:**

- Create: `artifacts/learned_cscan_same_perception/RESULTS_AND_LIMITATIONS.md`
- Create: `artifacts/learned_cscan_same_perception/CODEX_HANDOFF_LEARNED_CSCAN_SAME_PERCEPTION.md`

- [ ] Report real model parameter counts, optimizer steps, losses, bank sizes, aggregation round, transition counts, VLM call/cache counts, compute, rule selection, and stop authorization.
- [ ] Report LOCATE and CHARACTERIZE separately for planner-only and autonomous phases, with all rules, BC, CTG, domains, paired intervals, failures, and sample denominators.
- [ ] State `formal_effect: null` and `reference: PROXY_ONLY` unless the frozen reference manifest proves otherwise. Never promote self-consistency to damage-detection validity.
- [ ] Document what was not run and why, annotation-resume commands, all-276/LODO future plan, exact replay commands, frozen paths, and source/data/model licensing boundaries.
- [ ] Commit results and handoff as `results: report same-perception learned C-scan pilot`.

## Task 14: Final verification and GitHub synchronization

- [ ] Read and apply `superpowers:verification-before-completion` and `superpowers:finishing-a-development-branch`.
- [ ] Run:

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_learned_cscan_*.py
python -m ruff check src/cmc_bbdm/learned_cscan tests/test_learned_cscan_*.py scripts/run_learned_cscan.py
git diff --check
sha256sum -c results/learned_cscan_same_perception/CHECKSUMS.sha256
git diff --name-only 59a67511c0ee6395b68220c6a1644f40c383dfa2 -- \
  src/cmc_bbdm/mva src/cmc_bbdm/mvd src/cmc_bbdm/mavis \
  src/cmc_bbdm/inspection_agent src/cmc_bbdm/inspection_agent_g1 \
  src/cmc_bbdm/vlm_cscan \
  results/inspection_agent/g0 results/inspection_agent/g1 \
  results/vlm_cscan_efficiency artifacts/vlm_cscan_efficiency
```

- [ ] Confirm the frozen-path diff is empty and inspect `git status --short` for only intentional files.
- [ ] Push without force: `git push -u origin research/learned-cscan-same-perception`.
- [ ] Verify `git rev-parse HEAD`, `git rev-parse @{upstream}`, and `git ls-remote origin refs/heads/research/learned-cscan-same-perception` are identical, then confirm a clean worktree.
- [ ] If network access still fails, retain verified local commits and report the exact failure; do not claim remote synchronization.
