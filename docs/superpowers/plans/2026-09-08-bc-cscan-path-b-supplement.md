# BC C-scan Path-B Supplement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the bounded BC Path-B supplementary study with BC replicas, two independently trained actor-input ablations, episode-first STOP calibration, retrospective TEST statistics, reference-rescore recovery, true-break evidence, figures, and a verified GitHub handoff.

**Architecture:** Keep the original learned-C-scan implementation and result tree read-only. Add four supplement modules that adapt frozen low-level interfaces, write only to a new result root, and make VALID calibration precede TEST evaluation. Reuse the frozen surface cache and base-BC rows; never invoke CTG bank construction or VLM inference.

**Tech Stack:** Python 3.13, PyTorch, NumPy, Polars, SciPy, Matplotlib, PyYAML, pytest, Ruff, Git.

**Spec:** `artifacts/bc_cscan_path_b_supplement/DESIGN_AND_REUSE.md`

## Global Constraints

- Base commit is exactly `d8b5b090891fc030931c6dc81e3619a80966f739`.
- The source roots `results/learned_cscan_same_perception/` and `artifacts/learned_cscan_same_perception/` are read-only.
- Frozen BC seed 1 and STOP hashes must remain `c95185aec70515f2578d671e520ebe9b57e600e91e70e5f6752567ae2f87fbe5` and `c6af5c383ef4c586a4ac85c7dc9afbcd5bc71dc07558c692fb295d0f5386892d`.
- No Reader, VLM prompt/cache, Actor architecture, task, grid, prior, rule priority, CTG, CAI, RL, or historical model/result changes.
- New Actor updates are at most 16,000; one conditional STOP fit may raise the total to at most 20,000.
- New base episode transitions are at most 180,000; confirmation transitions are separately capped at 60,000.
- Original-60 VLM calls are zero; optional functional calls are not needed by default.
- TRAIN fits models, VALID selects STOP thresholds, and retrospective TEST only evaluates frozen choices.
- Formal effects remain null without attributable independently reviewed references.

---

### Task 1: Supplement Configuration and Source Audit

**Files:**
- Create: `paper_v3/configs/bc_cscan_path_b_supplement.yaml`
- Create: `src/cmc_bbdm/learned_cscan/bc_supplement.py`
- Create: `tests/test_bc_cscan_supplement.py`
- Create: `scripts/run_bc_cscan_supplement.py`
- Generate: `results/bc_cscan_path_b_supplement/inventory.json`
- Generate: `results/bc_cscan_path_b_supplement/source_artifact_manifest.json`
- Generate: `results/bc_cscan_path_b_supplement/cohort_and_reference_coverage.csv`
- Generate: `artifacts/bc_cscan_path_b_supplement/inventory_and_bindings.md`

**Interfaces:**
- Consumes: frozen parent config, manifests, checkpoints, trajectories, reference queue, and source bank.
- Produces: `SupplementConfig`, `load_supplement_config(path, project_root)`, `audit_supplement(config_path, project_root, source_root)`.

- [ ] **Step 1: Write failing root/hash/split tests**

```python
def test_config_separates_frozen_source_and_destination(tmp_path):
    config = load_supplement_config(CONFIG, project_root=ROOT)
    assert config.source_result_root != config.output_root
    assert config.parent_config_sha256 == PARENT_CONFIG_SHA
    assert config.actor_update_cap == 16_000

def test_audit_rejects_changed_frozen_bc(tmp_path):
    changed = tmp_path / "l_bc.pt"
    changed.write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen BC"):
        verify_frozen_file(changed, FROZEN_BC_SHA)
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_bc_cscan_supplement.py`

Expected: collection failure because `bc_supplement` does not exist.

- [ ] **Step 3: Implement strict configuration, audit, and CLI audit route**

Implement a frozen `SupplementConfig` dataclass. Resolve all relative paths beneath `project_root`, reject source/destination overlap, verify base/config/model/bank hashes once, join the 60-member split to the 276-row reference manifest, and write the four audit artifacts atomically. Copy only canonical `base_bc` rows into the ignored destination `_work` cache.

- [ ] **Step 4: Verify GREEN**

Run: `PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_bc_cscan_supplement.py`

Expected: configuration and audit tests pass.

- [ ] **Step 5: Run the real audit and commit**

```bash
PYTHONPATH=src python scripts/run_bc_cscan_supplement.py audit \
  --config paper_v3/configs/bc_cscan_path_b_supplement.yaml \
  --source-root /home/ww/paper3/cmc_damage_inference
git add paper_v3/configs/bc_cscan_path_b_supplement.yaml \
  src/cmc_bbdm/learned_cscan/bc_supplement.py \
  scripts/run_bc_cscan_supplement.py tests/test_bc_cscan_supplement.py \
  results/bc_cscan_path_b_supplement \
  artifacts/bc_cscan_path_b_supplement/inventory_and_bindings.md
git commit -m "audit: freeze BC supplement evidence"
```

### Task 2: Actor-Input and Reviewed-Report Adapters

**Files:**
- Create: `src/cmc_bbdm/learned_cscan/supplement_adapters.py`
- Modify: `tests/test_bc_cscan_supplement.py`

**Interfaces:**
- Consumes: `ObservationPacket`, `PolicyTrainingExample`, `LearnedCellActor`, `TaskReportV2`.
- Produces: `ActorInputMode`, `mask_actor_tensors`, `transform_policy_example`, `SupplementActorView.score_cells`, `adapt_task_report_v2`, `save_supplement_actor`, `load_supplement_actor`.

- [ ] **Step 1: Write failing immutable-mask and reference-adapter tests**

```python
def test_no_us_mask_preserves_geometry_and_removes_content():
    masked = mask_actor_tensors(tensors, ActorInputMode.NO_US_FEEDBACK)
    np.testing.assert_array_equal(masked["cell_features"][:, :4], tensors["cell_features"][:, :4])
    assert not masked["cell_features"][:, 4:15].any()
    assert not masked["subblock_features"][:, :, 2:10].any()
    assert masked["global_features"][7] == 0.0
    assert tensors["cell_features"][0, 4] != 0.0

def test_review_adapter_preserves_mask_support_and_formal_scope():
    adapted = adapt_task_report_v2(report)
    np.testing.assert_array_equal(adapted.predicted_mask, report.predicted_mask)
    np.testing.assert_array_equal(adapted.support_positions, report.support_positions)
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_bc_cscan_supplement.py -k 'mask or adapter'`

Expected: import failure because `supplement_adapters` does not exist.

- [ ] **Step 3: Implement minimal immutable transforms and checkpoint metadata**

Use copied float32 arrays. NO_VLM zeros cell 15:17 and global 6. NO_US_FEEDBACK zeros cell 4:15, subblock 2:10, and global 7. `SupplementActorView` applies the same transform before `LearnedCellActor.forward`. Checkpoints record method, seed, parent config SHA, actor input mode, parameter count, and state dict.

- [ ] **Step 4: Verify GREEN and commit**

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_bc_cscan_supplement.py
git add src/cmc_bbdm/learned_cscan/supplement_adapters.py tests/test_bc_cscan_supplement.py
git commit -m "exp: add frozen BC input adapters"
```

### Task 3: Episode-First STOP and Paired Statistics

**Files:**
- Create: `src/cmc_bbdm/learned_cscan/episode_stop_calibration.py`
- Create: `src/cmc_bbdm/learned_cscan/supplement_analysis.py`
- Modify: `tests/test_bc_cscan_supplement.py`

**Interfaces:**
- Consumes: ordered visible trajectory rows and physical specimen metric rows.
- Produces: `first_stop_outcome`, `summarize_episode_risk`, `calibrate_episode_stop`, `paired_domain_bootstrap`, `cost_at_success_rate`.

- [ ] **Step 1: Write failing first-stop, exhaustion, denominator, non-monotone, and paired-bootstrap tests**

```python
def test_first_false_stop_is_not_replaced_by_later_success():
    outcome = first_stop_outcome(rows, probability_field="p", threshold=0.90)
    assert outcome.false_stop and not outcome.completed
    assert outcome.stop_cost == 0.20

def test_never_stopping_is_exhaustion():
    outcome = first_stop_outcome(rows, probability_field="p", threshold=0.99)
    assert outcome.exhausted and outcome.failure_penalized_cost == 1.0

def test_wrong_among_stops_uses_actual_stops():
    risk = summarize_episode_risk(outcomes)
    assert risk.wrong_among_stops == 1 / 2
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_bc_cscan_supplement.py -k 'stop or bootstrap or cost'`

Expected: imports fail because calibration/analysis modules do not exist.

- [ ] **Step 3: Implement fixed-threshold calibration and reusable bootstrap draws**

Scan thresholds `(0.90, 0.95, 0.99)` only. Require each planner to have wrong-among-stops <= 0.05, completion >= 0.50, completion no more than 0.05 below its S_RULE result, and at least six stops. Choose the qualifying threshold with minimum equal-planner failure cost, breaking ties upward. Bootstrap specimen effects within each of six domains with one fixed 5,000-draw index set and support both 95% and 97.5% intervals.

- [ ] **Step 4: Verify GREEN and commit**

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_bc_cscan_supplement.py
git add src/cmc_bbdm/learned_cscan/episode_stop_calibration.py \
  src/cmc_bbdm/learned_cscan/supplement_analysis.py \
  tests/test_bc_cscan_supplement.py
git commit -m "exp: add episode-level STOP calibration"
```

### Task 4: Replica and Ablation Training

**Files:**
- Modify: `src/cmc_bbdm/learned_cscan/bc_supplement.py`
- Modify: `scripts/run_bc_cscan_supplement.py`
- Modify: `tests/test_bc_cscan_supplement.py`
- Generate: `results/bc_cscan_path_b_supplement/models/*.pt`
- Generate: `results/bc_cscan_path_b_supplement/model_manifest.json`
- Generate: `results/bc_cscan_path_b_supplement/training_log.csv`

**Interfaces:**
- Consumes: canonical 192-row base-BC cache and immutable actor modes.
- Produces: `train_replicas` for FULL seeds 2/3 and `train_ablations` for NO_VLM/NO_US_FEEDBACK seed 1.

- [ ] **Step 1: Add a failing output-identity test**

```python
def test_supplement_checkpoint_paths_cannot_overwrite_frozen_models(config):
    for path in planned_checkpoint_paths(config):
        assert config.output_root in path.parents
        assert config.source_result_root not in path.parents
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_bc_cscan_supplement.py -k checkpoint`

Expected: failure because checkpoint planning is not implemented.

- [ ] **Step 3: Implement the two recoverable training stages**

Convert base rows to `PolicyTrainingExample`, preserve the exact 144/48 internal split, and call `fit_actor(..., route=BEHAVIOR_CLONING)` with AdamW 3e-4, weight decay 1e-4, gradient clip 1.0, at most 4,000 steps, 250-step validation, and patience four. Save seed 2/3 and the two seed-1 ablations separately; append real training logs and enforce the 16,000-update cap.

- [ ] **Step 4: Run focused tests and real training**

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_bc_cscan_supplement.py
PYTHONPATH=src python scripts/run_bc_cscan_supplement.py train-replicas --config paper_v3/configs/bc_cscan_path_b_supplement.yaml --source-root /home/ww/paper3/cmc_damage_inference
PYTHONPATH=src python scripts/run_bc_cscan_supplement.py train-ablations --config paper_v3/configs/bc_cscan_path_b_supplement.yaml --source-root /home/ww/paper3/cmc_damage_inference
```

- [ ] **Step 5: Verify model hashes, steps, losses, and commit**

```bash
python -m ruff check src/cmc_bbdm/learned_cscan/supplement_adapters.py src/cmc_bbdm/learned_cscan/bc_supplement.py scripts/run_bc_cscan_supplement.py tests/test_bc_cscan_supplement.py
git add src/cmc_bbdm/learned_cscan/bc_supplement.py scripts/run_bc_cscan_supplement.py tests/test_bc_cscan_supplement.py results/bc_cscan_path_b_supplement
git commit -m "exp: train BC replicas and perception ablations"
```

### Task 5: VALID Calibration, Optional STOP Refit, and TEST Evaluation

**Files:**
- Modify: `src/cmc_bbdm/learned_cscan/bc_supplement.py`
- Modify: `scripts/run_bc_cscan_supplement.py`
- Modify: `tests/test_bc_cscan_supplement.py`
- Generate: `results/bc_cscan_path_b_supplement/validation_episode_scores.parquet`
- Generate: `results/bc_cscan_path_b_supplement/stop_calibration.json`
- Generate: `results/bc_cscan_path_b_supplement/per_episode_metrics.csv`
- Generate: `results/bc_cscan_path_b_supplement/trajectories.parquet`
- Generate: `results/bc_cscan_path_b_supplement/report_manifest.json`

**Interfaces:**
- Consumes: frozen and supplement actors, frozen STOP, optional S_BC_CAL, common runtime/Reader/percepts.
- Produces: `calibrate_stop`, `evaluate_supplement`, `run_true_break_validation`.

- [ ] **Step 1: Write failing true-break and split-authorization tests**

```python
def test_true_break_performs_no_step_after_stop():
    result = run_true_break(fake_world, packets, stop_at=1)
    assert fake_world.step_count == 1
    assert result.terminal_step == 1

def test_test_split_cannot_enter_fit_or_calibration():
    with pytest.raises(ValueError, match="VALID"):
        require_calibration_split(Split.TEST)
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_bc_cscan_supplement.py -k 'true_break or calibration_split'`

Expected: failure because split/true-break functions are missing.

- [ ] **Step 3: Implement one-pass trajectory collection and optional STOP fit**

Generate ordered VALID trajectories only for BC seed 1 and R_BALANCED_P8. Calibrate the frozen head. If either task has no qualifying threshold, collect at most 16 fixed-quantile natural TRAIN states per specimen/task/planner, fit one unchanged `LearnedStopHead` with the 18/6 split, and calibrate once more. Persist all failed candidates and lock task-specific head/threshold identities before TEST.

- [ ] **Step 4: Run VALID calibration**

```bash
PYTHONPATH=src python scripts/run_bc_cscan_supplement.py calibrate-stop --config paper_v3/configs/bc_cscan_path_b_supplement.yaml --source-root /home/ww/paper3/cmc_damage_inference
```

- [ ] **Step 5: Implement and run the retrospective TEST matrix**

Evaluate R_BALANCED_P4/P8, BC seeds 1/2/3, BC_NO_VLM seed 1, and BC_NO_US_FEEDBACK seed 1 once per specimen/task. Record planner-only steps, S_RULE, S_OLD_090, and only qualified episode-calibrated STOP outcomes. Reuse existing rows for other historical rules in analysis. Generate six-domain deterministic true-break cases and require terminal report/cost/action identity with the matching cached prefix.

```bash
PYTHONPATH=src python scripts/run_bc_cscan_supplement.py evaluate --split test --config paper_v3/configs/bc_cscan_path_b_supplement.yaml --source-root /home/ww/paper3/cmc_damage_inference
```

- [ ] **Step 6: Verify caps and commit**

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_bc_cscan_supplement.py
git add src/cmc_bbdm/learned_cscan/bc_supplement.py scripts/run_bc_cscan_supplement.py tests/test_bc_cscan_supplement.py results/bc_cscan_path_b_supplement
git commit -m "results: evaluate BC Path-B stopping matrix"
```

### Task 6: Reference Recovery and Confirmation Queue

**Files:**
- Modify: `src/cmc_bbdm/learned_cscan/bc_supplement.py`
- Modify: `scripts/run_bc_cscan_supplement.py`
- Generate: `results/bc_cscan_path_b_supplement/annotation_queue.csv`
- Generate: `results/bc_cscan_path_b_supplement/confirmation_roster.csv`
- Generate: `results/bc_cscan_path_b_supplement/reviewed_rescore.csv`

**Interfaces:**
- Consumes: attributable reference JSON payloads and trajectory recovery indices.
- Produces: `import_references`, `rescore_reports`, `prepare_confirmation`.

- [ ] **Step 1: Implement strict reference import and deterministic confirmation hashing**

Accept only `EXPERT_REVIEWED` or `AUTHOR_PROVIDED` payloads with `review_state=reviewed`, a reviewer alias, matching specimen key, matching registered-C-scan hash, and matching native shape. Select four non-pilot specimens per domain by fixed SHA-256 ordering without opening outcomes.

- [ ] **Step 2: Run available-reference import, rescore status, and confirmation preparation**

```bash
PYTHONPATH=src python scripts/run_bc_cscan_supplement.py import-references --config paper_v3/configs/bc_cscan_path_b_supplement.yaml --source-root /home/ww/paper3/cmc_damage_inference
PYTHONPATH=src python scripts/run_bc_cscan_supplement.py rescore --references results/vlm_cscan_efficiency/annotation_queue --config paper_v3/configs/bc_cscan_path_b_supplement.yaml --source-root /home/ww/paper3/cmc_damage_inference
PYTHONPATH=src python scripts/run_bc_cscan_supplement.py prepare-confirm --config paper_v3/configs/bc_cscan_path_b_supplement.yaml --source-root /home/ww/paper3/cmc_damage_inference
```

Expected current status: zero attributable references, formal effects null, and `ADDITIONAL_CONFIRMATION_PENDING_REFERENCE`; no confirmation VLM or world rollout.

- [ ] **Step 3: Commit recovery artifacts**

```bash
git add src/cmc_bbdm/learned_cscan/bc_supplement.py scripts/run_bc_cscan_supplement.py results/bc_cscan_path_b_supplement
git commit -m "exp: add reviewed rescoring and confirmation queue"
```

### Task 7: Statistical Tables, Figures, Summary, and Handoff

**Files:**
- Modify: `src/cmc_bbdm/learned_cscan/supplement_analysis.py`
- Modify: `scripts/run_bc_cscan_supplement.py`
- Generate: `results/bc_cscan_path_b_supplement/paired_effects.csv`
- Generate: `results/bc_cscan_path_b_supplement/per_domain_effects.csv`
- Generate: `results/bc_cscan_path_b_supplement/ablation_effects.csv`
- Generate: `results/bc_cscan_path_b_supplement/risk_coverage.csv`
- Generate: `results/bc_cscan_path_b_supplement/figures/`
- Generate: `results/bc_cscan_path_b_supplement/summary.json`
- Generate: `results/bc_cscan_path_b_supplement/CHECKSUMS.sha256`
- Generate: `results/bc_cscan_path_b_supplement/reproduce.md`
- Create: `artifacts/bc_cscan_path_b_supplement/RESULTS_AND_CLAIM_BOUNDARIES.md`
- Create: `artifacts/bc_cscan_path_b_supplement/CODEX_HANDOFF_BC_CSCAN_PATH_B_SUPPLEMENT.md`

**Interfaces:**
- Consumes: all frozen and supplement metrics, calibration lock, reference coverage, model logs, and resource ledger.
- Produces: `analyze_existing`, `summarize_supplement`, four evidence-bound figures, and complete handoff artifacts.

- [ ] **Step 1: Run direct existing analysis before new summary**

```bash
PYTHONPATH=src python scripts/run_bc_cscan_supplement.py analyze-existing --config paper_v3/configs/bc_cscan_path_b_supplement.yaml --source-root /home/ww/paper3/cmc_damage_inference
```

- [ ] **Step 2: Implement result assembly and visual contract**

Produce a two-panel success-rate-versus-cost figure, a Path-B completion/risk/cost figure, a domain-effect figure, and an actor-input ablation figure. Use a color-vision-safe palette, identical task axes where comparison requires it, explicit proxy/reviewed labels, vector PDF plus 300-dpi PNG, and source CSV linkage in figure metadata.

- [ ] **Step 3: Run summary and inspect rendered figures**

```bash
PYTHONPATH=src python scripts/run_bc_cscan_supplement.py summarize --config paper_v3/configs/bc_cscan_path_b_supplement.yaml --source-root /home/ww/paper3/cmc_damage_inference
```

Open each PNG, verify nonblank pixels, legibility, no clipping/overlap, and exact agreement with source tables. State Path-B support separately for LOCATE and CHARACTERIZE; keep reviewed effects null at zero coverage.

- [ ] **Step 4: Commit results and documentation**

```bash
git add src/cmc_bbdm/learned_cscan/supplement_analysis.py scripts/run_bc_cscan_supplement.py results/bc_cscan_path_b_supplement artifacts/bc_cscan_path_b_supplement
git commit -m "docs: add BC Path-B results and handoff"
```

### Task 8: Completion Audit and GitHub Synchronization

**Files:**
- Verify all files from Tasks 1-7; do not modify frozen tracked sources.

**Interfaces:**
- Consumes: the completed branch.
- Produces: clean local/upstream/remote identity and final response evidence.

- [ ] **Step 1: Run final focused verification**

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_bc_cscan_supplement.py tests/test_learned_cscan*.py
python -m ruff check src/cmc_bbdm/learned_cscan/bc_supplement.py src/cmc_bbdm/learned_cscan/supplement_adapters.py src/cmc_bbdm/learned_cscan/episode_stop_calibration.py src/cmc_bbdm/learned_cscan/supplement_analysis.py scripts/run_bc_cscan_supplement.py tests/test_bc_cscan_supplement.py
git diff --check
sha256sum -c results/bc_cscan_path_b_supplement/CHECKSUMS.sha256
```

- [ ] **Step 2: Prove frozen tracked files are unchanged**

```bash
git diff --name-status --diff-filter=MDR d8b5b090891fc030931c6dc81e3619a80966f739 -- src/cmc_bbdm/learned_cscan src/cmc_bbdm/vlm_cscan src/cmc_bbdm/inspection_agent src/cmc_bbdm/inspection_agent_g1 src/cmc_bbdm/mva src/cmc_bbdm/mvd src/cmc_bbdm/mavis results/learned_cscan_same_perception artifacts/learned_cscan_same_perception
```

Expected: empty output. New supplement files are additions and are listed separately.

- [ ] **Step 3: Push without PR, merge, or force**

```bash
git push -u origin research/bc-cscan-path-b-supplement
git rev-parse HEAD
git rev-parse '@{upstream}'
git ls-remote origin refs/heads/research/bc-cscan-path-b-supplement
git status --short
```

Expected: local, upstream, and remote SHAs match; worktree is clean.
