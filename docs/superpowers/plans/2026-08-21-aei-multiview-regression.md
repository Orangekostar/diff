# AEI Multi-View Regression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute the gated E1--E3 mechanics-consistent multi-view CAI experiment, with E4/E5 created only when their registered evidence gates pass.

**Architecture:** Reuse the immutable A2 three-view feature bank and exact P1 fold-local estimator, then evaluate independent experts, joint agreement regression, and leakage-safe fusion under nested domain holdouts. Publish each completed stage as an atomic hash-bound artifact package and preserve A0--A5 unchanged.

**Tech Stack:** Python 3.13, NumPy, SciPy, scikit-learn, PyYAML, pytest, existing `cmc_bbdm` data and artifact authorities.

---

The directory is not a Git repository. Commit checkpoints in the generic skill
workflow are replaced by diff inspection plus focused test checkpoints.

### Task 1: Freeze Protocol and Reference Audit

**Files:**
- Create: `paper_v3/configs/aei_multiview_regression.yaml`
- Create: `docs/AEI_MULTIVIEW_SCIENTIFIC_PROTOCOL.md`
- Create: `docs/MULTIVIEW_REFERENCE_METHOD_AUDIT.md`
- Create: `docs/E1_CROSS_VIEW_AUDIT_PROTOCOL.md`
- Create: `docs/AEI_MULTIVIEW_CLAIM_EVIDENCE_MATRIX.md`
- Create: `tests/test_multiview_protocol.py`

- [ ] **Step 1: Write the failing protocol test**

```python
def test_multiview_protocol_freezes_views_and_stage_order():
    protocol = load_protocol(CONFIG, project_root=ROOT)
    assert protocol.views == ("FULL", "BILINEAR_50", "BILINEAR_25")
    assert protocol.stage_order == ("E1", "E2", "E3", "E4", "E5")
    assert protocol.baseline_mae == 0.08963580465761432
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest -q tests/test_multiview_protocol.py`

Expected: import or file-not-found failure because the new protocol loader and
configuration do not exist.

- [ ] **Step 3: Implement the exact protocol contract**

Define immutable fields for source paths and SHA-256 values, cohort and view
order, PCA/Ridge registry, consistency and complementarity grids, bootstrap
seed/resamples, stage gates, and output roots. Reject changed rosters, source
digests, unsupported views, and reordered stages.

```python
@dataclass(frozen=True, slots=True)
class MultiViewProtocol:
    views: tuple[str, str, str]
    stage_order: tuple[str, str, str, str, str]
    baseline_mae: float
    pca_dimensions: tuple[int, int, int]
    consistency_grid: tuple[float, ...]
```

- [ ] **Step 4: Write the four required pre-E2 documents**

Record claim boundaries, split semantics, exact reference mechanisms, registered
metrics, stage-specific GO/NO-GO rules, and the no-fabrication rule. Cite only
primary paper, proceedings, publisher, OpenReview, and official repository pages.

- [ ] **Step 5: Verify protocol and inspect changes**

Run: `python -m pytest -q tests/test_multiview_protocol.py`

Expected: all protocol tests pass and `rg -n 'TBD|TODO'` finds no placeholders
in the five new files.

### Task 2: Exact Independent View Experts

**Files:**
- Create: `src/cmc_bbdm/aei_multiview_regression/__init__.py`
- Create: `src/cmc_bbdm/aei_multiview_regression/protocol.py`
- Create: `src/cmc_bbdm/aei_multiview_regression/view_experts.py`
- Create: `src/cmc_bbdm/aei_multiview_regression/oof_predictions.py`
- Create: `tests/test_multiview_view_experts.py`
- Create: `tests/test_multiview_oof_predictions.py`

- [ ] **Step 1: Write failing baseline-equivalence and leakage tests**

```python
def test_full_expert_reproduces_frozen_p1_predictions(authoritative_inputs):
    result = evaluate_independent_views(authoritative_inputs)
    assert np.max(np.abs(result.predictions[:, 0] - FROZEN_FULL)) <= 1e-12

def test_fold_fit_never_contains_query_specimen(synthetic_inputs):
    events = []
    evaluate_independent_views(synthetic_inputs, fit_hook=events.append)
    assert all(set(event.fit_ids).isdisjoint(event.query_ids) for event in events)
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest -q tests/test_multiview_view_experts.py tests/test_multiview_oof_predictions.py`

Expected: import failure for the new package.

- [ ] **Step 3: Implement fold-local expert primitives**

```python
def fit_view_expert(
    embeddings: np.ndarray,
    metadata: np.ndarray,
    targets: np.ndarray,
    fit_indices: np.ndarray,
    *,
    pca_dimension: int,
    alpha: float = 10.0,
) -> FittedViewExpert: ...

def select_view_dimension(
    embeddings: np.ndarray,
    metadata: np.ndarray,
    targets: np.ndarray,
    domains: np.ndarray,
    source_indices: np.ndarray,
) -> ViewSelection: ...
```

Use float64 SVD on training embeddings, canonical component signs, metadata plus
PCA scores, training-only mean imputation and scaling, and the exact centered
Ridge solve used by P1. Cache one SVD per fold for all three dimensions.

- [ ] **Step 4: Implement strict outer OOF orchestration**

Return immutable `(n, 3)` predictions, selections, source-inner OOF predictions,
fit/query identities, and source-state digests. Assert exactly one outer
prediction per specimen and never flatten views into independent samples.

- [ ] **Step 5: Verify exact replay and isolation**

Run: `python -m pytest -q tests/test_multiview_view_experts.py tests/test_multiview_oof_predictions.py`

Expected: FULL maximum difference at most `1e-12`; all leakage and immutability
tests pass.

### Task 3: E1 Audit and Reliability

**Files:**
- Create: `src/cmc_bbdm/aei_multiview_regression/agreement_audit.py`
- Create: `src/cmc_bbdm/aei_multiview_regression/complementarity.py`
- Create: `src/cmc_bbdm/aei_multiview_regression/reliability.py`
- Create: `tests/test_multiview_agreement_audit.py`
- Create: `tests/test_multiview_reliability.py`

- [ ] **Step 1: Write failing metric tests with hand-computed arrays**

```python
def test_oracle_and_best_view_are_diagnostic():
    audit = audit_predictions(Y, PREDICTIONS, DOMAINS)
    assert audit.oracle_mae == pytest.approx(EXPECTED_ORACLE)
    assert audit.best_view_counts == (2, 1, 1)
    assert "oracle" not in audit.deployable_methods
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest -q tests/test_multiview_agreement_audit.py tests/test_multiview_reliability.py`

Expected: missing-module failure.

- [ ] **Step 3: Implement E1 metrics and GO/NO-GO**

Compute equal-domain and per-domain MAE, worst-domain MAE, domain SD, pooled
RMSE/R2, prediction Pearson correlations, residual correlations, mean absolute
disagreement, oracle MAE, best-view ties in registered view order, and grouped
frequencies by domain, ply count, layup, and registered damage descriptors.

- [ ] **Step 4: Implement reliability analysis**

```python
dispersion = np.std(predictions, axis=1, ddof=0)
absolute_error = np.abs(targets - deployable_prediction)
```

Report Pearson and Spearman correlation plus low-25%, middle-50%, and high-25%
error strata using deterministic quantile ranks.

- [ ] **Step 5: Verify metrics**

Run: `python -m pytest -q tests/test_multiview_agreement_audit.py tests/test_multiview_reliability.py`

Expected: all hand-computed values and gate cases pass.

### Task 4: Cooperative Multi-View Regression

**Files:**
- Create: `src/cmc_bbdm/aei_multiview_regression/cooperative_regression.py`
- Create: `src/cmc_bbdm/aei_multiview_regression/search.py`
- Create: `tests/test_multiview_cooperative_regression.py`
- Create: `tests/test_multiview_search.py`

- [ ] **Step 1: Write failing objective and collapse tests**

```python
def test_lambda_zero_matches_independent_ridge(designs, targets):
    fit = fit_cooperative(designs, targets, lambda_consistency=0.0, loss="mse")
    np.testing.assert_allclose(fit.train_predictions, INDEPENDENT, atol=1e-12)

def test_large_lambda_reduces_disagreement(designs, targets):
    low = fit_cooperative(designs, targets, lambda_consistency=0.0, loss="mse")
    high = fit_cooperative(designs, targets, lambda_consistency=1.0, loss="mse")
    assert pairwise_disagreement(high.train_predictions) < pairwise_disagreement(low.train_predictions)
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest -q tests/test_multiview_cooperative_regression.py tests/test_multiview_search.py`

Expected: missing implementation.

- [ ] **Step 3: Implement quadratic and Huber peer fits**

For MSE, build target rows for each expert, pairwise zero-target agreement rows
weighted by `sqrt(lambda_consistency)`, and coefficient-penalty rows weighted by
`sqrt(10)`. Keep three unpenalized intercepts. For Huber, minimize the identical
objective with `scipy.optimize.minimize(method="L-BFGS-B")` and an analytic
gradient, initializing from the MSE solution and failing closed on nonconvergence.

- [ ] **Step 4: Implement source-only consistency search**

For each outer domain, retain independently selected view dimensions, evaluate
all loss/consistency candidates on source-domain OOF folds, rank by ensemble
equal-domain MAE, worst-domain MAE, domain SD, loss order, and lambda order, then
refit once on all sources.

- [ ] **Step 5: Verify solver and selection**

Run: `python -m pytest -q tests/test_multiview_cooperative_regression.py tests/test_multiview_search.py`

Expected: lambda zero equivalence, agreement monotonicity, deterministic ties,
and query exclusion all pass.

### Task 5: Leakage-Safe E3 Fusion

**Files:**
- Create: `src/cmc_bbdm/aei_multiview_regression/late_fusion.py`
- Create: `src/cmc_bbdm/aei_multiview_regression/stacking.py`
- Create: `src/cmc_bbdm/aei_multiview_regression/gmvr_regression.py`
- Create: `tests/test_multiview_late_fusion.py`
- Create: `tests/test_multiview_stacking.py`
- Create: `tests/test_multiview_gmvr.py`

- [ ] **Step 1: Write failing simplex and meta-leakage tests**

```python
def test_validation_weights_are_nonnegative_simplex():
    fit = fit_validation_weights(SOURCE_OOF, Y_SOURCE)
    assert np.all(fit.weights >= 0.0)
    assert fit.weights.sum() == pytest.approx(1.0, abs=1e-12)

def test_meta_training_uses_only_base_oof_predictions(events):
    assert all(event.base_prediction_role == "source_oof" for event in events)
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest -q tests/test_multiview_late_fusion.py tests/test_multiview_stacking.py tests/test_multiview_gmvr.py`

Expected: missing implementation.

- [ ] **Step 3: Implement equal and validation-weighted fusion**

Solve MAE simplex weighting as a linear program with three weights and one
absolute-residual slack per specimen. Validate solver success, constraint
residuals, and finite outputs.

- [ ] **Step 4: Implement stacking with domain-held-out meta-selection**

Candidates are Ridge, non-negative Ridge, and Huber. Each candidate is evaluated
by holding out one source domain from the meta fit; after selection, refit on all
strict source OOF base predictions and apply once to outer base predictions.

- [ ] **Step 5: Implement lightweight GMvR**

Search the cooperative consistency grid and weight regularization
`[0, 1e-3, 1e-2, 0.1, 1.0]`. Fit non-negative simplex weights by minimizing
source OOF squared error plus weight concentration. Record per-view contribution
variance and reject a collapsed solution unless its MAE improves.

- [ ] **Step 6: Verify fusion behavior**

Run: `python -m pytest -q tests/test_multiview_late_fusion.py tests/test_multiview_stacking.py tests/test_multiview_gmvr.py`

Expected: constraints, meta-fold separation, deterministic ranking, and collapse
diagnostics pass.

### Task 6: Formal Outer Evaluation, Statistics, and Stress Tests

**Files:**
- Create: `src/cmc_bbdm/aei_multiview_regression/formal_outer.py`
- Create: `src/cmc_bbdm/aei_multiview_regression/statistics.py`
- Create: `tests/test_multiview_formal_outer.py`
- Create: `tests/test_multiview_statistics.py`

- [ ] **Step 1: Write failing gate-order and bootstrap tests**

```python
def test_e4_is_not_called_when_e3_gate_fails(monkeypatch, inputs):
    monkeypatch.setattr(module, "run_e4", forbidden)
    result = run_formal_chain(inputs)
    assert result.e4_status == "NO_GO"

def test_common_bootstrap_shape():
    result = common_domain_bootstrap(EFFECTS)
    assert result.indices.shape == (100_000, 6)
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest -q tests/test_multiview_formal_outer.py tests/test_multiview_statistics.py`

Expected: missing implementation.

- [ ] **Step 3: Implement E1--E3 ordered execution and gates**

Run E1, evaluate its gate, run E2 only on GO, then E3. E4 requires best fusion
below both `0.08963580465761432` and the best individual view with at least four
improved domains. E5 remains a recorded NO-GO unless all prompt conditions hold.

- [ ] **Step 4: Implement common four-effect bootstrap**

Use one `PCG64(20260811)` index matrix of shape `(100000, 6)`, ordinary quantiles
`[0.025, 0.975]`, and family-wise quantiles `[0.00625, 0.99375]` for four frozen
FULL-minus-candidate effect vectors.

- [ ] **Step 5: Implement engineering stress splits**

Evaluate frozen method families with outer groups `ply_count in {8,16,24}` and
`layup in {cross_ply, quasi_isotropic}`. All selection remains inner source-domain
LODO; no held-out group statistics enter preprocessing or weights.

- [ ] **Step 6: Verify orchestration**

Run: `python -m pytest -q tests/test_multiview_formal_outer.py tests/test_multiview_statistics.py`

Expected: branch order, gate outcomes, bootstrap indices, and stress split
isolation pass.

### Task 7: Atomic Artifacts and CLI

**Files:**
- Create: `src/cmc_bbdm/aei_multiview_regression/artifacts.py`
- Create: `src/cmc_bbdm/aei_multiview_regression/replay.py`
- Create: `scripts/run_aei_multiview_regression.py`
- Create: `tests/test_multiview_artifacts.py`
- Create: `tests/test_multiview_cli.py`

- [ ] **Step 1: Write failing artifact and CLI tests**

```python
def test_e1_oof_schema(package):
    rows = read_csv(package / "oof_predictions.csv")
    assert tuple(rows[0]) == REQUIRED_E1_COLUMNS
    assert len(rows) == 276

def test_replay_rejects_changed_output(package):
    (package / "summary.json").write_text("{}\n")
    with pytest.raises(ArtifactError):
        replay_stage(package)
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest -q tests/test_multiview_artifacts.py tests/test_multiview_cli.py`

Expected: missing modules and CLI.

- [ ] **Step 3: Implement stage serializers**

Publish E1, E2, and E3 with `config.yaml`, `aggregate_metrics.csv`,
`domain_metrics.csv`, `summary.json`, `REPORT.md`, `artifact_manifest.json`, and
`CHECKSUMS.sha256`. Add stage-specific OOF, selection, agreement, weight,
reliability, bootstrap, and stress tables. Use the existing atomic writer and
strict replay facade.

- [ ] **Step 4: Implement CLI**

Commands are `audit`, `run`, and `replay`. `run` refuses an existing output root;
`audit` verifies A0, A2, and A5 sources without fitting; `replay` hashes one named
stage without outer exposure.

- [ ] **Step 5: Verify artifacts and CLI**

Run: `python -m pytest -q tests/test_multiview_artifacts.py tests/test_multiview_cli.py`

Expected: schemas, atomic no-overwrite behavior, tamper rejection, and command
validation pass.

### Task 8: Execute Real Experiment and Complete Gate Documents

**Files:**
- Create: `results/multiview/e1_audit/`
- Create when authorized: `results/multiview/e2_cooperative/`
- Create when authorized: `results/multiview/e3_complementarity/`
- Create only if E3 passes: `results/multiview/e4_moe/`
- Modify: `docs/AEI_MULTIVIEW_CLAIM_EVIDENCE_MATRIX.md`
- Create: `docs/AEI_MULTIVIEW_RESULT_REPORT.md`

- [ ] **Step 1: Run focused tests before production**

Run: `python -m pytest -q tests/test_multiview_*.py`

Expected: all new tests pass.

- [ ] **Step 2: Audit frozen inputs**

Run: `python scripts/run_aei_multiview_regression.py audit --config paper_v3/configs/aei_multiview_regression.yaml`

Expected: `A0_BASELINE_PASS`, `A2_PAIRED_FEATURES_PASS`, and
`FACTORISATION_NO_GO` are reported with matching source hashes.

- [ ] **Step 3: Execute the formal chain**

Run: `python scripts/run_aei_multiview_regression.py run --config paper_v3/configs/aei_multiview_regression.yaml`

Expected: E1 always publishes; subsequent stage directories appear only after
their preceding gate returns GO.

- [ ] **Step 4: Replay every generated stage**

Run one `replay --stage ...` command per generated stage and require
`REPLAY_PASS` for every package.

- [ ] **Step 5: Replace planned claim states with evidence-bound outcomes**

Write exact metrics, intervals, gate conditions, warnings, NO-GO branches, and
claim boundaries from generated files. Never promote oracle performance or an
unrun conditional branch.

### Task 9: Final Integration Verification

**Files:**
- Modify only if needed: files from Tasks 1--8

- [ ] **Step 1: Run new tests**

Run: `python -m pytest -q tests/test_multiview_*.py`

Expected: all pass.

- [ ] **Step 2: Run affected legacy tests**

Run: `python -m pytest -q tests/test_aei_*.py tests/test_paired_view_semantics.py tests/test_feature_bank.py tests/test_cpb_sparse_scan_*.py`

Expected: all pass; existing A0--A5 and P5 semantics remain unchanged.

- [ ] **Step 3: Run static checks**

Run: `python -m ruff check src/cmc_bbdm/aei_multiview_regression scripts/run_aei_multiview_regression.py tests/test_multiview_*.py`

Run: `python -m mypy src/cmc_bbdm/aei_multiview_regression scripts/run_aei_multiview_regression.py`

Expected: both commands exit zero.

- [ ] **Step 4: Perform requirement-by-requirement audit**

Compare every controlling-prompt stage, output, metric, gate, stop rule,
independent-unit rule, leakage rule, stress test, and forbidden branch against
current files and replayed results. Treat missing evidence as incomplete.

- [ ] **Step 5: Inspect the final change set**

Run: `find src/cmc_bbdm/aei_multiview_regression tests docs paper_v3/configs results/multiview -type f -newermt '2026-08-21 00:00:00' -print | sort`

Expected: only the planned multi-view implementation, documentation, tests,
configuration, and result packages are present; A0--A5 files are byte-identical.
