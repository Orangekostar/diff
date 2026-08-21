# D8 Pilot Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the frozen internal-only baseline, regenerate a cross-fitted P6 residual bank, search morphology-preserving nuisance marginalization candidates with strict nested-domain isolation, and publish a validated D8 pilot decision package.

**Architecture:** A new `cpb_diffusion_marginalization` package separates immutable configuration and fold authority from residual generation, decomposition, morphology gating, feature/regressor evaluation, Optuna search, and transactional artifacts. The pilot never evaluates an outer-domain result; it produces one hash-frozen selected candidate per prospective outer fold plus an objective residual-diffusion escalation decision.

**Tech Stack:** Python 3.13, NumPy, SciPy, Pillow, PyTorch, torchvision, diffusers, scikit-learn, Optuna 4.9.0, PyWavelets 1.8.0, PyYAML, pytest, Ruff.

---

### Task 1: Freeze the D8 exploration configuration

**Files:**
- Create: `paper_v3/configs/d8_exploration.yaml`
- Create: `src/cmc_bbdm/cpb_diffusion_marginalization/__init__.py`
- Create: `src/cmc_bbdm/cpb_diffusion_marginalization/config.py`
- Test: `tests/test_d8_config.py`
- Create: `environment/requirements-d8.txt`

- [ ] **Step 1: Write the failing exact-contract tests**

```python
def test_d8_config_binds_prompt_design_plan_and_source_packages(project_root):
    config = load_d8_config(
        project_root / "paper_v3/configs/d8_exploration.yaml",
        project_root=project_root,
    )
    assert config.scope == "cpb_d8_morphology_preserving_marginalization"
    assert config.baseline_mae == 0.08963580465761432
    assert config.outer_domains == DOMAIN_ORDER
    assert config.optuna_trials == 60
    assert config.forced_trials == 12
    assert config.p6_draws == 8
    assert config.output_dir == "results/d8_search"


def test_d8_config_rejects_unknown_keys_and_source_drift(project_root, tmp_path):
    payload = yaml.safe_load(
        (project_root / "paper_v3/configs/d8_exploration.yaml").read_text()
    )
    payload["extra"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(D8ConfigError):
        load_d8_config(path, project_root=project_root)
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `pytest -q tests/test_d8_config.py`  
Expected: collection fails because `cmc_bbdm.cpb_diffusion_marginalization.config` does not exist.

- [ ] **Step 3: Add exact config values and immutable loader**

Implement this public API:

```python
@dataclass(frozen=True, slots=True)
class D8Config:
    scope: str
    schema_version: int
    seed: int
    outer_domains: tuple[str, ...]
    baseline_mae: float
    optuna_trials: int
    forced_trials: int
    rerank_seeds: tuple[int, ...]
    p6_draws: int
    sources: Mapping[str, SourceRecord]
    search_space: Mapping[str, object]
    output_dir: str
    config_sha256: str


def load_d8_config(path: str | Path, *, project_root: str | Path) -> D8Config:
    """Load duplicate-key-safe YAML and verify exact values, types, paths, hashes and runtime."""
```

The YAML must bind the target prompt, D8 design, exploration plan, P1 config/manifest/predictions/splits, P5 manifest, P6 config/manifest/checkpoints/uncertainty source, ResNet weights, runtime requirements, and every executed D8 source file as it is added.

- [ ] **Step 4: Pin missing dependencies**

Create a D8-specific lock that inherits the frozen P6 runtime lock without
changing it:

```text
-r requirements-runtime.txt
optuna==4.9.0
PyWavelets==1.8.0
```

Install them in the active experiment environment and verify module/distribution versions agree.

- [ ] **Step 5: Run focused verification**

Run: `pytest -q tests/test_d8_config.py`  
Expected: PASS.  
Run: `ruff check src/cmc_bbdm/cpb_diffusion_marginalization tests/test_d8_config.py`  
Expected: `All checks passed!`

### Task 2: Issue fold-authorized data views and reproduce I_frozen

**Files:**
- Create: `src/cmc_bbdm/cpb_diffusion_marginalization/authority.py`
- Create: `src/cmc_bbdm/cpb_diffusion_marginalization/baseline.py`
- Test: `tests/test_d8_baseline.py`

- [ ] **Step 1: Write RED tests for five-domain search views**

```python
def test_search_view_excludes_outer_domain_and_cannot_issue_evaluation(data, config):
    view = issue_search_view(data, outer_domain="74t7kcdgkr", config=config)
    assert set(view.dataset_ids) == set(config.outer_domains) - {"74t7kcdgkr"}
    assert "74t7kcdgkr" not in view.dataset_ids
    with pytest.raises(D8AuthorityError):
        issue_evaluation_view(data, selection=None, outer_domain="74t7kcdgkr", config=config)


def test_inner_view_excludes_query_domain_from_fit_authorities(search_view):
    inner = issue_inner_fold(search_view, query_domain="cgtnjyggtm")
    assert "cgtnjyggtm" not in inner.fit_dataset_ids
    assert set(inner.query_dataset_ids) == {"cgtnjyggtm"}
    assert set(inner.fit_specimen_ids).isdisjoint(inner.query_specimen_ids)
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `pytest -q tests/test_d8_baseline.py`  
Expected: FAIL because authority and baseline APIs are missing.

- [ ] **Step 3: Implement unforgeable process-local views**

```python
@dataclass(frozen=True, slots=True)
class D8SearchView:
    outer_domain: str
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    state_sha256: str


@dataclass(frozen=True, slots=True)
class D8InnerFold:
    outer_domain: str
    query_domain: str
    fit_indices: np.ndarray
    query_indices: np.ndarray
    state_sha256: str
```

Issue views from validated V3 loader state, keep arrays immutable, bind IDs/domains/source hashes, and reject copied, reconstructed, mutated, or cross-process authorities.

- [ ] **Step 4: Write RED baseline reproduction tests**

```python
def test_d8_baseline_reproduces_all_p1_internal_only_predictions(project_root, d8_data):
    result = reproduce_internal_only_baseline(d8_data, project_root=project_root)
    assert result.specimen_count == 276
    assert result.pca_dimensions == (8, 32, 8, 8, 8, 8)
    assert result.equal_domain_mae == pytest.approx(0.08963580465761432, abs=1e-12)
    assert result.maximum_prediction_error <= 1e-12
    assert result.maximum_target_error <= 1e-12
```

- [ ] **Step 5: Implement the baseline adapter**

```python
@dataclass(frozen=True, slots=True)
class D8BaselineResult:
    specimen_count: int
    pca_dimensions: tuple[int, ...]
    domain_mae: tuple[float, ...]
    equal_domain_mae: float
    maximum_prediction_error: float
    maximum_target_error: float
    state_sha256: str


def reproduce_internal_only_baseline(data: object, *, project_root: Path) -> D8BaselineResult:
    """Rebuild I_frozen through the registered encoder/PCA/Ridge path and compare all rows."""
```

Reuse the V3 data loader and frozen encoder, but not saved predictions as model outputs. Saved P1 predictions are comparison authority only.

- [ ] **Step 6: Run baseline verification**

Run: `pytest -q tests/test_d8_baseline.py`  
Expected: PASS with exact six-domain reproduction.

### Task 3: Regenerate and validate the cross-fitted P6 residual bank

**Files:**
- Create: `src/cmc_bbdm/cpb_diffusion_marginalization/residuals.py`
- Test: `tests/test_d8_residuals.py`

- [ ] **Step 1: Write RED checkpoint-isolation and posterior tests**

```python
def test_p6_residuals_are_cross_fitted_by_complete_domain(p6_bank, d8_data):
    for record in p6_bank.records:
        assert record.dataset_id == record.heldout_domain
        assert record.dataset_id not in record.checkpoint_train_domains
        assert record.specimen_id not in record.checkpoint_train_ids


def test_regenerated_posterior_matches_p6_authority(p6_bank):
    assert p6_bank.draw_count == 8
    assert p6_bank.maximum_mean_error <= 1e-6
    assert p6_bank.maximum_variance_error <= 1e-6
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `pytest -q tests/test_d8_residuals.py`  
Expected: FAIL because residual-bank APIs do not exist.

- [ ] **Step 3: Implement deterministic regeneration**

```python
@dataclass(frozen=True, slots=True)
class ResidualRecord:
    specimen_id: str
    dataset_id: str
    heldout_domain: str
    draw_index: int
    residual_64: np.ndarray
    source_sha256: str
    checkpoint_scientific_digest: str
    residual_sha256: str


@dataclass(frozen=True, slots=True)
class P6ResidualBank:
    records: tuple[ResidualRecord, ...]
    draw_count: int
    maximum_mean_error: float
    maximum_variance_error: float
    state_sha256: str


def build_cross_fitted_p6_residual_bank(
    data: object, *, p6_package: Path, device: str = "cuda"
) -> P6ResidualBank:
    """Regenerate each fold's eight P6 draws and subtract its measured 64x64 field."""
```

Use the existing `load_fold_checkpoint` and `sample_diffusion_fields`. Residual arrays are float32, immutable, finite, shape `(3,64,64)`, and bound to the P6 posterior mean/variance authority.

- [ ] **Step 4: Add tamper and wrong-fold tests**

Reject checkpoint metadata/state changes, specimen reorder, source changes, incorrect heldout domain, missing draw, duplicate draw, mean/variance mismatch, and nonfinite residuals.

- [ ] **Step 5: Run focused verification**

Run: `pytest -q tests/test_d8_residuals.py`  
Expected: PASS on a synthetic fixture; mark the full 276-row regeneration test `slow` for the registered run.

### Task 4: Implement frequency decomposition and non-diffusion controls

**Files:**
- Create: `src/cmc_bbdm/cpb_diffusion_marginalization/decomposition.py`
- Test: `tests/test_d8_decomposition.py`

- [ ] **Step 1: Write analytic RED tests**

```python
@pytest.mark.parametrize("family", ["gaussian", "fourier", "wavelet"])
def test_decomposition_reconstructs_residual_and_preserves_shape(family, synthetic_residual):
    bands = decompose_residual(synthetic_residual, family=family, parameters=PARAMS[family])
    assert bands.selected.shape == synthetic_residual.shape
    assert np.isfinite(bands.selected).all()
    assert bands.reconstruction_error <= 1e-6


def test_fourier_high_band_rejects_constant_field():
    residual = np.ones((3, 64, 64), dtype=np.float32)
    bands = decompose_residual(
        residual,
        family="fourier",
        parameters={"band": "high", "cutoff": 0.2, "transition": 0.05},
    )
    assert np.max(np.abs(bands.selected)) <= 1e-6
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `pytest -q tests/test_d8_decomposition.py`  
Expected: FAIL because decomposition APIs are missing.

- [ ] **Step 3: Implement three decomposition families**

```python
@dataclass(frozen=True, slots=True)
class ResidualBands:
    family: str
    selected_band: str
    selected: np.ndarray
    low: np.ndarray
    mid: np.ndarray
    high: np.ndarray
    energy_fraction: float
    reconstruction_error: float
    state_sha256: str


def decompose_residual(
    residual: np.ndarray, *, family: str, parameters: Mapping[str, object]
) -> ResidualBands:
    """Return Gaussian, raised-cosine Fourier, or PyWavelets frequency bands."""
```

- [ ] **Step 4: Implement B2-B4 control residuals**

```python
def gaussian_control(residual: np.ndarray, *, seed: int) -> np.ndarray: ...
def phase_randomized_control(residual: np.ndarray, *, seed: int) -> np.ndarray: ...
def empirical_control(
    bank: P6ResidualBank, *, fit_domains: tuple[str, ...], query_ids: tuple[str, ...], seed: int
) -> tuple[np.ndarray, ...]: ...
```

Tests must prove variance/spectrum matching, deterministic seeds, donor-domain exclusion, and no query ID self-donation.

- [ ] **Step 5: Run focused verification**

Run: `pytest -q tests/test_d8_decomposition.py`  
Expected: PASS.

### Task 5: Implement variant construction and morphology gates

**Files:**
- Create: `src/cmc_bbdm/cpb_diffusion_marginalization/variants.py`
- Test: `tests/test_d8_variants.py`

- [ ] **Step 1: Write RED gate tests**

```python
def test_zero_alpha_preserves_every_morphology_measure(source, rule, calibration):
    result = build_variant(source, np.ones_like(source), alpha=0.0, rule=rule, calibration=calibration)
    assert result.accepted
    assert result.area_deviation == 0.0
    assert result.width_deviation == 0.0
    assert result.height_deviation == 0.0
    assert result.centroid_shift_mm == 0.0


def test_variant_rejects_large_low_frequency_change(source, rule, calibration):
    residual = low_frequency_blob(source.shape)
    result = build_variant(source, residual, alpha=1.0, rule=rule, calibration=calibration)
    assert not result.accepted
    assert "low_frequency_correlation" in result.failed_conditions
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `pytest -q tests/test_d8_variants.py`  
Expected: FAIL because variant APIs are missing.

- [ ] **Step 3: Implement the gate using registered morphology rules**

```python
@dataclass(frozen=True, slots=True)
class VariantRecord:
    variant: np.ndarray
    accepted: bool
    area_deviation: float
    width_deviation: float
    height_deviation: float
    centroid_shift_mm: float
    low_frequency_correlation: float
    radial_profile_correlation: float
    failed_conditions: tuple[str, ...]
    state_sha256: str


def build_variant(
    source: np.ndarray,
    residual: np.ndarray,
    *,
    native_source: np.ndarray,
    alpha: float,
    rule: CscanMorphologyRule,
    calibration: PhysicalCalibration,
    thresholds: MorphologyThresholds,
) -> VariantRecord:
    """Build one clipped 64x64 variant, lift its delta to native, and rerun the registered footprint extractor."""
```

- [ ] **Step 4: Implement K-variant deterministic proposal/fallback**

Try at most 32 proposals, keep accepted variants in draw order, fill missing slots with raw source, and serialize acceptance/fallback counts. Mark a candidate ineligible below the registered acceptance thresholds.

- [ ] **Step 5: Run focused verification**

Run: `pytest -q tests/test_d8_variants.py`  
Expected: PASS.

### Task 6: Implement frozen features, marginalization, and regressors

**Files:**
- Create: `src/cmc_bbdm/cpb_diffusion_marginalization/features.py`
- Create: `src/cmc_bbdm/cpb_diffusion_marginalization/regression.py`
- Test: `tests/test_d8_models.py`

- [ ] **Step 1: Write RED encoder and aggregation tests**

```python
def test_raw_global_feature_matches_p1_encoder(d8_encoder, p1_field, p1_embedding):
    observed = d8_encoder.encode(((p1_field,),), layer="global")
    np.testing.assert_allclose(observed[0, 0], p1_embedding, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("aggregation", ["mean", "median", "trimmed", "mean_variance"])
def test_feature_aggregation_preserves_specimen_axis(variant_features, aggregation):
    result = aggregate_features(variant_features, method=aggregation)
    assert result.shape[0] == variant_features.shape[0]
    assert np.isfinite(result).all()
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `pytest -q tests/test_d8_models.py`  
Expected: FAIL because D8 feature/model APIs are missing.

- [ ] **Step 3: Implement frozen layer extraction and cache identity**

```python
class D8FrozenEncoder:
    def encode(self, variants: tuple[tuple[np.ndarray, ...], ...], *, layer: str) -> np.ndarray:
        """Return `(specimen,K,feature)` immutable embeddings with frozen ResNet weights."""


def aggregate_features(features: np.ndarray, *, method: str) -> np.ndarray:
    """Aggregate only the variant axis; never merge specimens or domains."""
```

- [ ] **Step 4: Write RED nested-regressor tests**

```python
def test_candidate_fit_never_receives_inner_query_rows(inner_fold, candidate):
    fit = fit_candidate(candidate, inner_fold=inner_fold)
    assert set(fit.fit_specimen_ids).isdisjoint(inner_fold.query_specimen_ids)
    assert fit.query_predictions.shape == (len(inner_fold.query_indices),)


def test_replicated_variants_have_total_specimen_weight_one(training_rows):
    weights = specimen_weight_sums(training_rows)
    np.testing.assert_allclose(weights, np.ones_like(weights), atol=0.0, rtol=0.0)
```

- [ ] **Step 5: Implement bounded regressor registry**

```python
@dataclass(frozen=True, slots=True)
class CandidatePrediction:
    query_specimen_ids: tuple[str, ...]
    targets: np.ndarray
    predictions: np.ndarray
    fit_state_sha256: str
    state_sha256: str


def fit_candidate(spec: CandidateSpec, *, inner_fold: D8InnerFold) -> CandidatePrediction:
    """Fit fold-local scaling/PCA/regressor and return one query prediction per specimen."""
```

Register exact constructors for Ridge, ElasticNet, PLS, Huber, kernel Ridge, SVR, HistGradientBoosting, and shallow MLP. Reject unsupported or nonfinite parameterizations.

- [ ] **Step 6: Run focused verification**

Run: `pytest -q tests/test_d8_models.py`  
Expected: PASS.

### Task 7: Implement Optuna objective and complete trial tracking

**Files:**
- Create: `src/cmc_bbdm/cpb_diffusion_marginalization/search.py`
- Create: `src/cmc_bbdm/cpb_diffusion_marginalization/tracking.py`
- Test: `tests/test_d8_search.py`

- [ ] **Step 1: Write RED objective and no-outer-access tests**

```python
def test_objective_uses_five_inner_domains_and_registered_formula(inner_scores):
    score = robust_inner_objective(inner_scores)
    expected = np.mean(inner_scores) + 0.25 * np.max(inner_scores) + 0.10 * np.std(inner_scores)
    assert score == pytest.approx(expected, abs=1e-15)


def test_search_runner_has_no_evaluation_view_parameter():
    assert "evaluation_view" not in inspect.signature(run_outer_search).parameters
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `pytest -q tests/test_d8_search.py`  
Expected: FAIL because search APIs are missing.

- [ ] **Step 3: Implement canonical candidate suggestions**

```python
def suggest_candidate(trial: optuna.Trial, config: D8Config) -> CandidateSpec:
    """Suggest one valid control/decomposition/gate/K/feature/regressor/consistency combination."""


def robust_inner_objective(domain_mae: np.ndarray) -> float:
    values = np.asarray(domain_mae, dtype=np.float64)
    return float(math.fsum(values) / len(values) + 0.25 * max(values) + 0.10 * np.std(values))
```

- [ ] **Step 4: Implement 12 warm starts plus 60 TPE trials**

```python
def run_outer_search(view: D8SearchView, *, config: D8Config, output: Path) -> SearchResult:
    """Run five-domain inner LODO and write every trial before returning selected candidates."""
```

Use Optuna SQLite storage with one study per outer fold. Set study/trial user attributes for source, split, residual-bank, candidate, feature-cache, prediction, and score hashes.

- [ ] **Step 5: Implement exact trial index schema**

`trial_index.csv` must include trial ID, outer fold, state, prune/failure reason, five inner MAEs, mean/worst/SD/J, decomposition, diffusion/control family, alpha, K values, gate values, aggregation, consistency, regressor, seed, runtime, and every authority hash.

- [ ] **Step 6: Add resume and tamper tests**

Prove interrupted trials resume without duplication; changed config/source/split/database rows are rejected; failed trials remain visible.

- [ ] **Step 7: Run focused verification**

Run: `pytest -q tests/test_d8_search.py`  
Expected: PASS on an in-memory synthetic six-domain study.

### Task 8: Rerank, ensemble, and freeze prospective outer selections

**Files:**
- Create: `src/cmc_bbdm/cpb_diffusion_marginalization/selection.py`
- Test: `tests/test_d8_selection.py`

- [ ] **Step 1: Write RED repeated-seed ranking tests**

```python
def test_rerank_uses_three_fixed_seeds_and_canonical_tie_break(candidates, search_view):
    result = rerank_candidates(candidates, view=search_view, seeds=(20260820, 20260821, 20260822))
    assert result.seed_count == 3
    assert result.selected.config_sha256 == min(
        row.config_sha256 for row in result.rows if row.rank_key == result.rows[0].rank_key
    )
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `pytest -q tests/test_d8_selection.py`  
Expected: FAIL because selection APIs are missing.

- [ ] **Step 3: Implement top-12/top-4 reranking**

Recompute the top twelve with three seeds, then the top four with `K_test=8,16`. Aggregate seed/domain scores with `math.fsum` in canonical order.

- [ ] **Step 4: Implement inner-OOF nonnegative ensemble**

```python
def fit_nonnegative_ensemble(
    predictions: np.ndarray, targets: np.ndarray, *, minimum_j_gain: float = 1e-4
) -> EnsembleResult:
    """Fit simplex-constrained weights from inner-OOF rows and reject negligible gain."""
```

Test nonnegative weights, unit sum, specimen alignment, no duplicate OOF row, and deterministic fallback to the best member.

- [ ] **Step 5: Freeze selected configuration records**

Write one immutable JSON per prospective outer fold containing selected candidate/ensemble, fit-domain evidence, search/rerank database hashes, dependency/runtime hashes, and `outer_evaluation_started=false`. No outer prediction is produced by this task.

- [ ] **Step 6: Run focused verification**

Run: `pytest -q tests/test_d8_selection.py`  
Expected: PASS.

### Task 9: Publish the pilot package transactionally

**Files:**
- Create: `src/cmc_bbdm/cpb_diffusion_marginalization/artifacts.py`
- Create: `scripts/run_d8_exploration.py`
- Create: `scripts/run_d8_exploration.sh`
- Test: `tests/test_d8_artifacts.py`
- Test: `tests/test_d8_cli.py`

- [ ] **Step 1: Write RED exact-package tests**

```python
def test_pilot_package_contains_required_tracking_and_selections(package):
    validated = validate_d8_search_package(package)
    assert validated.outer_domains == DOMAIN_ORDER
    assert validated.initial_trial_count == 72 * 6
    assert validated.trial_count >= validated.initial_trial_count
    assert validated.outer_evaluation_count == 0
    assert {
        "trial_index.csv",
        "study.db",
        "residual_bank_manifest.json",
        "search_summary.csv",
        "selected_configs.json",
        "pilot_report.md",
        "artifact_manifest.json",
    }.issubset(validated.required_files)
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `pytest -q tests/test_d8_artifacts.py tests/test_d8_cli.py`  
Expected: FAIL because artifact/CLI APIs are missing.

- [ ] **Step 3: Implement independent package recomputation**

```python
@dataclass(frozen=True, slots=True)
class D8ValidatedSearchPackage:
    outer_domains: tuple[str, ...]
    initial_trial_count: int
    trial_count: int
    outer_evaluation_count: int
    escalation_status: str
    scientific_digest: str


def validate_d8_search_package(path: str | Path) -> D8ValidatedSearchPackage:
    """Reload all authorities and recompute schemas, scores, selections and hashes."""
```

- [ ] **Step 4: Implement conservative publication**

Reuse the repository transaction pattern: `flock`, owner marker, sibling staging, full validation, publish, rollback, and crash recovery. Never delete the only valid prior package after a double rename failure.

- [ ] **Step 5: Implement CLI boundaries**

Commands:

```bash
scripts/run_d8_exploration.sh baseline
scripts/run_d8_exploration.sh residual-bank
scripts/run_d8_exploration.sh pilot
scripts/run_d8_exploration.sh validate
```

The wrapper fixes CUDA visibility, deterministic runtime flags, BLAS thread counts, `PYTHONPATH`, and the sole registered config path. It refuses an outer-evaluation command.

- [ ] **Step 6: Run focused verification**

Run: `pytest -q tests/test_d8_artifacts.py tests/test_d8_cli.py`  
Expected: PASS, including concurrency, tamper, rollback, and crash tests.  
Run: `bash -n scripts/run_d8_exploration.sh`  
Expected: exit 0.

### Task 10: Execute the registered pilot and decide the next frozen branch

**Files:**
- Generate: `results/d8_search/`
- Modify: `docs/D8_RESULT_ORIENTED_EXPLORATION_PLAN.md`
- Create after pilot: `docs/D8_PILOT_DECISION.md`

- [ ] **Step 1: Run exact baseline reproduction**

Run: `scripts/run_d8_exploration.sh baseline`  
Expected: 276 predictions reproduced, MAE `0.08963580465761432`, maximum row error `<=1e-12`.

- [ ] **Step 2: Build the full residual bank**

Run: `scripts/run_d8_exploration.sh residual-bank`  
Expected: 2,208 cross-fitted residual records, 8 per specimen, saved posterior mean/variance match within `1e-6`.

- [ ] **Step 3: Run six registered studies**

Run: `scripts/run_d8_exploration.sh pilot`  
Expected: 432 initial trials plus registered reruns, complete trial index, no outer predictions, validated selected configuration for every outer fold.

- [ ] **Step 4: Run independent validation**

Run: `scripts/run_d8_exploration.sh validate`  
Expected: PASS with exact row counts, recomputed objectives/selections, source/code/runtime hashes, and scientific digest.

- [ ] **Step 5: Apply the frozen escalation rule**

Create `docs/D8_PILOT_DECISION.md` directly from validated search outputs. It must state exactly one:

```text
TRAIN_RESIDUAL_DIFFUSION
FREEZE_PILOT_FOR_OUTER_EVALUATION
CLOSE_DIFFUSION_SPECIFIC_ROUTE
```

If `TRAIN_RESIDUAL_DIFFUSION`, write a separate implementation plan for fold-local `p(R|S)` training. Otherwise write the formal outer-evaluation plan using the already frozen selected candidates. Do not inspect outer results before that next plan and config are frozen.

- [ ] **Step 6: Verify the pilot implementation surface**

Run:

```bash
pytest -q tests/test_d8_*.py
ruff check src/cmc_bbdm/cpb_diffusion_marginalization tests/test_d8_*.py scripts/run_d8_exploration.py
python -m compileall -q src/cmc_bbdm/cpb_diffusion_marginalization scripts/run_d8_exploration.py
sha256sum -c results/d8_search/CHECKSUMS.sha256
```

Expected: all commands pass without warnings or unregistered files.
