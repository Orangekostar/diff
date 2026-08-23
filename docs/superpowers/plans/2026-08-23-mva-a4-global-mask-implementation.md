# MVA A4 Global Task-Aware Static Acquisition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a leakage-safe A4 global mechanical-value mask study, publish validated/replayed results, and issue separate global-mask and A5-authorization decisions.

**Architecture:** A separate A4 configuration and protocol bind the frozen A0-A3 package. Two checksum-bound initial-candidate banks amortize image encoding, while each outer fold regenerates mechanical labels with query-domain OOF predictors trained on only four of the five source domains. Pure ranking and static-trajectory modules feed outer-domain workers; aggregation, gates, figures, and replay derive only from validated tables.

**Tech Stack:** Python 3.13, NumPy, PyTorch/Torchvision, Polars/Parquet, SciPy, Matplotlib, PyYAML, pytest, Ruff.

---

### Task 1: Freeze A4 protocol and configuration

**Files:**
- Create: `docs/MVA_A4_PROTOCOL.md`
- Create: `docs/MVA_A4_CLAIM_EVIDENCE_MATRIX.md`
- Create: `paper_v3/configs/mva_a4_global_mask.yaml`
- Create: `src/cmc_bbdm/mva/a4_config.py`
- Test: `tests/test_mva_a4_config.py`

- [ ] **Step 1: Write a failing immutable-config test**

```python
def test_a4_config_freezes_outer_safe_static_mask_protocol() -> None:
    config = load_a4_config(CONFIG, project_root=ROOT)
    assert config.a3_status == "MVA_ORACLE_GO"
    assert config.cell_shape == (8, 8)
    assert config.rank_aggregation == "equal_domain_mean_normalized_rank"
    assert config.methods == (
        "global_appearance_mask",
        "global_reconstruction_mask",
        "global_mechanical_mask",
    )
    assert config.adaptive_gap_threshold == 0.03
    assert config.minimum_improved_domains == 4
    assert config.bootstrap_resamples == 100000
    assert not any(path.path.is_absolute() for path in config.sources.values())
```

- [ ] **Step 2: Run the test and require missing-module failure**

Run: `python -m pytest -q tests/test_mva_a4_config.py`

Expected: collection fails because `cmc_bbdm.mva.a4_config` does not exist.

- [ ] **Step 3: Write the protocol and claim-evidence matrix**

Freeze the exact source OOF roster, rank normalization, methods, checkpoints,
AUEBC interval, image metrics, bootstrap matrix, three A4 gate comparisons,
3% adaptive-gap gate, artifact layout, replay contract, and A4-only stop scope.
Record all result-dependent claims as `TO TEST` and A5-A7 as `LOCKED`.

- [ ] **Step 4: Implement a fail-closed A4 config loader**

```python
@dataclass(frozen=True, slots=True)
class A4Config:
    sources: Mapping[str, BoundSource]
    domain_order: tuple[str, ...]
    checkpoints: tuple[float, ...]
    methods: tuple[str, ...]
    rank_aggregation: str
    adaptive_gap_threshold: float
    minimum_improved_domains: int
    bootstrap_seed: int
    bootstrap_resamples: int
    output_dir: Path
```

Reject unknown keys, absolute paths, source hash drift, nonfinite thresholds,
changed A3 status, methods outside A4, and any A5-A7 implementation entry.

- [ ] **Step 5: Run config tests**

Run: `python -m pytest -q tests/test_mva_a4_config.py`

Expected: all tests pass.

### Task 2: Implement equal-domain static ranking

**Files:**
- Create: `src/cmc_bbdm/mva/global_mask.py`
- Test: `tests/test_mva_global_mask.py`

- [ ] **Step 1: Write failing ranking tests**

```python
def test_global_ranking_uses_equal_domain_normalized_ranks() -> None:
    rows = fixture_rows_with_unequal_domain_counts_and_value_scales()
    ranking = aggregate_global_ranking(
        rows,
        outer_domain="held_out",
        method="global_mechanical_mask",
        cell_count=4,
    )
    assert ranking.source_domains == ("d1", "d2")
    assert ranking.cell_order == (2, 0, 1, 3)
    assert ranking.target_rows_seen == 0
```

Also require deterministic lower-cell tie breaks, one row per
specimen/cell/method, finite values, five source domains, exactly 64 cells, and
leave-one-source-domain-out stability metrics.

- [ ] **Step 2: Run the ranking tests and confirm failure**

Run: `python -m pytest -q tests/test_mva_global_mask.py`

Expected: import failure for `global_mask`.

- [ ] **Step 3: Implement pure ranking primitives**

```python
@dataclass(frozen=True, slots=True)
class GlobalMaskRanking:
    outer_domain: str
    method: str
    cell_order: tuple[int, ...]
    cell_scores: tuple[float, ...]
    mean_raw_values: tuple[float, ...]
    mean_value_per_measurement: tuple[float, ...]
    source_domains: tuple[str, ...]
    source_specimen_count: int


def normalized_candidate_ranks(values: Sequence[float]) -> np.ndarray:
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    ranks = np.empty(len(order), dtype=np.float64)
    for position, index in enumerate(order):
        ranks[index] = 1.0 - position / (len(order) - 1)
    return ranks
```

Aggregate specimen ranks inside each domain, then average domain means equally.
Never accept target rows or use diagnostic raw values for primary ordering.

- [ ] **Step 4: Run ranking tests**

Run: `python -m pytest -q tests/test_mva_global_mask.py`

Expected: all tests pass.

### Task 3: Implement fixed-ranking trajectories

**Files:**
- Modify: `src/cmc_bbdm/mva/oracle_trajectory.py`
- Test: `tests/test_mva_static_mask_trajectory.py`

- [ ] **Step 1: Write failing geometry-only policy tests**

```python
def test_static_mask_uses_one_order_and_only_level_one_actions() -> None:
    trajectory = run_static_mask_trajectory(
        grid,
        initial_state(grid),
        cell_order=(5, 2, 0, *remaining_cells),
        checkpoints=(0.0625, 0.125, 0.25),
        method="global_mechanical_mask",
    )
    assert trajectory.actions[:3] == (
        RefinementAction(5, 0, 1),
        RefinementAction(2, 0, 1),
        RefinementAction(0, 0, 1),
    )
    assert all(action.to_level == 1 for action in trajectory.actions)
    assert all(
        snapshot.effective_budget <= snapshot.nominal_checkpoint
        for snapshot in trajectory.snapshots
    )
```

Require identical action order across all registered native shapes, unique
measurement accounting, no removed locations, checkpoint monotonicity, and no
image/CAI argument in the policy API.

- [ ] **Step 2: Run the trajectory test and confirm missing-function failure**

Run: `python -m pytest -q tests/test_mva_static_mask_trajectory.py`

- [ ] **Step 3: Implement `run_static_mask_trajectory`**

Walk the supplied permutation once. At each checkpoint, apply the next
level-0 to level-1 action only if the exact resulting unique-measurement count
fits the cap. Validate a 64-cell permutation and reject all level-2 behavior.

- [ ] **Step 4: Run static-trajectory and existing trajectory tests**

Run: `python -m pytest -q tests/test_mva_static_mask_trajectory.py tests/test_mva_oracle_trajectory.py tests/test_budget_curve_monotonic_measurement_count.py`

Expected: all tests pass.

### Task 4: Build checksum-bound initial candidate banks

**Files:**
- Create: `src/cmc_bbdm/mva/a4_candidate_bank.py`
- Test: `tests/test_mva_a4_candidate_bank.py`

- [ ] **Step 1: Write failing bank-contract tests**

```python
def test_candidate_bank_contains_exact_initial_actions() -> None:
    bank = build_candidate_bank_fixture(images, initial_budget=0.015625)
    assert bank.embeddings.shape == (len(images), 64, 512)
    assert bank.reconstruction_values.shape == (len(images), 64)
    assert bank.appearance_values.shape == (len(images), 64)
    assert bank.added_measurements.shape == (len(images), 64)
    assert bank.to_levels == (1,) * 64
    assert bank.authority_state_sha256 == AUTHORITY_SHA
```

Require specimen order, image/input hashes, initial-budget identity, finite
arrays, exact candidate reconstruction, measured-value restoration, and tamper
rejection.

- [ ] **Step 2: Run the bank test and confirm failure**

Run: `python -m pytest -q tests/test_mva_a4_candidate_bank.py`

- [ ] **Step 3: Implement bank construction and validation**

For each specimen and each cell, refine only that cell from level 0 to level 1,
encode the reconstruction, calculate reconstruction and appearance values, and
store the exact added-measurement count. Save float64 arrays in
`results/mva/.work/a4_candidate_bank_<budget>.npz` with canonical metadata and
a content digest.

- [ ] **Step 4: Run bank tests**

Run: `python -m pytest -q tests/test_mva_a4_candidate_bank.py tests/test_mva_incremental_reconstruction.py`

Expected: all tests pass.

### Task 5: Generate outer-specific source OOF labels and rankings

**Files:**
- Create: `src/cmc_bbdm/mva/a4_source_labels.py`
- Test: `tests/test_mva_a4_source_oof.py`

- [ ] **Step 1: Write failing roster and value tests**

```python
def test_source_label_predictor_excludes_outer_and_query_domains() -> None:
    result = generate_source_labels(synthetic_six_domain_authority, outer="d0")
    for audit in result.fit_audits:
        assert "d0" not in audit.fit_domains
        assert set(audit.query_domains).isdisjoint(audit.fit_domains)
        assert len(audit.fit_domains) == 4
    assert {row.dataset_id for row in result.rows} == {"d1", "d2", "d3", "d4", "d5"}
```

Also assert mechanical labels equal absolute-error reduction from the bank's
candidate embeddings, reconstruction/appearance labels match bank arrays,
every source specimen has 64 rows per method, and ranking rows contain no target
specimen ID or target-derived predictor hash.

- [ ] **Step 2: Run source-OOF tests and confirm failure**

Run: `python -m pytest -q tests/test_mva_a4_source_oof.py`

- [ ] **Step 3: Implement outer-specific source OOF generation**

For each query source domain, subset authority arrays to the five A4 source
domains and call the existing nested predictor with the query domain as its OOF
holdout. Predict all 64 bank candidates in batches, calculate mechanical values,
serialize fit rosters, and call `aggregate_global_ranking` for all three methods.

- [ ] **Step 4: Run leakage and source-label tests**

Run: `python -m pytest -q tests/test_mva_a4_source_oof.py tests/test_mva_oracle_uses_oof_predictor.py tests/test_outer_domain_not_used_for_oracle_training.py`

Expected: all tests pass.

### Task 6: Evaluate static masks on outer targets

**Files:**
- Create: `src/cmc_bbdm/mva/a4_execution.py`
- Test: `tests/test_mva_a4_outer_evaluation.py`

- [ ] **Step 1: Write failing outer-isolation and common-head tests**

```python
def test_a4_outer_worker_uses_source_rankings_and_common_pb_heads() -> None:
    result = run_outer_fixture(outer_domain="d0")
    assert result.ranking_source_domains == ("d1", "d2", "d3", "d4", "d5")
    for checkpoint in CHECKPOINTS:
        hashes = {
            row.p_b_predictor_state_sha256
            for row in result.states
            if row.nominal_checkpoint == checkpoint
        }
        assert len(hashes) == 1
    assert all(row.dataset_id == "d0" for row in result.states)
```

Require all three methods, actual budget fields, deterministic trajectories,
P-A and P-B predictions, normalized RGB MSE, SSIM, and no target contribution
to mask ranking or predictor fitting.

- [ ] **Step 2: Run outer-evaluation tests and confirm failure**

Run: `python -m pytest -q tests/test_mva_a4_outer_evaluation.py`

- [ ] **Step 3: Implement the A4 outer worker**

Load the selected initial budget, candidate bank, source ranking, uniform bank,
and one source-uniform P-B model per checkpoint. Materialize each fixed
trajectory on target images, encode snapshots, and write transactional shards:

```text
results/mva/.work/a4_domains/<outer>/
  source_values.parquet
  rankings.csv
  trajectories.parquet
  states.parquet
  fit_audits.csv
  ranking_stability.csv
  complete.json
```

- [ ] **Step 4: Run the outer worker tests**

Run: `python -m pytest -q tests/test_mva_a4_outer_evaluation.py tests/test_policy_never_reads_true_cai.py tests/test_policy_never_reads_unobserved_pixels.py`

Expected: all tests pass.

### Task 7: Aggregate metrics, bootstrap effects, and gates

**Files:**
- Create: `src/cmc_bbdm/mva/a4_evaluation.py`
- Test: `tests/test_mva_a4_gate.py`
- Test: `tests/test_mva_a4_aggregation.py`

- [ ] **Step 1: Write failing boundary tests for both decisions**

```python
def test_a5_is_authorized_only_above_the_three_percent_gap() -> None:
    below = evaluate_a4_gate(fixture(relative_adaptive_gap=0.029999999))
    exact = evaluate_a4_gate(fixture(relative_adaptive_gap=0.03))
    assert below.a5_status == "MVA_A5_NOT_AUTHORIZED"
    assert exact.a5_status == "MVA_A5_AUTHORIZED"
```

Test every A4 comparison, synchronized-bootstrap lower-bound equality,
4/6-domain boundary, missing data, adverse domain, and exact one-status output.

- [ ] **Step 2: Run gate tests and confirm failure**

Run: `python -m pytest -q tests/test_mva_a4_gate.py tests/test_mva_a4_aggregation.py`

- [ ] **Step 3: Implement aggregation**

Validate six complete shards. Reuse A2 uniform/random/oracle reference states,
calculate method/domain/equal-domain curves, AUEBC, `B_5%`, image curves,
source ranking stability, and specimen records. Reuse
`synchronized_bootstrap_indices(20260823, 100000, 6)` and publish its digest.

```python
@dataclass(frozen=True, slots=True)
class A4GateResult:
    global_mask_status: str
    a5_status: str
    uniform_effect: BootstrapEffect
    reconstruction_effect: BootstrapEffect
    appearance_effect: BootstrapEffect
    adaptive_gap_effect: BootstrapEffect
    relative_adaptive_gap: float
```

- [ ] **Step 4: Run aggregation tests**

Run: `python -m pytest -q tests/test_mva_a4_gate.py tests/test_mva_a4_aggregation.py tests/test_auebc.py tests/test_b5_metric.py tests/test_mva_statistics.py`

Expected: all tests pass.

### Task 8: Implement A4 artifacts and byte-identical replay

**Files:**
- Create: `src/cmc_bbdm/mva/a4_artifacts.py`
- Create: `src/cmc_bbdm/mva/a4_replay.py`
- Test: `tests/test_mva_a4_artifacts.py`
- Test: `tests/test_mva_a4_replay.py`

- [ ] **Step 1: Write failing artifact tests**

Require the formal layout, exact schemas, canonical CSV/JSON, repository-relative
provenance, source/upstream hashes, checksum coverage, transactional publication,
tamper rejection, A4/A5 status consistency, and byte-identical replay.

```python
def test_a4_replay_is_byte_identical(tmp_path: Path) -> None:
    formal = publish_a4_fixture(tmp_path / "formal")
    replay = replay_a4_package(formal, tmp_path / "replay", project_root=tmp_path)
    assert tree_digest(formal) == tree_digest(replay)
    assert replay.global_mask_status in A4_STATUSES
    assert replay.a5_status in A5_AUTHORIZATION_STATUSES
```

- [ ] **Step 2: Run artifact tests and confirm failure**

Run: `python -m pytest -q tests/test_mva_a4_artifacts.py tests/test_mva_a4_replay.py`

- [ ] **Step 3: Implement validators and replay**

Recompute budgets, target errors, curves, AUEBC, sufficiency, paired domain
effects, gates, source/target separation, ranking permutations, and artifact
hashes from raw tables. Copy only after validation into a temporary directory,
validate again, then atomically publish replay.

- [ ] **Step 4: Run artifact tests**

Run: `python -m pytest -q tests/test_mva_a4_artifacts.py tests/test_mva_a4_replay.py`

Expected: all tests pass.

### Task 9: Render evidence-bound figures and report

**Files:**
- Create: `docs/visual-composer/mva-a4/visual-contract.md`
- Create: `docs/visual-composer/mva-a4/qa-ledger.md`
- Create: `docs/visual-composer/mva-a4/iteration-log.md`
- Create: `src/cmc_bbdm/mva/a4_figures.py`
- Test: `tests/test_mva_a4_figures.py`

- [ ] **Step 1: Freeze the visual contract before reading formal outcomes**

Specify three aligned 8 x 8 ranking maps, the common CAI error-budget curve,
an image-fidelity-versus-CAI comparison, stable colors/markers, source-data
exports, PNG/SVG output, representative examples selected without outcome
inspection, and explicit retrospective-simulation captions.

- [ ] **Step 2: Write failing figure-contract tests**

```python
def test_a4_figures_are_table_derived_and_complete(tmp_path: Path) -> None:
    output = render_a4_figures(validated_fixture, tmp_path)
    assert REQUIRED_A4_FIGURES <= {path.name for path in output.iterdir()}
    assert (output / "source_data.csv").is_file()
    assert b"global_mechanical_mask" in (output / "source_data.csv").read_bytes()
```

- [ ] **Step 3: Implement table-only rendering and `REPORT.md` generation**

The report directly answers whether global MVoM beats uniform, reconstruction,
and appearance; how much oracle gap remains; which domains are adverse; whether
A5 is authorized; whether reconstruction quality trades off against CAI; and
which physical/deployment claims remain excluded.

- [ ] **Step 4: Run figure tests and visually inspect all outputs**

Run: `MPLBACKEND=Agg python -m pytest -q tests/test_mva_a4_figures.py`

Expected: all tests pass; every raster is nonblank and has no overlap or clipping.

### Task 10: Wire A4 CLI and execute formal shards

**Files:**
- Create: `src/cmc_bbdm/mva/a4_cli.py`
- Create: `scripts/run_mva_a4.py`
- Test: `tests/test_mva_a4_cli.py`

- [ ] **Step 1: Write failing parser tests**

Require commands `candidate-bank`, `domain`, `aggregate`, `figures`, `finalize`,
`validate`, and `replay`, with explicit config, project root, device, outer
domain, source, and destination arguments.

- [ ] **Step 2: Implement ordered A4 CLI routes**

Keep A4 separate from the frozen A0-A3 entry point. Reject an A3 status other
than `MVA_ORACLE_GO`, unregistered outer domains, incomplete shards, and any
A5 training flag.

- [ ] **Step 3: Run all A4 and inherited MVA tests**

Run: `python -m pytest -q $(rg -l 'cmc_bbdm\.mva' tests -g '*.py' | sort)`

Expected: all tests pass.

- [ ] **Step 4: Build both candidate banks**

Run:

```bash
python scripts/run_mva_a4.py candidate-bank --config paper_v3/configs/mva_a4_global_mask.yaml --initial-budget 0.015625 --device cuda:0
python scripts/run_mva_a4.py candidate-bank --config paper_v3/configs/mva_a4_global_mask.yaml --initial-budget 0.03125 --device cuda:0
```

Expected: two validated cache files with 276 specimens and 64 candidates each.

- [ ] **Step 5: Execute all six outer-domain workers**

Assign deterministic outer domains to available GPUs without sharing output
directories. Each worker must write `complete.json` only after shard validation.

- [ ] **Step 6: Aggregate, render, finalize, validate, and replay**

Run:

```bash
python scripts/run_mva_a4.py aggregate --config paper_v3/configs/mva_a4_global_mask.yaml
python scripts/run_mva_a4.py figures --config paper_v3/configs/mva_a4_global_mask.yaml
python scripts/run_mva_a4.py finalize --config paper_v3/configs/mva_a4_global_mask.yaml
python scripts/run_mva_a4.py validate --config paper_v3/configs/mva_a4_global_mask.yaml
python scripts/run_mva_a4.py replay --config paper_v3/configs/mva_a4_global_mask.yaml --source results/mva/a4_global_task_mask --destination results/mva/replay/a4_global_task_mask
```

Expected: formal and replay trees are byte-identical and contain one A4 status
and one A5-authorization status.

### Task 11: Final audit and compact Git publication

**Files:**
- Modify: `docs/MVA_A4_CLAIM_EVIDENCE_MATRIX.md`
- Create: `docs/MVA_A4_COMPLETION_AUDIT.md`
- Modify: `README.md` only in the compact export
- Copy: validated A4 code/docs/config/tests/results into the compact Git repository

- [ ] **Step 1: Fill claim statuses only from validated formal results**

Record all gate effects, confidence intervals, adverse domains, image-fidelity
comparisons, exact A4/A5 decisions, and the A5-A7 lock state.

- [ ] **Step 2: Run completion verification**

Run the full MVA-linked pytest suite, Ruff, compileall, formal validation,
formal/replay byte comparison, absolute-path scan, nonfinite-value scan, figure
inspection, source/config hash validation, and explicit search for A5-A7 models.

- [ ] **Step 3: Review the complete compact-repository diff**

Verify scope, public-interface compatibility, generated-artifact integrity,
GitHub file-size limits, root README checksum, and a clean remote base.

- [ ] **Step 4: Commit and push**

Commit the A4 code, evidence, and decision to compact `main`, push to
`origin/main` without force, and verify the remote branch resolves to the new
commit.
