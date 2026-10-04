# CAI Order Mechanism Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publish the frozen archived-stage and fixed-final-set order analysis bound to task `CAI_ORDER_MECHANISM_R1_37B3C404`.

**Architecture:** A phase-oriented CLI locks immutable inputs, derives archive-only evidence, freezes deterministic intervention orders, evaluates predictor prefixes through a bounded cache, computes the two analysis families, renders an offline report, and verifies every artifact. Scientific computation is concentrated in pure functions; real predictor calls exist only in the infer phase and are guarded by a durable resource ledger.

**Tech Stack:** Python 3.13, NumPy, pandas, PyTorch, matplotlib, pytest, Ruff, HTML/CSS, Git

**Spec:** `docs/superpowers/specs/2026-10-04-cai-order-mechanism-design.md`

## Global Constraints

- Work only on branch `research/cai-vlm-agent-v3-controlled-reuse`, descended from `37b3c40414c00c6633b64656c4bbb2b178ef9848`.
- Read only the 50-sample VALID cohort; never join labels or infer on TEST.
- Actor, Qwen, CNN, OOF predictor, optimizer, autograd, and TEST forwards remain zero.
- The only model forward is frozen W2 `MEAN_SC` update 1750 for Track B prefixes and prescribed QA.
- Track A uses archived float64 costs; Track B uses integer cumulative native pixels divided by total pixels.
- Reuse the fixed 5000 bootstrap weights; missing or mismatched weights are fatal and have no generation fallback.
- At most 5950 nominal prefix requests and 7000 evaluated rows including retries, padding, and QA.
- One visible GPU, four CPU threads, 1800-second GPU session, 5400-second total CPU process, and one resume attempt.
- Output roots are exactly `results/cai_agent_v3/order_mechanism/r1_37b3c404` and `artifacts/cai_agent_v3/order_mechanism/r1_37b3c404`.
- No manuscript/PDF edits, dependency additions, PR, merge, force push, or tag.

---

### Task 1: Scope, Context, And Immutable Input Lock

**Files:**
- Create: `docs/cai/order_mechanism/ORDER_MECHANISM_SCOPE.json`
- Create: `scripts/cai_order_mechanism/__init__.py`
- Create: `scripts/cai_order_mechanism/inputs.py`
- Create: `tests/test_cai_order_mechanism.py`

**Interfaces:**
- Produces: `TaskContext.load(root: Path, scope_path: Path) -> TaskContext`
- Produces: `TaskContext.from_mapping(root: Path, scope: Mapping[str, Any], branch: str, head: str) -> TaskContext`
- Produces: `atomic_json(path: Path, value: object) -> None`
- Produces: `prepare_inputs(context: TaskContext) -> dict[str, object]`
- Produces: `load_valid_features(context: TaskContext, keys: Sequence[str]) -> ValidFeatures`
- Produces: `load_bootstrap_weights(context: TaskContext, keys: Sequence[str], domains: np.ndarray, groups: np.ndarray) -> np.ndarray`

- [ ] **Step 1: Copy the packaged scope byte-for-byte and write Q1 tests**

```python
def test_q1_scope_rejects_wrong_branch_hash_and_nonvalid_cohort(tmp_path):
    scope = make_scope(tmp_path)
    scope["cohort"]["split"] = "TEST"
    with pytest.raises(ValueError, match="VALID"):
        TaskContext.from_mapping(tmp_path, scope, branch=EXPECTED_BRANCH, head=SOURCE)
```

- [ ] **Step 2: Run Q1 and confirm the missing module failure**

Run: `PYTHONPATH="$PWD/src:$PWD" /home/ww/miniconda3/bin/python3.13 -m pytest -q tests/test_cai_order_mechanism.py -k q1`

- [ ] **Step 3: Implement canonical scope validation, repository-relative path resolution, atomic writers, single-pass hashing, source bindings, cohort lock, model-selection lock, and runtime lock**

```python
@dataclass(frozen=True, slots=True)
class TaskContext:
    root: Path
    scope: dict[str, Any]
    scope_path: Path
    branch: str
    head: str

    def path(self, dotted: str) -> Path:
        value = reduce(operator.getitem, dotted.split("."), self.scope)
        return resolve_inside_root(self.root, value)
```

- [ ] **Step 4: Implement thin VALID feature reads and strict bootstrap reindexing**

```python
def load_bootstrap_weights(context, keys, domains, groups):
    with np.load(context.source("sources.bootstrap_weights"), allow_pickle=False) as z:
        order = exact_key_reindex(z["specimen_keys"], keys)
        assert_exact(z["domains"][order], domains, "bootstrap domains")
        assert_exact(z["capture_groups"][order], groups, "bootstrap groups")
        weights = np.asarray(z["weights"][:, order])
    if weights.shape != (5000, 50):
        raise ValueError("fixed bootstrap shape changed")
    return weights
```

- [ ] **Step 5: Run Q1 and commit the input lock**

Run: `PYTHONPATH="$PWD/src:$PWD" /home/ww/miniconda3/bin/python3.13 -m pytest -q tests/test_cai_order_mechanism.py -k q1`

Commit: `git commit -m "feat(cai): lock order mechanism inputs"`

### Task 2: Archived Trajectory Derivation And Stage Identities

**Files:**
- Create: `scripts/cai_order_mechanism/analysis.py`
- Modify: `tests/test_cai_order_mechanism.py`

**Interfaces:**
- Consumes: `TaskContext`, prepared cohort and immutable archive bindings
- Produces: `parse_episode(row: Mapping[str, str]) -> Episode`
- Produces: `timing_decomposition(costs: np.ndarray, predictions: np.ndarray, target: float) -> TimingResult`
- Produces: `derive_archived(context: TaskContext) -> dict[str, Path]`
- Produces: `domain_equal(values: np.ndarray, domains: np.ndarray) -> float`
- Produces: `bootstrap_domain_equal(values: np.ndarray, weights: np.ndarray, domains: np.ndarray) -> np.ndarray`

- [ ] **Step 1: Add Q2-Q3 tests for archive parsing, right-closed stages, negative terms, and both identities**

```python
def test_q3_stage_boundary_and_identities():
    result = timing_decomposition(
        np.array([0.0, 0.0625, 0.125, 0.25]),
        np.array([10.0, 8.0, 9.0, 4.0]),
        0.0,
    )
    assert result.stage_index.tolist() == [0, 1, 3]
    assert result.raw_stage.sum() == pytest.approx(6.0)
    assert result.weighted_stage.sum() == pytest.approx(10.0 - result.area_mpa)
    assert result.negative_stage[1] < 0
```

- [ ] **Step 2: Run Q2-Q3 and observe missing implementations**

Run: `PYTHONPATH="$PWD/src:$PWD" /home/ww/miniconda3/bin/python3.13 -m pytest -q tests/test_cai_order_mechanism.py -k 'q2 or q3'`

- [ ] **Step 3: Implement exact 500-row filter, selected-main equality, typed semicolon parsing, event rows, and archive outputs**

```python
completion = costs[1:]
delta = errors[:-1] - errors[1:]
weighted = (1.0 - completion / 0.25) * delta
stage = np.searchsorted(STAGE_EDGES, completion, side="left") - 1
assert abs(delta.sum() - (errors[0] - errors[-1])) <= 1e-8
assert abs(weighted.sum() - (errors[0] - area)) <= 1e-8
```

- [ ] **Step 4: Implement repeat-first RANDOM aggregation, specimen/domain summaries, stage contrasts, and terminal-set comparison**

```python
repeat_mean = frame.groupby(["method", "specimen_key", "stage"], sort=True)[metric].mean()
domain_mean = repeat_mean.groupby(["method", "dataset_id", "stage"], sort=True).mean()
estimate = domain_mean.groupby(["method", "stage"], sort=True).mean()
```

- [ ] **Step 5: Run Q2-Q3 and commit archive derivation**

Commit: `git commit -m "feat(cai): derive archived timing evidence"`

### Task 3: Deterministic Fixed-Set Order Plan

**Files:**
- Create: `scripts/cai_order_mechanism/orders.py`
- Modify: `tests/test_cai_order_mechanism.py`

**Interfaces:**
- Consumes: parsed Proposed episodes and `NativeCellGrid`
- Produces: `permuted_order(cells: Sequence[int], specimen_key: str, repeat: int, namespace: str) -> tuple[int, ...]`
- Produces: `build_order_plan(episodes: Sequence[Episode], namespace: str) -> list[dict[str, object]]`
- Produces: `prefix_state(order: Sequence[int], grid: NativeCellGrid) -> Iterator[PrefixState]`
- Produces: `prefix_cache_key(binding: CacheBinding, specimen_key: str, mask_bits: int, cost32: np.float32) -> str`

- [ ] **Step 1: Add Q4-Q6 tests for label-free SHA256 orders, duplicate retention, integer pixel clocks, and cache-key sensitivity**

```python
def test_q4_permutation_is_sha256_sort_of_sorted_set():
    expected = tuple(sorted({7, 2, 9}, key=lambda c: hashlib.sha256(
        f"NS|sample|0|{c}".encode("ascii")
    ).digest()))
    assert permuted_order([9, 2, 7], "sample", 0, "NS") == expected
```

- [ ] **Step 2: Run Q4-Q6 and observe missing functions**

- [ ] **Step 3: Implement native/reverse/five-permutation plan rows and atomic plan-plus-SHA lock**

```python
score = hashlib.sha256(f"{namespace}|{specimen_key}|{repeat}|{cell}".encode("ascii")).digest()
order = tuple(sorted(sorted(set(cells)), key=lambda cell: (score_for(cell), cell)))
```

- [ ] **Step 4: Implement prefix masks, exact pixel counts, float32 model costs, float64 metric costs, boundary audit, and full cache binding**

```python
pixels = np.cumsum([grid.cells[cell].pixel_count for cell in order], dtype=np.int64)
cost64 = pixels.astype(np.float64) / float(np.prod(grid.native_shape))
cost32 = cost64.astype(np.float32)
mask_bits = np.bitwise_or.accumulate(np.asarray([1 << cell for cell in order], dtype=np.uint64))
```

- [ ] **Step 5: Run Q4-Q6 and commit the plan layer**

Commit: `git commit -m "feat(cai): lock deterministic order interventions"`

### Task 4: Bounded Predictor Replay And Durable Cache

**Files:**
- Create: `scripts/cai_order_mechanism/replay.py`
- Modify: `tests/test_cai_order_mechanism.py`

**Interfaces:**
- Consumes: `ValidFeatures`, locked order plan, predictor checkpoint, cache keys
- Produces: `PredictorCache.load/save_atomic`
- Produces: `ReplayEngine.evaluate(requests: Sequence[PrefixRequest]) -> np.ndarray`
- Produces: `run_reproduction_trial(context: TaskContext, engine: ReplayEngine) -> RuntimeLock`
- Produces: `infer_orders(context: TaskContext, resume: bool) -> dict[str, Path]`

- [ ] **Step 1: Add Q7 tests with a fake predictor for batching, charge-before-call, cache reuse, interrupted batch recovery, and single fallback**

```python
def test_q7_forward_is_precharged_and_completed_atomically(tmp_path):
    ledger = ResourceLedger(tmp_path / "events.jsonl", cap=10)
    engine = ReplayEngine(FakePredictor(), ledger, PredictorCache.empty(), batch_size=4)
    engine.evaluate(make_requests(3))
    events = read_jsonl(ledger.path)
    assert [event["state"] for event in events] == ["INTENDED", "COMPLETE"]
    assert events[0]["evaluated_rows"] == 3
```

- [ ] **Step 2: Run Q7 without any real checkpoint forward and confirm failure**

- [ ] **Step 3: Implement cache arrays, model tensor assembly, no-grad/eval float32 inference, atomic batch commits, and resource events**

```python
with torch.inference_mode():
    prediction = model(surface, cscan, measured, cost=cost).detach().cpu().numpy()
```

- [ ] **Step 4: Implement three-case reproduction, one batch-one diagnostic, engine lock, all-native reproduction, 350 trajectories, endpoint invariance, and 12 uncached checks**

```python
trial = compare_native(engine, FIXED_CASE_KEYS, atol_prediction=1e-4, atol_cost=1e-12)
if not trial.passed:
    trial = compare_native(engine.with_batch_size(1), FIXED_CASE_KEYS, atol_prediction=1e-4, atol_cost=1e-12)
if not trial.passed:
    raise RuntimeError("native reproduction failed after the only diagnostic fallback")
```

- [ ] **Step 5: Run Q7 and commit replay infrastructure before the real run**

Commit: `git commit -m "feat(cai): add bounded prefix replay engine"`

### Task 5: Order Metrics, Bootstrap, Curves, And Matched Quality

**Files:**
- Modify: `scripts/cai_order_mechanism/analysis.py`
- Modify: `tests/test_cai_order_mechanism.py`

**Interfaces:**
- Consumes: archive outputs, reorder trajectories, fixed bootstrap matrix
- Produces: `analyze_results(context: TaskContext) -> dict[str, Path]`
- Produces: `curve_metrics(rows: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame`
- Produces: `first_quality_crossing(costs: np.ndarray, values: np.ndarray, target: float) -> dict[str, object]`

- [ ] **Step 1: Add Q8 tests for contrast signs, repeat-before-domain aggregation, all-one bootstrap identity, RMSE/R2 estimands, recrossing, null and negative savings**

```python
def test_q8_primary_order_contrast_positive_means_native_better():
    native = np.array([2.0, 4.0])
    permutations = np.array([[3.0, 5.0], [1.0, 7.0]])
    assert np.mean(permutations, axis=0) - native == pytest.approx([0.0, 2.0])
```

- [ ] **Step 2: Run Q8 and confirm the analysis additions fail**

- [ ] **Step 3: Implement order specimen/domain summaries, paired fixed-weight intervals, separate family event grids, cap metrics, and explicit difference definitions**

```python
permuted = frame[frame.variant == "PERMUTED"].groupby("specimen_key").area_mpa.mean()
native = frame[frame.variant == "NATIVE_REPLAY"].set_index("specimen_key").area_mpa
primary = permuted - native
draws = bootstrap_domain_equal(primary.to_numpy(), weights, domains)
```

- [ ] **Step 4: Implement integer q=41..61 plus 41.69001007080078 matched-quality rows without threshold or permutation selection**

```python
quality_targets = [*map(float, range(41, 62)), 41.69001007080078]
rows = [compare_first_crossings(curves, family, target) for family in FAMILIES for target in quality_targets]
```

- [ ] **Step 5: Run Q8 and commit analysis**

Commit: `git commit -m "feat(cai): compute order mechanism effects"`

### Task 6: Phase CLI And Resume State

**Files:**
- Create: `scripts/cai_order_mechanism/run.py`
- Modify: `tests/test_cai_order_mechanism.py`

**Interfaces:**
- Consumes: all phase functions from Tasks 1-5
- Produces: `main(argv: Sequence[str] | None = None) -> int`
- Produces: commands `prepare`, `derive`, `plan`, `infer`, `analyze`, `report`, `verify`, `all`

- [ ] **Step 1: Add Q9 tests for command ordering, phase signatures, output-hash reuse, and one-resume limit**

```python
def test_q9_all_dispatches_fixed_phase_order(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(run, "PHASE_FUNCTIONS", {name: lambda c, n=name: seen.append(n) for name in run.PHASES})
    assert run.main(["all", "--root", str(tmp_path), "--scope", str(scope)]) == 0
    assert seen == ["prepare", "derive", "plan", "infer", "analyze", "report", "verify"]
```

- [ ] **Step 2: Run Q9 and observe the missing CLI**

- [ ] **Step 3: Implement argparse, environment thread locks, state signatures, atomic completion records, and deterministic reuse checks**

```python
PHASES = ("prepare", "derive", "plan", "infer", "analyze", "report", "verify")
signature = sha256_bytes(canonical_json({"scope": context.scope_sha256, "phase": phase, "inputs": input_hashes}))
if state.reusable(phase, signature, expected_output_hashes):
    return 0
```

- [ ] **Step 4: Run Q1-Q9 and commit the orchestration layer**

Run: `PYTHONPATH="$PWD/src:$PWD" /home/ww/miniconda3/bin/python3.13 -m pytest -q tests/test_cai_order_mechanism.py`

Commit: `git commit -m "feat(cai): orchestrate order analysis phases"`

### Task 7: Publication Figures, Offline Report, And Handoff Documents

**Files:**
- Create: `scripts/cai_order_mechanism/reporting.py`
- Modify: `tests/test_cai_order_mechanism.py`
- Generate: `results/cai_agent_v3/order_mechanism/r1_37b3c404/figures/*.{png,svg}`
- Generate: `results/cai_agent_v3/order_mechanism/r1_37b3c404/cases/*`
- Generate: `results/cai_agent_v3/order_mechanism/r1_37b3c404/index.html`
- Generate: `artifacts/cai_agent_v3/order_mechanism/r1_37b3c404/{FINDINGS_ZH.md,METHODS_FACTS.md,FIGURE_CAPTIONS.md,REPRODUCE.md,CODEX_HANDOFF_CAI_ORDER_MECHANISM.md}`

**Interfaces:**
- Consumes: cached final tables only; performs zero predictor calls
- Produces: `render_report(context: TaskContext) -> dict[str, Path]`

- [ ] **Step 1: Read and apply the `nature-figure` skill before figure implementation**

- [ ] **Step 2: Add Q10 tests for required output inventory, five-family ceiling, PNG/SVG pairs, offline HTML, fixed cases, bilingual captions, and Chinese findings length**

```python
def test_q10_report_is_offline_and_uses_fixed_cases(rendered_report):
    html = rendered_report.index.read_text(encoding="utf-8")
    assert "https://" not in html and "http://" not in html
    assert set(rendered_report.case_keys) == set(FIXED_CASE_KEYS)
    assert len(rendered_report.figure_families) <= 5
```

- [ ] **Step 3: Implement restrained publication styling and deterministic plots from supplied tables**

```python
for stem, draw in FIGURE_FAMILIES.items():
    figure = draw(tables)
    figure.savefig(figures / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(figures / f"{stem}.svg", bbox_inches="tight")
```

- [ ] **Step 4: Implement native/reverse/permutation-0 case figures, offline track switch, linked inventory, and evidence-bound prose**

```python
case_rows = trajectories.query("specimen_key == @key and ((variant != 'PERMUTED') or repeat == 0)")
assert set(case_rows.variant) == {"NATIVE_REPLAY", "REVERSE", "PERMUTED"}
html_text = render_offline_html(tables=tables, figures=figures, cases=FIXED_CASE_KEYS)
```

- [ ] **Step 5: Run Q10 and commit reporting code**

Commit: `git commit -m "feat(cai): render order mechanism report"`

### Task 8: Real Execution, Acceptance, Visual QA, And Publication

**Files:**
- Create: `scripts/cai_order_mechanism/verify.py`
- Modify: `tests/test_cai_order_mechanism.py`
- Generate: all required files under the exact result and artifact roots

**Interfaces:**
- Consumes: complete local results and artifacts
- Produces: `verify_release(context: TaskContext) -> dict[str, object]`
- Produces: `acceptance_results.json`, `release_manifest.json`, and after Git publication `PUBLICATION_RECEIPT.json`

- [ ] **Step 1: Implement Q1-Q10 artifact verification, resource reconciliation, local status `LOCAL_VERIFIED_PENDING_PUSH`, and non-self-referential release manifest**

```python
checks = [check(context) for check in ACCEPTANCE_CHECKS]
if not all(row["passed"] for row in checks):
    raise RuntimeError("acceptance verification failed")
write_release_manifest(context, status="LOCAL_VERIFIED_PENDING_PUSH", exclude={"PUBLICATION_RECEIPT.json"})
```

- [ ] **Step 2: Run the focused test file once and Ruff once before real inference**

Run: `PYTHONPATH="$PWD/src:$PWD" /home/ww/miniconda3/bin/python3.13 -m pytest -q tests/test_cai_order_mechanism.py`

Run: `/home/ww/miniconda3/bin/python3.13 -m ruff check scripts/cai_order_mechanism tests/test_cai_order_mechanism.py`

- [ ] **Step 3: Execute the full authorized CLI with four CPU threads and at most one available GPU**

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
PYTHONPATH="$PWD/src:$PWD" \
/home/ww/miniconda3/bin/python3.13 -m scripts.cai_order_mechanism.run all \
  --root "$PWD" \
  --scope docs/cai/order_mechanism/ORDER_MECHANISM_SCOPE.json
```

- [ ] **Step 4: Run final acceptance once, inspect the offline report in a browser, and inspect a contact sheet plus three key figures for clipping, overlap, legibility, and scientific consistency**

```bash
PYTHONPATH="$PWD/src:$PWD" /home/ww/miniconda3/bin/python3.13 \
  -m scripts.cai_order_mechanism.run verify --root "$PWD" \
  --scope docs/cai/order_mechanism/ORDER_MECHANISM_SCOPE.json
```

- [ ] **Step 5: Review the full diff and map every package requirement to code, test, or output evidence**

Run: `git diff --check && git status --short && git diff --stat 37b3c404 -- scripts/cai_order_mechanism tests/test_cai_order_mechanism.py docs/cai/order_mechanism results/cai_agent_v3/order_mechanism artifacts/cai_agent_v3/order_mechanism`

- [ ] **Step 6: Selectively commit and push the content, then fetch and prove local HEAD, upstream HEAD, and remote ref equality**

```bash
git add scripts/cai_order_mechanism tests/test_cai_order_mechanism.py docs/cai/order_mechanism \
  results/cai_agent_v3/order_mechanism/r1_37b3c404 \
  artifacts/cai_agent_v3/order_mechanism/r1_37b3c404
git commit -m "feat(cai): publish order mechanism analysis"
git push origin research/cai-vlm-agent-v3-controlled-reuse
git fetch origin research/cai-vlm-agent-v3-controlled-reuse
test "$(git rev-parse HEAD)" = "$(git rev-parse @{upstream})"
test "$(git rev-parse HEAD)" = "$(git ls-remote origin refs/heads/research/cai-vlm-agent-v3-controlled-reuse | cut -f1)"
```

- [ ] **Step 7: Write the publication receipt with content SHA and remote checks, commit it separately, push, and repeat remote equality/sample checks**

```bash
git add artifacts/cai_agent_v3/order_mechanism/r1_37b3c404/PUBLICATION_RECEIPT.json
git commit -m "docs(cai): record order mechanism publication"
git push origin research/cai-vlm-agent-v3-controlled-reuse
git fetch origin research/cai-vlm-agent-v3-controlled-reuse
test "$(git rev-parse HEAD)" = "$(git rev-parse @{upstream})"
```

Commit: `git commit -m "docs(cai): record order mechanism publication"`
