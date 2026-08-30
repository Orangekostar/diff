# Damage-to-Failure Response P0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a checksum-bound P0 authority package that decides whether the exact 276-specimen cohort can be paired with raw CAI traces without guessing and whether published peak strength is reproducible.

**Architecture:** Add a focused `cmc_bbdm.damage_response` package whose P0 path reads external data as immutable regular-file snapshots, joins only exact specimen/domain identities, derives audit-only trace quantities, and writes an atomic replayable package. No estimator, model-selection, feature-learning, P1 target-selection, or training code is authorized in this plan.

**Tech Stack:** Python 3.10+, NumPy, openpyxl, PyYAML, pytest, Ruff, Git, SHA-256.

---

### Task 1: Freeze public-source and local-authority facts

**Files:**
- Create: `artifacts/damage_to_failure_response/LITERATURE_NOVELTY_LEDGER.md`
- Create: `artifacts/damage_to_failure_response/P0_SOURCE_DISCOVERY.md`

- [ ] **Step 1: Verify primary public records**

Use official DOI, publisher, and dataset pages to verify the named Mack, Hasebe,
Yang, Liu, Cai, IJMS, and Mezeix records. Record title, authors, year, venue,
DOI/PII, accessible source, task, inputs, response target, split protocol, and
the exact difference from the proposed route. Mark inaccessible full-method
details `UNKNOWN_FROM_ACCESSIBLE_PRIMARY_SOURCE`.

- [ ] **Step 2: Run the same-day closest-work search**

Search the eight preregistered public topic families with public keywords only.
Deduplicate by DOI/title, exclude policy-prohibited sources, and distinguish
search evidence from inference. The ledger must end with this bounded position:

```text
Novelty status: SEARCHED_NOT_ASSUMED
Direct novelty claim authorized: NO
P0/P1 scientific question retained: YES
Reason: closest work determines differentiation, while empirical gates determine support.
```

- [ ] **Step 3: Record machine-local authority facts**

Write exact base commit, feature-bank SHA, external dataset ID/version, all
locally available source-file hashes, and the fact that the historical full
tree lacks Git metadata. Do not publish personal absolute paths; label external
locations `local:historical_full_tree` and `local:hasebe_v3_root`.

- [ ] **Step 4: Verify documentation integrity**

Run:

```bash
rg -n 'UNKNOWN|SEARCHED_NOT_ASSUMED|Direct novelty claim authorized' \
  artifacts/damage_to_failure_response/LITERATURE_NOVELTY_LEDGER.md
rg -n '/(home|Users)/|[A-Z]:\\\\' artifacts/damage_to_failure_response
git diff --check
```

Expected: explicit unknowns are visible, the bounded novelty status is present,
no private absolute path is committed, and the diff check is clean.

### Task 2: Define fail-closed P0 contracts

**Files:**
- Create: `src/cmc_bbdm/damage_response/__init__.py`
- Create: `src/cmc_bbdm/damage_response/contracts.py`
- Test: `tests/test_damage_response_input_boundary.py`
- Test: `tests/test_damage_response_gates.py`

- [ ] **Step 1: Write failing input-boundary tests**

```python
from cmc_bbdm.damage_response.contracts import InputRole, validate_input_names


def test_post_cai_image_is_forbidden_as_input() -> None:
    names = ("laminate", "post_cai_image")
    try:
        validate_input_names(names)
    except ValueError as error:
        assert "post-CAI" in str(error)
    else:
        raise AssertionError("post-CAI input was accepted")


def test_true_response_is_never_a_deployable_input() -> None:
    assert InputRole.TRUE_CAI_TRACE.deployable is False
    assert InputRole.TRUE_PEAK_STRENGTH.deployable is False
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  tests/test_damage_response_input_boundary.py
```

Expected: collection fails because `cmc_bbdm.damage_response` does not exist.

- [ ] **Step 3: Implement immutable roles and gate transitions**

`contracts.py` must define exact primary domains/counts, conversion constants,
forbidden input names, and these statuses:

```python
class StageStatus(str, Enum):
    P0_GO = "P0_GO"
    P0_NO_GO = "P0_NO_GO"
    P0_REQUIRES_HUMAN_REVIEW = "P0_REQUIRES_HUMAN_REVIEW"
    NOT_RUN_NOT_AUTHORIZED = "NOT_RUN_NOT_AUTHORIZED"


PRIMARY_COUNTS = {
    "74t7kcdgkr": 45,
    "cgtnjyggtm": 49,
    "w68dtmpfyf": 43,
    "xcmzfsbd9t": 59,
    "yfxyg8jm46": 42,
    "ykhs7s2dck": 38,
}
LOAD_KN_PER_VOLT = 25.0
DISPLACEMENT_MM_PER_VOLT = 1.0
POST_CAI_IMAGE_INPUT_FORBIDDEN = True
```

The gate function accepts only explicit counts/reconciliation/source-binding
facts. Any missing domain, guessed identity, unbound source, or failed peak
reconciliation returns `P0_NO_GO`; fewer than 20 exact pairs in any domain or
more than 20% missing primary channels returns
`P0_REQUIRES_HUMAN_REVIEW`; only all-clear facts return `P0_GO`.

- [ ] **Step 4: Add and run gate tests**

Test every fail-closed reason independently and the one valid transition. Run:

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  tests/test_damage_response_input_boundary.py \
  tests/test_damage_response_gates.py
```

Expected: both files pass.

### Task 3: Snapshot external authority safely

**Files:**
- Create: `src/cmc_bbdm/damage_response/authority.py`
- Test: `tests/test_damage_response_authority.py`

- [ ] **Step 1: Write failing authority tests**

```python
def test_snapshot_rejects_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("a,b\n1,2\n", encoding="ascii")
    link = tmp_path / "link.csv"
    link.symlink_to(source)
    with pytest.raises(AuthorityError, match="regular file"):
        snapshot_file(link, max_bytes=1024)


def test_snapshot_binds_sha_and_size(tmp_path: Path) -> None:
    path = tmp_path / "source.csv"
    path.write_bytes(b"a,b\n1,2\n")
    record = snapshot_file(path, max_bytes=1024)
    assert record.size == 8
    assert record.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
```

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  tests/test_damage_response_authority.py
```

Expected: import failure for the missing authority module.

- [ ] **Step 3: Implement descriptor-based immutable snapshots**

Use `os.open` with `O_NOFOLLOW`, require a regular file, enforce per-file byte
limits, read once through the descriptor, compare pre/post `fstat` identity,
and return frozen records containing logical source label, size, and SHA-256.
External absolute paths may be used at runtime but must never be serialized;
manifests store dataset-relative paths only.

- [ ] **Step 4: Verify GREEN**

Run the authority tests and then Ruff on the new package/tests. Expected: pass.

### Task 4: Enforce exact specimen/domain pairing

**Files:**
- Create: `src/cmc_bbdm/damage_response/pairing.py`
- Test: `tests/test_damage_response_pairing.py`

- [ ] **Step 1: Write failing exact-pair tests**

```python
def test_pairing_rejects_order_only_match() -> None:
    features = (FeatureIdentity("c8-2", "74t7kcdgkr"),)
    traces = (TraceIdentity("c8-3", "74t7kcdgkr", "a" * 64),)
    with pytest.raises(PairingError, match="exact identity"):
        pair_exact(features, traces)


def test_pairing_rejects_duplicate_specimen_domain() -> None:
    item = TraceIdentity("c8-2", "74t7kcdgkr", "a" * 64)
    with pytest.raises(PairingError, match="duplicate"):
        pair_exact((FeatureIdentity("c8-2", "74t7kcdgkr"),), (item, item))
```

- [ ] **Step 2: Verify RED**

Run `tests/test_damage_response_pairing.py`; expected import failure.

- [ ] **Step 3: Implement canonical identity extraction and exact join**

Load specimen/domain vectors from the frozen feature bank without pickle,
normalize only case and surrounding whitespace, validate the exact 276 roster,
and join on `(specimen_id, domain_id)`. Trace identities must additionally carry
their raw-file SHA. Reject fuzzy names, numeric-only order, duplicate keys,
missing keys, unexpected primary keys, and any inferred domain.

- [ ] **Step 4: Verify GREEN and real-bank roster**

Add a real-bank test bound to SHA
`f2a69f0da75e20880202d7fc4a6a92f979978406ec21f9d83e4bc8db07fb72a8`.
Expected: 276 unique keys with domain counts `45,49,43,59,42,38`.

### Task 5: Decode raw CAI traces and reconcile published peaks

**Files:**
- Create: `src/cmc_bbdm/damage_response/raw_cai.py`
- Create: `src/cmc_bbdm/damage_response/targets.py`
- Test: `tests/test_damage_response_raw_cai.py`
- Test: `tests/test_damage_response_targets.py`

- [ ] **Step 1: Write failing raw-decoder tests**

Use small fixture CSV payloads with the registered channels. Assert exact row
count, finite-channel counts, peak row, `load_kN = Load[V] * 25`, and
`extension_mm = Extension[V]`. Add failures for a missing load channel,
duplicate header, non-finite peak, and fewer than two finite rows.

- [ ] **Step 2: Verify RED**

Run both test files; expected missing-module import failures.

- [ ] **Step 3: Implement schema-preserving decoding**

Decode only the source-documented delimiter/encoding. Preserve original header
spelling in QC output and map registered aliases through an explicit table.
Compute audit quantities without smoothing or curve repair. Four gauge channels
are retained as raw numeric arrays and their unit status is
`STRAIN_UNIT_UNRESOLVED` until independently authorized.

- [ ] **Step 4: Implement physical conversion and global reconciliation**

For each exact pair compute:

```python
load_kn = load_volts * 25.0
stress_mpa = load_kn * 1000.0 / (width_mm * thickness_mm)
```

Derive one global absolute tolerance from the decimal precision of the published
workbook, never per specimen. Reconciliation compares maximum absolute raw
stress with published CAI strength and records signed/absolute error, tolerance,
and pass/fail per row. JIS modulus and maximum strain remain unauthorized while
the strain unit/sign audit is unresolved.

- [ ] **Step 5: Verify GREEN**

Run the two focused test files. Expected: all parser, conversion, precision, and
unresolved-unit tests pass.

### Task 6: Write atomic P0 artifacts and replay them

**Files:**
- Create: `src/cmc_bbdm/damage_response/artifacts.py`
- Test: `tests/test_damage_response_artifacts.py`

- [ ] **Step 1: Write failing package tests**

```python
REQUIRED = {
    "summary.json",
    "REPORT.md",
    "pairing_manifest.csv",
    "source_hashes.csv",
    "raw_trace_qc.csv",
    "published_peak_reconciliation.csv",
    "strain_unit_audit.csv",
    "post_cai_image_manifest.csv",
}


def test_replay_rejects_extra_file(complete_package: Path) -> None:
    (complete_package / "extra.txt").write_text("x", encoding="ascii")
    with pytest.raises(ArtifactError, match="membership"):
        replay_p0(complete_package)
```

- [ ] **Step 2: Verify RED**

Run `tests/test_damage_response_artifacts.py`; expected missing module import.

- [ ] **Step 3: Implement atomic writer and exact replay**

Write required payloads into a sibling temporary directory, create a sorted
`artifact_manifest.json`, create `CHECKSUMS.sha256` over every payload plus the
manifest, then rename once into a nonexistent destination. Replay rejects
symlinks, missing files, extra files, schema drift, size drift, or hash drift.

- [ ] **Step 4: Verify GREEN and byte identity**

Generate two packages from identical in-memory records in separate temporary
roots and assert each corresponding file is byte-identical. Expected: pass.

### Task 7: Orchestrate the P0 audit without training

**Files:**
- Create: `src/cmc_bbdm/damage_response/pipeline.py`
- Create: `scripts/run_damage_response.py`
- Create: `paper_v3/configs/damage_to_failure_response.yaml`
- Test: `tests/test_damage_response_p0_pipeline.py`

- [ ] **Step 1: Write failing pipeline tests**

Test that `audit-p0` requires explicit external roots, refuses an already
existing output directory, emits downstream statuses exactly
`NOT_RUN_NOT_AUTHORIZED`, and never imports scikit-learn or torch. Test CLI exit
code 0 for every completed audit, including NO-GO/review, and exit code 1 only
for integrity, schema, or runtime errors.

- [ ] **Step 2: Verify RED**

Run `tests/test_damage_response_p0_pipeline.py`; expected missing pipeline/CLI.

- [ ] **Step 3: Implement minimal orchestration**

The CLI commands are:

```text
run_damage_response.py audit-p0 --config PATH --legacy-root PATH --hasebe-v3-root PATH --output PATH
run_damage_response.py replay-p0 --path PATH
```

`audit-p0` calls only authority, pairing, raw decoding, target audit, gate, and
artifact functions. It writes no cache, model, feature array, or raw mirror.
The YAML fixes dataset/version, primary domains/counts, calibrations, review
thresholds, and P0 failure reasons.

- [ ] **Step 4: Verify GREEN**

Run all P0 tests. Expected: pass with synthetic fixtures and the frozen compact
feature-bank roster.

### Task 8: Execute P0, preserve the gate, and commit evidence

**Files:**
- Create: `results/damage_to_failure_response/p0_data_audit/*`
- Create: `artifacts/damage_to_failure_response/P0_DATA_AND_AUTHORITY_AUDIT.md`
- Create: `artifacts/damage_to_failure_response/P0_GO_NO_GO.md`

- [ ] **Step 1: Execute the real audit once**

Run `audit-p0` against the verified external roots and the frozen compact
feature bank. Do not add or alter a pairing after viewing the outcome.

- [ ] **Step 2: Replay and independently verify checksums**

Run:

```bash
PYTHONPATH=src python scripts/run_damage_response.py replay-p0 \
  --path results/damage_to_failure_response/p0_data_audit
(cd results/damage_to_failure_response/p0_data_audit && \
  sha256sum -c CHECKSUMS.sha256)
```

Expected: replay passes and every checksum is OK.

- [ ] **Step 3: Enforce authorization**

If status is not `P0_GO`, do not create P1/P2/P3 model modules or results. Record
all later stages as `NOT_RUN_NOT_AUTHORIZED`. If status is `P0_GO`, first write a
separate P1 implementation plan and its test-first contract before adding P1
code.

- [ ] **Step 4: Run verification**

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_damage_response_*.py
python -m ruff check src/cmc_bbdm/damage_response tests/test_damage_response_*.py scripts/run_damage_response.py
git diff --check
git diff --name-only 3951f71f28b6efdf8c74eea0fe274b2a78a9cd57 -- \
  results/p1_full_field_oracle results/p3_spatial_specificity results/p5_sparse_scan \
  results/mvd results/mva results/mavis results/mavis_science_closure \
  artifacts/aei_information_hierarchy
```

Expected: focused tests and Ruff pass, diff check is clean, and frozen-path diff
is empty.

- [ ] **Step 5: Commit the P0 stage**

```bash
git add src/cmc_bbdm/damage_response scripts/run_damage_response.py \
  paper_v3/configs/damage_to_failure_response.yaml \
  tests/test_damage_response_*.py results/damage_to_failure_response \
  artifacts/damage_to_failure_response docs/superpowers/plans
git commit -m "research: audit damage-to-failure response authority"
```

Expected: one scoped P0 implementation/evidence commit; no raw data or large
model artifact is staged.

## Plan self-review

- Spec coverage: all P0 output files, hard failures, review trigger, exact
  pairing, raw trace QC, peak reconciliation, strain-unit boundary, post-CAI
  exclusion, source hashing, replay, and frozen-path gate are assigned.
- Placeholder scan: no deferred implementation marker is present.
- Type consistency: stage statuses, identity keys, source hashes, and artifact
  names are consistent across tasks.
- Scope check: P1-P5 implementation is intentionally excluded and requires a
  new plan only after authorization.
