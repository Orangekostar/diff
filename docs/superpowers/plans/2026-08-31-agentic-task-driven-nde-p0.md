# Agentic Task-Driven NDE P0 Implementation Plan

**Goal:** Produce a checksum-bound P0 package that decides whether released
surface images can be mapped to the frozen legal 8x8 ultrasound action space
without using hidden C-scan content, CAI, oracle values, manual target
alignment, or guessed specimen identity.

**Architecture:** Add a focused `cmc_bbdm.agentic_nde` P0 pipeline. It reads
external raw images through immutable snapshots, joins exact specimen/domain
identities to frozen compact-repository authorities, validates a typed
source-supported transform, renders legal actions, and emits a deterministic
GO/NO-GO audit. It contains no visual encoder, VLM call, estimator, training,
target-dependent registration, or P1 logic.

**Tech stack:** Python 3.10+, standard library, NumPy, Pillow, pandas/pyarrow
only for frozen structured data, PyYAML, pytest, Ruff, Git, SHA-256.

---

### Task 1: Freeze source-backed literature and external-code facts

**Files:**

- Create: `artifacts/agentic_task_driven_nde/EXTERNAL_REPOSITORY_AUDIT.md`
- Create: `artifacts/agentic_task_driven_nde/LITERATURE_NOVELTY_LEDGER.md`

- [ ] Verify current official repositories, commits, licenses, inspected files,
  declared reuse mode, and copied-code status for DriveAgent-R1, Qwen2.5-VL,
  US-VLA, ActiveVLA, GroundingDINO, SAM2, RoboSpection, and TADRED.
- [ ] Treat unclear licenses as `NO_CODE_REUSE_AUTHORIZED`; do not copy code.
- [ ] Search primary sources for Fuentes 2020, TADRED 2024, MoRAL 2026,
  US-VLA, DriveAgent/ActiveVLA, and RoboSpection. Record modality, task,
  action space, active/task-conditioned/hardware/code status, source date, and
  source URL. Do not use MDPI sources.
- [ ] Record facts separately from bounded inferences. Do not use a novelty
  superlative or a direct first claim.
- [ ] End the ledger with the exact status:

```text
Novelty status: SEARCHED_NOT_ASSUMED
Direct novelty claim authorized: NO
Empirical route retained: YES, subject to P0 and P1 gates.
```

**Validation:**

```bash
rg -n 'current commit|license|copied code|SEARCHED_NOT_ASSUMED' \
  artifacts/agentic_task_driven_nde
rg -n '/(home|Users)/|[A-Z]:\\\\' artifacts/agentic_task_driven_nde
git diff --check
```

Expected: current source identities and unknowns are explicit, no private
absolute path is committed, and no external code has been copied.

### Task 2: Define P0 contracts and fail-closed gates

**Files:**

- Create: `src/cmc_bbdm/agentic_nde/__init__.py`
- Create: `src/cmc_bbdm/agentic_nde/contracts.py`
- Test: `tests/test_agentic_nde_registration_contract.py`
- Test: `tests/test_agentic_nde_gate.py`

- [ ] Write failing tests for immutable primary domain counts, typed frames,
  orientation enumeration, evidence classes A/B/C, canonical transform hashes,
  forbidden evidence roles, P0 statuses, and downstream authorization.
- [ ] Verify RED because `cmc_bbdm.agentic_nde` does not exist.
- [ ] Implement frozen dataclasses/enums for source records, frame geometry,
  registration evidence, transform status, evidence role, and stage status.
- [ ] Define exact primary counts:

```python
PRIMARY_COUNTS = {
    "74t7kcdgkr": 45,
    "cgtnjyggtm": 49,
    "w68dtmpfyf": 43,
    "xcmzfsbd9t": 59,
    "yfxyg8jm46": 42,
    "ykhs7s2dck": 38,
}
```

- [ ] Reject transforms whose evidence roles include hidden C-scan pixels,
  C-scan masks, damage centroid, CAI, oracle value, target-domain labels, or
  manual target alignment.
- [ ] Gate GO only for per-domain coverage >=90%, total coverage >=240, exact
  identity/hash binding, resolved orientation, legal deterministic transform,
  deployable evidence, and successful replay. Otherwise emit the exact P0
  NO-GO reason and set P1-P4 to `NOT_RUN_NOT_AUTHORIZED`.

**Validation:**

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  tests/test_agentic_nde_registration_contract.py \
  tests/test_agentic_nde_gate.py
```

### Task 3: Snapshot external surface authority safely

**Files:**

- Create: `src/cmc_bbdm/agentic_nde/authority.py`
- Test: `tests/test_agentic_nde_surface_authority.py`

- [ ] Write failing tests that reject symlinks, non-regular files, oversize
  files, descriptor identity changes, duplicate logical paths, absolute
  serialized paths, and SHA mismatches.
- [ ] Verify RED for the missing module.
- [ ] Implement descriptor-based immutable snapshots using `O_NOFOLLOW`,
  pre/post `fstat`, byte limits, and SHA-256.
- [ ] Store only a logical root label plus dataset-relative POSIX path, never an
  external absolute path.
- [ ] Parse exact external authority rows keyed by `(domain_id, specimen_id)`;
  reject ordering, fuzzy matching, case-changing aliases, and duplicates.
- [ ] Bind the exact released surface path/hash, source C-scan path/hash,
  dataset ID/version, frozen registered crop identity/hash, and CAI row identity.

**Validation:**

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  tests/test_agentic_nde_surface_authority.py
python -m ruff check src/cmc_bbdm/agentic_nde \
  tests/test_agentic_nde_surface_authority.py
```

### Task 4: Audit surface files and geometry

**Files:**

- Create: `src/cmc_bbdm/agentic_nde/surface_qc.py`
- Test: `tests/test_agentic_nde_surface_qc.py`

- [ ] Write failing tests for PNG/JPEG decoding, exact dimensions, channel
  mode, truncated input, unexpected alpha, embedded metadata capture, specimen
  boundary evidence, annotation detection status, orientation evidence, and
  physical-extent status.
- [ ] Verify RED for the missing module.
- [ ] Implement read-only Pillow inspection with decompression-bomb protection,
  explicit accepted formats/modes, no EXIF autorotation, and no image mutation.
- [ ] Record raw orientation metadata rather than silently applying it.
- [ ] Distinguish `OBSERVED`, `SOURCE_DECLARED`, `INFERRED`, and `UNKNOWN` QC
  facts. Unknown physical extent or orientation remains unknown and blocks a
  geometry claim that depends on it.
- [ ] Test the primary real roster without committing images.

**Validation:**

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  tests/test_agentic_nde_surface_qc.py
```

### Task 5: Implement deterministic typed registration

**Files:**

- Create: `src/cmc_bbdm/agentic_nde/registration.py`
- Test: `tests/test_agentic_nde_registration.py`
- Test: `tests/test_agentic_nde_registration_leakage.py`

- [ ] Write failing synthetic tests for identity, scale/offset, all legal
  rotations/reflections, forward/inverse point mapping, boundary corners,
  out-of-frame rejection, degenerate geometry, box mapping, and transform hash
  stability.
- [ ] Write failing leakage tests that reject hidden C-scan, C-scan mask,
  damage centroid, CAI, oracle, manual target alignment, and target-domain fitted
  parameters in transform construction.
- [ ] Verify RED for the missing registration module.
- [ ] Implement an affine geometry transform restricted to explicitly
  enumerated axis swaps/reflections plus source-supported scale/offset. Store
  canonical decimal/string parameters and hash their sorted canonical JSON.
- [ ] Require evidence class A or B unless a separately preregistered class-C
  source-only procedure exists. Do not implement class C merely to rescue P0.
- [ ] Provide deterministic point and axis-aligned box mapping; inversion must
  reproduce legal points within a fixed numeric tolerance.

**Validation:**

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  tests/test_agentic_nde_registration.py \
  tests/test_agentic_nde_registration_leakage.py
```

### Task 6: Map registered geometry to legal 8x8 actions

**Files:**

- Create: `src/cmc_bbdm/agentic_nde/grid.py`
- Test: `tests/test_agentic_nde_grid_mapping.py`
- Test: `tests/test_agentic_nde_action_legality.py`

- [ ] Write failing tests for cell IDs 0-63, deterministic row/column order,
  half-open internal boundaries, closed final boundary, cell boxes, point-to-cell
  mapping, transformed boxes, empty intersections, full-grid conservation, and
  exact renderer replay.
- [ ] Verify RED for the missing grid module.
- [ ] Implement the legal 8x8 frame from frozen acquisition semantics rather
  than redefining the action space.
- [ ] Implement point, box, cell, and full-grid renderers that consume the same
  validated transform object.
- [ ] Reject any action outside 0-63 or any mapping with unresolved orientation.

**Validation:**

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  tests/test_agentic_nde_grid_mapping.py \
  tests/test_agentic_nde_action_legality.py
```

### Task 7: Bind the frozen A2 target authority without exposing it to P0

**Files:**

- Create: `src/cmc_bbdm/agentic_nde/frozen_bindings.py`
- Test: `tests/test_agentic_nde_frozen_a2_binding.py`

- [ ] Write failing tests that bind exact SHA-256
  `6b289f2f6f74ac75dde47ea7cbfefcda1c49f025e74227bfb34ef269182ff963`,
  schema, initial-state rows, specimen/domain keys, state identity, and 64 legal
  cells.
- [ ] Verify RED for the missing module.
- [ ] Implement read-only schema inspection and identity binding. P0 may report
  availability and hash only; transform/QC functions cannot accept target
  values or the loaded A2 table.
- [ ] Bind the strongest compatible frozen static baseline from existing
  authority before P1 protocol creation, without evaluating a new target.

**Validation:**

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  tests/test_agentic_nde_frozen_a2_binding.py
```

### Task 8: Write exact P0 artifacts and replay contract

**Files:**

- Create: `src/cmc_bbdm/agentic_nde/artifacts.py`
- Test: `tests/test_agentic_nde_artifacts.py`
- Test: `tests/test_agentic_nde_replay.py`

- [ ] Write failing tests for exact required membership:

```text
config.yaml
surface_manifest.csv
surface_qc.csv
registration.csv
registration_qc.csv
source_hashes.csv
summary.json
REPORT.md
artifact_manifest.json
CHECKSUMS.sha256
```

- [ ] Test rejection of symlinks, missing files, extra files, duplicate CSV
  keys, schema drift, size drift, hash drift, manifest recursion, and status/
  summary inconsistency.
- [ ] Verify RED for the missing artifact module.
- [ ] Implement canonical CSV/JSON/Markdown serialization, sorted rows, stable
  line endings, exact package membership, SHA-256 manifest, and atomic rename
  into a nonexistent destination.
- [ ] Replay validates all source/config hashes and recomputes the gate from
  machine-readable records. It never trusts the prose report as authority.
- [ ] Assert two synthetic executions are byte-identical.

**Validation:**

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  tests/test_agentic_nde_artifacts.py \
  tests/test_agentic_nde_replay.py
```

### Task 9: Orchestrate P0 without importing model code

**Files:**

- Create: `src/cmc_bbdm/agentic_nde/p0.py`
- Create: `scripts/run_agentic_nde.py`
- Create: `paper_v3/configs/agentic_nde_p0.yaml`
- Test: `tests/test_agentic_nde_p0_pipeline.py`
- Test: `tests/test_agentic_nde_cli.py`

- [ ] Write failing tests that require explicit external roots, refuse existing
  output paths, prevent serialized absolute paths, prohibit torch/transformers/
  sklearn imports, emit downstream `NOT_RUN_NOT_AUTHORIZED`, and distinguish a
  completed NO-GO audit from an integrity/runtime error.
- [ ] Verify RED for missing pipeline and CLI.
- [ ] Implement commands:

```text
run_agentic_nde.py audit-p0 --config PATH --surface-root PATH --output PATH
run_agentic_nde.py replay-p0 --path PATH --surface-root PATH
```

- [ ] The config freezes dataset IDs/versions, six domains/counts, expected
  source authority hashes, accepted surface formats/modes, registration
  hierarchy, action-grid semantics, coverage thresholds, forbidden evidence,
  and exact status vocabulary.
- [ ] `audit-p0` calls only authority, surface QC, registration, grid, frozen
  hash binding, gate, and artifact functions. It writes no image copy, cache,
  feature, embedding, model, or weight.

**Validation:**

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  tests/test_agentic_nde_p0_pipeline.py \
  tests/test_agentic_nde_cli.py
```

### Task 10: Execute real P0 and answer all twelve authority questions

**Files:**

- Create mechanically through the pipeline:
  `results/agentic_task_driven_nde/p0_registration/`
- Create: `artifacts/agentic_task_driven_nde/P0_SURFACE_CSCAN_AUTHORITY_AUDIT.md`
- Create: `artifacts/agentic_task_driven_nde/P0_REGISTRATION_DECISION.md`

- [ ] Run P0 against the immutable local Hasebe authority root. Do not commit
  raw images or absolute local paths.
- [ ] Verify exact specimen/domain counts and every hash against the source and
  compact-repository authorities.
- [ ] Answer, with machine/source evidence, the exact surface roster,
  C-scan/CAI exact match, formats/resolutions/hashes, surface physical metadata,
  registered-C-scan physical metadata, orientation authority, legal 8x8 map,
  frozen A2 file/schema, strongest frozen baseline, local model
  revision/license availability, external-code reuse need, and claims blocked
  by P0/P1 failure.
- [ ] If any required transform fact is unsupported, emit
  `P0_SPATIAL_REGISTRATION_NO_GO`. Do not substitute normalized coordinates,
  target overlap, a damage centroid, manual alignment, or a learned rescue.
- [ ] Record P1-P4 as `NOT_RUN_NOT_AUTHORIZED` after a P0 NO-GO.

**Validation:**

```bash
PYTHONPATH=src python scripts/run_agentic_nde.py replay-p0 \
  --path results/agentic_task_driven_nde/p0_registration \
  --surface-root /external/hasebe/root
sha256sum -c results/agentic_task_driven_nde/p0_registration/CHECKSUMS.sha256
```

The runtime command substitutes the actual external path. That path must not
appear in committed output.

### Task 11: Apply the P0 authorization gate

- [ ] If and only if P0 is GO, create and commit
  `paper_v3/configs/agentic_nde_p1.yaml` and
  `docs/AGENTIC_NDE_P1_PROTOCOL.md` before formal P1 target evaluation.
- [ ] If P0 is not GO, do not create P1-P4 configs, model code, model tests,
  embeddings, figures, paper directories, or training artifacts.
- [ ] In either case, retain the full P0 audit package and continue only with
  integrity verification, handoff, commit, and push.

### Task 12: Verify P0 and frozen-path integrity

Run:

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  tests/test_agentic_nde_*.py
python -m ruff check src/cmc_bbdm/agentic_nde \
  scripts/run_agentic_nde.py tests/test_agentic_nde_*.py
git diff --check
git diff --name-only 15db6edad14ef36364fbda17945ccc924f600e47 -- \
  results/mva results/mvd results/mavis results/mavis_science_closure \
  results/p1_full_field_oracle results/p3_spatial_specificity \
  results/p5_sparse_scan artifacts/mavis artifacts/mavis_science_closure \
  artifacts/mvd_authority artifacts/mavis_authority \
  artifacts/aei_information_hierarchy results/damage_to_failure_response \
  artifacts/damage_to_failure_response paper_aei_information_hierarchy \
  src/cmc_bbdm/mva src/cmc_bbdm/mvd src/cmc_bbdm/mavis
git status --short
```

Expected: all P0 tests and Ruff pass, diff check is clean, frozen-path output is
empty, and only intended new-route files are present.

### Task 13: Commit the P0 decision

```bash
git add docs/superpowers/specs/2026-08-31-agentic-task-driven-nde-design.md \
  docs/superpowers/plans/2026-08-31-agentic-task-driven-nde-p0.md \
  src/cmc_bbdm/agentic_nde scripts/run_agentic_nde.py \
  paper_v3/configs/agentic_nde_p0.yaml tests/test_agentic_nde_*.py \
  results/agentic_task_driven_nde/p0_registration \
  artifacts/agentic_task_driven_nde
git commit -m "research: audit agentic NDE spatial authority"
```

Do not force-push, merge main, create a PR, or amend a published commit.
