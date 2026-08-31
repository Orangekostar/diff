# Agentic NDE Author Registration P0R Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and formally replay a separate P0R stage that binds the later author correspondence, proves the historical raw-panel extraction chain, and authorizes P1 only if every registration gate passes.

**Architecture:** Reuse the frozen P0 package as the identity authority. Keep author-orientation evidence in an immutable API that cannot accept C-scan or mechanical targets, keep raw/processed C-scan pixels inside a separate fixed-recipe provenance verifier, then compose the globally fixed ROT90 full-frame transform with the existing canonical 8x8 grid. Write and replay a deterministic P0R package without modifying historical P0 or frozen scientific paths.

**Tech Stack:** Python 3.13, dataclasses/enums, pathlib, hashlib/json/csv, Pillow, PyYAML, pytest, Ruff, existing `cmc_bbdm.agentic_nde` contracts.

---

## File Map

- Create `paper_v3/configs/agentic_nde_p0r_author_registration.yaml`: frozen P0R inputs, evidence hashes, crop recipes, thresholds, and QC selection.
- Modify `src/cmc_bbdm/agentic_nde/contracts.py`: add author evidence role and closed P0R gate types without changing old P0 decisions.
- Create `src/cmc_bbdm/agentic_nde/author_authority.py`: immutable correspondence record and exact statement validation.
- Create `src/cmc_bbdm/agentic_nde/scan_frame_provenance.py`: fixed historical crop recipes and decoded-pixel verifier.
- Create `src/cmc_bbdm/agentic_nde/p0r_artifacts.py`: exact P0R package writer and replay verifier using existing canonical artifact primitives.
- Create `src/cmc_bbdm/agentic_nde/p0r_qc.py`: deterministic post-freeze overlay rendering.
- Create `src/cmc_bbdm/agentic_nde/p0r.py`: full P0R orchestration.
- Modify `scripts/run_agentic_nde.py`: add `audit-p0r` and `replay-p0r` while preserving old commands.
- Add the five prompt-required tests plus focused gate, artifact, CLI, QC, and historical-freeze tests.
- Generate the required P0R result package and three decision/provenance artifacts only after tests pass.

### Task 1: Freeze Author Authority And Evidence Role

**Files:**
- Create: `paper_v3/configs/agentic_nde_p0r_author_registration.yaml`
- Modify: `src/cmc_bbdm/agentic_nde/contracts.py`
- Create: `src/cmc_bbdm/agentic_nde/author_authority.py`
- Create: `tests/test_agentic_nde_author_authority.py`

- [ ] **Step 1: Write the failing author-authority tests**

Test exact constants and make the constructor reject any deviation:

```python
def test_author_authority_binds_exact_user_attested_statement() -> None:
    authority = build_author_registration_authority()
    assert authority.source_type == USER_ATTESTED_SOURCE
    assert authority.statement_sha256 == EXPECTED_STATEMENT_SHA256
    assert authority.orientation is Orientation.ROT90
    assert authority.mapping_basis == "AUTHOR_FULL_FRAME_PIXEL_CORRESPONDENCE"
    assert authority.physical_mm_used_for_cross_modal_mapping is False


def test_author_authority_api_has_no_result_or_cscan_inputs() -> None:
    forbidden = {
        "cscan_pixels", "cai", "mechanical_value", "oracle_action",
        "damage_mask", "damage_centroid", "target_domain_label",
    }
    assert forbidden.isdisjoint(inspect.signature(AuthorRegistrationAuthority).parameters)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_agentic_nde_author_authority.py
```

Expected: collection/import failure because `author_authority.py` does not exist.

- [ ] **Step 3: Add the closed role and immutable record**

Add only this legal role to the existing enum:

```python
AUTHOR_CORRESPONDENCE = "AUTHOR_CORRESPONDENCE"
```

Implement a frozen, slotted dataclass whose `__post_init__` requires the exact
source type, statement text/hash, `ROT90`, no specimen-frame crop, mapping
basis, examples, false physical-mm flag, and either a valid optional SHA-256 or
`None`. `as_dict()` must use only JSON-safe primitives.

- [ ] **Step 4: Create the hash-bound config**

The config must include these fixed identities:

```yaml
schema_version: 1
stage: P0R_AUTHOR_SURFACE_CSCAN_REGISTRATION
repository_base_sha: 3cb63b544b6c13047773c0eda045558ff4466afa
controlling_prompt_sha256: 37265bb06eef238dca2325b590d1353a8159514ac693e4b4d070a637fb3b8eb8
author_statement_sha256: 3560662d4509ea3e059d597cedca15950cce02f706a992330b161381acfba6ba
author_authority_artifact_sha256: 60a3aeb39256d1b88692439677a20f03f3b987fde3faa1c69224aea1c1f76bf8
historical_preprocessor_sha256: 3bd0c56adb78b65cc09cec340b06c40c7ec6e73988e187067a76f1128ddf83f0
paired_manifest_sha256: f81002981bf2f6aed84818b48da87cd57e6336f5f3da8d78df1a58d26dd8026f
```

Store no absolute paths.

- [ ] **Step 5: Verify GREEN and old enum contracts**

Run the new test and the existing registration/no-leakage tests. Expected: all
pass, with the old P0 status vocabulary unchanged.

### Task 2: Prove Fixed Raw-Panel Processing Provenance

**Files:**
- Create: `src/cmc_bbdm/agentic_nde/scan_frame_provenance.py`
- Create: `tests/test_agentic_nde_scan_frame_provenance.py`
- Create: `tests/test_agentic_nde_raw_panel_to_crop.py`

- [ ] **Step 1: Write failing recipe and pixel-equivalence tests**

Cover the exact source recipes:

```python
@pytest.mark.parametrize(
    ("size", "boxes"),
    [
        ((891, 891), ((31, 33, 706, 707),)),
        ((669, 885), ((39, 33, 469, 708),)),
        ((996, 581), ((30, 33, 370, 371), (464, 33, 816, 371))),
    ],
)
def test_historical_crop_recipes_are_closed(size, boxes) -> None:
    recipe = recipe_for_screenshot_size(size)
    assert recipe.panel_boxes == boxes


def test_dual_panel_index_zero_is_left_and_one_is_right(tmp_path: Path) -> None:
    raw = tmp_path / "c8-2and3.jpg"
    image = Image.new("RGB", (996, 581), "white")
    ImageDraw.Draw(image).rectangle((30, 33, 369, 370), fill="red")
    ImageDraw.Draw(image).rectangle((464, 33, 815, 370), fill="green")
    image.save(raw, quality=100, subsampling=0)
    left = tmp_path / "c8-2.png"
    right = tmp_path / "c8-3.png"
    with Image.open(raw) as decoded:
        decoded.convert("RGB").crop((30, 33, 370, 371)).save(left)
        decoded.convert("RGB").crop((464, 33, 816, 371)).save(right)
    assert verify_registered_crop(raw, left, panel_index=0).panel_box == (30, 33, 370, 371)
    assert verify_registered_crop(raw, right, panel_index=1).panel_box == (464, 33, 816, 371)


def test_registered_crop_requires_exact_decoded_rgb(tmp_path: Path) -> None:
    raw = tmp_path / "q24-7astm.jpg"
    Image.new("RGB", (669, 885), "blue").save(raw, quality=100, subsampling=0)
    crop = tmp_path / "q24-7astm.png"
    with Image.open(raw) as decoded:
        decoded.convert("RGB").crop((39, 33, 469, 708)).save(crop)
    result = verify_registered_crop(raw, crop, panel_index=0)
    assert result.decoded_pixel_equal is True
    Image.new("RGB", (430, 675), "black").save(crop)
    with pytest.raises(ProcessingProvenanceError, match="decoded RGB"):
        verify_registered_crop(raw, crop, panel_index=0)
```

Also test unknown geometry, negative/out-of-range panel index, symlinks,
expected-hash drift, altered crop pixels, and non-RGB decode failure.

- [ ] **Step 2: Run both files and verify RED**

Expected: import failure for the missing provenance module.

- [ ] **Step 3: Implement the minimal closed verifier**

Implement:

```python
@dataclass(frozen=True, slots=True)
class CropRecipe:
    screenshot_width: int
    screenshot_height: int
    panel_boxes: tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True, slots=True)
class ScanProcessingProvenance:
    panel_index: int
    panel_box: tuple[int, int, int, int]
    raw_decoded_rgb_sha256: str
    recovered_panel_rgb_sha256: str
    registered_crop_rgb_sha256: str
    decoded_pixel_equal: bool
    resize: str
    interpolation: str
    rotation: str
    reflection: str
```

`verify_registered_crop` may decode raw and registered C-scan pixels, but
its signature contains no surface image, orientation, CAI, oracle, or target
outcome. It selects a crop recipe only from exact screenshot dimensions and the
already-authorized panel index.

- [ ] **Step 4: Run GREEN and Ruff**

Expected: both provenance test files pass; Ruff reports no findings.

- [ ] **Step 5: Commit the authority/provenance implementation**

```bash
git add paper_v3/configs/agentic_nde_p0r_author_registration.yaml \
  src/cmc_bbdm/agentic_nde/contracts.py \
  src/cmc_bbdm/agentic_nde/author_authority.py \
  src/cmc_bbdm/agentic_nde/scan_frame_provenance.py \
  tests/test_agentic_nde_author_authority.py \
  tests/test_agentic_nde_scan_frame_provenance.py \
  tests/test_agentic_nde_raw_panel_to_crop.py
git commit -m "audit: verify C-scan panel and crop provenance"
```

### Task 3: Lock CW90 And 64-Cell Transform Semantics

**Files:**
- Create: `tests/test_agentic_nde_rot90_semantics.py`
- Create: `tests/test_agentic_nde_p0r.py`

- [ ] **Step 1: Add explicit asymmetric corner tests**

Use different source/destination dimensions and assert:

```python
assert transform.forward_point((0, 0)) == pytest.approx((dst_w - 1, 0))
assert transform.forward_point((src_w - 1, 0)) == pytest.approx((dst_w - 1, dst_h - 1))
assert transform.forward_point((src_w - 1, src_h - 1)) == pytest.approx((0, dst_h - 1))
assert transform.forward_point((0, src_h - 1)) == pytest.approx((0, 0))
```

Parameterize `3357x3357 -> 675x674` and `1500x1500 -> 340x338`.

- [ ] **Step 2: Run and verify existing behavior passes**

This is a characterization test of the frozen transform convention. Expected:
pass without changing `registration.py`.

- [ ] **Step 3: Add all-cell round-trip and sentinel API tests**

For every cell, inverse-map its registered box and forward-map it back within
`1e-9`. Assert the P0R transform factory has no CAI/oracle/damage arguments and
always constructs `Orientation.ROT90` from the author record.

### Task 4: Implement Closed P0R Gate Types

**Files:**
- Modify: `src/cmc_bbdm/agentic_nde/contracts.py`
- Extend: `tests/test_agentic_nde_p0r.py`

- [ ] **Step 1: Write failing decision tests**

Cover GO, author conflict, unresolved provenance, no-target-evidence failure,
per-domain 90% failure, total 240 failure, and compatibility status.

- [ ] **Step 2: Verify RED**

Expected: imports for `P0RGateFacts`, `P0RDecision`, or `decide_p0r` fail.

- [ ] **Step 3: Implement the closed gate**

Use a separate enum:

```python
class P0RStatus(str, Enum):
    GO = "P0R_AUTHOR_REGISTRATION_GO"
    NO_GO = "P0R_AUTHOR_REGISTRATION_NO_GO"
    CONFLICT = "P0R_AUTHOR_EVIDENCE_CONFLICT"
    PROVENANCE_UNRESOLVED = "P0R_PROCESSING_PROVENANCE_UNRESOLVED"
```

Conflict takes precedence, then unresolved provenance, then remaining gate and
coverage failures. GO maps compatibility to `P0_REGISTRATION_GO`; every other
status maps to `P0_SPATIAL_REGISTRATION_NO_GO` and P1-P4
`NOT_RUN_NOT_AUTHORIZED`.

- [ ] **Step 4: Verify GREEN**

Run P0R, old P0 gate, registration, and no-leakage tests. Expected: all pass.

### Task 5: Build Deterministic P0R Package And Replay

**Files:**
- Create: `src/cmc_bbdm/agentic_nde/p0r_artifacts.py`
- Create: `tests/test_agentic_nde_p0r_artifacts.py`
- Create: `tests/test_agentic_nde_p0r_replay.py`

- [ ] **Step 1: Write failing exact-membership tests**

Require exactly:

```python
REQUIRED_P0R_FILES = {
    "config.yaml", "author_authority.json", "surface_manifest.csv",
    "scan_processing_provenance.csv", "registration.csv",
    "registration_qc.csv", "grid_mapping_qc.csv", "summary.json",
    "REPORT.md", "artifact_manifest.json", "CHECKSUMS.sha256",
}
```

Test atomic no-overwrite, no absolute paths, deterministic byte identity,
duplicate keys, extra/missing/symlink members, payload drift, and status/gate
inconsistency.

- [ ] **Step 2: Verify RED**

Expected: missing module/import failure.

- [ ] **Step 3: Implement writer and structure replay**

Reuse the canonical JSON/CSV/hash helpers from `artifacts.py`. Use a staging
directory and atomic rename. The artifact manifest stage is `P0R`, and replay
must recompute the P0R gate from serialized facts.

- [ ] **Step 4: Add source-aware replay**

When `surface_root` and `project_root` are supplied, replay must rebind the
historical P0 package, author artifact, paired manifest, preprocessing source,
every surface/raw/crop file, every panel crop, every transform hash, the
authorized roster, and all 64-cell rows. A structural replay without source
roots may verify package integrity only and must not claim source revalidation.

- [ ] **Step 5: Verify GREEN and old P0 replay**

Run the new package/replay tests and all old artifact/replay tests.

### Task 6: Orchestrate P0R End To End

**Files:**
- Create: `src/cmc_bbdm/agentic_nde/p0r.py`
- Extend: `tests/test_agentic_nde_p0r.py`

- [ ] **Step 1: Write failing pipeline boundary tests**

Test missing external root, existing output, invalid config, changed author
artifact, changed historical preprocessor, changed old P0 package, processing
contradiction, and import isolation from model frameworks.

- [ ] **Step 2: Verify RED**

Expected: `audit_p0r` is unavailable.

- [ ] **Step 3: Implement minimal orchestration**

The orchestrator must:

```text
replay old P0 -> require historical NO-GO
bind author record
bind paired manifest and historical source
join exact old surface/QC rows
verify all processing chains
construct fixed ROT90 transforms
render canonical 64-cell mappings
decide P0R
write package
source-aware replay package
```

No model framework or A2 target value may be imported or read.

- [ ] **Step 4: Verify GREEN on synthetic fixtures**

Expected: exact GO package for a valid synthetic roster and closed failure for
every mutated authority.

### Task 7: Add Diagnostic QC And CLI

**Files:**
- Create: `src/cmc_bbdm/agentic_nde/p0r_qc.py`
- Modify: `scripts/run_agentic_nde.py`
- Create: `tests/test_agentic_nde_p0r_qc.py`
- Create: `tests/test_agentic_nde_p0r_cli.py`

- [ ] **Step 1: Write failing QC and CLI tests**

Test two deterministic specimens per domain, four required panels per overlay,
stable hashes, no transform mutation, and command exit/status behavior.

- [ ] **Step 2: Verify RED**

Expected: missing QC renderer and subcommands.

- [ ] **Step 3: Implement deterministic overlays**

Use Pillow only. Select specimens by SHA-256 ordering from a config seed. Render
surface original, surface CW90, original surface with inverse-mapped grid, and
registered C-scan with grid. Downsampling affects diagnostics only.

- [ ] **Step 4: Extend CLI without changing old paths**

Add:

```text
audit-p0r --config --surface-root --output [--project-root] [--qc-output]
replay-p0r --path [--surface-root]
```

Preserve the existing `audit-p0` and `replay-p0` arguments, output status, and
success/error exit semantics exactly.

- [ ] **Step 5: Verify GREEN and all 81 old tests**

Run all `tests/test_agentic_nde_*.py` plus Ruff.

### Task 8: Run Formal P0R And Record Decision

**Files:**
- Generate: `results/agentic_task_driven_nde/p0r_author_registration/`
- Generate: `artifacts/agentic_task_driven_nde/p0r_qc_overlays/`
- Create: `artifacts/agentic_task_driven_nde/P0R_SCAN_PROCESSING_PROVENANCE.md`
- Create: `artifacts/agentic_task_driven_nde/P0R_AUTHOR_REGISTRATION_DECISION.md`

- [ ] **Step 1: Run the formal audit once**

```bash
python scripts/run_agentic_nde.py audit-p0r \
  --config paper_v3/configs/agentic_nde_p0r_author_registration.yaml \
  --surface-root /home/ww/paper3/cmc_damage_inference \
  --output results/agentic_task_driven_nde/p0r_author_registration \
  --qc-output artifacts/agentic_task_driven_nde/p0r_qc_overlays
```

Expected best case: `P0R_AUTHOR_REGISTRATION_GO`; do not alter registration if
the observed result differs.

- [ ] **Step 2: Replay with all authorities**

```bash
python scripts/run_agentic_nde.py replay-p0r \
  --path results/agentic_task_driven_nde/p0r_author_registration \
  --surface-root /home/ww/paper3/cmc_damage_inference
```

Expected: the exact observed P0R decision, exit 0.

- [ ] **Step 3: Rebuild into a temporary directory and compare**

Generate a second package and require `diff -qr` to be empty. Move the temporary
directory to trash after verification.

- [ ] **Step 4: Inspect all QC overlays without changing transforms**

Record any contradiction as a stage failure. Do not tune orientation, crop,
offset, or roster after inspection.

- [ ] **Step 5: Write provenance and decision artifacts**

Record the exact source file/hash, missing historical Git metadata, crop boxes,
panel counts, 276 replay outcome, Q24-7 exemplar, transform coverage, exclusions,
gate facts, status, and downstream authorization.

- [ ] **Step 6: Commit P0R**

```bash
git add src/cmc_bbdm/agentic_nde scripts/run_agentic_nde.py \
  paper_v3/configs/agentic_nde_p0r_author_registration.yaml \
  tests/test_agentic_nde_*.py \
  results/agentic_task_driven_nde/p0r_author_registration \
  artifacts/agentic_task_driven_nde/P0R_*.md \
  artifacts/agentic_task_driven_nde/p0r_qc_overlays
git commit -m "audit: reopen agentic NDE registration"
```

### Task 9: Apply The Stage Gate

**Files:**
- Conditional: `docs/AGENTIC_NDE_P1_VISUAL_OBSERVABILITY_PROTOCOL.md`
- Conditional: `paper_v3/configs/agentic_nde_p1_visual_observability.yaml`

- [ ] **Step 1: If P0R is not GO**

Set P1-P4 exactly to `NOT_RUN_NOT_AUTHORIZED`, create the final handoff, run the
frozen diff and all verification, commit, push, and stop.

- [ ] **Step 2: If P0R is GO**

Do not run P1 yet. First inspect the frozen MVD/MAVIS machine artifacts, create
a separate P1 design and plan, then commit the exact P1 protocol/config before
any formal target-domain evaluation.

## P0R Completion Checks

Run after the P0R commit:

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_agentic_nde_*.py
python -m ruff check src/cmc_bbdm/agentic_nde scripts/run_agentic_nde.py tests/test_agentic_nde_*.py
git diff --check
```

Run the exact frozen-path diff from the controlling prompt against
`3cb63b544b6c13047773c0eda045558ff4466afa`; it must be empty. Re-run old P0
replay and new P0R source-aware replay. Do not claim a downstream stage until
its prerequisite decision and protocol commit exist.
