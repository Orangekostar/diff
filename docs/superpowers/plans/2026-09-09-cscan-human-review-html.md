# C-scan Human Review HTML Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Chinese HTML tool that produces attributable registered-C-scan references and blinded frozen-first-STOP report decisions without model inference or scientific-result changes.

**Architecture:** A Python adapter owns frozen-source identity, real packet generation, STOP-prefix recovery, return conversion, and existing-contract validation. A dependency-free HTML/CSS/JavaScript application reads self-contained local JSON packets and exports portable session JSON; `build-ui` inlines the source assets into a double-clickable delivery file. Large real packets and private mappings remain under the ignored `.local` tree, while small manifests, validation evidence, documentation, and a synthetic practice packet are committed.

**Tech Stack:** Python 3.13, Pillow, NumPy, Polars, PyYAML, pytest, plain HTML/CSS/JavaScript, SVG, Playwright Chromium for one targeted browser smoke.

**Spec:** `/home/ww/paper3/expert/CODEX_CSCAN_HUMAN_REVIEW_HTML_EXECUTION.md`

## Global Constraints

- Base exactly `2102cc4a1726910931dfaaf20e29ad29a20eaf2e`; branch `research/cscan-human-review-html`.
- Training, VLM calls, Actor calls, STOP forward calls, threshold search, and W0-W5 re-analysis are all zero.
- Do not modify Reader, weights, perception caches, split, measurement geometry, task criteria, stored actions, STOP points, or historical results.
- Reference coordinates are normalized pixel-center coordinates: `u=x/(W-1)`, `v=y/(H-1)`.
- Blind packets contain sanitized measured evidence and report/support layers only; they contain no full C-scan or private method mapping.
- Real human return counts stay zero until real people provide returns; synthetic acceptance data stays `TEST_ONLY`.

---

### Task 1: Contracts, Configuration, and CLI Surface

**Files:**
- Create: `paper_v3/configs/cscan_human_review_tool.yaml`
- Create: `src/cmc_bbdm/learned_cscan/human_review_tool.py`
- Create: `scripts/cscan_human_review.py`
- Create: `tests/test_cscan_human_review_tool.py`

**Interfaces:**
- Consumes: `load_frozen_process_config`, `load_study_config`, `load_study_roster`, `reference_from_payload`, and `summarize_blind_reviews`.
- Produces: `load_human_review_config(path, project_root)`, packet/session validators, packet builders, return exporters, and CLI subcommands `build-ui`, `prepare-references`, `prepare-blind`, `validate-return`, `export-references`, and `export-blind-reviews`.

- [ ] **Step 1: Write failing tests for config identity and all six parser commands**

```python
def test_cli_exposes_all_workflows() -> None:
    parser = build_parser()
    for command in EXPECTED_COMMANDS:
        assert parser.parse_args([command, "--help"]) is not None
```

- [ ] **Step 2: Run the focused tests and verify they fail because the new module is absent**

Run: `PYTHONPATH=src pytest -q tests/test_cscan_human_review_tool.py`

- [ ] **Step 3: Implement the frozen config loader and lazy CLI dispatch**

The loader resolves project-relative paths, verifies the frozen process config SHA, validates packet sizes and seed, and records the fixed research base. The CLI bootstraps the local package path using the established analysis-script pattern and imports runtime code only after argument parsing.

- [ ] **Step 4: Run focused tests and Ruff on the new Python files**

Run: `PYTHONPATH=src pytest -q tests/test_cscan_human_review_tool.py`

Run: `python -m ruff check scripts/cscan_human_review.py src/cmc_bbdm/learned_cscan/human_review_tool.py tests/test_cscan_human_review_tool.py`

### Task 2: Registered-C-scan Reference Packets and Conversion

**Files:**
- Modify: `src/cmc_bbdm/learned_cscan/human_review_tool.py`
- Modify: `tests/test_cscan_human_review_tool.py`

**Interfaces:**
- Consumes: frozen TEST roster, cohort coverage CSV, and registered crop identity.
- Produces: `prepare_reference_packets(...)`, `validate_reference_session(...)`, and `export_reference_session(...)`.

- [ ] **Step 1: Add failing tests for TEST identity, lossless PNG separation, state semantics, source mismatch, polygons, and per-reviewer output isolation**

Use a non-square fixture and assert that `CONFIRMED`, `CONFIRMED_NO_CERTAIN_REGION`, `UNABLE_TO_JUDGE`, `DRAFT`, and `UNANNOTATED` produce distinct audit/export outcomes. Assert canonical output is accepted by `reference_from_payload` and no metadata JSON is written inside `references/`.

- [ ] **Step 2: Implement deterministic self-contained packets**

Load only frozen TEST assignments, compare keys with the frozen cohort CSV, verify registered crop dimensions and SHA, encode lossless PNG, keep source and display hashes separate, group by configured packet size, and emit a 24-row manifest plus a clearly labeled asymmetric `TEST_ONLY` practice packet.

- [ ] **Step 3: Implement strict return validation and canonical export**

Bind sessions to packet/item/source identity, reject non-finite or out-of-range coordinates and zero-area polygons, require explicit human provenance for confirmed records, place one canonical JSON per specimen under `converted/<reviewer>/<export_id>/references/`, and write annotation audit/metadata beside that directory.

- [ ] **Step 4: Run focused reference tests**

Run: `PYTHONPATH=src pytest -q tests/test_cscan_human_review_tool.py -k 'reference or polygon or state'`

### Task 3: Frozen First-STOP Packet Recovery

**Files:**
- Modify: `src/cmc_bbdm/learned_cscan/human_review_tool.py`
- Modify: `tests/test_cscan_human_review_tool.py`

**Interfaces:**
- Consumes: first-stop table, stored trajectory rows, `zero_state`, `action_added_positions_from_mask`, `apply_action`, `FrozenVisibleReportReader`, `_report_digest`, and the frozen TEST context.
- Produces: `prepare_blind_packets(...)`, private report index, public packet manifest, no-STOP coverage, and recovery summary.

- [ ] **Step 1: Add failing tests for prefix row semantics and public-field allowlist**

Assert only action rows with `step < stop_step` are applied, the `stop_step` action is not applied, repeated report digests do not deduplicate report IDs, non-stopped runs stay out of public packets, and private fields/method tokens never appear in serialized public packet bytes.

- [ ] **Step 2: Implement one-pass per-specimen recovery**

Load context once, reuse one grid/Reader per specimen, rebuild each stored prefix from zero state, verify action legality, measured cost, shape, support-inside-measured, and original report digest, then record actual recovery/Reader/action counts. Cache by frozen input/config/code/report identity without treating cache use as zero computation.

- [ ] **Step 3: Render privacy-preserving layers and deterministic packets**

Render measured pixels over an opaque checker field, a transparent frozen-report mask, and measured support points. Shuffle all eligible report IDs with one fixed seed before chunking; keep the private method/specimen mapping physically separate under `.local`.

- [ ] **Step 4: Run focused STOP and privacy tests**

Run: `PYTHONPATH=src pytest -q tests/test_cscan_human_review_tool.py -k 'stop or blind or public'`

### Task 4: Local HTML Application

**Files:**
- Create: `web/cscan_human_review/index.template.html`
- Create: `web/cscan_human_review/styles.css`
- Create: `web/cscan_human_review/app.js`
- Create: `dist/cscan_human_review/index.html`
- Create: `dist/cscan_human_review/README_ZH.md`
- Modify: `src/cmc_bbdm/learned_cscan/human_review_tool.py`
- Modify: `tests/test_cscan_human_review_tool.py`

**Interfaces:**
- Consumes: `REFERENCE_ANNOTATION` and `BLIND_FIRST_STOP_REVIEW` packets.
- Produces: `CSCAN_REFERENCE_SESSION` and `CSCAN_BLIND_REVIEW_SESSION` JSON exports with explicit confirmation state and full draft history.

- [ ] **Step 1: Add failing build assertions**

Assert the built HTML is self-contained, has no CDN/network references, includes both packet modes, and contains the required file import/session export controls.

- [ ] **Step 2: Implement the quiet three-column work surface**

Use an item navigator at left, an aspect-preserving SVG viewport in the center, and mode-specific tools/provenance at right. Keep dimensions stable and use neutral, green, amber, and red semantics rather than a one-hue palette.

- [ ] **Step 3: Implement reference interaction**

Support certain/uncertain polygon drawing, multiple components, vertex selection/editing, deletion, undo/redo, unfinished-polygon cancel, SVG-CTM coordinate inversion, zoom/pan/reset, layer visibility/opacity, navigation draft retention, explicit state confirmation, portable session export, and session re-import. Keyboard actions must ignore text inputs.

- [ ] **Step 4: Implement blind-review interaction**

Render measured evidence, report, and support layers without exposing reference/full-scan/method metadata. Support exactly three decisions, notes/problem/basis fields, explicit confirmation, progress counters, export, and re-import; do not compute rankings or consensus.

- [ ] **Step 5: Build and run static tests**

Run: `PYTHONPATH=src python scripts/cscan_human_review.py build-ui`

Run: `PYTHONPATH=src pytest -q tests/test_cscan_human_review_tool.py -k 'build or html'`

### Task 5: Blind-Return Conversion and Existing Interface Validation

**Files:**
- Modify: `src/cmc_bbdm/learned_cscan/human_review_tool.py`
- Modify: `tests/test_cscan_human_review_tool.py`

**Interfaces:**
- Consumes: raw blind session plus private report index.
- Produces: UTF-8 CSV accepted by `summarize_blind_reviews`, validation summary, and raw-session preservation.

- [ ] **Step 1: Add failing tests for matching, duplicate keys, untouched pending items, UTF-8 punctuation/newlines, and `UNABLE_TO_JUDGE`**

- [ ] **Step 2: Implement export with Python's `csv` module**

Retain one current confirmed decision per `(report_id, reviewer_id)`, preserve raw JSON, reject unknown IDs and duplicates, call `summarize_blind_reviews`, and label proxy objective fields `PROXY_LEGACY` rather than independent accuracy.

- [ ] **Step 3: Exercise every CLI help and small synthetic round trip**

Run: `for command in build-ui prepare-references prepare-blind validate-return export-references export-blind-reviews; do PYTHONPATH=src python scripts/cscan_human_review.py "$command" --help >/dev/null; done`

### Task 6: Real Packets and Targeted Acceptance

**Files:**
- Create locally: `.local/cscan_human_review_html/reference_packets/*`
- Create locally: `.local/cscan_human_review_html/blind_packets/*`
- Create locally: `.local/cscan_human_review_html/private_report_index/*`
- Create: `results/cscan_human_review_html/build_summary.json`
- Create: `results/cscan_human_review_html/reference_packet_manifest.csv`
- Create: `results/cscan_human_review_html/blind_packet_manifest.csv`
- Create: `results/cscan_human_review_html/no_stop_coverage.csv`
- Create: `results/cscan_human_review_html/validation_summary.json`
- Create: `artifacts/cscan_human_review_html/screenshots/reference_workflow.png`
- Create: `artifacts/cscan_human_review_html/screenshots/blind_workflow.png`

**Interfaces:**
- Consumes: real source root `/home/ww/paper3/cmc_damage_inference`.
- Produces: all T1-T8 acceptance evidence and actual package/recovery counts.

- [ ] **Step 1: Generate all 24 real reference tasks and all eligible real first-STOP tasks once**

Run the two prepare commands with the fixed config/source root. Confirm six domains, 187 expected stopped rows from current source data, five no-STOP rows, no missing registered crop, and actual recovery counts from the generated summary.

- [ ] **Step 2: Run one real browser smoke in available Chromium**

For reference mode, import the asymmetric practice packet, draw after zoom/pan, edit a vertex, confirm, export, reopen, and re-import. For blind mode, import a real public packet, make one `TEST_ONLY` local decision, export/re-import, and capture screenshots without importing that decision into research results.

- [ ] **Step 3: Run focused Python verification**

Run: `PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_cscan_human_review_tool.py tests/test_vlm_cscan_evaluation.py tests/test_learned_cscan_runtime.py`

Run: `python -m ruff check scripts/cscan_human_review.py src/cmc_bbdm/learned_cscan/human_review_tool.py tests/test_cscan_human_review_tool.py`

Run: `git diff --check`

### Task 7: Documentation, Scope Audit, Commit, and Push

**Files:**
- Create: `artifacts/cscan_human_review_html/SOURCE_BINDINGS_AND_TASKS.md`
- Create: `artifacts/cscan_human_review_html/USER_GUIDE_ZH.md`
- Create: `artifacts/cscan_human_review_html/ACCEPTANCE_AND_LIMITATIONS.md`
- Create: `artifacts/cscan_human_review_html/CODEX_HANDOFF_CSCAN_HUMAN_REVIEW_HTML.md`

**Interfaces:**
- Consumes: verified commands, manifests, recovery counts, browser evidence, and Git state.
- Produces: a reproducible handoff and a pushed GitHub branch.

- [ ] **Step 1: Record P0-P7 evidence and exact user commands**

Document actual counts/status, local-only packet paths and sizes, committed files, source bindings, no-human-input boundary, no model/training calls, browser/version/screenshots, tests, and later formal-finalize inputs without claiming expert validation.

- [ ] **Step 2: Prove frozen-path scope**

Compare the task branch against `2102cc4a1726910931dfaaf20e29ad29a20eaf2e`; only new tool/config/test/dist/result/artifact/plan paths may differ. Historical results and shared scientific modules must have no diff.

- [ ] **Step 3: Stage only explicit task paths and commit**

Run: `git add -- docs/superpowers/plans/2026-09-09-cscan-human-review-html.md paper_v3/configs/cscan_human_review_tool.yaml scripts/cscan_human_review.py src/cmc_bbdm/learned_cscan/human_review_tool.py tests/test_cscan_human_review_tool.py web/cscan_human_review dist/cscan_human_review results/cscan_human_review_html artifacts/cscan_human_review_html`

Run: `git commit -m "Add local C-scan annotation and blinded report review tool"`

- [ ] **Step 4: Push once without force and verify SHA equality**

Run: `git push -u origin research/cscan-human-review-html`

Verify `HEAD`, `@{upstream}`, and `git ls-remote origin refs/heads/research/cscan-human-review-html` are identical and the worktree is clean.
