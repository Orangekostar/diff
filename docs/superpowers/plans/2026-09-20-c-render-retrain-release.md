# C Render Retrain Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute the complete C=P0+R1 prior, three-actor retraining, evidence, manuscript and GitHub release workflow.

**Architecture:** A new stage-oriented CLI owns paths, state and budgets while importing the frozen repository's pure renderer, model, training and numeric functions. Each stage is signature-addressed and writes only the new C output, artifact and paper roots; old A/W2/pilot/evidence/paper inputs remain read-only.

**Tech Stack:** Python 3.13, PyTorch 2.12, Transformers 4.49, NumPy, Pillow, Matplotlib, pytest, Pandoc/LuaLaTeX, Git LFS.

**Spec:** `docs/superpowers/specs/2026-09-20-c-render-retrain-release-design.md`; binding external instruction: `/home/ww/paper3/docs/0920/CAI_C_RENDER_RETRAIN_RELEASE_FINAL_PACKAGE.zip::CODEX_C_RENDER_RETRAIN_RELEASE_FINAL.md`.

## Global Constraints

- Run only on `research/cai-vlm-agent-v3-controlled-reuse`, based on `331f52952b93bd9442ad2e44243cc71c7d966b4d`.
- C identity is exactly `C_P0_R1_GLOBAL_V1`: byte-identical P0, exact pilot R1 renderer/font, pilot parser and at most one format repair.
- VLM operates on exactly 161 TRAIN + 50 VALID records; TEST inference, labels, models and images are forbidden.
- Retrain only `VLM_SPATIAL_FEEDBACK`, `VLM_SPATIAL_OPEN_LOOP` and `VLM_MEAN_FEEDBACK`, each to 1250 logical updates from fixed seeds.
- Candidate updates are exactly 250, 500, 750, 1000 and 1250; selection is domain-equal left-error area with strict improvement greater than `1e-12`.
- Preserve all 15 candidate weights and 750 candidate trajectories. Selected 150 rows are a subset.
- Main matrix is 500 immutable control rows plus 150 new C rows. Historical A VLM rows remain separate.
- New optimizer use is 3750 normally and at most 4500 with one 250-update replay reserve per method. VLM has at most 211 new primary jobs and 422 total attempts.
- At most one visible GPU and four CPU threads. Qwen and Actor execute in separate processes and interpreters recorded by absolute path.
- Preserve old W2, old A results, old C pilot, old evidence and old paper byte-for-byte. Never silently fill new outputs with old results.
- Do not run the full repository test suite, TEST evaluation, extra seeds, W2/CNN/Qwen training, attention, GDFS, STOP, hyperparameter search, PR, merge, force push or release tag.
- Completion requires actual W0-W6 execution, PDF visual review, tracked weights/results/manuscript/handoff, commit, push and local/upstream/remote SHA equality.

---

### Task 1: Scope, Context, State And Unified CLI

**Files:**
- Create: `docs/cai/c_render_retrain/C_RETRAIN_RELEASE_SCOPE.json`
- Create: `docs/cai/c_render_retrain/CODEX_C_RENDER_RETRAIN_RELEASE_FINAL.md`
- Create: `scripts/cai_c_retrain/__init__.py`
- Create: `scripts/cai_c_retrain/context.py`
- Create: `scripts/cai_c_retrain/cli.py`
- Create: `tests/test_cai_c_retrain_context.py`

**Interfaces:**
- Produces: `TaskContext.load(config_path: Path) -> TaskContext`, `TaskContext.root`, `TaskContext.path(name)`, `TaskContext.phase_signature(name, inputs)`, atomic JSON/CSV writers, phase state transitions and append-only resource events.
- Produces CLI commands: `prepare`, `vlm`, `train`, `assemble`, `analyze`, `paper`, `verify`, `publish`, `export-inputs`, `all`.

- [x] **Step 1: Add failing context tests**

```python
def test_scope_rejects_test_and_wrong_branch(tmp_path):
    scope = minimal_scope(tmp_path)
    scope["cohort"]["vlm_authorized_splits"] = ["TRAIN", "VALID", "TEST"]
    with pytest.raises(ValueError, match="TEST"):
        TaskContext.from_mapping(tmp_path, scope, branch="wrong")


def test_phase_signature_rejects_changed_inputs(context, tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"first")
    signature = context.phase_signature("prepare", [source])
    source.write_bytes(b"second")
    with pytest.raises(ValueError, match="signature"):
        context.require_phase_signature("prepare", signature, [source])
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src pytest -q tests/test_cai_c_retrain_context.py`

Expected: collection fails because `scripts.cai_c_retrain.context` does not exist.

- [x] **Step 3: Implement strict scope/path/state context and CLI dispatch**

```python
@dataclass(frozen=True)
class TaskContext:
    root: Path
    scope: dict[str, object]
    config_path: Path

    @classmethod
    def load(cls, config_path: Path) -> "TaskContext": ...
    def path(self, name: str) -> Path: ...
    def phase_signature(self, phase: str, inputs: Sequence[Path]) -> str: ...
    def transition(self, phase: str, status: str, **details: object) -> None: ...
    def append_resource_event(self, event: Mapping[str, object]) -> None: ...
```

The implementation validates task ID, branch, source ancestry, all frozen
counts/caps/methods/seeds, path isolation and TEST exclusion. `all` invokes the
same CLI with the phase-specific absolute interpreter and stops on nonterminal
phase state. `publish` dispatches no earlier phase.

- [x] **Step 4: Verify GREEN and CLI help**

Run: `PYTHONPATH=src pytest -q tests/test_cai_c_retrain_context.py && python scripts/cai_c_retrain/cli.py --help`

Expected: tests pass and all ten commands appear.

### Task 2: Exact C Inputs And Bounded VLM Cache

**Files:**
- Create: `scripts/cai_c_retrain/prepare.py`
- Create: `scripts/cai_c_retrain/vlm.py`
- Create: `tests/test_cai_c_retrain_vlm.py`

**Interfaces:**
- Consumes: `TaskContext`, candidate queue, feature-bank manifest, pilot lock/states and exact P0 text.
- Produces: `prepare_stage(context)`, `vlm_stage(context, backend=None)`, `export_inputs(context, specimen_key)`, `vlm_actor_features_fit.csv`, `vlm_manifest_fit.json`.

- [x] **Step 1: Add failing behavior tests**

```python
def test_terminal_cache_uses_true_false_and_64_float_features(tmp_path): ...
def test_no_cue_is_terminal_without_repair(tmp_path): ...
def test_duplicate_cells_repairs_once_then_schema_failure(tmp_path): ...
def test_started_without_raw_consumes_attempt_and_never_gets_third_call(tmp_path): ...
def test_pilot_reuse_requires_full_signature_and_preserves_attempt_history(
    tmp_path,
): ...
def test_known_six_r1_hashes_match_pilot_records(project_root): ...
def test_roster_contains_211_train_valid_and_no_test(project_root): ...
```

Each fake backend returns real-schema text and the assertions target saved
state/raw/token/features, not mock call existence.

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src pytest -q tests/test_cai_c_retrain_vlm.py`

Expected: failures identify missing renderer, state machine and manifest.

- [x] **Step 3: Implement exact rendering, signatures and state machine**

```python
def render_c_inputs(source: Image.Image) -> RenderedCInputs: ...
def parse_contract(raw: str) -> ContractResult: ...
def execute_case(
    context: TaskContext, case: CaseInput, backend: Backend
) -> CaseState: ...
def build_actor_features(states: Sequence[CaseState]) -> list[dict[str, object]]: ...
```

Atomically persist STARTED before inference, raw/token IDs before parsing, and
terminal state last. Reuse exact six C pilot records only after full signature
comparison. Preserve INCOMPLETE separately from schema failure and no-cue.

- [x] **Step 4: Verify GREEN and preparation smoke**

Run: `PYTHONPATH=src pytest -q tests/test_cai_c_retrain_vlm.py`

Run: `PYTHONPATH=src python scripts/cai_c_retrain/cli.py prepare --config docs/cai/c_render_retrain/C_RETRAIN_RELEASE_SCOPE.json`

Expected: tests pass; W0 writes runtime/protocol/cohort locks, reports 211 fit
records, six exact reusable cases, no TEST access and no model calls.

### Task 3: Three-Actor Training, Atomic Candidates And One Resume

**Files:**
- Create: `scripts/cai_c_retrain/train.py`
- Create: `tests/test_cai_c_retrain_train.py`

**Interfaces:**
- Consumes: complete C manifest/features, frozen W2 P_all/OOF checkpoints/folds, existing actor-training pure functions.
- Produces: per-method initial state, five candidates, five 50-row episode files, snapshots, selected checkpoint/episodes, manifests and three release smoke records.

- [x] **Step 1: Add failing deterministic training-control tests**

```python
def test_tie_within_tolerance_keeps_earliest_candidate(tmp_path): ...
def test_snapshot_resume_restores_python_numpy_and_torch_rng(tmp_path): ...
def test_lost_segment_and_replay_are_each_charged_once(tmp_path): ...
def test_second_resume_is_rejected(tmp_path): ...
def test_changed_c_or_w2_signature_is_not_resumable(tmp_path): ...
def test_disk_reloaded_winner_reproduces_actions_costs_predictions(tmp_path): ...
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src pytest -q tests/test_cai_c_retrain_train.py`

Expected: failures identify missing snapshot, segment ledger and disk smoke.

- [x] **Step 3: Implement the new training shell**

```python
METHODS = {
    "VLM_SPATIAL_FEEDBACK": 2026091301,
    "VLM_SPATIAL_OPEN_LOOP": 2026091303,
    "VLM_MEAN_FEEDBACK": 2026091305,
}


def train_method(context: TaskContext, method: str, *, resume: bool) -> ActorResult: ...
def restore_snapshot(path: Path, actor, optimizer, rng) -> ResumeState: ...
def select_candidate(candidates: Sequence[Candidate]) -> Candidate: ...
def release_smoke(context: TaskContext, result: ActorResult) -> dict[str, object]: ...
```

Call the existing `seeded_actor`, `_sample_specimens`,
`_training_rollout_loss`, `evaluate_policy` and `_load_oof_predictors`.
Entropy uses logical update. Close each candidate with a commit marker after
weights, 50 episodes and snapshot are durable. Never extend logical training
beyond 1250.

- [x] **Step 4: Verify GREEN and unchanged legacy tests**

Run: `PYTHONPATH=src pytest -q tests/test_cai_c_retrain_train.py tests/test_cai_agent_v3_w3_pilot.py`

Expected: all focused training and legacy W3 tests pass.

### Task 4: Immutable Control Reuse And 650-Row Assembly

**Files:**
- Create: `scripts/cai_c_retrain/assemble.py`
- Create: `tests/test_cai_c_retrain_assemble.py`

**Interfaces:**
- Consumes: old 650-row A matrix, old protocol/input identities, three C selected episode files.
- Produces: 650-row C matrix, `row_provenance.csv`, reuse audit and `historical_A_comparison/`.

- [x] **Step 1: Add failing assembly tests**

```python
def test_assembly_keeps_500_controls_byte_semantically_unchanged(tmp_path): ...
def test_assembly_rejects_target_key_predictor_or_cost_mismatch(tmp_path): ...
def test_historical_a_vlm_rows_are_not_in_primary_matrix(tmp_path): ...
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src pytest -q tests/test_cai_c_retrain_assemble.py`

- [x] **Step 3: Implement provenance-preserving assembly**

```python
def audit_reusable_controls(context: TaskContext) -> ReuseAudit: ...
def assemble_primary(context: TaskContext) -> AssemblyResult: ...
```

Copy all original control identity fields, add provenance externally, and
assert method counts `50,50,50,250,50,50,50,50,50`.

- [x] **Step 4: Verify GREEN**

Run: `PYTHONPATH=src pytest -q tests/test_cai_c_retrain_assemble.py`

Expected: all assembly tests pass.

### Task 5: C Evidence, Figures, Cases And HTML

**Files:**
- Create: `scripts/cai_c_retrain/evidence.py`
- Create: `tests/test_cai_c_retrain_evidence.py`

**Interfaces:**
- Consumes: 650-row C matrix, separate A rows, old bootstrap weights after key reorder, full P_all NPZ, three case inputs and C priors.
- Produces: same-cost/domain/paired/A-C/full/equal-quality/timing tables, figures, new case PNGs, `analysis_manifest.json` and local HTML.

- [x] **Step 1: Add failing numeric and integration tests**

```python
def test_random_repeats_average_losses_not_predictions(): ...
def test_reused_bootstrap_weights_are_reindexed_by_key(): ...
def test_equal_quality_keeps_unreached_negative_and_recrossing(): ...
def test_timing_keeps_negative_terms_and_satisfies_identity(): ...
def test_event_grid_is_derived_not_fixed_length(): ...
def test_old_six_controls_recompute_to_frozen_values(project_root): ...
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src pytest -q tests/test_cai_c_retrain_evidence.py`

- [x] **Step 3: Implement parameterized analysis and rendering**

```python
def analyze_stage(context: TaskContext) -> AnalysisManifest: ...
def compute_timing(episodes, stage_edges=(0.0, 0.0625, 0.125, 0.1875, 0.25)): ...
def render_case(context: TaskContext, specimen_key: str) -> list[Path]: ...
def render_index(context: TaskContext) -> Path: ...
```

Reuse `paper_evidence_math` and `metrics`; remove old A-specific hardcoded
quality assertions. Recompute all intervals from C errors with the matched
draw weights. Generate all cases from selected C trajectories.

- [x] **Step 4: Verify GREEN**

Run: `MPLBACKEND=Agg PYTHONPATH=src pytest -q tests/test_cai_c_retrain_evidence.py tests/test_cai_agent_paper_evidence.py`

Expected: focused C and unchanged original evidence tests pass.

### Task 6: Independent C Manuscript And Two PDFs

**Files:**
- Create: `scripts/cai_c_retrain/paper.py`
- Create: `tests/test_cai_c_retrain_paper.py`
- Create at runtime: `paper_cai_aei/r2_c_331f5295/**`

**Interfaces:**
- Consumes: old source-only manuscript and completed C evidence.
- Produces: updated abstract, six sections, supplementary, declarations,
  tables/figures/maps, MD/HTML/TeX, `build/main.pdf` and
  `build/supplementary.pdf`.

- [x] **Step 1: Add failing source/result synchronization tests**

```python
def test_paper_copy_excludes_old_generated_build_outputs(tmp_path): ...
def test_result_tokens_resolve_only_from_c_evidence(tmp_path): ...
def test_build_scripts_have_no_old_evidence_or_old_paper_runtime_path(tmp_path): ...
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src pytest -q tests/test_cai_c_retrain_paper.py`

- [x] **Step 3: Implement source copy, bindings and build orchestration**

```python
def create_paper_source(context: TaskContext) -> Path: ...
def update_result_bindings(context: TaskContext, paper_root: Path) -> None: ...
def build_paper(context: TaskContext, paper_root: Path) -> BuildResult: ...
```

The real paper text is edited only after evidence exists. Preserve references
and unknown author metadata, label the draft `AUTHOR_REVIEW_DRAFT`, and state
VALID/single-seed/exploratory limits without hiding adverse results.

- [x] **Step 4: Verify GREEN**

Run: `PYTHONPATH=src pytest -q tests/test_cai_c_retrain_paper.py`

Expected: paper pipeline tests pass without compiling a fake scientific paper.

### Task 7: Q1-Q8 Verification, Release Manifest And Publish Guard

**Files:**
- Create: `scripts/cai_c_retrain/validate.py`
- Create: `tests/test_cai_c_retrain_validate.py`
- Create at runtime: required ART handoff/review/Git files and OUT release manifest.

**Interfaces:**
- Consumes: every prior phase manifest and artifact.
- Produces: structured Q1-Q8 verdicts, release status, tracked-file manifest,
  handoff, Git delivery evidence and guarded push.

- [x] **Step 1: Add failing completeness and publish tests**

```python
def test_complete_release_requires_all_211_15_750_3_150_650_and_two_pdfs(tmp_path): ...
def test_partial_release_cannot_claim_complete(tmp_path): ...
def test_publish_never_invokes_compute_and_rejects_untracked_required_files(
    tmp_path,
): ...
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src pytest -q tests/test_cai_c_retrain_validate.py`

- [x] **Step 3: Implement validation and publish-only Git flow**

```python
def verify_stage(context: TaskContext) -> VerificationReport: ...
def build_release_manifest(context: TaskContext) -> dict[str, object]: ...
def publish_stage(context: TaskContext) -> GitDelivery: ...
```

Publish checks clean expected scope, targeted staging, LFS policy, local
upstream identity and fast-forward push. It never merges, rebases, tags or
force-pushes.

- [x] **Step 4: Verify all focused development tests**

Run: `PYTHONPATH=src pytest -q tests/test_cai_c_retrain_*.py tests/test_cai_agent_v3_w3_pilot.py tests/test_cai_agent_paper_evidence.py`

Expected: zero failures and no warnings attributable to the new pipeline.

### Task 8: Execute W0-W6 And Audit Against The Instruction

**Files:**
- Modify only through stage CLIs: new OUT, ART, PAPER, authorization, global ledger and targeted Git index/commits.

**Interfaces:**
- Consumes: completed implementation and the exact scope.
- Produces: the complete scientific release or a truthful partial release with one exact resume command.

- [x] **Step 1: Commit the tested implementation identity**

Run: targeted `git add` for scope, design, plan, scripts and tests; inspect
`git diff --cached`; commit once. Record the commit and scientific file hashes
in `protocol_lock.json` before model execution.

- [x] **Step 2: Run W0 and W1**

```bash
python scripts/cai_c_retrain/cli.py prepare --config docs/cai/c_render_retrain/C_RETRAIN_RELEASE_SCOPE.json
python scripts/cai_c_retrain/cli.py vlm --config docs/cai/c_render_retrain/C_RETRAIN_RELEASE_SCOPE.json
```

Require 211 terminal rows and `C_PRIOR_COMPLETE` before training.

- [x] **Step 3: Run W2 and W3**

```bash
python scripts/cai_c_retrain/cli.py train --config docs/cai/c_render_retrain/C_RETRAIN_RELEASE_SCOPE.json
python scripts/cai_c_retrain/cli.py assemble --config docs/cai/c_render_retrain/C_RETRAIN_RELEASE_SCOPE.json
```

Use `train --resume --method <METHOD>` only for the single bounded recovery.

- [x] **Step 4: Run W4 and W5**

```bash
python scripts/cai_c_retrain/cli.py analyze --config docs/cai/c_render_retrain/C_RETRAIN_RELEASE_SCOPE.json
python scripts/cai_c_retrain/cli.py paper --config docs/cai/c_render_retrain/C_RETRAIN_RELEASE_SCOPE.json
```

Render both PDFs to images and visually inspect the overview plus abstract,
algorithm, main table, most changed figure and SI table.

- [x] **Step 5: Run fresh verification and requirement audit**

```bash
python scripts/cai_c_retrain/cli.py verify --config docs/cai/c_render_retrain/C_RETRAIN_RELEASE_SCOPE.json
```

Map every W0-W6, Q1-Q8, required file, cap and prohibition from the instruction
to current artifacts and command evidence. Passing unit tests alone is not
completion evidence.

- [ ] **Step 6: Commit results/handoff and publish**

Use targeted staging for all required new weights, records, evidence, paper and
handoff; inspect LFS pointers and staged diff; create at most two post-
implementation commits. Then run:

```bash
python scripts/cai_c_retrain/cli.py publish --config docs/cai/c_render_retrain/C_RETRAIN_RELEASE_SCOPE.json
```

Verify local HEAD, upstream and selected remote files resolve to the same final
SHA. Report W0-W6 status and all requested counts/effects/paths/SHA.
