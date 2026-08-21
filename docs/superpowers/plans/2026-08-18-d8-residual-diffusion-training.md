# D8 Residual Diffusion Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train, select, replay, and hash-freeze one leakage-safe residual-diffusion pipeline for each prospective outer domain without issuing any formal outer evaluation.

**Architecture:** A separate residual-diffusion branch consumes the validated D8 Pilot package and loader-issued D8 fold authorities. It derives exact measured-field nuisance targets, trains compact conditional UNets inside nested four-fit/one-query folds, compares them with the frozen Pilot and raw incumbents, and publishes six pre-outer pipeline documents through a validated transaction. Formal outer arrays and targets are structurally unavailable to this branch.

**Tech Stack:** Python 3.13, NumPy, SciPy, PyTorch, diffusers, scikit-learn, PyYAML, pytest, Ruff, three NVIDIA A40 GPUs.

**Repository note:** `/home/ww/paper3/cmc_damage_inference` is not a Git worktree. Each completed task records SHA-256 values in the test output or artifact manifest instead of creating Git commits.

---

### Task 1: Freeze the residual-training configuration

**Files:**
- Create: `paper_v3/configs/d8_residual_diffusion.yaml`
- Create: `src/cmc_bbdm/cpb_diffusion_marginalization/residual_config.py`
- Preserve byte-for-byte: `src/cmc_bbdm/cpb_diffusion_marginalization/__init__.py`
- Test: `tests/test_d8_residual_config.py`

- [ ] **Step 1: Write exact-contract RED tests**

```python
def test_registered_residual_config_is_preouter_and_exact(project_root):
    config = load_residual_diffusion_config(
        project_root / "paper_v3/configs/d8_residual_diffusion.yaml",
        project_root=project_root,
    )
    assert config.scope == "cpb_d8_residual_diffusion_preouter"
    assert config.outer_evaluation_count == 0
    assert config.candidate_ids == tuple(f"RD{i}" for i in range(8))
    assert config.screening_epochs == 24
    assert config.rerank_epochs == 120
    assert config.training_seeds == (20260823, 20260824, 20260825)
    assert config.promotion_margin == 1.0e-4
    assert config.output_dir == "results/d8_residual_diffusion_search"


def test_registered_residual_config_rejects_unknown_keys(project_root, tmp_path):
    payload = yaml.safe_load(
        (project_root / "paper_v3/configs/d8_residual_diffusion.yaml").read_text()
    )
    payload["unregistered"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ResidualConfigError, match="unknown"):
        load_residual_diffusion_config(path, project_root=project_root)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
/home/ww/miniconda3/bin/python -m pytest -q tests/test_d8_residual_config.py
```

Expected: collection fails because `residual_config` is missing.

- [ ] **Step 3: Add the frozen YAML and immutable loader**

The loader must expose immutable records for the eight candidate definitions and reject duplicate YAML keys, unknown keys, nonfinite numbers, path escape, source hash drift, Pilot decision drift, a nonzero outer count, and timing fields inside scientific configuration. It must bind:

```text
docs/D8_PILOT_DECISION.md
docs/superpowers/specs/2026-08-18-d8-residual-diffusion-training-design.md
paper_v3/configs/d8_exploration.yaml
results/d8_search/artifact_manifest.json
results/d8_search/escalation_evidence.json
```

- [ ] **Step 4: Verify GREEN and source drift rejection**

Run:

```bash
/home/ww/miniconda3/bin/python -m pytest -q tests/test_d8_residual_config.py
/home/ww/miniconda3/bin/ruff check src/cmc_bbdm/cpb_diffusion_marginalization/residual_config.py tests/test_d8_residual_config.py
```

Expected: all tests pass and Ruff reports `All checks passed!`.

---

### Task 2: Build exact measured-field residual targets

**Files:**
- Create: `src/cmc_bbdm/cpb_diffusion_marginalization/residual_targets.py`
- Test: `tests/test_d8_residual_targets.py`

- [ ] **Step 1: Write decomposition and isolation RED tests**

```python
def test_residual_target_round_trip_is_exact(inner_fold, selected_pilot_scaffold):
    batch = build_residual_target_batch(
        inner_fold, selected_pilot_scaffold, role="fit"
    )
    np.testing.assert_allclose(batch.stable + batch.residual, batch.measured, atol=1e-7)
    np.testing.assert_allclose(batch.training_target * 2.0, batch.residual, atol=1e-7)
    assert batch.training_target.min() >= -1.0
    assert batch.training_target.max() <= 1.0


def test_fit_target_builder_rejects_query_and_outer_rows(
    inner_fold, selected_pilot_scaffold
):
    batch = build_residual_target_batch(
        inner_fold, selected_pilot_scaffold, role="fit"
    )
    assert set(batch.specimen_ids) == set(inner_fold.fit_specimen_ids)
    assert not set(batch.specimen_ids) & set(inner_fold.query_specimen_ids)
    assert inner_fold.outer_domain not in batch.dataset_ids
```

- [ ] **Step 2: Verify RED**

Run:

```bash
/home/ww/miniconda3/bin/python -m pytest -q tests/test_d8_residual_targets.py
```

Expected: collection fails because `build_residual_target_batch` is missing.

- [ ] **Step 3: Implement immutable target batches**

Implement:

```python
@dataclass(frozen=True, slots=True)
class ResidualTargetBatch:
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    measured: np.ndarray
    stable: np.ndarray
    stable_condition: np.ndarray
    residual: np.ndarray
    training_target: np.ndarray
    decomposition_state_sha256: str
    state_sha256: str


def build_residual_target_batch(
    authority: D8InnerFold | D8SearchView,
    scaffold: PilotScaffold,
    *,
    role: Literal["fit", "query", "outer_fit"],
) -> ResidualTargetBatch:
    validated = validate_residual_target_authority(authority, role=role)
    indices = validated.indices
    measured = load_registered_fields(validated.data_view, indices)
    selected = tuple(
        decompose_residual(
            field,
            family=scaffold.decomposition_family,
            parameters=dict(scaffold.decomposition_parameters),
        ).selected
        for field in measured
    )
    residual = readonly_field_batch(np.stack(selected))
    stable = readonly_field_batch(measured - residual)
    stable_condition = readonly_field_batch(np.clip(stable, -1.0, 1.0))
    target = readonly_field_batch(residual / np.float32(2.0))
    return issue_residual_target_batch(
        validated=validated,
        measured=measured,
        stable=stable,
        stable_condition=stable_condition,
        residual=residual,
        training_target=target,
        scaffold=scaffold,
    )
```

Use the already registered Gaussian, wavelet, or Fourier implementation from `decomposition.py`; do not consume P6 reconstruction residuals. Query construction must require a frozen checkpoint token. No function may accept CAI targets.

Retain raw `stable = measured - residual` for exact reconstruction. The model
condition is exactly `clip(stable, -1, 1)` because registered wavelet
decompositions can overshoot the normalized image range. Bind both arrays in
the batch state and reject any condition that is not the exact registered
clip. Preflight every fit authority and fail closed if `residual / 2` is not
bounded by `[-1,1]`.

- [ ] **Step 4: Add variant reconstruction and morphology-gate tests**

Assert:

```text
D_variant = clip(D + alpha * (2 * sampled_target - residual), -1, 1)
```

and verify that the existing native-frame gate is rerun for every variant, with overall acceptance `>=0.80` and per-query-domain acceptance `>=0.60` required for eligibility.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
/home/ww/miniconda3/bin/python -m pytest -q tests/test_d8_residual_targets.py tests/test_d8_decomposition.py tests/test_d8_variants.py
```

Expected: all tests pass.

---

### Task 3: Implement the registered conditional diffusion model

**Files:**
- Create: `src/cmc_bbdm/cpb_diffusion_marginalization/residual_model.py`
- Test: `tests/test_d8_residual_model.py`

- [ ] **Step 1: Write model-roster RED tests**

```python
@pytest.mark.parametrize("candidate_id", [f"RD{i}" for i in range(8)])
def test_registered_model_has_exact_shape_and_no_response_input(config, candidate_id):
    model = build_residual_unet(config.candidate(candidate_id))
    assert model.config.sample_size == 64
    assert model.config.in_channels == 6
    assert model.config.out_channels == 3
    assert "response" not in inspect.signature(model.forward).parameters
```

- [ ] **Step 2: Verify RED**

Run:

```bash
/home/ww/miniconda3/bin/python -m pytest -q tests/test_d8_residual_model.py
```

Expected: collection fails because `residual_model` is missing.

- [ ] **Step 3: Implement RD0--RD7 and loss reconstruction**

Implement exact UNet construction, squared-cosine and linear schedulers, epsilon/v/direct prediction, analytic clean-target reconstruction, orthonormal FFT magnitude L1, and sigma-2 low-pass L1. The public loss API is:

```python
def residual_diffusion_loss(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    clean_target: torch.Tensor,
    stable_condition: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    candidate: ResidualCandidate,
) -> LossBreakdown:
    validate_loss_inputs(
        clean_target=clean_target,
        stable_condition=stable_condition,
        timesteps=timesteps,
        noise=noise,
    )
    noisy = scheduler.add_noise(clean_target, noise, timesteps)
    prediction = model(
        torch.cat((noisy, stable_condition), dim=1), timesteps
    ).sample
    clean_prediction, base_target = reconstruct_registered_targets(
        prediction=prediction,
        clean_target=clean_target,
        noise=noise,
        noisy=noisy,
        timesteps=timesteps,
        scheduler=scheduler,
        prediction_type=candidate.prediction_type,
    )
    diffusion = torch.nn.functional.mse_loss(prediction, base_target)
    spectral = fft_magnitude_l1(clean_prediction, clean_target)
    low_pass = gaussian_low_pass_l1(clean_prediction, clean_target, sigma=2.0)
    total = (
        diffusion
        + candidate.spectral_weight * spectral
        + candidate.low_pass_weight * low_pass
    )
    return LossBreakdown(total, diffusion, spectral, low_pass)
```

- [ ] **Step 4: Add deterministic DDIM and checkpoint RED tests**

The same model state, condition, specimen identity, seed, 25 steps, and `eta=1.0` must return byte-identical arrays. Checkpoint loading must reject changed architecture, candidate ID, config hash, split hash, tensor digest, or runtime-major version.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 /home/ww/miniconda3/bin/python -m pytest -q tests/test_d8_residual_model.py
```

Expected: all tests pass on CPU and CUDA-specific tests pass when CUDA is available.

---

### Task 4: Train one inner-fold candidate deterministically

**Files:**
- Create: `src/cmc_bbdm/cpb_diffusion_marginalization/residual_training.py`
- Test: `tests/test_d8_residual_training.py`

- [ ] **Step 1: Write RED tests for fit/query boundaries**

```python
def test_training_never_consumes_query_or_response(inner_fold, candidate, recorder):
    result = train_inner_residual_model(
        inner_fold,
        candidate,
        epochs=1,
        seed=20260823,
        recorder=recorder,
    )
    assert set(result.fit_specimen_ids) == set(inner_fold.fit_specimen_ids)
    assert not set(result.fit_specimen_ids) & set(inner_fold.query_specimen_ids)
    assert recorder.response_reads == 0
```

- [ ] **Step 2: Verify RED**

Run:

```bash
/home/ww/miniconda3/bin/python -m pytest -q tests/test_d8_residual_training.py
```

Expected: collection fails because `train_inner_residual_model` is missing.

- [ ] **Step 3: Implement AdamW training and final-epoch selection**

Use batch size 32, learning rate `2e-4`, weight decay `1e-4`, 1,000 diffusion timesteps, registered identity-derived dataloader/noise seeds, no early stopping, and only the final epoch checkpoint. Serialize per-epoch finite loss components, sample counts, split state, model state, optimizer settings, runtime, and checkpoint tensor digest.

- [ ] **Step 4: Add replay, nonfinite, interruption, and seed tests**

Run a two-epoch reduced fixture twice and require identical tensor/scientific digests. Reject NaN/Inf, partial batches with identity loss, query access before freeze, checkpoint overwrite, and an unregistered seed.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 /home/ww/miniconda3/bin/python -m pytest -q tests/test_d8_residual_training.py
```

Expected: all tests pass.

---

### Task 5: Evaluate Stage A and Stage B without outer leakage

**Files:**
- Create: `src/cmc_bbdm/cpb_diffusion_marginalization/residual_search.py`
- Test: `tests/test_d8_residual_search.py`

- [ ] **Step 1: Write Stage-A Cartesian-contract RED tests**

Require exactly:

```text
6 prospective outers x 5 inner queries x 8 candidates x 1 seed = 240 model fits
```

and exact key `(outer_domain, query_domain, candidate_id, seed)`.

- [ ] **Step 2: Verify RED**

Run:

```bash
/home/ww/miniconda3/bin/python -m pytest -q tests/test_d8_residual_search.py -k stage_a
```

Expected: collection fails because the search module is missing.

- [ ] **Step 3: Implement Stage-A ranking**

For every candidate compute:

```text
J = mean(inner-domain MAE) + 0.25 * worst MAE + 0.10 * domain SD
```

Only eligible morphology-gate candidates rank. Break ties by mean MAE, worst MAE, parameter count, then candidate ID. Promote exactly two per prospective outer.

- [ ] **Step 4: Write Stage-B RED tests**

Require exactly:

```text
6 prospective outers x 5 inner queries x 2 finalists x 3 seeds = 180 model fits
```

Average the three specimen predictions before domain MAE and `J`; reject scoring code that treats seeds, variants, or specimens as independent domains.

- [ ] **Step 5: Implement Stage-B, incumbent comparison, and optional ensemble**

Compare the best residual candidate, frozen Pilot pipeline, and raw B0 using the same inner identities. Promote residual diffusion only at `1e-4` improvement; promote a two-member nonnegative cross-fit ensemble only for a further `1e-4`. Train three final checkpoints on each complete five-domain outer-fit view only when residual diffusion is selected.

- [ ] **Step 6: Verify GREEN and leakage mutation tests**

Run:

```bash
/home/ww/miniconda3/bin/python -m pytest -q tests/test_d8_residual_search.py
```

Mutating an outer image or outer CAI must not change any Stage-A/B state; mutating an inner-query CAI may change its score but not its generator checkpoint.

---

### Task 6: Publish and validate the pre-outer package

**Files:**
- Create: `src/cmc_bbdm/cpb_diffusion_marginalization/residual_artifacts.py`
- Test: `tests/test_d8_residual_artifacts.py`

- [ ] **Step 1: Write exact package-schema RED tests**

Require exactly:

```text
config.yaml
candidate_index.csv
training.csv
inner_predictions.csv
inner_metrics.csv
checkpoint_index.csv
selected_generators.json
frozen_pipelines.json
models/
REPORT.md
artifact_manifest.json
CHECKSUMS.sha256
```

- [ ] **Step 2: Verify RED**

Run:

```bash
/home/ww/miniconda3/bin/python -m pytest -q tests/test_d8_residual_artifacts.py
```

Expected: collection fails because `residual_artifacts` is missing.

- [ ] **Step 3: Implement independent recomputation and transaction publication**

The validator must rederive exact row sets, unique keys, finite metrics, Stage-A/B scores, promotions, model tensor hashes, runtime/code/source bindings, six pipeline states, and `outer_evaluation_count=0`. Publication must use an exclusive lock, owner-marked staging, validation before commit, conservative rollback, crash recovery, and preservation of the only valid previous package on double failure.

- [ ] **Step 4: Add tamper and fault-injection tests**

Cover single- and synchronized tampering of source hashes, split identities, scores, checkpoint config/state, selected pipelines, scientific digest, unknown paths, symlinks, FIFO, interrupted first publication, concurrent publishers, publish rename failure, rollback rename failure, and restart recovery.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
/home/ww/miniconda3/bin/python -m pytest -q tests/test_d8_residual_artifacts.py
```

Expected: all tests pass without leaving transaction directories.

---

### Task 7: Add the registered three-GPU CLI

**Files:**
- Create: `scripts/run_d8_residual_diffusion.py`
- Create: `scripts/run_d8_residual_diffusion.sh`
- Test: `tests/test_d8_residual_cli.py`

- [ ] **Step 1: Write CLI RED tests**

Require commands:

```text
smoke
train
validate
replay
```

The shell wrapper must freeze BLAS threads to one, require three visible A40 GPUs for `train`, assign exactly two prospective outers per worker, and reject direct outer evaluation arguments.

- [ ] **Step 2: Verify RED**

Run:

```bash
/home/ww/miniconda3/bin/python -m pytest -q tests/test_d8_residual_cli.py
```

Expected: tests fail because the scripts are missing.

- [ ] **Step 3: Implement isolated workers and validated merge**

Workers write to distinct temporary leaves and publish signed worker manifests. The parent validates all three manifests, exact outer allocation, code/config/source hashes, and zero outer evaluations before merging. `replay` writes a distinct output leaf and never overwrites production.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
/home/ww/miniconda3/bin/python -m pytest -q tests/test_d8_residual_cli.py
bash -n scripts/run_d8_residual_diffusion.sh
```

Expected: all tests pass and shell syntax is valid.

---

### Task 8: Run the complete pre-outer verification gate

**Files:**
- Modify: `docs/D8_PILOT_DECISION.md`
- Create: `docs/D8_RESIDUAL_DIFFUSION_PREOUTER_DECISION.md`
- Create: `results/d8_residual_diffusion_search/`
- Create: `results/replay/d8_residual_diffusion_search/`

- [ ] **Step 1: Run focused and compatibility tests**

```bash
/home/ww/miniconda3/bin/python -m pytest -q tests/test_d8_*.py
/home/ww/miniconda3/bin/ruff check src/cmc_bbdm/cpb_diffusion_marginalization scripts/run_d8_residual_diffusion.py tests/test_d8_*.py
/home/ww/miniconda3/bin/python -m compileall -q src/cmc_bbdm/cpb_diffusion_marginalization scripts/run_d8_residual_diffusion.py
```

Expected: all tests and static checks pass.

- [ ] **Step 2: Run the registered smoke command**

```bash
scripts/run_d8_residual_diffusion.sh smoke
```

Expected: reduced Stage-A/B, model load, package publication, and package validation pass with `outer_evaluation_count=0`.

- [ ] **Step 3: Run production pre-outer training**

```bash
scripts/run_d8_residual_diffusion.sh train
```

Expected: `results/d8_residual_diffusion_search` publishes atomically with six frozen pipeline records and no outer predictions.

- [ ] **Step 4: Run independent replay**

```bash
scripts/run_d8_residual_diffusion.sh replay
scripts/run_d8_residual_diffusion.sh validate
```

Expected: both packages validate, all checkpoint scientific records match, and canonical scientific digests are identical after excluding only registered timing paths.

- [ ] **Step 5: Freeze the gate decision**

Record the production/replay hashes, selected pipeline per prospective outer, residual-versus-incumbent decisions, test commands, and explicit `formal_outer_status=BLOCKED_PENDING_AUTHORIZED_ONE_WAY_RUN` in `docs/D8_RESIDUAL_DIFFUSION_PREOUTER_DECISION.md`.

Formal outer evaluation may begin only after all five steps pass. No command in this plan performs an outer evaluation.
