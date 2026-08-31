# Codex Handoff: Agentic Task-Driven NDE

Date: 2026-08-31 UTC

## Repository identity

- Repository: `git@github.com:Orangekostar/diff.git`
- Base SHA: `15db6edad14ef36364fbda17945ccc924f600e47`
- Branch: `research/agentic-task-driven-nde`
- Worktree: `local:.worktrees/agentic-task-driven-nde`
- Protocol commit: `27554566de2297714a721f8d4763ef56611e2376`
- P0 stage commit: `bc943a9f6f9494c2068f23c9ecabad3fc5af749c`
- Final handoff commit: `SELF` (the commit containing this file)
- Remote SHA: `SELF` after the required push; the exact resolved SHA is reported
  by `git rev-parse HEAD` and `git ls-remote` in the final user response.
- Clean status: required and verified after the final push.

`SELF` is necessary because a Git commit cannot contain its own literal hash:
changing this file to insert that hash would create a different commit hash.

## External-repository audit

No external code was copied. Full file hashes and source observations are in
`EXTERNAL_REPOSITORY_AUDIT.md`.

| Repository | Revision | License at audit | Files inspected | Copied code | Purpose |
| --- | --- | --- | --- | --- | --- |
| DriveAgent-R1 | `a1abd75fe80cc6868f74f308f71042f7f0d054a8` | no top-level license identified | tool registry, tool utils, modified GRPO trainer | NO | conditional P4 architecture reference only |
| Qwen2.5-VL | `96588727e44c78b25ba03ea03b8e12f7e64fd0da` | Apache-2.0 | `README.md`, `LICENSE` | NO | conditional frozen legal-grid diagnostic reference |
| US-VLA | `dde576ae9992d33231f3de925ff44a1fc818ab00` | Apache-2.0 | ultrasound fusion source, `README.md`, `LICENSE` | NO | conditional fusion reference only |
| ActiveVLA | `4450b1188a24f3f1d1d3fb3e47d226d150b9e470` | no top-level license identified | `README.md` | NO | coarse-to-fine conceptual reference only |
| GroundingDINO | `856dde20aee659246248e20734ef9ba5214f5e44` | Apache-2.0 | `README.md`, `LICENSE` | NO | optional source-side geometry helper |
| SAM 2 | `2b90b9f5ceec907a1c18123530e92e794ad901a4` | Apache-2.0 | `README.md`, `LICENSE` | NO | optional prompted-region helper |
| RoboSpection | `cacd70e8aa734383b6046ec0818d9af6d06bfcbf` | Apache-2.0 | `README.md`, `LICENSE`, `src/` listing | NO | high-/low-level control boundary reference |
| TADRED | `7cf5ab47e2504da7bf7075a40f852fe8be8b951f` | Apache-2.0 | `README.md`, `LICENSE`, package listing | NO | task-driven experimental-design prior-art boundary |

P0-P3 have no external runtime repository dependency. No audited repository
supplies the missing Hasebe surface-to-C-scan transform.

## Literature / novelty boundary

The literature ledger covers autonomous ultrasonic point selection, task-driven
channel design, multimodal ultrasonic localization, ultrasound VLA, VLM active
perception, LLM inspection planning, the Hasebe authority, and direct
surface-to-C-scan prediction. MDPI sources were excluded.

```text
Novelty status: SEARCHED_NOT_ASSUMED
Direct novelty claim authorized: NO
Empirical route retained: YES, subject to P0 and P1 gates.
```

P0 subsequently failed, so the empirical route was not executed beyond P0.

## P0

- Surface pair count: 276 exact identities and hashes.
- Per-domain count: `45 / 49 / 43 / 59 / 42 / 38`.
- Surface format: 276 RGB PNG files.
- Surface resolutions: `1500x1500` (36), `1679x1679` (10),
  `1680x1680` (1), `3357x3357` (229).
- Registered C-scan crop resolutions: `340x338` (17), `352x338` (19),
  `675x674` (240).
- Registration method: evidence classes A/B/C audited; no class yielded an
  authorized real transform. Class C was not trained as a rescue.
- Registration coverage: 0/276; every primary domain is 0% authorized.
- Orientation evidence: unresolved eight-way ambiguity for 276/276.
- Package replay: deterministic negative-decision replay passes with all 818
  source authority rows rehashed.
- New training: NO.
- CAI/hidden C-scan content used for registration: NO.
- Decision: `P0_SPATIAL_REGISTRATION_NO_GO`.

Key hashes:

| Authority | SHA-256 |
| --- | --- |
| P0 config | `d117d01c4ffc113cf3bdc5b8ce46694e24f46a8e3b43395580dd65efbf113c0b` |
| surface manifest | `31aaf123d6f7b684566ef19387d11a5ef2756e5bc51e666f8b1cdcc3b9ed5fe1` |
| surface QC | `fe64077852ffea9af42e91f49b2aff03090ba9ba0900b5cacdc5454c95597b8c` |
| registration | `fc7a3b9ceef28e371a6a340444db992f1bf8aa87596cb1155c8ab48957c61f96` |
| registration QC | `36a43f4b3f57a279a6a9ce429da3d25a81929073eabb8c8b56ab9d737d8d7630` |
| source hashes | `f31fbdd9494fb23124094102181305342498b1e32e6c33c3407de5e5b7436f2a` |
| summary | `9d0032828614c79c2ed9faaf0fcfdbbb2e7c39382fa7e4cb1fc1d28ec4a9a826` |
| artifact manifest | `2ad59787b8c7c8726ce3d1dc0475540e7db56bfecb7dc0604f0b982557338523` |
| frozen A2 | `6b289f2f6f74ac75dde47ea7cbfefcda1c49f025e74227bfb34ef269182ff963` |
| legal 8x8 grid | `7a8b7b3c61b9f74597a5d1196b2e8561d10f6105b4df86a22cabbe3a5e9f56e2` |

The 12 mandatory pre-visual-model questions and local model inventory are
answered in `P0_SURFACE_CSCAN_AUTHORITY_AUDIT.md`. Only the repository
ResNet18 satisfies the strict local asset + revision + license-evidence gate;
this fact does not override the failed registration gate.

## P1

Status: `NOT_RUN_NOT_AUTHORIZED`

No visual candidate, outer-fold selection, model/prompt, no-image/shuffled/
center comparison, AUEBC, regret, interval, domain result, or oracle-gap result
was produced. No P1 config or protocol was created.

## P2

Status: `NOT_RUN_NOT_AUTHORIZED`

No `T_CAI`, `T_FIELD`, correct-/wrong-task, task-conditioned, or task-agnostic
evaluation was produced.

## P3

Status: `NOT_RUN_NOT_AUTHORIZED`

No fusion rule, fold-selected lambda, closed-loop AUEBC, baseline/MAVIS/
surface-only comparison, shuffled control, oracle-gap result, or manuscript
directory was produced.

## P4

Status: `NOT_RUN_NOT_AUTHORIZED`

No VLM revision, tool schema implementation, prompt set, illegal-call result,
task-consistency result, non-inferiority test, SFT, or RL was produced.

## Frozen-path audit

Exact command, run from the worktree:

```bash
git diff --name-only 15db6edad14ef36364fbda17945ccc924f600e47 -- \
  results/mva results/mvd results/mavis results/mavis_science_closure \
  results/p1_full_field_oracle results/p3_spatial_specificity \
  results/p5_sparse_scan artifacts/mavis artifacts/mavis_science_closure \
  artifacts/mvd_authority artifacts/mavis_authority \
  artifacts/aei_information_hierarchy results/damage_to_failure_response \
  artifacts/damage_to_failure_response paper_aei_information_hierarchy \
  src/cmc_bbdm/mva src/cmc_bbdm/mvd src/cmc_bbdm/mavis
```

Observed result: empty.

`tests/test_agentic_nde_frozen_paths.py` additionally checks both tracked diff
and worktree/untracked status within every frozen root.

## Tests and replay

Preimplementation base evidence:

```text
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_damage_response_*.py
300 passed

Selected reusable MVA/MVD/MAVIS contract tests
42 passed
```

The scientific suites were not rerun after P0 because no frozen/shared runtime
code changed. They are not represented as newly passing.

Final P0 commands and observed results:

```text
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_agentic_nde_*.py
81 passed in 1.54s

python -m ruff check src/cmc_bbdm/agentic_nde scripts/run_agentic_nde.py tests/test_agentic_nde_*.py
All checks passed!

python scripts/run_agentic_nde.py replay-p0 --path results/agentic_task_driven_nde/p0_registration --surface-root <external_hasebe_root>
P0_SPATIAL_REGISTRATION_NO_GO

(cd results/agentic_task_driven_nde/p0_registration && sha256sum -c CHECKSUMS.sha256)
9/9 payload records OK

python scripts/run_agentic_nde.py audit-p0 --config paper_v3/configs/agentic_nde_p0.yaml --surface-root <external_hasebe_root> --output <temporary_root>/p0
diff -qr results/agentic_task_driven_nde/p0_registration <temporary_root>/p0
PASS: fresh current-code package is byte-identical to the formal package

git diff --check
PASS
```

Replay rehashed 818 logical authority rows: 813 external Hasebe rows and five
compact-repository authorities. Formal package membership is exactly ten
files. Package-local checksum paths are intentional so two executions written
to different destination directories remain byte-identical.

## Changed files

```text
docs/superpowers/specs/2026-08-31-agentic-task-driven-nde-design.md
docs/superpowers/plans/2026-08-31-agentic-task-driven-nde-p0.md
artifacts/agentic_task_driven_nde/CODEX_HANDOFF_AGENTIC_TASK_DRIVEN_NDE.md
artifacts/agentic_task_driven_nde/EXTERNAL_REPOSITORY_AUDIT.md
artifacts/agentic_task_driven_nde/LITERATURE_NOVELTY_LEDGER.md
artifacts/agentic_task_driven_nde/P0_REGISTRATION_DECISION.md
artifacts/agentic_task_driven_nde/P0_SURFACE_CSCAN_AUTHORITY_AUDIT.md
paper_v3/configs/agentic_nde_p0.yaml
results/agentic_task_driven_nde/p0_registration/CHECKSUMS.sha256
results/agentic_task_driven_nde/p0_registration/REPORT.md
results/agentic_task_driven_nde/p0_registration/artifact_manifest.json
results/agentic_task_driven_nde/p0_registration/config.yaml
results/agentic_task_driven_nde/p0_registration/registration.csv
results/agentic_task_driven_nde/p0_registration/registration_qc.csv
results/agentic_task_driven_nde/p0_registration/source_hashes.csv
results/agentic_task_driven_nde/p0_registration/summary.json
results/agentic_task_driven_nde/p0_registration/surface_manifest.csv
results/agentic_task_driven_nde/p0_registration/surface_qc.csv
scripts/run_agentic_nde.py
src/cmc_bbdm/agentic_nde/__init__.py
src/cmc_bbdm/agentic_nde/artifacts.py
src/cmc_bbdm/agentic_nde/authority.py
src/cmc_bbdm/agentic_nde/contracts.py
src/cmc_bbdm/agentic_nde/frozen_bindings.py
src/cmc_bbdm/agentic_nde/grid.py
src/cmc_bbdm/agentic_nde/p0.py
src/cmc_bbdm/agentic_nde/registration.py
src/cmc_bbdm/agentic_nde/surface_qc.py
tests/test_agentic_nde_action_legality.py
tests/test_agentic_nde_artifacts.py
tests/test_agentic_nde_cli.py
tests/test_agentic_nde_frozen_a2_binding.py
tests/test_agentic_nde_frozen_paths.py
tests/test_agentic_nde_gates.py
tests/test_agentic_nde_grid_mapping.py
tests/test_agentic_nde_no_leakage.py
tests/test_agentic_nde_p0_pipeline.py
tests/test_agentic_nde_registration.py
tests/test_agentic_nde_registration_contract.py
tests/test_agentic_nde_replay.py
tests/test_agentic_nde_surface_authority.py
tests/test_agentic_nde_surface_qc.py
```

No raw public image, foundation-model weight, feature cache, embedding, or
manuscript directory was added.

## GitHub synchronization

Required push command:

```bash
git push -u origin research/agentic-task-driven-nde
```

Verification commands:

```bash
git rev-parse HEAD
git rev-parse @{upstream}
git ls-remote origin refs/heads/research/agentic-task-driven-nde
git status --short
```

Acceptance requires local HEAD, upstream, and remote branch SHA to be the same
`SELF` commit and the final worktree status to be empty. No force push, PR, or
merge is authorized.
