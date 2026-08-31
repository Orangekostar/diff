# Codex Handoff: Agentic NDE Author Registration Reopen

## Repository Identity

- Repository: `git@github.com:Orangekostar/diff.git`
- Worktree: `/home/ww/diff/.worktrees/agentic-task-driven-nde-author-registration`
- Branch: `research/agentic-task-driven-nde-author-registration`
- Exact base: `3cb63b544b6c13047773c0eda045558ff4466afa`
- Frozen P1 evidence commit before this handoff: `5cf1e6086bf1901125c97fd0c135f548c9a27e97`
- Final branch and remote tips: verify with the commands in `GitHub Verification`; the
  enclosing handoff commit cannot contain its own content-dependent Git SHA.

## Stage Status

- Historical P0: `P0_SPATIAL_REGISTRATION_NO_GO` (preserved, not overwritten).
- P0R: `P0R_AUTHOR_REGISTRATION_GO`.
- P1: `P1_SURFACE_VISUAL_OBSERVABILITY_NO_GO`.
- P2: `NOT_RUN_NOT_AUTHORIZED`.
- P3: `NOT_RUN_NOT_AUTHORIZED`.
- P4: `NOT_RUN_NOT_AUTHORIZED`.
- No post-result rescue, encoder change, threshold change, registration change, or
  target-driven re-selection was run.

## Historical P0

- Exact identities bound: 276.
- Authorized cross-instrument registrations: 0.
- Historical status remains `P0_SPATIAL_REGISTRATION_NO_GO`.
- Historical P0 package and decision/audit artifacts are byte-preserved by the
  frozen-path diff gate.

## P0R Author Registration

- Status: `P0R_AUTHOR_REGISTRATION_GO`.
- Authorized roster: 276/276.
- Orientation: one global `ROT90` transform.
- Mapping basis: `AUTHOR_FULL_FRAME_PIXEL_CORRESPONDENCE`.
- No per-specimen orientation selection, reflection, resize, interpolation, or
  manual target alignment.
- Cross-modal physical millimetres were not used.
- CAI, oracle, target-domain labels, and damage outcomes were not accessed for
  orientation or registration.
- Authorized roster SHA-256:
  `4fd8c6076dd3fcdf908a73739251db215fcb01f570f1a930b7faf250fe6d285a`.
- Registration authority SHA-256:
  `38ab3cf32e866cda447a5edf2637fa502406c4c5c574bc966c13cc1cbbd2553a`.
- Grid mapping state SHA-256:
  `40b9a2aa98a70ac1a5929e73b0b25b69e865ac80ae2035d625585b11dd85447c`.
- Surface manifest SHA-256:
  `e3274cacfb50b5b136e81ed3c107ac79d134e37d5d14ccc296b4e1ae2a51b44b`.
- Registration CSV SHA-256:
  `43bafc8819a1bea0df7f7ea9fa2b1dc194591ef9a4b58e8dcf3e64c13622e265`.
- Grid mapping CSV SHA-256:
  `fb076daefbb0b748184d10b777268dfd13777a3e78cb7fed104252e6d62afd35`.
- Package summary SHA-256:
  `a19cc2f0528da4842e04a521711d590f44bd226e92bc69e5f84de4fcb97cd02f`.
- Package checksum-ledger SHA-256:
  `470a1df4ff19930f5924493fb6a51c49084c76a37376fa266a6499baa9ffbb82`.
- Author authority artifact SHA-256:
  `60a3aeb39256d1b88692439677a20f03f3b987fde3faa1c69224aea1c1f76bf8`.
- Author statement SHA-256:
  `3560662d4509ea3e059d597cedca15950cce02f706a992330b161381acfba6ba`.

Per-domain authorized registrations:

| Domain | Specimens |
|---|---:|
| `74t7kcdgkr` | 45 |
| `cgtnjyggtm` | 49 |
| `w68dtmpfyf` | 43 |
| `xcmzfsbd9t` | 59 |
| `yfxyg8jm46` | 42 |
| `ykhs7s2dck` | 38 |

Raw/processed provenance:

- 259 unique raw screenshots.
- 19 unique multi-panel screenshots.
- 17 multi-panel screenshots resolve both selected panels; 2 resolve one selected
  panel.
- 276/276 processed surface crops replay as RGB decode plus deterministic
  axis-aligned crop.
- Historical processing rotation is `IDENTITY`; P0R then applies the separately
  authorized global `ROT90` surface-to-C-scan registration.
- No processing resize, interpolation, or reflection was found.

## P1 Frozen Protocol

- Config SHA-256:
  `00f3e0cf45d45dd64c20852513d9b23c69a3c29ad7e0d0d7220fb13f86bfe92e`.
- Protocol SHA-256:
  `6b73c4d179482feb46ff6d0c15c9b043ff23d44d1ad88f0ccc65bc21b90721b5`.
- The protocol was committed before formal target-domain evaluation.
- Target: 17,664 frozen A2 post-scout mechanical-value rows (276 x 64).
- Outer split: leave one domain out; inner split: leave one source domain out.
- Encoder: frozen ImageNet1K-V1 ResNet18, RGB, 512-dimensional penultimate
  embedding. The encoder was not trained.
- Source-only heads: preregistered Ridge and bounded MLP candidates. These heads
  were fitted only on the five source domains for each outer fold.
- Surface feature-bank state SHA-256:
  `01c17be6447b3b567db63005a056b0e1f8b73be82be9ee18d7a82a192c938a47`.
- Feature-manifest SHA-256:
  `0106cbb00fd5e07c216d8852fcc7077e5c66bcc3af9b6895d0c2ffabfd48cc8b`.
- Global array SHA-256:
  `7776822bb6aa3bad4ac182806faa313225131895e4ee73b329d36e26bbf1cfda`.
- Correct-local array SHA-256:
  `d5133a43ae1a9053d42c29f23e20649fc871b0f0718c26f87c495f86e1e4b00c`.
- Wrong-orientation-local array SHA-256:
  `dc2ad11858680b48cd0363cb59c17e0218fc934b1e14465b22697dc17222e7be`.
- Every outer score table was atomically checksum-frozen before its target-domain
  mechanical labels were read.

## P1 Result

- Status: `P1_SURFACE_VISUAL_OBSERVABILITY_NO_GO`.
- Authorized route: none.
- Decision state SHA-256:
  `e563a42cea940e2ebb9b82a7f2757d46ac54c688416556e3e6a351607601bd95`.
- Aggregate state SHA-256:
  `6a372a226825abe926b7a10119c5cc1753316e73db729cb407dc6429f26df59d`.
- Bootstrap: 100,000 synchronized specimen resamples, equal-domain aggregate.
- Strong old reference CAI AUEBC: `0.09202411704871961`.
- Proposed CAI AUEBC: `0.09202411587521975`.
- Proposed minus old is scientifically negligible: the selected proposed fusion
  used `lambda=0` in all six outer folds and therefore retained the old ranking.
- Paired old-minus-proposed effect: `1.1734998728786495e-09`.
- 95% CI: `[-1.4800368926610731e-09, 3.835603117861021e-09]`.
- Improved domains: 4/6, but the paired lower bound is not positive.
- Mechanical-oracle gap closure: `9.798758058215481e-08`, below 20%.
- Proposed NDCG@10: `0.6876884535291307` (old: same).
- Proposed next-action regret: `0.015083156595862604` (old: same).
- Global-context CAI AUEBC: `0.09218686262516539`, worse than old.
- Old-minus-global effect: `-0.00016274557644576599`.
- 95% CI: `[-0.00119033670349159, 0.0008634693037827474]`.
- Global improved domains: 3/6; global gate failed all four conditions.
- Shuffled/wrong-orientation controls do not establish registered spatial value;
  every required spatial lower confidence bound is non-positive.
- Ranking improvement gate is false.

Outer proposed selections:

| Outer domain | Representation | Head | Fusion lambda |
|---|---|---|---:|
| `74t7kcdgkr` | `LOCAL_GLOBAL` | `ridge_alpha_100` | 0.0 |
| `cgtnjyggtm` | `LOCAL` | `ridge_alpha_100` | 0.0 |
| `w68dtmpfyf` | `LOCAL_GLOBAL` | `ridge_alpha_100` | 0.0 |
| `xcmzfsbd9t` | `LOCAL_GLOBAL` | `ridge_alpha_100` | 0.0 |
| `yfxyg8jm46` | `LOCAL` | `ridge_alpha_100` | 0.0 |
| `ykhs7s2dck` | `LOCAL_GLOBAL` | `ridge_alpha_100` | 0.0 |

## P1 Package and Replay

- Formal package: `results/agentic_task_driven_nde/p1_visual_observability/`.
- Deterministic replay package:
  `results/agentic_task_driven_nde/replay/p1_visual_observability/`.
- Each package contains exactly the 14 preregistered files.
- Formal/replay `diff -qr`: empty.
- Summary SHA-256:
  `f312d1c7ab0f6096d56e3f9bc46107929ea83976701147e1ac3990cd29d976ea`.
- Artifact manifest SHA-256:
  `c4058abd0dfbc12fe44193eff604f705d9e658d17e86ecd0d89bc6e9c2136d65`.
- Checksum ledger SHA-256:
  `6d856b4630aaf06500273538e48f970e81c620d4c13548370eeaa9eea9eaf4fc`.
- Replay recomputed source hashes, P0R registration/roster, feature cache hashes,
  all source-only model-selection identities, all ranking metrics, CAI AUEBC,
  100,000-resample bootstrap effects, and the P1 gate.
- Replay then rebuilt all 14 files byte-identically.

## Commands and Evidence

P0/P0R replay:

```bash
PYTHONPATH=src python scripts/run_agentic_nde.py replay-p0 \
  --path results/agentic_task_driven_nde/p0_registration
PYTHONPATH=src python scripts/run_agentic_nde.py replay-p0r \
  --path results/agentic_task_driven_nde/p0r_author_registration
```

Result: `P0_SPATIAL_REGISTRATION_NO_GO` and
`P0R_AUTHOR_REGISTRATION_GO`.

Feature-cache reproducibility command:

```bash
PYTHONPATH=src python scripts/run_agentic_nde.py materialize-p1-features \
  --config paper_v3/configs/agentic_nde_p1_visual_observability.yaml \
  --surface-root /home/ww/paper3/cmc_damage_inference \
  --output results/agentic_task_driven_nde/.work/p1_visual_observability/features-replay \
  --project-root .
```

Formal P1 used the same code path and produced the feature state recorded above.
The feature replay command is retained separately because foundation encoder
inference is expensive; feature arrays are hash-bound and excluded from Git.

Formal P1:

```bash
PYTHONPATH=src python scripts/run_agentic_nde.py run-p1 \
  --config paper_v3/configs/agentic_nde_p1_visual_observability.yaml \
  --research-root /home/ww/paper3/cmc_damage_inference \
  --project-root . \
  --feature-root results/agentic_task_driven_nde/.work/p1_visual_observability/features \
  --device cuda:0
```

Scientific replay:

```bash
PYTHONPATH=src python scripts/run_agentic_nde.py replay-p1 \
  --path results/agentic_task_driven_nde/p1_visual_observability \
  --project-root . \
  --feature-root results/agentic_task_driven_nde/.work/p1_visual_observability/features \
  --replay-output results/agentic_task_driven_nde/replay/p1_visual_observability
```

Actual replay result: `P1_SURFACE_VISUAL_OBSERVABILITY_NO_GO` and byte-identical
formal/replay packages.

Tests:

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  $(rg --files tests | rg '/test_agentic_nde.*\.py$' | sort)
```

Result: `196 passed in 41.42s`.

Static checks:

```bash
python -m ruff check src/cmc_bbdm/agentic_nde scripts/run_agentic_nde.py \
  $(rg --files tests | rg '/test_agentic_nde.*\.py$' | sort)
git diff --check
```

Result: passed.

## Frozen-Path Audit

Command:

```bash
git diff --name-only 3cb63b544b6c13047773c0eda045558ff4466afa -- \
  results/mva results/mvd results/mavis results/mavis_science_closure \
  results/p1_full_field_oracle results/p3_spatial_specificity \
  results/p5_sparse_scan artifacts/mavis artifacts/mavis_science_closure \
  artifacts/mvd_authority artifacts/mavis_authority \
  artifacts/aei_information_hierarchy results/damage_to_failure_response \
  artifacts/damage_to_failure_response paper_aei_information_hierarchy \
  src/cmc_bbdm/mva src/cmc_bbdm/mvd src/cmc_bbdm/mavis \
  results/agentic_task_driven_nde/p0_registration \
  artifacts/agentic_task_driven_nde/P0_REGISTRATION_DECISION.md \
  artifacts/agentic_task_driven_nde/P0_SURFACE_CSCAN_AUTHORITY_AUDIT.md
```

Result: empty.

## Changed-File Inventory

The base-to-evidence diff contains 101 files. Complete groups are:

- Governance/config: `.gitignore`, five P0R/P1 design/plan/protocol documents,
  and the two frozen P0R/P1 YAML configs.
- New/extended runtime: `scripts/run_agentic_nde.py` and the Agentic NDE modules
  `author_authority.py`, `contracts.py`, `p0r.py`, `p0r_artifacts.py`,
  `p0r_qc.py`, `p1.py`, `p1_artifacts.py`, `p1_cai.py`, `p1_execution.py`,
  `p1_pipeline.py`, `scan_frame_provenance.py`, `surface_cells.py`,
  `surface_encoder.py`, and `visual_observability.py`.
- Tests: the 19 P0R/P1 files matching
  `tests/test_agentic_nde_{author_authority,p0r*,p1*,raw_panel_to_crop,rot90_semantics,scan_frame_provenance,surface_cells,surface_encoder,visual_observability}.py`.
- P0R package: all 11 files under
  `results/agentic_task_driven_nde/p0r_author_registration/`.
- P1 formal package: all 14 files under
  `results/agentic_task_driven_nde/p1_visual_observability/`.
- P1 replay package: the same 14 files under
  `results/agentic_task_driven_nde/replay/p1_visual_observability/`.
- Audit artifacts: three author/P0R Markdown authorities, two P1 Markdown
  decisions, and the complete 12-image P0R QC overlay package with manifest and
  checksums.

Exact machine-readable inventory command:

```bash
git diff --name-only 3cb63b544b6c13047773c0eda045558ff4466afa..HEAD | sort
```

## GitHub Verification

Push command:

```bash
git push -u origin research/agentic-task-driven-nde-author-registration
```

Verification commands:

```bash
git rev-parse HEAD
git rev-parse '@{upstream}'
git ls-remote origin refs/heads/research/agentic-task-driven-nde-author-registration
git status --short
```

Acceptance requires identical local/upstream/remote SHAs and an empty status.
No force push, PR creation, or merge to `main` is part of this handoff.
