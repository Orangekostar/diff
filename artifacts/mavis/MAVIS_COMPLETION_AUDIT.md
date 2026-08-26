# MAVIS Completion Audit

Status: `COMPLETE`. Frozen claim tier: `Tier B`.

## Frozen Result

| Item | Frozen value | Authority |
|---|---:|---|
| Specimens / held-out domains | 276 / 6 | `results/mavis/p7_final_frozen_eval/summary.json` |
| Strongest deployable baseline | `mvd_m1_o2` | `results/mavis/p7_final_frozen_eval/claim_evidence.csv` |
| Baseline CAI AUEBC | 0.1249920401 | `results/mavis/p7_final_frozen_eval/claim_evidence.csv` |
| Aggregated MAVIS CAI AUEBC | 0.1250531822 | `results/mavis/p7_final_frozen_eval/claim_evidence.csv` |
| Safe MAVIS CAI AUEBC | 0.1250488183 | `results/mavis/p7_final_frozen_eval/claim_evidence.csv` |
| Improved held-out domains | 2 / 6 | `results/mavis/p7_final_frozen_eval/claim_evidence.csv` |
| P7 state SHA-256 | `6bf473e811314b5c2897cdf789f71854ea9899bcbe13c8e16664ae32c03ed78c` | `results/mavis/p7_final_frozen_eval/summary.json` |

Tier S is not supported because MAVIS does not beat the strongest deployable
baseline. Tier A is not supported because safe routing is not demonstrably
non-degrading and the high-confidence benefit interval crosses zero. Tier B is
the conservative frozen ceiling; unsupported positive component claims remain
explicitly unclaimed.

## Provenance And Causal Isolation

- Final configuration SHA-256: `e99b47e161663fdaefe28719d16321a010f95b4ad8cf8f506a6e18d1d7f57b9d`.
- Development package SHA-256: `89c96fdf6d9da3301569dad301477e86b75ef8a0107ad5795c977d4608e6a6fe`.
- P1/P4/P5/P6 state hashes are bound in `results/mavis/p7_final_frozen_eval/summary.json`.
- All six P5 folds record `target_state_count=0` and `target_data_used_for_training=false`.
- All six P6 folds record `target_outcomes_used_for_selection=false`.
- All six P7 workers record `future_target_content_used_by_policy=false` and `target_true_cai_used_by_policy=false`.
- The final package records `target_data_used_for_training_or_selection=false`.

## Reproducibility

- `results/mavis/p5_aggregation/` and `results/mavis/p6_safety/` were generated twice and compared byte-for-byte.
- `results/mavis/replay/` independently regenerated the complete P7 package.
- Replay result: 31 files, 7,891,077 bytes, `byte_identical=true`.
- Replay tree SHA-256: `931dc86c26caf1c7246709c4706a7cd0428e3a1533b6ff1ad3c2ad8f9517d1e4`.
- Formal and replay `CHECKSUMS.sha256` files pass without mismatch.

## Verification Record

| Check | Result |
|---|---|
| Mandatory test-name roster | 15 / 15 present |
| `PYTHONPATH=src python -m pytest -q tests/test_mavis*.py` | 115 passed |
| `PYTHONPATH=src python -m pytest -q tests/test_mvd*.py` | 29 passed |
| Historical MVA regression in the authority repository | 126 passed |
| `p5-verify`, `p6-verify`, `p7-verify` | passed |
| `p7-replay-verify` | passed, byte-identical |
| `ruff check .` | passed |
| `git diff --check` | passed |
| Final figure render inspection | passed; PNG is nonblank and labels are legible |

## Claim Boundaries

The evidence supports only retrospective normalized-raster closed-loop
feasibility. External method performance, scanner path or time reduction,
industrial deployment, and external generalization were not evaluated and are
not claimed. The detailed claim-support matrix is
`artifacts/mavis/MANUSCRIPT_EVIDENCE_MAP.md`.

## Stage Commits

- `88fa427` audit: map MAVIS authority and frozen evidence
- `0e372e6` data: add MAVIS authority and causal reveal
- `bbd97d0` data: add strict-OOF MAVIS state bank
- `d9e41aa` model: add MRIS and mechanics head
- `bd9fc04` fix: include frozen P2 checkpoints
- `2f5d458` model: add dynamic mechanical VoI
- `2fb237c` policy: add cost-aware closed-loop rollout
- `9e7774f` safety: add source-selected MAVIS fallback
- `5c91073` train: add source-only on-policy aggregation
- `191f824` eval: complete MAVIS baselines and ablations

P5 and P6 remain separate scientific commits. P6 completed first while the
long-running P5 folds were still executing, so their commit chronology reflects
completion order rather than conceptual stage numbering.
