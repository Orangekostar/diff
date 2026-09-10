# Codex Handoff: CAI VLM-Guided Agent v2

## Repository identity

- Repository: `git@github.com:Orangekostar/diff.git`
- Branch: `research/cai-vlm-guided-agent-v2`
- Evidence base: `29b3249610c821e8f9c4f58d740588181c443dd5`
- Protocol SHA-256: `5ed1911bfe029a40962a4a1f391dc1ced1ddc6cff7fe8bb1f0953f69c8ce9726`
- Output roots: `results/cai_active_image/v2/`, `artifacts/cai_active_image/v2/`

## Executed design

- Main method: `VLM_CAI_FEEDBACK_AGENT`.
- Frozen VLM: `Qwen/Qwen2.5-VL-7B-Instruct` revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- VLM cache: 60/60 compatible hits; 0 new calls; prior execution recorded 62 calls including 2 format repairs.
- VLM cells are consumed in the once-registered 8x8 frame without a second rotation.
- First action uses only the highest available reliable VLM confidence level in C0; later actions unlock all affordable unmeasured cells.
- C0 fallback reasons are explicit in every trajectory and initialization row.
- Action: `NATIVE_8X8_FULL_CELL_V1`; endpoint cost 0.25 by exact native pixels.
- Actor state protocol `VISIBLE_ORDERED_HISTORY_BUDGET_V1` includes spent/remaining cost and ordered acquired-cell geometry history.
- Common predictor: frozen ResNet18 surface/cell tokens plus only acquired C-scan tokens; one `P_all` checkpoint is shared by every method.
- The persisted feature bank contains TRAIN/VALID MPa targets but redacts all 24 TEST targets; TEST MPa is joined from the hash-bound authority only in P5 scoring.
- Learned controls: independently trained `NO_VLM_FEEDBACK`, `VLM_OPEN_LOOP`, and `LEARNED_STATIC`.
- Fixed controls: `SERPENTINE`, `CENTER_FIRST`, `GEOMETRY_SPREAD`, and `RANDOM`.
- `BEST_NONADAPTIVE` was selected on VALID as `CENTER_FIRST` before TEST comparison.
- VALID main/no-VLM/best-nonadaptive AUEC: `78.4046` / `77.9723` / `76.3720` MPa; directional gates main-beats-no-VLM=`False`, main-beats-best-nonadaptive=`False`.
- Cached VLM region-cell counts by confidence: `{'high': 23, 'low': 15, 'medium': 226, 'unknown': 17}`; no-reliable-cue fraction: `0.0000`.
- Frozen CNN encoding took `127.987` s for the registered cohort. Historical VLM latency and per-step Actor/predictor latency are recorded separately; none is presented as scanner acquisition time.
- Domain-level MAE and each domain's gap to the common full-input predictor are exported in `domain_cai_performance.csv`.

## Primary results

| Quantity | MAE (MPa) | RMSE (MPa) | R2 |
|---|---:|---:|---:|
| Main, zero C-scan | 76.2564 | 104.4922 | -0.2205 |
| Main, cost 0.25 | 70.1317 | 96.5271 | -0.0415 |
| Common predictor, full 64 cells | 65.7945 | 89.9283 | 0.0960 |

Primary effects are comparator error minus main error; positive values favor the main method. Intervals are 98.333333% two-sided, 5000-replicate, paired hierarchical bootstrap intervals with equal-domain aggregation.

| Claim | Estimate (MPa) | 98.333333% CI | Evidence |
|---|---:|---:|---|
| VLM_EARLY_GUIDANCE | -0.4742 | [-3.0676, 1.5234] | NOT_SUPPORTED |
| FEEDBACK_ACQUISITION | 0.1651 | [-0.3894, 0.8026] | NOT_SUPPORTED |
| BEST_NONADAPTIVE_ACQUISITION | 0.8745 | [-1.9920, 4.2145] | NOT_SUPPORTED |

Engineering status remains `ENGINEERING_ACCURACY_CRITERION_NOT_SPECIFIED`; no engineering-readiness threshold was invented.

At cost 0.25, the main MAE is `70.1317` MPa: `0.2597` MPa worse than `NO_VLM_FEEDBACK`, `0.5958` MPa better than `VLM_OPEN_LOOP`, and `0.5946` MPa better than `CENTER_FIRST`. These endpoint contrasts are descriptive and do not override the unsupported primary intervals.

## Compute and chronology

- Formal registered predictor updates: `8000`.
- Formal registered actor updates: `13500`.
- Formal registered total: `21500` / 21,500.
- Measured encoder / Actor time: `127.987` s / `1202.074` s. The exact formal predictor duration was not persisted, so no fabricated cumulative runtime is reported; the formal execution completed below the 12-hour limit.
- During implementation, an earlier 8,000-update predictor development execution was invalidated before Actor training because it omitted 32/64-cell states required for the registered full-input report. It was regenerated under the corrected pre-Actor protocol. This was a coverage correction, not outcome-driven hyperparameter or seed selection.
- A preceding 13,500-update Actor execution was invalidated during final contract audit because C0 combined medium/high candidates instead of retaining only the highest reliable confidence level. The retained Actor run follows `HIGHEST_RELIABLE_CONFIDENCE_C0_THEN_UNLOCK_V1`. No result was inspected to choose this correction.
- No VLM, ResNet, source image, CAI label, prior learned-C-scan result, or old scientific code was modified.
- No TEST target is stored in the training feature bank or trajectory log.

## Source bindings

| Source | Path | SHA-256 |
|---|---|---|
| learned_cscan_config | `paper_v3/configs/learned_cscan_same_perception.yaml` | `12268dcaf470f769c326007702f7b0b6f4a13cf326447eef652a6dda972b5662` |
| vlm_cache | `results/learned_cscan_same_perception/surface_percepts.jsonl` | `85f400119093b362e2eb76489c9ae158ecdd453ce84ad04b6dfab7f04384693e` |
| vlm_manifest | `results/learned_cscan_same_perception/perception_manifest.json` | `b87b85240ee96c31b7fd9f8b46e0ca2338c7e11065fe531bd1f9e286aa942454` |
| cai_mpa_authority | `results/multiview/e1_audit/oof_predictions.csv` | `1d65d9f7c40f561727329ab6b7f922973765cc57f6ae36b5a94c572b20e8b054` |
| resnet18_weights | `paper_v3/assets/resnet18-f37072fd.pth` | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |

## Generated figures

- `results/cai_active_image/v2/figures/case_74t7kcdgkr_c8-24.png` (`0e93658dbe7febba36d8a0aaf2e6901cc6590e2c719a7e6612dd8d4752c83fa0`)
- `results/cai_active_image/v2/figures/case_cgtnjyggtm_q24-21.png` (`9b75d1ca4d4dda2f872bd140c4a45c2a4785c7643ea2c4b007f6e9bcd3b4b837`)
- `results/cai_active_image/v2/figures/case_w68dtmpfyf_q16-21.png` (`01004f04a5db92e8f97823dfd1f93b86f7932aa17f6fab1fd8dd1226b76c517f`)

The three cases were fixed in the protocol before TEST evaluation; full C-scans are not shown in these acquisition figures.

## Verification

- Ruff: `All checks passed!`
- Tests: `..............................................                           [100%]
46 passed in 9.88s`
- `git diff --check`: pass
- Frozen prior-result/shared-code diff from evidence base: empty
- Git push and local/upstream/remote SHA equality are recorded in the final Codex response after push.
