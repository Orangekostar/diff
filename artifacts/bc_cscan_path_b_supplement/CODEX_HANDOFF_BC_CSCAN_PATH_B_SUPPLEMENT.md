# BC C-scan Path-B Supplement Handoff

## Repository and evidence identity

- Branch: `research/bc-cscan-path-b-supplement`.
- Frozen base: `d8b5b090891fc030931c6dc81e3619a80966f739`.
- Evidence commits through TEST/reference preparation: `b4a3bcfcefb97415f89198ba68d60a2dfad7893d`.
- Evidence scope: retrospective supplement on the reused 60-specimen, six-domain cohort; TEST has 24 physical specimens.
- Frozen tracked method/result files have no modification or deletion relative to the base.

## 1. BC replicas and training identity

The frozen seed-1 BC checkpoint was not changed. Its SHA-256 remains `c95185aec70515f2578d671e520ebe9b57e600e91e70e5f6752567ae2f87fbe5`.

| Model | Seed | Updates | Best internal-valid CE | SHA-256 |
|---|---:|---:|---:|---|
| BC replica | 2 | 1250 | 4.585256 | `138e74bc4ad1225b715d599310dc31eab5e651d6a33a1f5abcee13c17418044d` |
| BC replica | 3 | 1250 | 3.457240 | `38e77a7c1b382d4689e665c972f04404114bfa8ec3c50f2ddee1cd846c871425` |
| BC no-VLM Actor | 1 | 1250 | 3.693880 | `c6aa2c482eafe18d126130c7ed00b396c8fbdd559d863e0e3fa6adac9c4911ab` |
| BC no-US-feedback Actor | 1 | 1250 | 2.870559 | `ff7be36fe47cc2bf5df2c02c2afb2d2718c3ac745ae2712958de8bdcf75f613a` |
| Conditional STOP head | 1 | 2500 | 0.337141 | `29de4c8bb26d0b2df43da1d56d4d9f3d57061c382a4263f3dc7f4619360eb907` |

Total optimizer updates were 7,500/20,000. No CAI, VLM, Reader, action-space, CTG, or teacher-library training occurred.

## 2. Reference coverage

Independently reviewed references remain 0/60 for the reused pilot and 0/24 for the prepared confirmation cohort. No reviewer alias exists and `REVIEWED_V1` effects remain null. Every reported numerical effect below is `PROXY_LEGACY`, using the frozen same-Reader proxy. The 24 confirmation templates are blank, blinded review packages and contain no model-derived region proposals.

## 3. Direct planner effects and seed variation

| Task | Effect | Estimate | 97.5% CI | Positive domains |
|---|---|---:|---:|---:|
| LOCATE | BC three-seed mean minus P4 | 0.108383 | [0.020994, 0.200318] | 6/6 |
| LOCATE | BC three-seed mean minus P8 | 0.097059 | [0.012096, 0.187249] | 6/6 |
| CHARACTERIZE | BC three-seed mean minus P4 | 0.045740 | [0.021526, 0.069452] | 5/6 |
| CHARACTERIZE | BC three-seed mean minus P8 | 0.040324 | [0.015609, 0.064537] | 5/6 |

BC seed-1/2/3 effects against P8 were LOCATE `0.180136/0.088574/0.022467` and CHARACTERIZE `0.080889/0.048166/-0.008084`. The interval is conditional on three trained seeds; seeds do not increase the physical sample count.

## 4. Actor-input ablations

Full seed-1 BC minus no-VLM was `0.080795` for LOCATE and `0.058830` for CHARACTERIZE, with 97.5% CIs `[0.030208, 0.135362]` and `[0.003782, 0.118155]`. Full BC minus no-US-feedback was `0.459773` and `0.441817`, with 97.5% CIs `[0.380911, 0.536593]` and `[0.343865, 0.543677]`.

These are seed-1 Actor-input increments. The common Reader and STOP retain their registered inputs, so they are not whole-system no-VLM or no-ultrasound conclusions.

## 5. STOP calibration and refit

The frozen old 13,697-parameter STOP head had no qualifying VALID threshold under episode-level first-STOP calibration. This triggered the single allowed same-architecture, TRAIN-proxy refit. The new head used 1,152 fit and 384 internal-validation states, stopped at 2,500 updates, and locked threshold `0.99` for both tasks before TEST. The old head remains the `S_OLD_090` historical control.

`S_BC_CAL` does not dominate `S_RULE`: LOCATE completion increased but stopping occurred later and failure-penalized cost increased; CHARACTERIZE STOP increments remained imprecise. A calibrated-STOP improvement claim is not supported.

## 6. Autonomous TEST result

| Task | Planner | Completion | False-stop episodes | Exhaustion | Failure cost |
|---|---|---:|---:|---:|---:|
| LOCATE | P8 + S_BC_CAL | 0.958333 | 0.041667 | 0.000000 | 0.687637 |
| LOCATE | BC three-seed mean + S_BC_CAL | 0.958333 | 0.041667 | 0.000000 | 0.686657 |
| CHARACTERIZE | P8 + S_BC_CAL | 0.875000 | 0.000000 | 0.125000 | 0.813523 |
| CHARACTERIZE | BC three-seed mean + S_BC_CAL | 0.902778 | 0.069444 | 0.027778 | 0.792728 |

For LOCATE, BC-minus-P8 completion was `0.000000` (97.5% CI `[-0.055556, 0.069444]`) and P8-minus-BC failure cost was `0.000979` (`[-0.035969, 0.041334]`). Both joint gates failed. For CHARACTERIZE, completion was `0.027778` (`[-0.069444, 0.152778]`) and cost reduction was `0.020795` (`[0.005284, 0.036780]`); the cost gate passed but completion noninferiority did not. Path B is not supported for either task.

## 7. True-break evidence

Actual true-break execution, not only prefix replay, covered BC seed 1 and P8 on one hash-selected TEST specimen per domain and both tasks: 24 episodes, six physical specimens. All action prefixes and terminal reports matched the cached trajectories, and no `world.step` occurred after STOP. Full TEST statistics for the other configurations use declared prefix replay. The detailed route, timing, and per-episode records are in `report_manifest.json`; these are image-plane/process measurements, not hardware scan time.

## 8. Path-B decision

Planner-only proxy evidence supports a positive BC signal against P4 and P8. The joint autonomous criterion is false for LOCATE and CHARACTERIZE, and all formal reviewed-reference effects are null. The final status is `BC_PLANNING_SUPPORTED_STOP_NOT_SUPPORTED_PROXY_ONLY`.

## 9. Additional confirmation

The `ADDITIONAL_BC_CONFIRM_24` roster contains four fixed-hash, nonpilot specimens per domain. It was selected before opening model outcomes. Inference was not run because 0/24 references are independently reviewed; confirmation resource use is therefore zero world transitions and zero VLM calls. These specimens may have appeared in older G0/G1 development and are not described as untouched external data.

## 10. Minimum remaining evidence

The minimum next evidence is independent, attributable C-scan review for the frozen TEST or the already prepared 24-specimen confirmation roster, followed only by report rescoring or the registered confirmation run. No new Actor, Reader, VLM prompt, action space, threshold search, or large safety audit is required.

## Reproduction and verification

- `evaluate --split test`: recovered 336 episodes and 64,848 trajectory rows without new inference.
- `summarize`: produced 88 paired-effect rows, 120 domain rows, 8 ablation rows, 20 risk rows, and five PDF/PNG figure pairs.
- Reviewed-reference recovery: a temporary one-reference functional check replayed and scored 2,702 frozen report states without retraining or action reselection; final delivered state was restored to the real 0-reference result.
- New semantic suite: `12 passed`.
- Original learned-C-scan focused suite: `18 passed` across the six existing test files.
- Ruff: passed for all supplement source, CLI, figure, and test files.
- All five PDFs are one page with embedded DejaVu Sans fonts; PNG/PDF renders were visually checked for nonblank content, clipping, overlap, and legibility.
- Frozen tracked-file diff: empty.
- Resource use: 105,208/180,000 world transitions, 7,500/20,000 optimizer updates, and zero new VLM calls on the original 60 specimens.

Exact commands are in `results/bc_cscan_path_b_supplement/reproduce.md`; file identities are in `results/bc_cscan_path_b_supplement/CHECKSUMS.sha256`.
