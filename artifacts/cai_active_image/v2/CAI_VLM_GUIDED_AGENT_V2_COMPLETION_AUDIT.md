# CAI VLM-Guided Agent v2 Completion Audit

- [x] 60 frozen specimens retain TRAIN/VALID/TEST = 24/12/24.
- [x] Author CAI strength is evaluated in MPa.
- [x] All 24 TEST targets are absent from the training feature bank and joined only in P5 scoring.
- [x] Frozen Qwen cache is bound by model, prompt, schema, render, and image identities.
- [x] No second cell rotation is applied.
- [x] C0 uses only the highest reliable confidence level, constrains only the first main/open-loop action, and has a defined fallback.
- [x] Hidden C-scan tokens are masked from predictor and feedback actor.
- [x] `VLM_OPEN_LOOP` cannot read acquired C-scan content or current prediction.
- [x] `NO_VLM_FEEDBACK` cannot read VLM features or C0.
- [x] One common `P_all` predictor is used across all TEST methods.
- [x] Exact native-raster cost and absolute-zero state are logged.
- [x] Initial/legal action masks and per-step Actor/predictor latency are logged.
- [x] Actor state includes explicit spent/remaining cost and ordered action geometry history.
- [x] Formal optimizer schedule is 21,500 updates.
- [x] VALID selects `CENTER_FIRST` as `BEST_NONADAPTIVE`.
- [x] Three Bonferroni-adjusted primary comparisons use 5,000 paired bootstrap replicates.
- [x] Absolute and domain-level CAI performance are exported.
- [x] Three cases were selected before TEST evaluation and exported.
- [x] Prior results and shared learned-C-scan/VLM/MAVIS code have an empty diff from base.
- [x] Scoped tests, Ruff, and `git diff --check` pass.

Resource disclosure: one 8,000-update development predictor execution was invalidated before Actor training due to missing full-input state coverage. A subsequent 13,500-update Actor execution was invalidated during final contract audit due to the broader-than-specified C0 rule. The formal retained run follows the corrected frozen 21,500-update schedule. Neither correction was result-driven.
