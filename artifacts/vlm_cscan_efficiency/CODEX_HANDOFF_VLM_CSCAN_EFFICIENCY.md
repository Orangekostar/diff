# Codex handoff: VLM C-scan success-efficiency benchmark

## Repository identity

- Repository: `git@github.com:Orangekostar/diff.git`
- Branch: `research/vlm-cscan-success-efficiency`
- Base: `78453de3fe01c261887bc56c41e04aa505526fa1`
- Implementation commits: `316590d`, `097f26d`, `59450ec`, `175b561`,
  `d43809b`, `1a5591a`
- Pilot evidence commit: `87ce114`
- The enclosing documentation commit is intentionally not embedded here; the
  final local/upstream/remote SHA is authoritative in the terminal handoff.
- Historical G0/G1 artifacts, conclusions, and frozen science paths are
  preserved. No old result was renamed, replaced, or reinterpreted.

## Delivered system

- Code: `src/cmc_bbdm/vlm_cscan/`
- CLI: `scripts/run_vlm_cscan.py`
- Frozen configuration: `paper_v3/configs/vlm_cscan_efficiency.yaml`
- Protocol: `artifacts/vlm_cscan_efficiency/BENCHMARK_PROTOCOL.md`
- Review instructions: `artifacts/vlm_cscan_efficiency/ANNOTATION_GUIDE.md`
- Results: `results/vlm_cscan_efficiency/`

The implementation loads actual impacted-surface RGB and registered C-scan
files from `/home/ww/paper3/cmc_damage_inference`. The external Hasebe package
contains 276 registered pairs across six domains. Raw images are referenced by
manifest and are not copied into Git.

One frozen model was used: `Qwen/Qwen2.5-VL-7B-Instruct`, revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, Apache-2.0, bfloat16,
deterministic decoding. Its model-config SHA-256 is
`77d9ec7321cc572e3579e2c84799c9cadaded63c49ce93b101733349fc330c43`;
its preprocessor-config SHA-256 is
`f2058c716eef96ccaed1cc1e2d0c08306b62586d535b28d9d08e691b2fab7ca0`.
All surface renders were 1024x1024 and processed at 980x980, 1,225 merged
visual tokens per view. Exact evidence-image dimensions are in
`input_manifest.csv`.

## Executed scope

- Cohort: fixed pilot, 10 specimens per domain, 60 physical specimens total.
- Methods: B0-B7.
- Tasks: LOCATE and CHARACTERIZE.
- Trajectories: 960; primitive trajectory rows: 184,320; report rows: 185,280.
- Every trajectory reached exact acquisition cost 1.0 for the anytime branch.
- Real frozen-VLM inference: yes. New training: no.
- Full 276-specimen run: not executed because reviewed-reference coverage is
  incomplete.
- Appearance stress run: not executed for the same reason.

The run command executed 108 uncached B6 trajectories and made 108 new replan
calls; 12 smoke B6 trajectories were resumed from valid shards. Total wall
time was 7,377.553 s with four CPU workers. The cumulative deployment record is
60 initial calls plus 120 replan calls, 984.363 s original model latency, zero
fallbacks, 404,942 input tokens, and 8,243 output tokens.

## Reference status

No attributable expert-reviewed or author-provided C-scan indication masks were
found for this roster. Coverage is 0/60 (0%). Automatic inner-border RGB
distance polygons remain `ALGORITHM_DERIVED_NOT_REVIEWED`. Therefore:

- status: `PILOT_EXECUTED_PROXY_ONLY`;
- every formal success metric and formal H1/H2/H3 estimate is null;
- surface-plan benefit: `INCONCLUSIVE`;
- ultrasound-feedback benefit: `INCONCLUSIVE`;
- VLM-replanning benefit: `INCONCLUSIVE`;
- autonomous-completion efficiency: `INCONCLUSIVE`.

All numbers below are same-reader self-consistency diagnostics. They are not
damage ground truth and cannot support a scientific performance claim.

## Proxy pilot results

| Task | Method | Anytime AUSC | Autonomous SR | Autonomous AUSC | Failure-penalized cost |
|---|---:|---:|---:|---:|---:|
| LOCATE | B0 | 0.481913 | 0/60 | 0.000000 | 1.000000 |
| LOCATE | B1 | 0.450189 | 11/60 | 0.182861 | 0.817139 |
| LOCATE | B2 | 0.473522 | 9/60 | 0.149750 | 0.850250 |
| LOCATE | B3 | 0.487879 | 0/60 | 0.000000 | 1.000000 |
| LOCATE | B4 | 0.357645 | 0/60 | 0.000000 | 1.000000 |
| LOCATE | B5 | 0.359928 | 1/60 | 0.016564 | 0.983436 |
| LOCATE | B6 | 0.360654 | 3/60 | 0.049696 | 0.950304 |
| LOCATE | B7 | 0.357090 | 0/60 | 0.000000 | 1.000000 |
| CHARACTERIZE | B0 | 0.167154 | 0/60 | 0.000000 | 1.000000 |
| CHARACTERIZE | B1 | 0.194112 | 0/60 | 0.000000 | 1.000000 |
| CHARACTERIZE | B2 | 0.170694 | 0/60 | 0.000000 | 1.000000 |
| CHARACTERIZE | B3 | 0.168120 | 0/60 | 0.000000 | 1.000000 |
| CHARACTERIZE | B4 | 0.118626 | 0/60 | 0.000000 | 1.000000 |
| CHARACTERIZE | B5 | 0.118966 | 0/60 | 0.000000 | 1.000000 |
| CHARACTERIZE | B6 | 0.118833 | 0/60 | 0.000000 | 1.000000 |
| CHARACTERIZE | B7 | 0.118626 | 0/60 | 0.000000 | 1.000000 |

All anytime final proxy success rates are 1.0 only because the diagnostic uses
the same reader on a completed full scan. This does not demonstrate autonomous
completion. All methods triggered a public stop on every episode, so incomplete
rate is zero; most stops were wrong under the proxy. B6 early wrong-stop rates
were 95% for LOCATE and 100% for CHARACTERIZE. Its mean stop costs were
0.003068 and 0.106486, respectively.

Paired, within-domain, equal-domain-weighted 5,000-replicate proxy comparisons:

| Hypothesis | Task | Comparison | Anytime AUSC difference (95% CI) | Domain direction |
|---|---|---|---:|---:|
| H1 | LOCATE | B3-B0 | +0.005966 [-0.002629, +0.020849] | 3+/3- |
| H1 | LOCATE | B3-B1 | +0.037690 [-0.002456, +0.075456] | 6+/0- |
| H1 | LOCATE | B3-B2 | +0.014357 [-0.022241, +0.047561] | 4+/2- |
| H2 | LOCATE | B5-B3 | -0.127951 [-0.185665, -0.072521] | 0+/6- |
| H3 | LOCATE | B6-B5 | +0.000726 [-0.005511, +0.007574] | 3+/3- |
| H1 | CHARACTERIZE | B3-B0 | +0.000966 [-0.000002, +0.002331] | 3+/3= |
| H1 | CHARACTERIZE | B3-B1 | -0.025992 [-0.069222, +0.013005] | 1+/5- |
| H1 | CHARACTERIZE | B3-B2 | -0.002575 [-0.032548, +0.025352] | 2+/4- |
| H2 | CHARACTERIZE | B5-B3 | -0.049154 [-0.075618, -0.020960] | 0+/6- |
| H3 | CHARACTERIZE | B6-B5 | -0.000133 [-0.000577, +0.000120] | 1+/4=/1- |

The non-VLM feedback control B4-B0 was likewise negative for proxy anytime
AUSC: -0.124268 [-0.181602, -0.070427] for LOCATE and -0.048528
[-0.074843, -0.020504] for CHARACTERIZE. This indicates that the first
deterministic feedback policy, rather than only its VLM initialization, is
poorly aligned with this proxy efficiency curve. No post-result prompt,
threshold, method, or endpoint was changed.

## VLM behavior and failures

Initial inference consumed 161,220 input and 5,632 output tokens over 527.683 s.
There were only six unique raw responses: 45/60 plans selected `(27,)`, 8/60
selected `(35,)`, and the remaining seven selected one of three three-cell
patterns. Every plan used `SURVEY_ROI`, one low-confidence
`indentation_like` region, and `illumination_possible`; no call abstained.

Replanning consumed 243,722 input and 2,611 output tokens over 456.680 s. The
events were `NEW_INDICATION_OUTSIDE_INITIAL_ROI` (112) and
`INITIAL_ROI_UNSUPPORTED` (8), but all 120 responses selected the first legal
menu item `m1` with the same generic reason code. The high-level VLM choice
therefore collapsed and provides no credible evidence of useful dynamic
reasoning.

B6 had three proxy-successful LOCATE stops: `xcmzfsbd9t:c24-9t` and
`xcmzfsbd9t:c24-33` at step 9/cost 0.006080 with stop IoU 0.587685 and
0.609889, and `ykhs7s2dck:q8-33` at step 9/cost 0.006104 with IoU 0.818169.
These are examples of self-consistency only. For CHARACTERIZE, all 60 B6
stops exceeded the 10% area-error threshold; 58/60 also missed IoU 0.70.
The worst example, `yfxyg8jm46:c16-16`, stopped at step 66/cost 0.022976
with IoU 0.011944 and relative area error 81.465. LOCATE failures were mainly
insufficient measured support or box IoU below 0.50. Detailed counts are in
`failure_types.csv`.

## Validation evidence

Executed on the final code/result tree before commit:

```bash
python -m ruff check src/cmc_bbdm/vlm_cscan scripts/run_vlm_cscan.py \
  tests/test_vlm_cscan_*.py
PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  tests/test_vlm_cscan_core.py tests/test_vlm_cscan_evaluation.py \
  tests/test_vlm_cscan_planning.py
python scripts/run_vlm_cscan.py verify
git diff --check
git diff --name-only 78453de3fe01c261887bc56c41e04aa505526fa1 -- \
  results/p1_full_field_oracle results/p5_sparse_scan results/mvd \
  results/mavis results/mavis_science_closure artifacts/mavis \
  artifacts/mavis_science_closure artifacts/mvd_authority \
  artifacts/mavis_authority
```

Results: Ruff passed; 23 tests passed in 3.84 s; result verification passed
with 147 files; `git diff --check` was clean; the frozen-science diff was
empty. Old G0/G1 and other historical scientific suites were not rerun because
no old/shared/frozen runtime code changed.

Key artifact SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| `summary.json` | `4fb819a02466587d4772ec9743ba45bc2955540374aab763d5dd753f282889a5` |
| `episodes.parquet` | `13970aab2c54c8a48b3caf1e8716e18daecafd1cc988a411c2c15e9c2ea8e31a` |
| `reports.parquet` | `c18ca40934b6bf00ed9c4727d9c6a799f08386332b673d2a07fdf9d26b92e305` |
| `aggregate_metrics.csv` | `984b420b8ccbdd169a37d69962e9ae714ce993f75e8bb776024439b1a4244682` |
| `comparisons.csv` | `f32bb3478f79e7f585e57ecc427d06ccfc43328e52791015349b11a466536b13` |
| `failure_types.csv` | `b31c75363fc59b3e446940966a7eba4f6d5dc15a334e0116527869af4a97ae3f` |
| `CHECKSUMS.sha256` | `95f6f530dd54ce09ef3417c0b9cb7109d2f7637d356730b140447ffc3b3876b6` |

## Reproduction and review resume

```bash
python scripts/run_vlm_cscan.py prepare
python scripts/run_vlm_cscan.py export-annotations
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/run_vlm_cscan.py infer-surface --cohort pilot
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/run_vlm_cscan.py run --cohort pilot --workers 4
python scripts/run_vlm_cscan.py evaluate --cohort pilot
python scripts/run_vlm_cscan.py verify
```

After independent review, set only attributable C-scan templates to
`EXPERT_REVIEWED`/`reviewed`, retain reviewer identity, then rerun `prepare`,
`run`, `evaluate`, and `verify`. Reviewed-reference bytes change the execution
identity while initial VLM responses remain cached. Do not run the full cohort
or stress stage until coverage supports their formal comparisons.

## GitHub synchronization

The requested delivery command is:

```bash
git push -u origin research/vlm-cscan-success-efficiency
```

The remote branch did not exist before this delivery. After the enclosing
handoff commit, the final process verifies equality of `git rev-parse HEAD`,
`git rev-parse @{upstream}`, and `git ls-remote origin
refs/heads/research/vlm-cscan-success-efficiency`, plus a clean working tree.
No PR, merge, or force push is performed. The final terminal/assistant handoff
records the resulting SHA because this file cannot embed its own commit SHA.
