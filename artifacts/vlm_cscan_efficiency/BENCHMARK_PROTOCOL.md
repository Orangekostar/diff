# VLM C-scan success-efficiency benchmark protocol

## Fixed study identity

- Base: `78453de3fe01c261887bc56c41e04aa505526fa1`.
- Data: 276 registered Hasebe surface/C-scan pairs in six domains.
- Pilot: ten SHA-256-selected specimens per domain (60 total); the first per
  domain forms the six-specimen technical smoke subset.
- Evidence scope: `INTERNAL_REUSED_COHORT_EVALUATION`.
- Model: one frozen `Qwen/Qwen2.5-VL-7B-Instruct` snapshot at revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`, bfloat16, deterministic decode.
- Input geometry: both surface views render at 1024x1024 and are processed at
  980x980 (`image_grid_thw=[1,70,70]`, 1,225 visual tokens each). Evidence
  views retain native aspect ratio; exact per-specimen processor dimensions
  are recorded in `input_manifest.csv`.
- No model, reader, policy, or stopping-rule training occurs.

The controlling configuration is
`paper_v3/configs/vlm_cscan_efficiency.yaml`. Thresholds, method roster,
cohort, prompt, and cost checkpoints were fixed before the valid pilot.

## Tasks and references

`LOCATE` requires main-box IoU at least 0.50 and at least three truly measured
positive support coordinates spanning two rows and two columns; a full-frame
box is rejected. `CHARACTERIZE` requires valid-pixel IoU at least 0.70,
certain-positive recall at least 0.90, relative area error at most 0.10, and
measured support for each predicted component. Uncertain reference pixels are
excluded from IoU/recall and define the allowed area interval.

Only attributable `EXPERT_REVIEWED` or `AUTHOR_PROVIDED` registered C-scan
polygons with a nonempty certain indication are formal references. With no
such reference, formal success is null. The full-input output of the same
public RGB-distance reader is retained only as a self-consistency proxy and
cannot support a damage-detection claim.

## Causal acquisition and methods

Every method starts from zero measurement in the existing 8x8 nested causal
world and uses legal `-1 -> 0 -> 1 -> 2` cell transitions until the exact
native-raster endpoint. Every fourth primitive slot uses the same dispersed
geometry schedule. Each macro contains at most four primitive actions.

| ID | Initial order | Ultrasound changes route | VLM replan |
|---|---|---|---|
| B0 | geometry | no | no |
| B1 | centre outward | no | no |
| B2 | surface colour saliency | no | no |
| B3 | frozen VLM surface plan | no | no |
| B4 | geometry | deterministic feedback | no |
| B5 | same VLM plan as B3 | deterministic feedback | no |
| B6 | same VLM plan as B3/B5 | deterministic feedback | event-triggered menu choice |
| B7 | geometry | observed RBF/uncertainty feedback | no |

B6 receives only the real surface, measured/unknown C-scan rendering, current
task, visible evidence summary, cost, and legal macro menu. It never receives
the hidden reference or unmeasured pixels. Replanning is bounded by the frozen
maximum of four calls and falls back to the first legal option after one
format-repair attempt.

## Reader, reports, and stopping

All methods share one frozen reader. Its background is the equal-domain mean
of inner-border medians from the other five domains; target-domain pixels are
excluded from that prior. Only revealed RGB positions are classified.
Unmeasured pixels remain `UNKNOWN` and never become negative evidence through
display filling.

`ANYTIME_REPORT` records the latest causal report at each exact acquisition
cost and continues to the endpoint. `AUTONOMOUS_REPORT` is a logically ended
branch: its report and cost freeze at the first common public STOP, while the
same physical prefix may continue solely to obtain the anytime curve. No
post-STOP observation can change autonomous success.

## Cost and statistics

Acquisition cost is unique revealed native-raster positions divided by native
raster size. The common route compiler separately records transit, scan,
total normalized planar length, turns, and explicit revisit count from a
common `(0,0)` start. Route length is normalized by the same specimen's fixed
full-raster snake, not interpreted as physical time or a global optimum.

Primary outputs are final success, AUSC, C@SR=0.80/0.90,
failure-penalized autonomous completion cost, early wrong-stop/incomplete
rates, and normalized route cost. Comparisons are paired by physical specimen,
resampled within domain, and aggregate six domain means equally with 5,000
bootstrap replicates. Proxy intervals remain diagnostic; component conclusions
stay `INCONCLUSIVE` until formal reference coverage supports them.

## Execution

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

The complete 276-specimen and appearance-stress stages are blocked while
reviewed reference coverage is incomplete. Cached inference retains original
latency/tokens and does not count a cache hit as zero deployment cost.
