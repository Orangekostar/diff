# VLM C-scan success-efficiency evidence and design

## Frozen scope

- Repository base: `78453de3fe01c261887bc56c41e04aa505526fa1`.
- Branch: `research/vlm-cscan-success-efficiency`.
- Study: real impacted-surface RGB -> frozen VLM initial plan -> legal causal
  C-scan acquisition -> measured-evidence update -> route revision -> report.
- No policy, reader, or VLM training is authorized. G0/G1 results and scientific
  paths remain unchanged.

## Local evidence

The P0R package contains 276 authorized pairs across six domains. The surface,
registration, and 64-cell mapping manifests contain 276, 276, and 17,664 rows,
respectively; every referenced impacted-surface PNG, raw C-scan, and registered
C-scan exists under the external research root. P0R fixes one clockwise `ROT90`
surface-to-scan-frame correspondence and does not provide physical millimetre
calibration.

The Hasebe provenance declares CC BY 4.0 and DOI
`10.1016/j.dib.2022.108462`. Its paired manifest has no mask paths. The only
Hasebe-wide shape asset is an algorithmic color-map morphology summary without
stored per-pixel masks; its own provenance says it is not an expert label.
Unrelated FLAT-C COCO polygons do not overlap the 276-specimen Hasebe roster.
The author-registration authority supports correspondence only, not damage
annotation. Therefore reviewed LOCATE/CHARACTERIZE reference coverage is zero.

Consequences fixed before execution:

- formal task `success` is null for every unreviewed episode;
- full-cohort and appearance-stress experiments are not run;
- a 60-specimen hash-fixed real pilot is still run;
- automatic full-C-scan proposals are
  `ALGORITHM_DERIVED_NOT_REVIEWED`, and same-reader proxy scores are reported
  only as self-consistency diagnostics;
- an annotation queue preserves clean surface, full C-scan, proposal, and JSON
  template for later independent review.

## Model decision

One model is fixed: `Qwen/Qwen2.5-VL-7B-Instruct`, official Hugging Face
revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, Apache-2.0. The model is
used with bfloat16, deterministic decoding, `max_new_tokens=700`, at most one
format repair, and at most four event-triggered replans per episode. No second
model, prompt search, quantization search, or paid API is used. Mock responses
may test software but never enter experiment results.

## System boundaries

`BenchmarkTask` is an outer envelope; both tasks use the existing
`InspectionTask.FIELD` causal world. Episodes start from the existing 64-cell
zero state and may use the full endpoint. The planner receives surface inputs,
the observation's acquired positions/values, explicit UNKNOWN state, route
head/cost, and legal macro menus. It never receives specimen metadata, hidden
reference masks, unmeasured C-scan values, CAI, domain identity, or evaluator
outcomes.

All methods share one reader, report builder, stop rule, action legality gate,
and route compiler. B3/B5/B6 reuse the byte-identical cached initial VLM plan.
B4/B5 use the same deterministic feedback rule. B6 can select only a legal
menu ID. B7 uses fixed geometry/RBF uncertainty and measured evidence only.

The reader is a frozen, inexpensive RGB background-distance proxy because no
reviewed source labels exist. It has three states: measured indication,
measured no-indication, and unknown. Filled/interpolated pixels remain unknown
unless their coordinates were actually revealed.

## Route and evaluation contract

Each primitive action's exact newly revealed positions are compiled before
world execution. Four deterministic row/column snake candidates are evaluated
from a common `(0,0)` start; the shortest is selected with deterministic ties.
Information cost is unique revealed positions. Transit, active scan distance,
total distance, turns, and revisits are separate fields; transit never reveals.
Distances use normalized scan-frame coordinates and are normalized separately
by the same specimen's full-raster snake length.

LOCATE and CHARACTERIZE criteria follow the frozen YAML. `ANYTIME_REPORT` uses
the most recent report at each cost without cumulative-max hindsight.
`AUTONOMOUS_REPORT` freezes report and cost after an actual public STOP.
Hindsight sufficiency, if computed, is diagnostic only. Statistics use one
5,000-replicate paired, within-domain bootstrap and equal domain weighting once
reviewed references exist; it is not applied to null formal outcomes.

## External design sources

Accessed 2026-09-07:

- RUSSAgent, initial planning / execution / ultrasound-feedback refinement:
  https://arxiv.org/html/2603.14393v1
- US-VLA, success plus timesteps and feedback ablation:
  https://arxiv.org/html/2608.16074v1
- Fuentes et al., autonomous ultrasonic sampling precedent:
  https://strathprints.strath.ac.uk/72351/
- SayCan, language proposals grounded in executable skills:
  https://say-can.github.io/
- Inner Monologue, environment-feedback replanning:
  https://innermonologue.github.io/
- Hasebe dataset record:
  https://data.mendeley.com/datasets/ykhs7s2dck/1
- Qwen2.5-VL model card and documentation:
  https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
  and https://qwenlm.github.io/blog/qwen2.5-vl/

These sources motivate interfaces and comparisons; none establishes CFRP
performance for this implementation.
