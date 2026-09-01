# G1 Repository Binding Map

Base: `7a10cd425de582fa158bf6639285731ccd8ff7a7`. Existing G0 modules below are
read-only dependencies. G1 adapters live only in `inspection_agent_g1`.

| Concern | Existing authority | G1 binding | Status |
|---|---|---|---|
| Cohort, full scans, true CAI | `mavis/authority.py:MAVISAuthority` | Teacher/evaluation code receives guarded views; actor never receives authority | Wrap, privileged |
| Causal reveal | `inspection_agent/world.py:CausalInspectionWorld` | Rollout calls `reset`, `legal_actions`, and `step`; private reveal remains inside world | Reuse frozen |
| Policy observation | `inspection_agent/contracts.py:InspectionObservation` | Sole measured-data input to `features.py` | Reuse frozen |
| Task identity | `inspection_agent/contracts.py:InspectionTask` | Encode FIELD/CAI as a two-entry task token; NO_TASK and WRONG_TASK are evaluation controls | Reuse frozen |
| Zero-start state and actions | `inspection_agent/state.py` | Canonical 192 slots wrap `InspectionCellAction`; exact cost uses mask/count functions | Reuse frozen |
| Scanner grid | `mva/acquisition_grid.py:build_acquisition_grid` | Always call with geometry and nominal budget `0.015625` | Reuse frozen |
| Geometry-spread order | `mva/oracle.py:uniform_cell_order` | First 8 cells form the primary warm start; K4/K16 are sensitivity only | Reuse frozen |
| Surface hypothesis | `inspection_agent/surface_hypothesis.py` | Scores/top cells enter observable cell tokens; controls replace only these fields | Reuse frozen |
| Surface cell boxes | `agentic_nde/surface_cells.py:SurfaceCellRecord.cell_boxes` | Map acquired native locations to the fixed 8x8 owned cells | Reuse authority |
| Source reconstruction | `inspection_agent/generalized_reconstruction.py` | G1 `crossfit.py` fits equivalent priors with explicit dual exclusion and calls frozen reconstruction | New wrapper |
| Frozen embedding | `mva/encoder_session.py:MVAEncoderSession` | Encode current source-safe reconstruction into 512-D; weights stay frozen | Reuse frozen |
| CAI state features | `inspection_agent/cai_assessor.py` | G1 builds label-independent post-K8 rows and dual-exclusion assessor fits | New wrapper |
| G0 state policies | `inspection_agent/state_bank.py` | Reuse deterministic policy semantics after K8; rematerialize every state under G1 grid/split | Semantic reuse |
| FIELD teacher | `inspection_agent/oracle.py:choose_field_action` | Query every legal candidate on source-only privileged views with crossfit prior | Reuse frozen |
| CAI teacher | `inspection_agent/oracle.py:choose_cai_action` | Query every legal candidate with crossfit prior/assessor | Reuse frozen |
| G0 stopping evidence | `inspection_agent/stopping.py` | Reuse the 1.05 sufficiency definition only; learn a new observable STOP model/threshold | Semantic reuse |
| FIELD loss | `mva/reconstruction_value.py:normalized_rgb_mse` | Same loss for teacher, source selection, bridge, and target evaluation | Reuse frozen |
| CAI loss | absolute error through frozen CAI evaluation semantics | Same evaluator for fixed/learned/oracle bridge curves | Reuse frozen |
| AUEBC | `mva/budget_metrics.py` and G0 checkpoint semantics | G1 zero/warm-inclusive curves use common bridge checkpoints and carry-forward rule | New wrapper |
| Bootstrap | `inspection_agent/statistics.py` | Same specimen-within-domain synchronized bootstrap, 100,000 replicates | Reuse/wrap |
| Artifact publication | `inspection_agent/artifacts.py` | New G1 schema, source/model/trajectory freeze identities, replay comparison | New implementation |

## New Namespace Ownership

| File | Sole responsibility |
|---|---|
| `contracts.py` | Immutable G1 actor/teacher/integrity records, action-slot and freeze contracts |
| `warm_start.py` | Geometry audit and K=4/8/16 deterministic bridge |
| `crossfit.py` | Outer/inner rosters, dual-exclusion priors/assessors, final five-source fits |
| `features.py` | Observable-only global/cell/candidate tensors and privilege sentinels |
| `teacher.py` | Source privileged FIELD/CAI/STOP labels and full candidate utilities |
| `teacher_bank.py` | Deterministic 13+4 state generation, separated serialization, manifests |
| `policy_model.py` | `SharedActionMLP`, `StructuredInspectionPolicy`, exact legal masking |
| `policy_training.py` | Deterministic hard BC, source validation, model persistence |
| `utility_distillation.py` | Robust utility distribution and soft-distillation objective |
| `dagger.py` | Source-world-only actor rollout/relabeling |
| `privileged_awr.py` | Source-only authorization gate and independently implemented AAWR-style critic route |
| `stopping_policy.py` | Observable STOP labels/model/threshold selection |
| `rollout.py` | Causal learned/fixed rollouts, task/surface variants, target freeze guard |
| `metrics.py` | Common-geometry curves, AUEBC, gap closure, diagnostics |
| `statistics.py` | Paired equal-domain bootstrap and component gates |
| `artifacts.py` | Formal/replay schema, hashes, privacy checks, comparison |
| `g1.py` | Ordered orchestration only; no scientific definitions hidden in CLI |

## Frozen-Path Rule

No file listed in prompt Section 37 may change. If a G1 adapter cannot preserve a
frozen API, execution stops and records a separate bug audit before any edit.
