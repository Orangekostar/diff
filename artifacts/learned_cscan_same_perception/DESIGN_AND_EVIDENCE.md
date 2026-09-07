# Learned C-scan Same-Perception Study: Design and Evidence

## Study identity

- Repository: `git@github.com:Orangekostar/diff.git`
- Branch: `research/learned-cscan-same-perception`
- Base commit: `59a67511c0ee6395b68220c6a1644f40c383dfa2`
- Execution prompt SHA-256: `2b9a1812b16c232564dfc9bde3847d2732495706e32532a4c82ed36b6418315d`
- Evidence scope: `INTERNAL_REUSED_COHORT_NEW_PROTOCOL`
- Data root: `/home/ww/paper3/cmc_damage_inference`
- Study question: under the same visible surface and sparse C-scan perception, does a trained action policy organize LOCATE and CHARACTERIZE inspections more reliably or efficiently than prespecified rule policies?

This is a new, bounded learning study. Historical G0, G1, and VLM pilot results motivate the design but are not outcomes of this protocol.

## W0 inventory

### Data and split

The source inventory contains 276 paired surface images and specimen-level C-scan crops in six domains, with domain counts 45, 49, 43, 59, 42, and 38. The new pilot reuses the frozen 60-specimen roster from the preceding VLM study, ten specimens per domain. It does not select specimens again.

Within each domain, the roster is sorted by:

```text
sha256("learned-cscan-same-perception-pilot-v1|<specimen_key>")
```

The first four specimens are TRAIN, the next two are VALID, and the final four are TEST. All tasks, views, states, and trajectories for a specimen remain in the same split.

| Domain | TRAIN | VALID | TEST |
|---|---:|---:|---:|
| `74t7kcdgkr` | 4 | 2 | 4 |
| `cgtnjyggtm` | 4 | 2 | 4 |
| `w68dtmpfyf` | 4 | 2 | 4 |
| `xcmzfsbd9t` | 4 | 2 | 4 |
| `yfxyg8jm46` | 4 | 2 | 4 |
| `ykhs7s2dck` | 4 | 2 | 4 |
| **Total** | **24** | **12** | **24** |

The generated `split_manifest.csv` is the specimen-level authority. A single TRAIN-only background prior, feature normalization, and reader calibration are fit once. VALID may select the checkpoint, rule coverage period, and learned-stop threshold. TEST is evaluation-only.

### Reference status and evidence track

The existing 60-specimen annotation queue contains 60 `pending` records with `ALGORITHM_DERIVED_NOT_REVIEWED` provenance. No independently reviewed task reference was found in the scoped data, results, or artifact inventories at W0.

Therefore this run uses the `PROXY_ONLY` track:

- `model_trained` may be true after actual optimization.
- Proxy task labels may train and score the bounded engineering pilot.
- `proxy_success` is labeled `SAME_READER_SELF_CONSISTENCY` when the task proxy and public reader share derivation.
- `formal_task_success`, formal effect estimates, and formal confidence intervals remain null.
- Proxy improvements cannot be promoted to reviewed-reference evidence.

The `reference_manifest.csv` records provenance and review status independently from model performance. Later reviewed TEST references may rescore frozen reports if they do not alter training or reader calibration. Reviewed references that alter training or calibration require a new versioned training run.

### Model and compute

- Surface model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Local revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`
- Model configuration SHA-256: `77d9ec7321cc572e3579e2c84799c9cadaded63c49ce93b101733349fc330c43`
- Preprocessor configuration SHA-256: `f2058c716eef96ccaed1cc1e2d0c08306b62586d535b28d9d08e691b2fab7ca0`
- Surface model policy: frozen inference only; no download, fine-tuning, or model comparison.
- Learned planner policy: at most one GPU, at most four CPU workers, and less than one million trainable parameters.
- Logical transition ceiling: 400,000 across bank construction, aggregation, training-side validation rollouts, and evaluation.

At W0, three NVIDIA A40 GPUs were present and idle. Resource availability is an execution fact, not evidence of model performance.

## Existing-code binding map

| ID | Existing authority | Bound use in this study |
|---|---|---|
| E01 | `artifacts/vlm_cscan_efficiency/CODEX_HANDOFF_VLM_CSCAN_EFFICIENCY.md` | Historical roster, reference coverage, backend, and pilot facts only |
| E02 | `vlm_cscan.runtime` loading/rendering helpers | Real surface PNG and registered C-scan loading through a new thin runtime adapter |
| E03 | `agentic_nde.surface_cells` | Existing 8x8 cell authority and crop coordinates |
| E04 | `inspection_agent.state` | Zero state, legal progressive actions, acquired-position accounting |
| E05 | `inspection_agent.world.CausalInspectionWorld` | Deterministic reveal service; each counterfactual branch resets or uses an independent world |
| E06 | `vlm_cscan.reader` | Historical reader reference only; Reader v2 is new and shared |
| E07 | `vlm_cscan.reporting` | Historical behavior used to define corrected v2 contracts, not reused as the final reporter |
| E08 | `vlm_cscan.vlm.QwenVLBackend` | Frozen inference backend with a new prompt, schema, parser, and cache namespace |
| E09 | `vlm_cscan.planning` | Rule-priority source for `R_LEGACY`; no VLM action menu in the learned planner |
| E10 | `vlm_cscan.route.compile_route` | Normalized route diagnostics with its documented non-hardware limitations |
| E11 | `vlm_cscan.references` | Provenance states and task evaluation concepts; proxy remains non-formal |
| E12 | `vlm_cscan.metrics` | Bootstrap pattern; new metrics add exact action-step integration |
| E13 | `inspection_agent_g1.policy_model` | Architectural pattern for a compact masked cell scorer; no old weights or input contract |
| E14 | `inspection_agent_g1.features` | Visible-feature and action-mapping ideas, adapted to the new packet |
| E15 | `inspection_agent_g1.policy_training` | Negative implementation reference; new training performs an optimizer update per mini-batch |
| E16 | `vlm_cscan.benchmark` | Historical full-prefix runner reference; new autonomous runner breaks at STOP |

The bound legacy modules and their historical results are frozen. New behavior lives in `cmc_bbdm.learned_cscan`.

## Source-grounded method boundary

Sources were accessed on 2026-09-07.

| Source | Idea used | Claim boundary |
|---|---|---|
| [Ross et al., DAgger, AISTATS 2011](https://proceedings.mlr.press/v15/ross11a.html) | Add one bounded batch of states visited by the learned policy | One aggregation round does not establish DAgger's asymptotic guarantees |
| [Ross and Bagnell, 2014](https://arxiv.org/abs/1406.5979) | Supervise queried actions with downstream cost-to-go rather than only action identity | Finite deterministic suffix rollouts are not exact optimal Q-values |
| [Sun et al., AggreVaTeD, ICML 2017](https://proceedings.mlr.press/v70/sun17d.html) | Combine a differentiable policy with cost-sensitive action preferences | This implementation is inspired by, not a reproduction of, AggreVaTeD or its theory |
| [Warrington et al., ICML 2021](https://proceedings.mlr.press/v139/warrington21a.html) | Keep the policy input strictly observable even when labels use training-world outcomes | An oracle gap alone does not imply an observable policy can imitate the oracle |
| [Fuentes et al., MSSP 2020](https://doi.org/10.1016/j.ymssp.2020.106897) | Prior evidence that ultrasound feedback can drive sequential inspection | No claim of the first autonomous or adaptive ultrasound inspection |
| [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923) | Frozen structured visual localization backend | Generic localization does not establish CFRP damage perception validity |

The method is described as **cost-sensitive interactive imitation in a deterministic retrospective inspection environment**. It is neither a full online robot experiment nor conventional fixed-transition offline reinforcement learning.

## Frozen study design

### Common perception

Both rule and learned planners receive the exact same immutable `ObservationPacket` for the same specimen, task, action history, and perception version. Closed-loop policies may subsequently observe different pixels only because they chose different legal actions.

The packet contains only:

- measured native positions, their RGB values, and an explicit measured mask;
- per-cell measurement level, measured fraction, observed RGB mean and standard deviation;
- observed-only 4x4 sub-block aggregates, observation counts, and missingness flags;
- public Reader v2 candidate, estimate, support, validity, spacing, and boundary fields;
- frozen `SurfacePercept` regions and self-reported ordinal confidence;
- normalized geometry, probe position, recent four actions, acquisition cost, route diagnostics;
- a LOCATE or CHARACTERIZE task token.

It excludes unmeasured values, full C-scan content, reference masks, task loss, CAI ground truth or estimates, domain IDs, specimen IDs, paths, and hashes. UNKNOWN is explicit and is never encoded as observed background.

### Surface percept

`SurfacePercept` contains candidate regions, visible cue type, alternative explanation, ordinal confidence, and `no_reliable_cue`. It contains no action, menu, scan-skill, cell-27 example, indentation answer, or task-specific recommendation.

The cache key includes model revision, source-image digest, rendering version, prompt text, and schema version. One percept is reused across tasks, methods, and trajectories.

Functional checks use one TRAIN specimen per domain under original and deterministically permuted display numbering, plus a small blank/placeholder diagnostic. Extra diagnostic calls are capped at 24. Main-roster generation is capped at 60 new calls with at most one format repair per image. A single prompt/schema repair is allowed only for demonstrated copying or parse collapse and must be recorded before VALID/TEST use.

### Reader v2 and public report

Reader v2 is transparent and cell-local:

- actual measured positions retain their exact observed values;
- an entirely unmeasured cell has no estimate and cannot become a report candidate;
- level-0/1 interpolation stays within the measured cell and inside measured coordinate support;
- singleton support interpolates only on the measured point or measured line;
- estimates carry separate validity, support count, and spacing fields;
- candidate classification, continuous estimate, support, and unverified boundary are distinct;
- signal strength is never named completion probability.

The common report and task evaluator are identical for every planner. A TRAIN-only common background prior and the frozen threshold 0.18 are used in the proxy track.

### Rule stop

`S_rule` is a shared report-level stop controller.

For LOCATE it requires full level-0 coverage, at least three supporting observations spanning at least two rows and two columns, candidate cells at level 1 or higher, and their neighboring ring at level 0 or higher. For CHARACTERIZE it additionally requires candidate cells at level 2 and the neighboring ring at level 1. The last two relevant report updates must have area change at most 0.05 and normalized bounding-box change at most 1/64. Measurements unrelated to a current candidate or its verification ring do not advance stability.

### Action space and planners

At any state, the actor scores at most 64 legal progressive primitives. A primitive is `(cell_index, next_level)`, with `next_level` derived from the current cell state. STOP is a separate decision and is not a 65th scan action.

Rule controls share the packet, reader, reporter, action legality, and cost model:

- `R_GEOM`: spatially dispersed coverage before refinement.
- `R_CENTER`: center-prior coverage followed by deterministic completion.
- `R_VLM_OPEN`: fixed surface-cue ordering followed by completion.
- `R_LEGACY`: historical feedback priority adapted to common v2 evidence.
- `R_BALANCED`: low-density cue verification, visible-boundary expansion, supported refinement, and a mandatory coverage primitive every four actions.

Only `R_BALANCED` coverage periods 4 and 8 may be compared on VALID. `R_BALANCED` with period 4 is the fixed behavior policy `mu` used to generate labels. All rule results are reported, and the selected rule is locked before TEST.

The learned actor uses per-cell tokens of width 128, at most two Transformer encoder layers, four attention heads, shared masked scoring, and fewer than one million parameters. `L_BC` and `L_CTG` use the same architecture. `L_BC` imitates `mu`; `L_CTG` uses queried cost-to-go preferences. No specimen or domain identity enters the actor.

### Training bank and targets

For each TRAIN specimen and task, the base bank contains at most four states: zero state and reachable states near acquisition costs 0.025, 0.10, and 0.40 along `mu`. Each state queries at most six deduplicated legal candidates drawn from `mu`, geometry coverage, visible boundary, low-cost refinement, and two fixed-seed random actions. Candidate construction uses only visible packet fields.

Each queried candidate is executed in a clean branch, followed by `mu` to full input or a valid endpoint. Its main target is exact action-step cost-to-go:

```text
integral from current acquisition cost to 1 of (1 - task_success(report)) dc
```

A separately stored auxiliary term is 0.05 times a bounded continuous task-loss integral. The preference distribution is a softmax of normalized queried costs with temperature 0.10. Normalization uses `max(1 - current_cost, 1e-6)`. Unqueried legal actions receive neither positive nor negative labels.

After `L_CTG_0`, one and only one aggregation round adds at most two learned-policy-visited states per TRAIN specimen and task, labels the same candidate set with the same branch procedure, and produces `L_CTG_1`. VALID selects between the two; TEST never decides whether to aggregate again.

Training uses AdamW, learning rate `3e-4`, weight decay `1e-4`, gradient clipping at 1.0, actual per-mini-batch optimizer steps, and specimen/task-balanced sampling. A fixed subset of at most 32 states must first demonstrate decreasing training loss. Each fit is capped at 4,000 steps, validates every 250 steps, and stops after four validations without improvement. One seed is run initially; two confirmation seeds are permitted only after a positive VALID action-only signal.

### Learned stop

Learned stopping is trained separately from action selection on visible TRAIN states labeled by the training-side proxy or reviewed reference. Candidate thresholds 0.90, 0.95, and 0.99 are assessed on VALID, with support count and false-stop behavior recorded. If no threshold has credible support, the status is `LEARNED_STOP_NOT_AUTHORIZED`, and TEST learned-policy episodes use the declared resource endpoint rather than claiming successful autonomous stopping.

If authorized, one shared learned-stop model and threshold are used with both rule and learned action planners. No planner-specific TEST calibration is allowed.

### Evaluation and statistics

Phase A evaluates planner-only anytime behavior with exact right-continuous action-step integration through acquisition cost 1.0. Reports at display costs are descriptive and do not determine actions. Phase B evaluates true autonomous stopping: execution breaks immediately and no later reveal may repair the stopped report.

Primary comparisons are `L_CTG` versus `R_BALANCED`, the VALID-selected rule, and `L_BC`, separately for LOCATE and CHARACTERIZE. Reported diagnostics include:

- exact `AUSC_any` and success at fixed acquisition costs;
- full-input report quality and cost to success targets 0.8 and 0.9, or `NOT_REACHED`;
- autonomous completion AUSC, false-stop rate, exhaustion rate, stop cost, and failure-penalized cost;
- normalized route length, turns, endpoint, and inference time with no hardware-time claim;
- per-domain paired differences and one shared 5,000-repeat paired bootstrap with six domains equally weighted.

The physical specimen is the statistical unit. Tasks, policy seeds, and trajectories do not increase `n`. Proxy estimates remain explicitly proxy-scoped. With zero reviewed references, the formal planner and stop effects are `INCONCLUSIVE` regardless of proxy direction.

`L_NO_VLM` is trained only if the main VALID comparison is positive. `L_OPEN_INIT` is lower priority and resource-scoped. Neither is added after inspecting TEST outcomes.

## Acceptance and output contract

The run is complete only when it produces actual trained weights, real optimizer steps, a frozen VALID selection, untouched TEST evaluation, and the required compact artifacts in `results/learned_cscan_same_perception/`. The result summary keeps independent status fields for execution, reference, formal planner effect, proxy planner effect, stop effect, and VLM increment.

Required generated files are:

```text
config.yaml
inventory.json
split_manifest.csv
reference_manifest.csv
perception_manifest.json
surface_percepts.jsonl
readout_validation.csv
training_bank_manifest.json
training_log.csv
validation_selection.json
model_manifest.json
models/
per_episode_metrics.csv
trajectories.parquet
comparisons.csv
failure_cases.csv
summary.json
CHECKSUMS.sha256
```

The CLI provides `prepare`, `perception`, `build-train-bank`, `train`, `validate`, `evaluate --split test`, and `summarize`. Tests cover same-history packet determinism, data non-leakage, progressive cost, Reader v2 support, relevant-update stopping, legal masking, real gradient updates, branch isolation, queried-action cost-to-go, terminal STOP behavior, exact metrics, physical-specimen bootstrap accounting, and formal-null provenance.

## Frozen boundaries and non-goals

- Do not modify historical source or result roots named by the execution prompt.
- Do not train or fine-tune the VLM, old G1 policy, or CAI assessor.
- Do not rebuild G0/G1 teacher banks or run AAWR, PPO, GRPO, or architecture search.
- Do not use TEST to repair the prompt, reader, policy, stop threshold, or comparator.
- Do not claim millimeter calibration, hardware time, robot path optimality, acoustic waveform realism, damage causality, new-material transfer, or reviewed-reference detection accuracy.
- Do not copy source images or foundation-model weights into Git.
- Do not expand this pilot to all 276 specimens or six leave-domain-out folds in this run.

The final scientific conclusion must answer only whether a trained planner improves proxy-scoped inspection organization under this frozen common perception interface, while naming reviewed-reference coverage, readout quality, learning gain, and statistical power as separate limitations.
