# BC C-scan Path-B Supplement: Design and Reuse Contract

## Identity and question

- Repository base: `d8b5b090891fc030931c6dc81e3619a80966f739`.
- Branch: `research/bc-cscan-path-b-supplement`.
- Controlling prompt SHA-256: `ff4e2986d904127225a9fa0c834b38fddcb55860a995d4378031cde2d43a06c9`.
- Question: can the existing behavior-cloned C-scan actor complete LOCATE and CHARACTERIZE with lower failure-penalized acquisition cost without a material completion-rate loss under the same visible perception contract?
- Scope: retrospective supplement on the reused 60-specimen cohort, not a new method, external test, hardware trial, or CTG redesign.

## Frozen evidence

The source configuration SHA-256 is `12268dcaf470f769c326007702f7b0b6f4a13cf326447eef652a6dda972b5662`. The frozen BC seed-1 checkpoint is `c95185aec70515f2578d671e520ebe9b57e600e91e70e5f6752567ae2f87fbe5`; the frozen STOP checkpoint is `c6af5c383ef4c586a4ac85c7dc9afbcd5bc71dc07558c692fb295d0f5386892d`. Neither checkpoint will be overwritten or retrained.

The pilot has six domains with 24 TRAIN, 12 VALID, and 24 TEST physical specimens. The original BC bank contains 192 rows, with 144 rows from 18 fitting specimens and 48 rows from six internal-validation specimens. The internal-validation keys are:

```text
74t7kcdgkr:c8-32
cgtnjyggtm:q24-4
w68dtmpfyf:q16-42
xcmzfsbd9t:c24-43
yfxyg8jm46:c16-42
ykhs7s2dck:q8-7
```

The source bank SHA-256 is `52167731b60bbd6db759a1e74a85bd0eb0bc8ea208bd51e981bff44d841281c7`; its canonical `base_bc` content digest is `9d73c3e27de235ffe8337738517aec48f3f4a3267eb5f687ef0f6eeaae3566d1`. The source bank remains read-only and only its `base_bc` rows are consumed.

All 276 reference-manifest rows are `ALGORITHM_DERIVED_NOT_REVIEWED`; attributable reviewed coverage is 0/60 in the pilot. Formal effects therefore remain null unless independently reviewed payloads are imported later.

## Chosen architecture

The supplement will add four focused modules under `cmc_bbdm.learned_cscan`:

- `supplement_adapters.py`: immutable FULL, NO_VLM, and NO_US_FEEDBACK actor views; frozen report conversion; supplement checkpoint identity.
- `episode_stop_calibration.py`: ordered first-stop outcomes, episode-risk summaries, the fixed three-threshold qualification rule, and physical-specimen confidence intervals.
- `bc_supplement.py`: strict supplement configuration, source audit, base-BC loading, replica/ablation training, VALID/TEST rollout, reference import/rescore, confirmation roster, and true-break replay.
- `supplement_analysis.py`: seed aggregation, paired domain-balanced bootstrap, tables, figures, summary, checksums, and reproduction instructions.

The CLI is `scripts/run_bc_cscan_supplement.py`. It dispatches explicit recoverable stages and always separates the frozen source root from `results/bc_cscan_path_b_supplement/`.

## Data flow

```text
frozen config/model/cache/bank/results
  -> audit and direct existing-result analysis
  -> base_bc only
  -> BC seed 2/3 + NO_VLM seed 1 + NO_US_FEEDBACK seed 1
  -> VALID trajectories for BC seed 1 and R_BALANCED_P8
  -> episode-first threshold calibration
  -> at most one TRAIN-only S_BC_CAL refit if S_EP is not qualified
  -> frozen threshold/head
  -> retrospective TEST matrix and prefix-derived autonomous outcomes
  -> six-domain true-break equivalence replay
  -> paired statistics, figures, claim boundaries, and handoff
```

No VLM calls are needed for the old 60 specimens because the frozen surface cache is complete. TEST labels are not used in fitting or STOP calibration. The optional 24-specimen confirmation cohort is only rostered and queued while reviewed references are absent.

## Input ablations

`BC_NO_VLM` uses the existing `LearnedCellActor(use_surface_features=False)` during both training and inference. `BC_NO_US_FEEDBACK` uses a deterministic transform during both training and inference:

- preserve cell features 0:4 and 15:17; zero 4:15;
- preserve subblock features 0:2; zero 2:10;
- zero global feature 7 and preserve the other global features;
- preserve action-history features and the legal mask.

The transform returns copied arrays and never mutates an `ObservationPacket` or the frozen bank.

## STOP contract

`S_RULE` and `S_OLD_090` remain frozen controls. `S_EP` scans only thresholds 0.90, 0.95, and 0.99 on ordered VALID trajectories. It fixes the first eligible crossing and never replaces a false first stop with a later successful report. A threshold must meet the specified wrong-among-stops, completion, rule-margin, and stop-count criteria for both BC seed 1 and R_BALANCED_P8 within each task.

If either task has no qualified old-head threshold, exactly one 13,697-parameter `S_BC_CAL` may be fitted from TRAIN natural trajectories using the unchanged 18/6 internal split, then calibrated once with the same candidate thresholds. TEST cannot trigger a fit or threshold change.

## Statistics and claim scope

Planner-only results preserve non-monotone current-report success. The main BC result averages seeds 1/2/3 within each specimen before a domain-balanced paired bootstrap. Ablations compare seed 1 to seed 1. Autonomous evidence reports completion, false-stop episode rate, wrong among actual stops, exhaustion, autonomous AUSC, and failure-penalized acquisition cost together.

Path-B support requires both a completion-rate lower confidence bound of at least -0.05 and a positive lower confidence bound for rule-minus-BC failure-penalized cost, separately by task. Ninety-five-percent intervals are descriptive; a fixed 97.5% familywise interval is also reported for joint two-task language. With 0/60 reviewed references, all scientific effect claims are explicitly proxy-scoped.

## Alternatives considered

1. Modify the original `benchmark.py` and regenerate the prior run. Rejected because it violates the frozen-source contract and risks changing historical identities.
2. Copy the full benchmark into a second pipeline. Rejected because duplicated reader, world, and metric logic would make fairness and maintenance harder to verify.
3. Use thin adapters and replayable stage outputs. Selected because it reuses the exact low-level world, packet, Reader, rule, actor, and metric implementations while isolating all new outputs.

## Verification

The focused supplement suite will contain 8-12 semantic tests. The final gate will also rerun the existing 18 learned-C-scan tests, Ruff on new files, checksum verification, source checkpoint hashes, the frozen tracked-file diff, transition/update/VLM caps, representative true-break equivalence, and local/upstream/remote SHA equality.
