# MVD M0 One-Shot Oracle Protocol

Date frozen: 2026-08-24
Status before execution: `TO_TEST`

## Question

M0 tests whether a perfect Mechanical Value map computed once at the initial
coarse state can be converted into a useful specimen-specific fixed refinement
plan. It does not train or evaluate a deployable student.

## Information and split contract

- Outer evaluation is leave-one-dataset-out over the frozen six-domain,
  276-specimen cohort.
- Initial budgets are the source-only A1 selections frozen by outer domain.
- Initial Mechanical Values use the A4 strict label semantics: the P-A CAI
  predictor for a queried domain is fitted without that query domain and the
  outer target domain.
- Held-out target oracle values use true target CAI only after the outer P-A
  predictor is frozen on the five source domains.
- PCA parameter byte hashes are retained but are not used as the sole numerical
  identity because repeated SVD fits can rotate an unidentifiable subspace. Each
  actual P-B model used for M0 is therefore bound to its raw state hash and must
  reproduce the historical P-B predictions on the exact authoritative Uniform
  embeddings for every target specimen within the frozen MVA baseline tolerance
  of `1e-12`. The historical reference hash, new raw hash, prediction deltas,
  fit-domain roster, and specimen count are published for every outer fold and
  checkpoint. This compatibility rule was frozen before M0 performance was
  aggregated or inspected.
- Candidate embeddings are privileged label inputs. They are never student
  inputs.
- Each specimen's 64 values are scored exactly once at `S0`. No value or
  ranking is recomputed after any action.

## Exact-cost selection

Candidate action costs are unequal. For each checkpoint, traverse the frozen
descending score order with lower cell index as the tie break. Select every
unselected `level0 -> level1` action that fits the exact unique-location cap;
skip actions that do not fit. Reconsider unselected actions only when a later,
larger checkpoint is reached. No knapsack optimization is permitted.

Registered checkpoints are `3.125%`, `6.25%`, `9.375%`, `12.5%`, `18.75%`,
and `25%`. A checkpoint at or below the realized initial survey records the
unchanged initial state under historical MVA semantics. AUEBC uses the frozen
`6.25%-25%` interval.

## Comparators

The complete report includes Uniform, the median of 100 registered Random
seeds, specimen-specific initial Reconstruction ranking, historical A4 Global
Mechanical, the M0 One-Shot Mechanical Oracle, historical sequential A3
Mechanical Oracle, and FULL. Primary comparisons are one-shot oracle versus
Uniform and specimen-specific Reconstruction.

## Interaction audit

For each source specimen, outer fold, method, and registered checkpoint, report

`additive_gain = sum(initial individual Mechanical Values)`

and

`joint_gain = abs(y - prediction(S0)) - abs(y - prediction(S0 + A))`.

The audited sets are one-shot mechanical, specimen-specific reconstruction,
Uniform, and Random. Report Pearson, Spearman, signed bias
`additive_gain - joint_gain`, and MAE. Interaction is descriptive and
non-gating.

## Metrics and gate

Primary prediction protocol is the historical checkpoint-specific P-B head.
Report per-budget and per-domain equal-domain CAI MAE, AUEBC, B5, worst-domain
effects, and synchronized 100,000-resample held-out-domain intervals.

M0 is `MVD_ONE_SHOT_GO` only when all conditions hold:

1. Uniform-minus-one-shot AUEBC point estimate and 95% lower bound are positive.
2. Reconstruction-minus-one-shot AUEBC point estimate and lower bound are positive.
3. One-shot improves in at least four of six domains for each comparison.
4. Headroom retention is at least `0.20`, using the stronger primary baseline:
   `min(AUEBC_uniform, AUEBC_reconstruction)`.

Headroom retention at least `0.50` is `MVD_ONE_SHOT_STRONG_GO`; it does not
replace the four required GO conditions. Any failed condition yields
`MVD_ONE_SHOT_NO_GO`, immediately locks M1/M2/M3, and forbids student training.

## Outputs

The formal package is `results/mvd/m0_one_shot_oracle/` and contains the files
required by the controlling prompt, plus compact raw state/ranking evidence
needed to recompute every derived table. A byte-identical replay package is
stored under `results/mvd/replay/m0_one_shot_oracle/`.
