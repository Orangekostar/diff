# MVD M1 Mechanical-Value Observability Protocol

Date frozen: 2026-08-24
Authorization: only after `MVD_ONE_SHOT_GO`

## Question and information barrier

M1 tests whether the 64 initial Mechanical Values are predictable from the
deployable coarse state. Student inputs are restricted to the initial 512-D
embedding, current OOF CAI prediction, and eight observed-only candidate
features. Candidate refined embeddings, full images, unobserved RGB, true CAI,
and future states are rejected by the data contract.

The outer target domain is absent from architecture, loss, feature, hidden-size,
regularization, and model selection. Source teacher values remain query-domain
OOF and outer-safe. Target Mechanical Values are used only after the model and
ranking are frozen.

## Baselines and models

- O0: historical A4 global mechanical ranking.
- O1: candidate-only Ridge and a small candidate MLP from the eight features.
- O2: primary global-plus-candidate scorer. The global branch is
  `513 -> 64 -> 32`, candidate branch `8 -> 32 -> 16`, and scorer
  `48 -> 32 -> 1` with ReLU activations.
- O3: the historical A5 579/8 shared scorer evaluated only at its initial state.
- Observed uncertainty: exact historical score
  `added_fraction * (local_variance + nearest_measured_distance)`.
- Random: 100 PCG64 score vectors using the registered MVA random seeds.

O2 compares three supervision variants on the same architecture: L0 selected
top-1 versus all pairwise logistic loss; L1 Huber value regression; and L2 Huber
plus margin-weighted pairwise ranking. L2 uses lambda candidates
`0.1, 0.5, 1.0`. Ridge alphas are `0.1, 1.0, 10.0, 100.0`. The fixed optimizer
is deterministic float64 Adam, 50 epochs, learning rate `1e-3`, weight decay
`1e-4`, batch size 128 specimens, gradient clip 5, and no early stopping.
All neural models remain below 100,000 parameters.

## Source-only selection

For each outer fold, leave each source domain out in turn. Rank configurations
by equal-domain mean NDCG@10, then Spearman, then lower parameter count, then
lexicographic configuration ID. Refit the selected O2 configuration on all five
source domains. O1, O3, global, uncertainty, and random are reported but cannot
choose the primary O2 configuration.

Training weights give equal mass to domains and specimens. Candidate values are
not clipped to positive values. Selection ties use lower cell index.

## Metrics

Compute per specimen Spearman, NDCG@5/10, Recall@5/10, Regret@1, exact-cost
budgeted set regret, and true Mechanical Value captured at all M0 checkpoints.
Aggregate by equal-domain mean. All gate intervals use synchronized 100,000
resamples of the six held-out domains.

After source-only selection is frozen, a non-selection diagnostic applies the
predicted initial ranking in the M0 simulator and reports its AUEBC advantage
capture. Target CAI outcomes never tune the observability model.

## Gate

M1 is `MVD_OBSERVABILITY_GO` only when all conditions hold:

1. The selected O2 mean Spearman point estimate and synchronized 95% lower
   bound are positive.
2. O2-minus-global NDCG@10 has positive point estimate and lower bound and is
   positive in at least four of six domains.
3. Global-minus-O2 mean exact-budget set regret has positive point estimate and
   lower bound and is positive in at least four domains.
4. Random-minus-O2 mean exact-budget set regret has positive point estimate and
   lower bound and is positive in at least four domains.

Observed uncertainty is a required comparison but not a formal GO condition;
failure to beat it is reported as a limitation. Predicted AUEBC advantage
capture of at least `0.35` with the same direction in at least four domains is
`MVD_OBSERVABILITY_STRONG_GO`, not a replacement for the four GO conditions.

Any failed GO condition yields `MVD_OBSERVABILITY_NO_GO`. Capacity rescue,
Transformer, GNN, RL, diffusion, M2, and M3 remain forbidden in this execution.

## Outputs

The formal package is `results/mvd/m1_observability/` with the required
prediction, metric, regret, bootstrap, summary, report, manifest, and checksum
files plus compact model-selection audits. Replay is byte-identical under
`results/mvd/replay/m1_observability/`.
