# MVA A5 Oracle-Imitation Policy Design

Date: 2026-08-23
Status: frozen after formal A4 and before A5 result generation

## Authorization and goal

Formal A4 issued `MVA_A5_AUTHORIZED`: the global-mechanical-to-oracle AUEBC
gap is 39.766%, its synchronized bootstrap lower bound is positive, and the
oracle improves in all six domains. A5 may therefore test whether a small
deployable policy can imitate source-domain mechanical oracles. A6 and A7
remain locked.

The A4 global mask itself issued `MVA_A4_GLOBAL_NO_GO`. Consequently, an A5
policy is scientifically useful only if it improves both the registered global
mechanical mask and uniform acquisition. Merely outperforming the failed A4
mask is insufficient for promotion.

## Information barriers

For held-out target domain `d`, policy fitting and model selection use only the
other five source domains. Teacher labels are regenerated for A5 rather than
reusing A2 oracle rows. For each query source domain `q`, the teacher P-A model
is trained on the four domains excluding both `d` and `q`; its PCA dimension is
selected inside those four domains. Thus target-domain images, CAI, metadata,
embeddings, predictors, and A2 teacher values cannot influence A5 labels.

Offline source CAI and full source images may define teacher values only. They
must never enter the policy-state or candidate-feature tensors. Target CAI is
used only after a complete target trajectory has been fixed, to evaluate MAE.

## Teacher states and labels

Each source specimen begins at the A1 initial survey selected for the current
A5 outer fold. The registered nested action space contains every feasible
one-level refinement `0 -> 1` or `1 -> 2` that fits the current checkpoint.
At every state, the source-only teacher evaluates all feasible candidates by
absolute CAI-error reduction under its query-domain OOF P-A predictor. The
largest value is the selected action; ties prefer lower cell index and then
lower target level.

The cache retains the complete candidate value vector, selected action,
observed-state features, fit audits, exact budgets, and content hashes. Full
source image pixels and source CAI are not serialized into policy inputs.

## Policy inputs

The global state is:

```text
current reconstructed-image ResNet18 embedding: 512
current 8 x 8 refinement levels / 2:             64
current source-safe P-A CAI prediction:            1
used effective budget:                             1
remaining budget to 25%:                           1
                                                    ---
                                                    579
```

The eight candidate features are deterministic functions of the current
reconstruction, measurement mask, grid, and candidate geometry:

```text
normalized cell row
normalized cell column
current refinement level / 2
candidate added-measurement fraction
current measured fraction inside the cell
local reconstructed-image gradient magnitude
local reconstructed-image variance
distance to the nearest measured location
```

No impact-center feature is used because no physical impact-coordinate
authority is registered. Continuous features are standardized with statistics
fit only on the five source domains of the current outer fold.

## Architecture and optimization

The shared scorer is fixed before target evaluation:

```text
global 579 -> 64 -> 32
candidate 8 -> 32 -> 16
concat 48 -> 32 -> 1
```

All hidden layers use ReLU. The model has fewer than 50,000 trainable
parameters. Training uses float64 CPU tensors, one BLAS/Torch thread, a fixed
outer-specific seed, Adam with learning rate 1e-3 and weight decay 1e-4,
gradient norm cap 5, 50 epochs, and batches of 128 states. There is no early
stopping or target-informed hyperparameter selection.

For each state, the primary loss averages
`softplus(score_other - score_teacher)` over feasible nonteacher candidates.
State weights give equal mass to domains, then specimens within domains, then
states within specimens. One optimizer update is made after accumulating the
complete weighted epoch gradient, preserving the registered weighting exactly.

## Deployment and comparators

On a target specimen, the policy sees only the current reconstruction and
measurement geometry, chooses one feasible action, receives the corresponding
new measured RGB values from the simulator, and repeats. Current P-A prediction
uses the five-source outer-safe model. Checkpoint CAI evaluation uses the exact
A2 P-B head for that outer domain and checkpoint.

Registered comparators are uniform, all 100 random seeds, center-first,
observed-gradient appearance heuristic, observed-uncertainty reconstruction
heuristic, global mechanical mask, the imitation policy, and the retrospective
mechanical oracle. Only the oracle may read true target CAI or unmeasured target
pixels for action selection, and it is always styled as an upper bound.

## Metrics and decisions

Primary evidence is equal-domain P-B MAE and AUEBC over 6.25%-25%. Secondary
evidence includes B2.5, B5, B7.5, policy top-1 teacher accuracy, pairwise
agreement, and complete per-domain effects.

The oracle gap closure is:

```text
(AUEBC_global_mechanical - AUEBC_policy)
------------------------------------------------
(AUEBC_global_mechanical - AUEBC_mechanical_oracle)
```

One synchronized 100000 x 6 PCG64 domain-bootstrap matrix with seed 20260823
is reused for all A5 effects. `MVA_A5_POLICY_GO` requires all of:

1. global-mechanical-minus-policy AUEBC point effect > 0;
2. its 95% bootstrap lower bound > 0 and at least 4/6 domain effects > 0;
3. uniform-minus-policy AUEBC point effect > 0;
4. its 95% bootstrap lower bound > 0 and at least 4/6 domain effects > 0;
5. point oracle gap closure >= 20%.

Otherwise the status is `MVA_A5_POLICY_NO_GO`. `MVA_A6_AUTHORIZED` is issued
only with `MVA_A5_POLICY_GO`; otherwise `MVA_A6_NOT_AUTHORIZED`. B5 improvement
is reported but is not a gate because the prompt marks it as desirable rather
than mandatory.

## Claim boundary

A positive A5 result supports a source-trained policy in retrospective
normalized-raster simulation. It does not establish real-time control,
physical scanner coordinates, inspection-time reduction, or industrial
deployment. Current-cohort researcher exposure is reported explicitly.

