# G1 Observable Inspection Policy Design

Date: 2026-09-01  
Base: `7a10cd425de582fa158bf6639285731ccd8ff7a7`  
Status: frozen design before production training

## Objective

Measure whether a small task-conditioned policy can recover privileged G0
inspection decisions while receiving only surface hypothesis, already acquired
ultrasound, task identity, scanner geometry, and exact measurement state. FIELD and
CAI are separate engineering objectives. A negative observability result is valid.

## Fixed Scientific Boundary

- Six leave-one-Hasebe-domain-out outer folds, 276 specimens, equal domain weight.
- Universal geometry-only grid `0.015625`, K=8 farthest-spread warm start, endpoint
  0.25, with K4/K16 sensitivity only.
- Fixed action slots: 64 cells times transitions `-1->0`, `0->1`, `1->2` = 192.
- FOCUS/BROADEN/REFINE is derived from the selected primitive; STOP is separate.
- G0 and all historical science paths are immutable.
- Target truth is unavailable until the complete target trajectory is sealed.

## Split and Dependency Design

For outer target `D_o` and teacher-labeled source `D_s`, fit the prior and CAI
assessor on exactly the other four domains. The assessor uses only the 13
label-independent post-K8 states per fit specimen; oracle checkpoint states never
enter it. After source-only model selection, refit final dependencies and policy on
all five sources, hash/freeze them, then roll the target without target truth.

## Observable Tensor Contract

`G1PolicyState` is immutable and contains:

```text
reconstruction_embedding: float64[512]
global_scalars:            float64[17]
task_token:                float64[2]
cell_features:             float64[64,18]
candidate_features:        float64[192,12]
legal_action_mask:         bool[192]
observation/state hashes:  integrity only
```

The 17 global scalars are effective/remaining budget, observed-cell fraction,
mean observed level, raw count and fraction for unmeasured/level0/level1/level2,
observable CAI estimate plus presence mask, and normalized height, width, and log
aspect. Raw counts are scaled in the fold-local normalizer; fractions remain
explicit to satisfy the acquisition-state contract.

The 18 cell fields are row, column, surface score, top-eight flag, four-level
one-hot, observed unique-pixel fraction, observed RGB mean (3), RGB std (3), mean
absolute deviation from source background, observed-value mask, and previous-cell
flag. Unmeasured RGB fields are zero; reconstruction values at unseen locations are
never used here.

The 12 candidate fields are three-way from-level one-hot, three-way to-level
one-hot, exact added cost/native count, remaining budget, candidate
budget/endpoint, and FOCUS/BROADEN/REFINE one-hot. Slot index is
`transition_index*64+cell_index`. Illegal logits are exactly `-inf` before softmax.

CAI context has two preregistered modes: FIELD-zeroed/task-specific mask and the
same observable assessor estimate for both tasks as generic state context. Only
inner-source validation may choose; NO_TASK/WRONG_TASK controls use the selected
mode unchanged.

## Teacher Bank

Each outer/source fold, specimen, and task receives one K=8 state, 12 deterministic
continuation snapshots, and up to four cross-fitted oracle checkpoint states. Every
state stores all legal candidate utilities. Physical specimens and tasks receive
equal weight. Large tables use Zstandard-compressed Parquet outside Git, with a
committed manifest binding config, rosters, input/dependency/state hashes,
candidate counts, software versions, and rematerialization command.

Teacher distributions use `regret=best-utility`, scale
`max(IQR(utility),1e-12)`, temperatures `{0.25,0.5,1,2}`, and uniform mass when
utilities tie within `1e-12`.

## Models

`SharedActionMLP` is the no-attention baseline. The primary
`StructuredInspectionPolicy` projects the 512-D reconstruction to 128, embeds task
to 16 and global scalars to 32, fuses one 128-D global token, encodes each cell with
a shared 18->128->128 MLP, and contextualizes 65 tokens using two Transformer
layers (`d=128`, four heads, FF=256, dropout 0.1). A shared scorer consumes
contextual cell/global tokens and 12 candidate fields through 128->64->1. A
separate 128->64->1 head predicts sufficiency. Trainable parameters must be below
one million; frozen ResNet18 parameters are excluded.

## Training and Selection

Stage A trains masked hard BC. Stage B trains masked soft teacher cross entropy and
reports expected teacher regret. The best supervised source method seeds source-only
DAgger; iteration counts 0/1/2 are compared, at most 16 deterministic policy-visited
quantile states per specimen/task/iteration. Each candidate uses deterministic
initialization and is selected by equal-domain source engineering AUEBC, gap
closure, improved domains, then simplicity.

AAWR-style learning is conditionally authorized only if source validation shows
positive action observability but less than 20% source gap closure for an authorized
task. Actor inputs stay observable; a small source-only critic may add full-scan
embedding and true CAI/loss. No external AAWR source is copied. If the condition is
false, status is `NOT_RUN_NOT_AUTHORIZED`.

STOP is trained separately from observable global context. Source privileged labels
use the 1.05 fixed-reference criterion. Select the highest-saving source threshold
meeting loss ratio <=1.05 and premature rate <=5%; otherwise disable STOP and use
the fixed endpoint.

## Target Freeze and Evaluation

For each frozen model variant, roll every target specimen causally from K=8 to STOP
or 0.25. Before any target truth access, seal actions, logits/scores, STOP scores,
states, acquired positions/values, dependency/model/threshold hashes, and selection
identity. Only then compute same-geometry fixed/learned/oracle FIELD and CAI curves,
task variants, surface variants, and frozen misleading strata.

Inference pairs methods by physical specimen, bootstraps specimens synchronously
within domain for 100,000 replicates, computes domain effects, and weights six
domains equally. Cells/actions/states/seeds are not independent units.

## Decision

A task policy passes only when it beats the strongest gate-eligible fixed baseline,
the paired 95% CI lower bound is positive, at least four domains improve, gap closure
is at least 20%, leakage is absent, and replay validates. Task conditioning requires
both correct-vs-WRONG_TASK and correct-vs-NO_TASK effects for both tasks. STOP is a
separate component. Final status uses exactly the six prompt-defined G1 values; no
post-target rescue or architecture change is allowed.
