# Agentic NDE P1 Visual Observability Protocol

## Frozen identity

P1 asks whether correctly registered specimen-specific impacted-surface RGB adds
information about the frozen 64-cell mechanical measurement-value map beyond
the old deployable state. It is a source-trained, held-out-domain visual
observability experiment. It is not an LLM, VLM-agent, reinforcement-learning,
or scanner-time study.

This protocol was frozen only after `P0R_AUTHOR_REGISTRATION_GO`. Formal P1
target-domain scoring and evaluation are forbidden before the commit containing
this file and
`paper_v3/configs/agentic_nde_p1_visual_observability.yaml`.

## P0R authority

- Authorized roster: 276/276 specimens, six domains, no exclusions.
- Roster state: `4fd8c6076dd3fcdf908a73739251db215fcb01f570f1a930b7faf250fe6d285a`.
- Registration authority: `38ab3cf32e866cda447a5edf2637fa502406c4c5c574bc966c13cc1cbbd2553a`.
- Registration file: `43bafc8819a1bea0df7f7ea9fa2b1dc194591ef9a4b58e8dcf3e64c13622e265`.
- Grid mapping file: `fb076daefbb0b748184d10b777268dfd13777a3e78cb7fed104252e6d62afd35`.
- Global author-resolved orientation: `ROT90`.
- Mapping basis: full-frame, normalized edge-to-edge pixel correspondence; no
  physical-mm calibration and no target-derived alignment.

Every surface path and file hash comes from the frozen P0R surface manifest.
Every local cell patch comes from the inverse P0R cell mapping. Splitting the
unrotated surface into an independently assumed 8x8 grid is forbidden.

## Target and leakage barrier

The only P1 teacher is `primary_value` from
`results/mva/a2_oracle_value/oracle_values.parquet`, SHA-256
`6b289f2f6f74ac75dde47ea7cbfefcda1c49f025e74227bfb34ef269182ff963`,
after the exact filter:

```text
method = mechanical_oracle
step = 0
from_level = 0
to_level = 1
nominal_checkpoint = 0.0625
```

The pre-exclusion roster must be exactly 17,664 rows (276 x 64). The target
column is `primary_value`; `current_prediction` is an observed old-state input.

For outer domain `D`, only the other five domains may supply labels for fitting,
inner selection, representation selection, or fusion selection. The target
surface and old-state inputs are label-free inference inputs. Target scores and
their model/config hashes must be frozen before target `primary_value`, target
CAI, target selected actions, or target action utility is exposed to evaluation.

The frozen C0 scores are exact `mvd_m1_o2` / `o2_global_candidate` outputs.
Their historical teacher used outer-specific initial budgets. The P1 teacher is
the A2 `0.0625` authority above. These definitions are deliberately retained as
distinct authorities: C0 is not refit, its scores are not changed, and its
historical `teacher_value` column is not used as the P1 target.

## Visual features

The sole encoder candidate is the repository's hash-bound torchvision
ResNet18 ImageNet1K V1 asset, SHA-256
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
It is frozen and returns the 512-D post-average-pool, pre-classifier vector.
DINOv2, SigLIP/SigLIP2, and Qwen2.5-VL are excluded because an exact local
asset, revision, and license were not all resolved before this freeze.

Surface preprocessing is RGB uint8, exact deterministic P0R crop, CPU
bilinear-antialiased resize to 224 x 224, division by 255, and ImageNet RGB
normalization. Its canonical transform SHA-256 is
`2b275ebbc220e6a0376d305d0996f4ffe80509fc8b27223fd919331a100acbe5`.
Continuous P0R pixel-center boxes become half-open integer crops: lower bounds
are `ceil(lower)`; upper bounds are `ceil(upper)`, except a frame maximum maps
to the full axis length. Bounds are clamped, padding is forbidden, and the
global crop is the full P0R surface frame.

Three hash-bound feature arrays are materialized without labels: whole-surface
global embeddings, correctly registered 64-cell local embeddings, and the
preregistered wrong-orientation 64-cell embeddings. The formal feature manifest
binds array paths, shapes, dtypes, hashes, source hashes, transform hash, model
hash, and runtime versions. Feature arrays are reproducible cache, not new
trainable weights.

## Old state and model candidates

The exact old state has 521 values per candidate:

```text
512-D frozen initial C-scan embedding
+ 1 current OOF CAI prediction
+ 8 observed-only candidate descriptors
```

The MVD initial budget is `0.03125` only for outer domain `74t7kcdgkr` and
`0.015625` for the other five outer folds. Candidate-bank and observed-feature
states are hash-bound in the YAML configuration.

Representations are `OLD`, `GLOBAL = OLD + global`, `LOCAL = OLD + local`, and
`LOCAL_GLOBAL = OLD + local + global`. Correct spatial candidates are `LOCAL`
and `LOCAL_GLOBAL`. `GLOBAL` is the context control; `OLD` is a same-target
refit diagnostic, while C0 remains the formal frozen deployable reference.

Each representation uses the same preregistered low-capacity head roster:

- Ridge with alpha in `{0.1, 1, 10, 100}`;
- a shared `input -> 32 -> 16 -> 1` ReLU MLP, Smooth-L1 loss, Adam,
  50 epochs, and fewer than 100,000 trainable parameters.

Standardization is fit on source-training data only. Specimens have equal mass
within each domain and domains have equal mass. Outer splitting is leave-one-
domain-out; model and representation selection use inner leave-one-source-
domain-out. Selection is lexicographically fixed to higher equal-domain
NDCG@10, lower equal-domain next-action regret, lower parameter count, then
configuration ID. Formal CAI AUEBC is not used for hyperparameter selection.

## Fusion and controls

Rank-space fusion uses stable descending ordinal ranks, lower cell ID first on
exact ties, percentile `(63-rank)/63`, and lambda in
`{0, 0.25, 0.5, 0.75, 1}`. Lambda is selected on source domains only. Controls
reuse the selected correct head hyperparameters and lambda; controls are refit
on their controlled source inputs so a target-only distribution shift cannot
manufacture an advantage.

- C0: exact frozen `mvd_m1_o2` score, no refit.
- C1: negative squared distance from the 8x8 grid center.
- C2: source-selected `GLOBAL` context representation.
- C3: within-domain shuffled surface. The sorted roster is shifted by one
  nonzero domain-level SHA-256-derived offset, producing a bijection with no
  self donor and no cross-domain donor. If local+global is selected, both come
  from the donor surface.
- C4: one wrong D4 orientation per specimen. SHA-256 selects from
  `IDENTITY, ROT180, ROT270, FLIP_X, FLIP_Y, TRANSPOSE,
  ANTI_TRANSPOSE`; correct `ROT90` is impossible. Global context remains the
  recipient's full surface; only spatial correspondence is wrong.
- C5: a deterministic SHA-256-driven Sattolo 64-cycle per specimen, with no
  fixed cell. Global context remains unchanged.
- Mechanical oracle: evaluation-only diagnostic, never a candidate or selector.

The exact hash strings and seeds are frozen in the YAML configuration.

## Metrics, acquisition, and statistics

Primary action metrics are next-action regret and selected one-step CAI utility
(`primary_value`). Secondary metrics are Spearman, NDCG@10, recall@5, top-10%
overlap, and top-1 oracle match.

All formal methods issue 64 static scores and use the existing exact-cost
frozen-ranking/skip-nonfitting machinery at nominal checkpoints
`0.0625, 0.09375, 0.125, 0.1875, 0.25`. The primary engineering metric is
normalized trapezoidal CAI-error AUEBC over `[0.0625, 0.25]`; lower is better.
Cost is unique native-raster acquisition, not scanner time.

The first statistical unit is the physical specimen. A synchronized paired
bootstrap resamples specimens with replacement within each domain, then gives
each of the six held-out domains equal aggregate weight. Seed is `20260831`,
resamples are 100,000, and the interval is the percentile `[0.025, 0.975]`.
Cells, patches, seeds, prompts, and orientations are not independent units.

## Decision tree

Statuses are evaluated in this order.

`P1_SPATIAL_VISUAL_OBSERVABILITY_GO` requires all seven controlling conditions:

1. proposed CAI AUEBC is lower than C0;
2. paired CI lower bound for `C0 - proposed` is positive;
3. at least 4/6 held-out domains improve;
4. at least 20% of the C0-to-mechanical-oracle AUEBC gap is closed;
5. paired CI lower bound for `C3 shuffled - proposed` is positive;
6. paired CI lower bounds for both `C4 wrong - proposed` and
   `C5 deranged - proposed` are positive;
7. paired CI lower bound for `C2 global - proposed` is positive.

`P1_GLOBAL_VISUAL_CONTEXT_GO` is considered only if the spatial gate fails. It
requires C2 to improve C0 in at least 4/6 domains, a positive lower CI for
`C0 - C2`, and a positive lower CI against the shuffled-global control. It
authorizes only registration-free/context-conditioned downstream work and
forbids a local "see here -> scan here" claim.

`P1_DESCRIPTIVE_SPATIAL_SIGNAL_ONLY` applies when the correctly aligned method
improves the preregistered map-ranking point metrics over C0 but the applicable
CAI engineering gate fails. It authorizes no P2/P3/P4.

Otherwise the status is `P1_SURFACE_VISUAL_OBSERVABILITY_NO_GO`; the existing-
data VLM/agent route stops and cannot be rescued with a larger model.

## Outputs and replay

The formal package contains exactly the files prescribed under
`results/agentic_task_driven_nde/p1_visual_observability/`; the decision and
spatial-vs-global analysis are written under
`artifacts/agentic_task_driven_nde/`. A clean replay must reproduce package
bytes and checksums from the frozen config and external Hasebe root. No P2,
P3, or P4 work starts until the P1 status is issued and its authorization is
checked.
