# MVD Repository Authority Audit

Date: 2026-08-24
Baseline commit: `d0e0ebfca1f1de6b04e9cb43a5065de3435aee5b`
Decision scope: MVD M0/M1 feasibility only

## Reproduction status

The complete research workspace reproduced the registered FULL evaluator with
equal-domain MAE `0.08963580465761434`, maximum prediction delta
`4.440892098500626e-16`, and selected PCA dimensions `(8,32,8,8,8,8)`.
The MVA authority state is
`3ef44b3379a4377758443d6fa6ef23d8aae2a83f32b5f1ce86a42b18398c5f3a`;
the fresh FULL reproduction state is
`2bfff653ffbdc26a6f0332e4a1bf0b9c7dd78b09559475057a847c10df3177c6`.

The Git repository is a privacy-sanitized compact export. It intentionally
omits raw C-scan files. Formal MVD computation therefore reads the frozen raw
cohort from the complete workspace, while code, compact authority arrays,
results, manifests, and replay evidence are published here.

## Required authority answers

1. **Native acquisition shapes.** The 276 specimens use `(674,675)` for 240
   specimens, `(338,352)` for 19, and `(338,340)` for 17.
2. **Initial budgets.** Registered candidates are `1.5625%`, `3.125%`, and
   `6.25%`. Source-only A1 selected `3.125%` for outer domain `74t7kcdgkr` and
   `1.5625%` for the other five domains.
3. **Candidate-cell semantics.** Sixty-four normalized cells form an `8 x 8`
   partition. Cell index is row-major (`row * 8 + column`). Each M0 candidate
   is exactly one `level0 -> level1` refinement of one cell from the initial
   state.
4. **Nested lattice.** Level 0 is the endpoint-preserving initial survey;
   level 1 is the registered P5 25% endpoint-preserving lattice; level 2 is
   every native raster location inside the cell. Cell boundaries are retained
   at every level.
5. **Measurement counting.** `measurement_mask` unions native raster
   locations across all cells. `budget_record` counts unique true mask entries,
   and candidate cost is the number of newly true locations. Shared boundaries
   are never double-counted.
6. **Bilinear reconstruction.** Initial states use the registered P5 bilinear
   path. Mixed-cell refinements use deterministic rectangular bilinear patches
   on the native RGB raster.
7. **Measured restoration.** Every observed RGB triplet is overwritten from
   the source after interpolation and checked exactly. MVA operates on
   pseudocolor raster observations, not calibrated ultrasonic amplitudes.
8. **Frozen encoder.** The ImageNet ResNet18 weights SHA-256 is
   `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
   The complete crop is resized to `224 x 224`; the final embedding is 512-D.
9. **FULL reproduction.** Nested LODO, fold-local preprocessing/PCA, and Ridge
   reproduce the registered MAE within the `1e-12` tolerance, as stated above.
10. **CandidateBank dimensions.** For each initial budget it contains
    `initial_embeddings[N,512]`, `embeddings[N,64,512]`, reconstruction and
    appearance values `[N,64]`, and exact added measurements `[N,64]`.
11. **Candidate embedding meaning.** Each candidate embedding is the encoder
    output of the real post-refinement state constructed with the source
    specimen's unobserved RGB. It is privileged oracle-label material and is
    forbidden as a student input.
12. **Strict A4 source OOF.** For every outer target and every query source
    domain, the label predictor excludes both domains. Its PCA, preprocessing,
    PCA-dimension selection, and Ridge fit use only the other four domains.
13. **Query-domain exclusion.** `SourceLabelResult` requires the query source
    domain to be disjoint from every final predictor fit and verifies disjoint
    specimen rosters.
14. **Outer-target isolation.** The outer target is absent from source-label
    fitting, global ranking construction, evaluator selection, and M1 model
    selection/training. Target true CAI is allowed only to construct held-out
    diagnostic labels and compute held-out evaluation metrics.
15. **A5 policy input.** The global vector is 579-D: current 512-D embedding,
    64 normalized cell levels, current P-A prediction, used budget, and
    remaining budget. Each feasible action has eight observed-only features:
    normalized row/column, current level, added fraction, measured fraction,
    local gradient, local variance, and nearest-measured distance.
16. **A5 supervision retention.** Teacher caches save the complete value vector
    for every feasible action. The historical A5 training conversion discards
    the magnitudes and trains only from the selected top-1 index against all
    other candidates.
17. **Historical hashes.** A2 tree:
    `71d279dcd2dc1da9a09d08164669e9dc9432eea1da6476f5037fbad5aecc7595`;
    A4 tree:
    `2ed736568419ff549981430b202889d3f5b89e0971a8298ae9e125ddf9681e78`;
    A5 tree:
    `3712e6ebe09517565204a15b034cd3eb03da1fe3737a9a56e62cb8548f9fcbe8`.
    Their manifest hashes are respectively `a5499078463dccd3092bcedb795dd872e5196da85527201a291883aea5fe545c`,
    `bc2556041d5de9d42522830784ea328aed5a3857eec5e36d464f8237145a5ec5`,
    and `c59db1608d6f180587bb24b994bef2f1049b576e072cc4edbe0113f92407dd73`.
18. **Direct reuse.** MVD reuses grid/state/action/budget operations, bilinear
    reconstruction and restoration, frozen encoder, CAI evaluator and
    cross-fitting, CandidateBank validation, A4 strict-OOF source values,
    candidate features, Mechanical Value, AUEBC/B5, synchronized bootstrap,
    manifest hashing, checksums, and byte replay.
19. **Frozen code.** All files under `src/cmc_bbdm/mva/`, the A0-A5 configs,
    formal results, and replay results are historical authority and must not be
    modified.
20. **New modules.** MVD code is isolated under `src/cmc_bbdm/mvd/`. This run
    implements authority bindings, initial-value data, exact action-cost audit,
    fixed one-shot oracle selection/evaluation, interaction audit,
    observability data/models/metrics, artifacts, and replay. Formal MVD
    deployment, M2, and M3 remain absent and locked.

## Candidate-bank bindings

The `1.5625%` bank state is
`2b17097a85fdb41b4413fecfa7f5b141b2e132cc479adb127fceb28c2c444fc4`;
the `3.125%` bank state is
`e44d7b8e1ef1b1f715eeddc3ab9af485d1a993e0acfd792d86086a282b2fa0c0`.
All arrays are immutable and content-hashed. Action costs are not equal, so M0
must use exact-cost greedy traversal and must not use simple Top-K selection.

## Claim boundary

The authority supports normalized-raster acquisition simulation only. It does
not establish scanner pitch, physical measurement time, online control, or
prospective CAI performance.
