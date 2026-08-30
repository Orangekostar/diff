# Codex Handoff: Damage-to-Failure Response Audit

Audit date: 2026-08-30

## Final scientific outcome

The 276-specimen authority supports three response descriptors beyond ultimate
CAI strength (`P1_GO`), but the preregistered strict cross-domain P2 gate does
not support an extension from scalar damage to transferable spatial
damage-to-response inference. The final P2 status is
`MACK_EXTENSION_NO_GO`. P3, P4, and P5 are
`NOT_RUN_NOT_AUTHORIZED`; no new paper or manuscript rewrite is authorized.

## Repository identity

- Repository: `git@github.com:Orangekostar/diff.git`
- Exact base: `3951f71f28b6efdf8c74eea0fe274b2a78a9cd57`
- Branch: `research/aei-damage-to-failure-response`
- Worktree: `/home/ww/diff/.worktrees/aei-damage-to-failure-response`
- P2 execution implementation commit: `bdf5604`
- P2 negative-result commit: `dc8fc57`
- Handoff-creation commit: `b998e2a`
- Remote: `origin`, branch `research/aei-damage-to-failure-response`

The synchronized branch SHA is reported after push in the operator completion
response. A Git commit cannot embed its own final SHA or a subsequent remote
HEAD without changing that SHA; this handoff therefore records the commit that
first contains the handoff and leaves final local/remote equality to the
post-commit verification record.

Commits after the exact base, through the P2 result:

```text
109346a docs: define damage-response research gates
b315c64 docs: plan damage-response P0 authority audit
47133ec audit: establish damage-response data authority
e46b4e5 test: freeze damage-response P0 contracts
01f56da feat: bind external damage-response sources
0c57dd5 feat: enforce exact damage-response pairing
5d7060e feat: decode and reconcile raw CAI response
bcac68c feat: write replayable P0 audit packages
52be0c4 feat: audit Hasebe response source records
93f5f3b research: audit damage-to-failure response authority
cbf28ea docs: plan damage-response P1 richness audit
24dd6e4 feat: add standards-grounded CAI response extraction
47aa21e feat: bind fold-local P1 design metadata
8a0620a feat: add strict P1 LODO redundancy evaluation
5bdc020 feat: freeze P1 response-richness decision rules
130875f feat: generalize deterministic P1 artifact packages
de00a52 feat: orchestrate replayable P1 response audit
69b67c7 research: establish P1 response richness
34a8bef docs: plan damage-response P2 baselines
aff8e1c feat: bind P2 damage-response feature authority
7de540f feat: enforce fold-local P2 feature views
af8b0a9 feat: add strict nested P2 response evaluation
1154490 feat: freeze P2 response inference and gate
4e325ba feat: add deterministic P2 response packages
bdf5604 feat: orchestrate replayable P2 response audit
dc8fc57 research: establish P2 damage-response no-go
```

## Data authority

- Dataset: Hasebe et al., Mendeley Data `8scdmfdcfb`, version 3,
  DOI `10.17632/8scdmfdcfb.3`; accompanying Data in Brief DOI
  `10.1016/j.dib.2025.111509`.
- Local historical authority root: `/home/ww/paper3/cmc_damage_inference`.
  It exists but has no Git metadata, so its source files are hash-bound
  discovery authority, not Git history authority.
- Local v3 data root:
  `/home/ww/paper3/cmc_damage_inference/data/public/hasebe_cai/raw/8scdmfdcfb/v3`.
- Official inventory: 1,341 records: 446 raw CAI CSVs, 892 post-CAI JPEG
  records, and three workbooks. Raw bytes are not committed.
- Exact primary pairs: 276/276. Domain counts are
  `45,49,43,59,42,38` for `74t7kcdgkr`, `cgtnjyggtm`, `w68dtmpfyf`,
  `xcmzfsbd9t`, `yfxyg8jm46`, and `ykhs7s2dck`.
- Strict raw decoding: 445/446 overall and 276/276 primary. Non-primary
  `q8-17` is truncated and fails closed.
- Published peak reconciliation: 276/276 pass one 0.005 MPa tolerance;
  maximum absolute difference is `2.2737367544323206e-13` MPa.
- Post-CAI images: 892 official records are size/SHA bound; local image bytes
  were not downloaded and their use as model input is forbidden.
- Spatial extension: 281 raw/spatial identities, 280 with valid raw response;
  only four valid pairs are outside the primary 276. The 79 `r0/r45` raw
  identities have no exact spatial pair and are not an authorized extension.
- Strain status: `STRAIN_UNIT_UNRESOLVED`. Gauge/JIS modulus, maximum strain,
  gauge asymmetry, and gauge-derived mechanism claims remain prohibited.

Selected source authority hashes:

| Source | SHA-256 |
|---|---|
| Official inventory | `5ad443c1171e5cd34d85999822b7232c826b7a9b11c6f11bd6308c4e14be2ff7` |
| LVI workbook | `e6d98c968f57ac5748e104dc1da5e112114d25d77c03c7222ba3a0d93ac23cf1` |
| Size workbook | `72c6dc7e1e8790883dba2b4b1e3ee1259fdfbad61d22d241a866d4621834fa65` |
| Published CAI workbook | `0de44feca06294c33f2c6bc98ce6e9a8035476ff8fbef21d29f852e820cf4e2d` |
| Frozen feature bank | `f2a69f0da75e20880202d7fc4a6a92f979978406ec21f9d83e4bc8db07fb72a8` |
| Historical paired manifest | `f81002981bf2f6aed84818b48da87cd57e6336f5f3da8d78df1a58d26dd8026f` |

The complete 446-file registry is
`results/damage_to_failure_response/p0_data_audit/source_hashes.csv`.

## Response definition

Raw conversion and published peak:

```text
load_kN = Load[V] * 25
stress_MPa = load_kN * 1000 / (measured_width_mm * measured_thickness_mm)
published-peak check = max(abs(stress_MPa))
```

For each primary trace, only rows through the registered absolute-stress peak
are retained. Extension and stress offsets are the medians of the first 50
samples. Each axis is oriented toward its peak; positions outside
`[0, extension_peak]` are excluded; duplicate extension positions are reduced
by median stress. The response is linearly interpolated to 101 points at
`u = 0, 0.01, ..., 1`, with exact zero and peak anchors. There is no smoothing
or clipping.

- `extension_peak_mm`: oriented peak extension minus its baseline median, in
  mm.
- `slope_u20_u60_mpa_per_mm`: ordinary least-squares slope of zeroed stress
  against extension over inclusive normalized positions `0.2 <= u <= 0.6`, in
  MPa/mm.
- `normalized_prepeak_auc`: trapezoidal integral over `u` of zeroed stress
  divided by zeroed peak stress; dimensionless.
- Normalized curve: the 101-point pre-peak stress-extension shape with
  endpoints fixed to 0 and 1. It was audited in P1 but never modeled in P3.

No strain-derived endpoint is defined or used.

## Leakage boundary

Primary P2 inputs are outcome-free:

- F0: laminate category, ply count, measured width, measured thickness.
- F1: F0 plus 21 pre-CAI surface-profile statistics.
- F2: F0 plus projected damage area, damage height, and damage width.
- F3: F0 plus the frozen 512-dimensional full C-scan embedding, with PCA fit
  only on each source fold.
- F4: F0 plus surface statistics and the full C-scan embedding.

F5 adds true total impact energy and impactor identity only as a privileged
sensitivity. It is not a deployable primary view. True CAI strength, response
targets, raw CAI traces, post-CAI images, and target-domain preprocessing,
PCA, or hyperparameter state are forbidden inference inputs. Sentinel tests
enforce these boundaries.

## Stage decisions

| Stage | Decision | Exact basis | Evidence |
|---|---|---|---|
| P0 | `P0_GO` | 276/276 exact pairs; 276/276 peaks within 0.005 MPa | `results/damage_to_failure_response/p0_data_audit/` |
| P1 | `P1_GO` | all three registered non-strength descriptors pass richness screen | `results/damage_to_failure_response/p1_response_richness/` |
| P2 | `MACK_EXTENSION_NO_GO` | no F3/F4-vs-F2 contrast passes all three registered criteria | `results/damage_to_failure_response/p2_response_baselines/` |
| P3 | `NOT_RUN_NOT_AUTHORIZED` | P2 non-strength gate failed | no P3 output created |
| P4 | `NOT_RUN_NOT_AUTHORIZED` | P3 was not authorized | no P4 output created |
| P5 | `NOT_RUN_NOT_AUTHORIZED` | no transferable P2/P3 signal passed | no P5 output created |

### P1 response richness

All endpoints have 276/276 coverage. The fixed strength-only outer LODO
reference uses Ridge alpha `1e-6`, with no search.

| Endpoint | Range | Strength-only pooled R2 | Nonredundant domains | Gate |
|---|---:|---:|---:|---|
| Extension at peak (mm) | 1.4367192500000001 | 0.6815059345568462 | 3/6 | PASS |
| Slope u20-u60 (MPa/mm) | 555.1680866867356 | -1.2037088518000298 | 6/6 | PASS |
| Normalized pre-peak AUC | 0.34018986134233753 | 0.24905174995994184 | 6/6 | PASS |

This establishes response richness only; it does not establish predictability
from pre-CAI damage.

### P2 strict nested LODO

Protocol: six outer held-out domains; inner source-domain LODO; raw-unit
equal-domain MAE selection; Ridge alphas `0.1,1,10,100`; fold-local PCA
dimensions `8,16,32`; 100,000 PCG64 specimen-within-domain synchronized
bootstrap replicates, seed `20260830`; six familywise primary contrasts.

Primary equal-domain MAE values:

| Endpoint | F0 | F1 | F2 | F3 | F4 | F5 privileged |
|---|---:|---:|---:|---:|---:|---:|
| Extension peak (mm) | 0.2516507311 | 0.2187704944 | 0.2025732174 | 0.1893866300 | 0.2127036476 | 0.2147430430 |
| Slope u20-u60 (MPa/mm) | 35.34664724 | 48.28588007 | 35.94042925 | 32.74562519 | 45.16885654 | 49.57001233 |
| Normalized pre-peak AUC | 0.03427369811 | 0.03657101626 | 0.03333553424 | 0.03150477344 | 0.03568472007 | 0.03634925778 |

Registered F3/F4 versus F2 gates:

| Endpoint / candidate | Relative MAE improvement | Improved domains | Familywise CI for absolute improvement | Result |
|---|---:|---:|---|---|
| Extension / F3 | 6.5095% | 5/6 | [-0.00669432, 0.03512838] | FAIL |
| Extension / F4 | -5.0009% | 3/6 | [-0.03122250, 0.01282235] | FAIL |
| Slope / F3 | 8.8892% | 4/6 | [1.11846863, 5.30243137] | FAIL: below 10% |
| Slope / F4 | -25.6770% | 2/6 | [-13.01968968, -5.55846862] | FAIL |
| AUC / F3 | 5.4919% | 3/6 | [-0.000255969, 0.004045336] | FAIL |
| AUC / F4 | -7.0471% | 2/6 | [-0.005402578, 0.000695268] | FAIL |

The best near-gate result is F3 for slope: positive familywise interval and
4/6 domain direction, but only 8.8892% relative improvement. The registered
threshold is at least 10%; it was not relaxed. F4 is worse than F3 for all
three endpoints, so surface/C-scan complementarity is not established.

## Negative results and publication boundary

- Response descriptors beyond strength exist, but no preregistered spatial
  damage increment passes the P2 engineering gate.
- The evidence does not support a multi-property damage-to-response paper,
  functional response modeling, a multimodal superiority claim, failure
  phenotype inference, or uncertainty/engineering assessment.
- No neural response model was trained. P2 executed only the registered
  fold-local low-capacity Ridge fits.
- No manuscript, figure, table, claim map, or new paper directory was created.
- The existing AEI Paper 1 and all frozen evidence remain unchanged.

## Literature boundary

The final 2026-08-30 primary-source refresh is recorded in
`artifacts/damage_to_failure_response/LITERATURE_NOVELTY_LEDGER.md` (SHA-256
`83d777871674bb09ffe7c2960d8f3441fb5f7626eddf24712af47294086322a7`).
Geiselman (2011) and Mack et al. (2026) establish experimental pre-CAI C-scan
to ultimate-load/strength scalar prediction. Other close records use
simulation, microstructure-to-curve learning outside CAI, or answer-side
CAI-stage sensing. No accessible primary record found in the refresh exactly
matched experimental pre-CAI C-scan to full CAI-response learning, but this is
not proof of absence and no direct novelty claim is authorized after P2.

## Artifact identities

| Artifact | SHA-256 |
|---|---|
| P0 config | `f2129358d05e2ad540d88f5eb44bad8b00084c23eb991e5eb0ddda791899b12d` |
| P0 summary | `9d44ead975119db2181a91efbf14b74165671a9d25b7b576d90f6e104757a633` |
| P0 checksums | `80b98322ee27dae185654429c9f7a6f17343ef2763fd3cbbb88e981b52b315c5` |
| P1 config | `360d8cbfffcc65fb210e39e1c753fc50b667735a6606111143dbc26a9c6cfb16` |
| P1 summary | `37da95962395a0915f586820ab03f06d8d859856e8637d975bc302b1d555ebc7` |
| P1 checksums | `29bea7be282fec60badfaab1b4c3fa162b84fea2cddfc6556887ffe9bb728924` |
| P2 config | `c04a206be7fc6847dbcb43b1eb9252733dce173901276ee4e62dcfc5f3494d92` |
| P2 feature authority | `d112e0c23b975513952bf0476f6d269c7446d5218449741cd41ba7332907b8d5` |
| P2 OOF predictions | `a6641db63243964270a9874a1aefa525bd5608c1f477a9d1997b404fd33ed64f` |
| P2 bootstrap matrix | `0da00c47f1c2f265024e8994e8175e15d4307958112d732987152fecb92cc6f9` |
| P2 summary | `c3942d728b43a11ab270e7c9e600bc459be21471a915f16b1deb8e15a77db628` |
| P2 checksums | `796fe228c214632080303e0100a7076e56782728d61dbcab96a3fdf092f9944c` |
| P2 decision | `d64eda14e080e8d9fa11b985a0f2424cd374b91179129460f401f97f2ddb7ad4` |

## Test and replay evidence

Commands and observed results:

```text
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_damage_response_*.py
300 passed

python -m ruff check src/cmc_bbdm/damage_response tests/test_damage_response_*.py scripts/run_damage_response.py
All checks passed

python -m compileall -q src/cmc_bbdm/damage_response scripts/run_damage_response.py
PASS

PYTHONPATH=src python scripts/run_damage_response.py replay-p0 --path results/damage_to_failure_response/p0_data_audit
P0_REPLAY_OK payloads=8

PYTHONPATH=src python scripts/run_damage_response.py replay-p1 --path results/damage_to_failure_response/p1_response_richness
P1_REPLAY_OK payloads=8

PYTHONPATH=src python scripts/run_damage_response.py replay-p2 --path results/damage_to_failure_response/p2_response_baselines
P2_REPLAY_OK payloads=10

(cd results/damage_to_failure_response/p2_response_baselines && sha256sum -c CHECKSUMS.sha256)
11/11 OK, including artifact_manifest.json

git diff --check
PASS
```

P2 was executed twice, once in an isolated temporary directory and once in
the formal result path. Recursive file comparison and decision-file `cmp`
were byte-identical. An independent standard-library recomputation from the
committed OOF CSV verified 4,968 predictions, 864 inner candidate rows, 18
aggregate metrics, 108 domain metrics, nine contrasts, zero passing primary
contrasts, and `paper_route_authorized=false`.

`python -m ruff format --check` was also run over the damage-response source,
tests, and CLI. It reported 34 files would be reformatted and 11 already
formatted. No formatter mutation was applied: the repository did not use
this formatter as a gate for these files, broad mechanical churn would violate
the minimal-diff contract, and changing the result-bound source after P2 would
invalidate its implementation hashes. This is a formatting-state report, not
a Ruff lint failure.

The exact-base full repository suite was sampled before implementation and
reported `734 passed, 87 failed, 32 errors`; the failures/errors were caused by
large external data, weights, and manifests absent from the compact worktree.
They were recorded as baseline environment gaps and were not represented as
passing. The task-focused 300-test suite is the final regression authority.

## Frozen science audit

The required diff against the exact base is empty for:

```text
results/p1_full_field_oracle
results/p3_spatial_specificity
results/p5_sparse_scan
results/mvd
results/mva
results/mavis
results/mavis_science_closure
artifacts/aei_information_hierarchy
paper_aei_information_hierarchy
```

`artifacts/aei_information_hierarchy/PAPER_CANONICAL_METRICS.csv` has SHA-256
`59ce986b56961370dcee5772e199f2d897bc1bcdc04bacae4f5b772af31a5408`
at both the exact base and this branch. No old scientific output was changed,
deleted, recomputed, or relabeled.

## Changed files

The complete branch changes relative to the exact base are confined to the
new research namespace and its plans/configuration:

```text
artifacts/damage_to_failure_response/CODEX_HANDOFF_DAMAGE_TO_FAILURE_RESPONSE.md
artifacts/damage_to_failure_response/LITERATURE_NOVELTY_LEDGER.md
artifacts/damage_to_failure_response/P0_DATA_AND_AUTHORITY_AUDIT.md
artifacts/damage_to_failure_response/P0_GO_NO_GO.md
artifacts/damage_to_failure_response/P0_SOURCE_DISCOVERY.md
artifacts/damage_to_failure_response/P1_RESPONSE_RICHNESS_DECISION.md
artifacts/damage_to_failure_response/P2_DAMAGE_TO_RESPONSE_DECISION.md
docs/superpowers/plans/2026-08-30-damage-to-failure-response-p0.md
docs/superpowers/plans/2026-08-30-damage-to-failure-response-p1.md
docs/superpowers/plans/2026-08-30-damage-to-failure-response-p2.md
docs/superpowers/specs/2026-08-30-damage-to-failure-response-design.md
paper_v3/configs/damage_to_failure_response.yaml
paper_v3/configs/damage_to_failure_response_p1.yaml
paper_v3/configs/damage_to_failure_response_p2.yaml
results/damage_to_failure_response/p0_data_audit/CHECKSUMS.sha256
results/damage_to_failure_response/p0_data_audit/REPORT.md
results/damage_to_failure_response/p0_data_audit/artifact_manifest.json
results/damage_to_failure_response/p0_data_audit/pairing_manifest.csv
results/damage_to_failure_response/p0_data_audit/post_cai_image_manifest.csv
results/damage_to_failure_response/p0_data_audit/published_peak_reconciliation.csv
results/damage_to_failure_response/p0_data_audit/raw_trace_qc.csv
results/damage_to_failure_response/p0_data_audit/source_hashes.csv
results/damage_to_failure_response/p0_data_audit/strain_unit_audit.csv
results/damage_to_failure_response/p0_data_audit/summary.json
results/damage_to_failure_response/p1_response_richness/CHECKSUMS.sha256
results/damage_to_failure_response/p1_response_richness/REPORT.md
results/damage_to_failure_response/p1_response_richness/artifact_manifest.json
results/damage_to_failure_response/p1_response_richness/descriptor_qc.csv
results/damage_to_failure_response/p1_response_richness/descriptor_table.csv
results/damage_to_failure_response/p1_response_richness/domain_summary.csv
results/damage_to_failure_response/p1_response_richness/representative_pair_manifest.csv
results/damage_to_failure_response/p1_response_richness/response_curve_manifest.csv
results/damage_to_failure_response/p1_response_richness/strength_redundancy_oof.csv
results/damage_to_failure_response/p1_response_richness/summary.json
results/damage_to_failure_response/p2_response_baselines/CHECKSUMS.sha256
results/damage_to_failure_response/p2_response_baselines/REPORT.md
results/damage_to_failure_response/p2_response_baselines/aggregate_metrics.csv
results/damage_to_failure_response/p2_response_baselines/artifact_manifest.json
results/damage_to_failure_response/p2_response_baselines/bootstrap_contrasts.csv
results/damage_to_failure_response/p2_response_baselines/config.yaml
results/damage_to_failure_response/p2_response_baselines/domain_metrics.csv
results/damage_to_failure_response/p2_response_baselines/feature_authority.csv
results/damage_to_failure_response/p2_response_baselines/feature_provenance.json
results/damage_to_failure_response/p2_response_baselines/inner_selection.csv
results/damage_to_failure_response/p2_response_baselines/oof_predictions.csv
results/damage_to_failure_response/p2_response_baselines/summary.json
scripts/run_damage_response.py
src/cmc_bbdm/damage_response/__init__.py
src/cmc_bbdm/damage_response/artifacts.py
src/cmc_bbdm/damage_response/authority.py
src/cmc_bbdm/damage_response/contracts.py
src/cmc_bbdm/damage_response/feature_views.py
src/cmc_bbdm/damage_response/nested_eval.py
src/cmc_bbdm/damage_response/p1_gate.py
src/cmc_bbdm/damage_response/p2_evaluation.py
src/cmc_bbdm/damage_response/p2_features.py
src/cmc_bbdm/damage_response/p2_gate.py
src/cmc_bbdm/damage_response/p2_pipeline.py
src/cmc_bbdm/damage_response/p2_statistics.py
src/cmc_bbdm/damage_response/p2_views.py
src/cmc_bbdm/damage_response/pairing.py
src/cmc_bbdm/damage_response/pipeline.py
src/cmc_bbdm/damage_response/post_cai.py
src/cmc_bbdm/damage_response/raw_cai.py
src/cmc_bbdm/damage_response/representative_pairs.py
src/cmc_bbdm/damage_response/response_extraction.py
src/cmc_bbdm/damage_response/sources.py
src/cmc_bbdm/damage_response/targets.py
tests/test_damage_response_artifacts.py
tests/test_damage_response_authority.py
tests/test_damage_response_feature_views.py
tests/test_damage_response_gates.py
tests/test_damage_response_input_boundary.py
tests/test_damage_response_nested_eval.py
tests/test_damage_response_no_leakage.py
tests/test_damage_response_p0_pipeline.py
tests/test_damage_response_p1_gates.py
tests/test_damage_response_p1_pipeline.py
tests/test_damage_response_p2_evaluation.py
tests/test_damage_response_p2_features.py
tests/test_damage_response_p2_gates.py
tests/test_damage_response_p2_pipeline.py
tests/test_damage_response_p2_statistics.py
tests/test_damage_response_p2_views.py
tests/test_damage_response_pairing.py
tests/test_damage_response_post_cai.py
tests/test_damage_response_raw_cai.py
tests/test_damage_response_representative_pairs.py
tests/test_damage_response_response_extraction.py
tests/test_damage_response_sources.py
tests/test_damage_response_targets.py
```

## GitHub synchronization contract

Final command:

```text
git push -u origin research/aei-damage-to-failure-response
```

Post-push verification must show identical output for `git rev-parse HEAD` and
`git ls-remote origin refs/heads/research/aei-damage-to-failure-response`, plus
an empty `git status --short`. No force push, PR, merge, or old-history rewrite
is authorized or performed.
