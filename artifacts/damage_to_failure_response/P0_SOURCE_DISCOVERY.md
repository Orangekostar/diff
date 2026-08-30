# P0 Source Discovery and Authority Audit

Audit date: 2026-08-30  
Scope: source discovery and immutable facts before any model or training code.  
Operator prompt SHA-256:
`bbdc4e26e70dcf22cc6a34186064b3924804c40ac3e78cd113cad72f9cffe44d`.

## Repository authority

- Exact Git base: `3951f71f28b6efdf8c74eea0fe274b2a78a9cd57`.
- Base remote branch: `origin/aei-signal-saliency-reframe`.
- Research branch: `research/aei-damage-to-failure-response`.
- Frozen paired-feature bank:
  `results/aei_selective_invariance/a2_paired_features/paired_features.npz`.
- Feature-bank SHA-256:
  `f2a69f0da75e20880202d7fc4a6a92f979978406ec21f9d83e4bc8db07fb72a8`.
- The historical full tree, labeled `local:historical_full_tree`, exists but has
  no `.git` authority. It therefore has no verifiable HEAD and cannot be
  imported wholesale as source authority.

The compact base lacks these P0-relevant historical candidates:

| Historical candidate | SHA-256 in `local:historical_full_tree` |
|---|---|
| `src/cmc_bbdm/cpb_v3/data.py` | `4d294d3b047fc32540adb499402b7171700bfec152dac71263d10f92808b8e03` |
| `src/cmc_bbdm/cpb_v3/config.py` | `be07e8313127bb76b31b2b498a131fda151268e7c0d86a0ced36d65f4676f2e3` |
| `src/cmc_bbdm/cpb_v3/evaluation.py` | `2165376d1ee6459329da748f725b0a12d4b3d94d3138b811448b2649190688c9` |
| `src/cmc_bbdm/cpb_v3/morphology.py` | `953f37282149b4fd616dfafb9c9babd3afc600ead252ba99c5f4b85d83b91542` |
| `src/cmc_bbdm/cpb_redesign/mechanical_utility.py` | `feb57997c4e9e824a50c9ea7dd8f1922b2fb68c4489e528a2e4d7429159e9e73` |
| `src/cmc_bbdm/hasebe_cai.py` | `e788f77e01fd534b14168f06d3177ba81c24abc3fe53ca8ab7feebeb0d0dbf07` |
| `src/cmc_bbdm/mendeley.py` | `c7bdbe66f6337c51f06d674f5de860a4749d42830b3814f0d6b892a6a98bf44e` |
| `scripts/run_cpb_v3_p1.py` | `1def1cd19a1bab162a9291bba621cd2ef9feb9ca0e91f7b1e461a3ef1309c5a7` |

These hashes are discovery evidence only. New focused code must be reviewed and
tested in the current Git repository rather than treating the historical files
as versioned authority.

## Official dataset authority

- Dataset: Hasebe CAI data, Mendeley Data DOI
  [`10.17632/8scdmfdcfb.3`](https://doi.org/10.17632/8scdmfdcfb.3).
- Article: Hasebe et al., Data in Brief 60, 111509,
  [`10.1016/j.dib.2025.111509`](https://doi.org/10.1016/j.dib.2025.111509).
- External root label: `local:hasebe_v3_root`.
- Official raw-trace folder ID:
  `9559b236-6f4b-4389-baac-b8b42335cea2`.
- Official post-CAI-image folder ID:
  `7564c3ad-daa6-4762-9071-a2773bd5b2f9`.

The official API inventory and local verification produced:

| Source | Files | Bytes | Identity result | Sorted official-record digest |
|---|---:|---:|---|---|
| Raw CAI traces | 446 CSV | 101,241,312 | 446/446 size and SHA-256 matches; no extras | `1528a0faa428bc7aa0d1fa415a2caba8a91d792c9a7c08e0a60fb4056fbd7210` |
| Post-CAI images | 892 JPEG | 1,855,508,776 | Official inventory: exactly front/back for all 446 trace identities | `ad977a4bc4673ce44305e28af8e667f216900d49923d2733c79e39cbaf4dafdc` |

The sorted-record digest is SHA-256 over canonical JSON records containing
official file ID, filename, status, size, and official SHA-256. Raw family
counts are `c8=61`, `c16=58`, `c24=78`, `q8=52`, `q16=58`, `q24=60`,
`r0=39`, and `r45=40`. All 446 filename specimen prefixes are unique.

Locally available official workbook hashes are:

| Workbook role | SHA-256 |
|---|---|
| Low-velocity-impact conditions and observations | `e6d98c968f57ac5748e104dc1da5e112114d25d77c03c7222ba3a0d93ac23cf1` |
| Measured specimen dimensions | `72c6dc7e1e8790883dba2b4b1e3ee1259fdfbad61d22d241a866d4621834fa65` |
| Published CAI strength | `0de44feca06294c33f2c6bc98ce6e9a8035476ff8fbef21d29f852e820cf4e2d` |

## Identity and cohort facts

The frozen feature bank and historical authoritative roster agree on 276 exact
primary `(specimen_id, domain_id)` keys. Exact filename identity finds one and
only one raw trace for every key, with no order-based or fuzzy match:

| Domain | Exact primary pairs |
|---|---:|
| `74t7kcdgkr` | 45 |
| `cgtnjyggtm` | 49 |
| `w68dtmpfyf` | 43 |
| `xcmzfsbd9t` | 59 |
| `yfxyg8jm46` | 42 |
| `ykhs7s2dck` | 38 |
| **Total** | **276** |

The official outcome workbook contains 448 rows. `c8-1` and `c8-10` have no raw
trace and a nonnumeric outcome; the remaining 446 trace identities have numeric
published CAI strength. Of the 170 trace identities outside the primary 276,
144 are impacted specimens with numeric projected-delamination area and dent
depth, while 26 are intact. The hash-bound historical spatial manifest
intersects 281 raw-file identities: all 276 primary specimens plus five
additional identities. One additional identity, `q8-17`, has a truncated raw
CSV, leaving four additional spatial identities with a valid decoded response.
The remaining 139 impacted identities have scalar observations but no
established spatial pair. No `r0/r45` identity has an exact spatial pair in the
available manifest, so those families are not an authorized extension.

## Raw schema and unresolved semantics

Direct CP932 decoding of the official CSVs records 50 Hz sampling and the source
headers `Extension[V]`, `Load[V]`, `Strain-FL`, `Strain-FR`, `Strain-BL`, and
`Strain-BR`. The source calibration is 1 mm/V for extension and 25 kN/V for
load. The four strain columns declare `με` in the CSV unit row. The Data in
Brief prose labels those columns `[µm]`, and the sign/averaging/preload
convention is not independently resolved. The conservative audit state is
therefore `STRAIN_UNIT_UNRESOLVED`. JIS modulus and maximum-strain endpoints
remain unauthorized; load/extension auditing may proceed.

Strict decoding succeeds for 445/446 official raw files and for all 276 primary
files. The non-primary `q8-17` file declares 15,711 rows and 314.22 s but
contains 3,840 rows ending at 76.78 s. The primary `c24-12` file has internal
title `c24-112`; canonical `c24-12` is independently supported by the official
filename, dataset version, file SHA-256, and all three workbooks, and the title
conflict remains explicit in `raw_trace_qc.csv`.

Post-CAI images are identity/integrity audit outputs only. They are not model
inputs: `POST_CAI_IMAGE_INPUT_FORBIDDEN = true`.

## Ten-question pre-model audit

| # | Required question | Source-backed answer at discovery freeze | Status |
|---:|---|---|---|
| 1 | Exact base SHA? | `3951f71f28b6efdf8c74eea0fe274b2a78a9cd57` | CLOSED |
| 2 | Which compact authority modules/raw inputs were missing? | Eight P0-relevant historical source candidates above; all 446 raw CAI CSVs were initially absent from the historical external folder and are now official-SHA verified outside Git. | CLOSED |
| 3 | Does the full authority repository exist, and what is its HEAD? | A historical full tree exists, but it is not a Git repository; HEAD is `NO_GIT_AUTHORITY`. | CLOSED |
| 4 | What is the actual raw-folder structure? | One official v3 folder, 446 unique specimen CSVs across eight named families, CP932 metadata/data rows, six registered channels; 445 decode strictly and the one truncated non-primary file is retained in QC. | CLOSED |
| 5 | How many current 276 have exact raw traces? | 276/276, with exact per-domain counts `45/49/43/59/42/38`; package replay and checksums pass. | CLOSED |
| 6 | Which of 446 additionally exact-pair to pre-CAI C-scan? | The existing hash-bound spatial manifest intersects 281 raw identities: primary 276 plus five additional file identities. Because `q8-17` is truncated, four additional identities have valid decoded responses. Another 139 impacted identities have scalar-only observations; `r0/r45` has no established spatial pair. | CLOSED |
| 7 | Are strain unit and sign unambiguous? | No. CSV unit metadata says `με`, article prose conflicts, and sign semantics remain unresolved. | CLOSED_AS_STRAIN_UNIT_UNRESOLVED |
| 8 | Does raw peak reproduce published CAI strength? | Yes for 276/276 primary specimens under one formula and 0.005 MPa global tolerance; maximum absolute error is `2.2737367544323206e-13` MPa. | CLOSED |
| 9 | Which response endpoints may legally enter P1? | Stress-extension peak extension, a uniformly defined pre-peak slope/integral, and normalized pre-peak stress-extension shape. All strain/gauge-derived endpoints remain disabled. | CLOSED |
| 10 | What work is closest and how is this route different? | Mack (pre-CAI C-scan, scalar), Lu (pre-CAI CT to calibrated simulated full response), Du (CAI-stage AE), and composite full-curve work outside experimental post-impact CAI are closest. | CLOSED_SEARCHED_NOT_ASSUMED |

All ten questions are now source-backed. P0 status is `P0_GO`. P1 execution
still requires a separate preregistered implementation plan and failing tests
before any estimator code or training is added.

## Baseline execution state

- `python -m ruff check src tests scripts`: passed at the exact base.
- Full base test run: 734 passed, 87 failed, 32 errors in 372.92 s.
- The failures/errors are bound to unavailable large external datasets,
  manifests, weights, and adjacent prompt artifacts in the compact isolated
  worktree. They are recorded as baseline environment failures and are not
  attributed to damage-response changes.
- New training performed: **NO**.
