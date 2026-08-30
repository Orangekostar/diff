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
| `src/cmc_bbdm/cpb_v3/data.py` | `4d294d9abceee655f2ea5fbf2f9058658ad822ad8e87f14178a3e57671e9dbf7` |
| `src/cmc_bbdm/cpb_v3/config.py` | `be07e84bc95622833ea86149946950209450870247fd868772177950385137a5d` |
| `src/cmc_bbdm/cpb_v3/evaluation.py` | `216537469d426d0025859c624cf38d4b05d7684fcc4d29fc5485190a6d33c4f5` |
| `src/cmc_bbdm/cpb_v3/morphology.py` | `953f37b602127b1f73e65f7f2cde1530bc10bf1829055b3be8bd3e619c9acc06` |
| `src/cmc_bbdm/cpb_redesign/mechanical_utility.py` | `feb57922560238823340430917f9c8022edbf601074e8030cd0b4950dc0f4d13` |
| `src/cmc_bbdm/hasebe_cai.py` | `e788f76db5c23d4558b4d94cd01fd32ed95507281f66b962771349864487e958` |
| `src/cmc_bbdm/mendeley.py` | `c7bdbe7a45b050feae8246352fac3c9922d4b42958596c032250815661cc12a8` |
| `scripts/run_cpb_v3_p1.py` | `1def1ce609394d16eb7b39ff55105e2a76757bdb0aa3f871f0a7a3d7bbf2444c` |

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
depth, while 26 are intact. This proves additional exact raw-trace plus scalar
pre-CAI-observation pairs. It does **not** prove exact full C-scan or surface-
profile availability for those 170 identities. In particular, `r0/r45` remains
a candidate extension, not an authorized training cohort or an "untouched"
test set.

## Raw schema and unresolved semantics

Direct CP932 decoding of an official CSV records 50 Hz sampling and the source
headers `Extension[V]`, `Load[V]`, `Strain-FL`, `Strain-FR`, `Strain-BL`, and
`Strain-BR`. The source calibration is 1 mm/V for extension and 25 kN/V for
load. The four strain columns declare `με` in the CSV unit row. The Data in
Brief prose labels those columns `[µm]`, and the sign/averaging/preload
convention is not independently resolved. The conservative audit state is
therefore `STRAIN_UNIT_UNRESOLVED`. JIS modulus and maximum-strain endpoints
remain unauthorized; load/extension auditing may proceed.

Post-CAI images are identity/integrity audit outputs only. They are not model
inputs: `POST_CAI_IMAGE_INPUT_FORBIDDEN = true`.

## Ten-question pre-model audit

| # | Required question | Source-backed answer at discovery freeze | Status |
|---:|---|---|---|
| 1 | Exact base SHA? | `3951f71f28b6efdf8c74eea0fe274b2a78a9cd57` | CLOSED |
| 2 | Which compact authority modules/raw inputs were missing? | Eight P0-relevant historical source candidates above; all 446 raw CAI CSVs were initially absent from the historical external folder and are now official-SHA verified outside Git. | CLOSED |
| 3 | Does the full authority repository exist, and what is its HEAD? | A historical full tree exists, but it is not a Git repository; HEAD is `NO_GIT_AUTHORITY`. | CLOSED |
| 4 | What is the actual raw-folder structure? | One official v3 folder, 446 unique specimen CSVs across eight named families, CP932 metadata/data rows, six registered channels. | CLOSED_PENDING_ALL_FILE_SCHEMA_QC |
| 5 | How many current 276 have exact raw traces? | 276/276, with exact per-domain counts `45/49/43/59/42/38`. | CLOSED_PENDING_MANIFEST_REPLAY |
| 6 | Which of 446 additionally exact-pair to pre-CAI C-scan? | 144 additional impacted identities have exact raw trace plus scalar pre-CAI damage observations; exact full C-scan/profile pairing is not yet established. | OPEN_REQUIRES_SPATIAL_SOURCE_AUDIT |
| 7 | Are strain unit and sign unambiguous? | No. CSV unit metadata says `με`, article prose conflicts, and sign semantics remain unresolved. | CLOSED_AS_STRAIN_UNIT_UNRESOLVED |
| 8 | Does raw peak reproduce published CAI strength? | Formula and source calibration are fixed; real 446-row global-tolerance reconciliation has not yet run. | OPEN_REQUIRES_P0_RECONCILIATION |
| 9 | Which response endpoints may legally enter P1? | Peak stress and extension-derived quantities require P0 reconciliation/QC; all strain-dependent endpoints are disabled. Final legal endpoint list is not yet authorized. | OPEN_REQUIRES_P0_GATE |
| 10 | What work is closest and how is this route different? | Mack (pre-CAI C-scan, scalar), Lu (pre-CAI CT to calibrated simulated full response), Du (CAI-stage AE), and composite full-curve work outside experimental post-impact CAI are closest. | CLOSED_SEARCHED_NOT_ASSUMED |

No model/training code is authorized while questions 6, 8, and 9 remain open.
P0 authority, parsing, reconciliation, and replay code is audit infrastructure,
not model code, and is permitted solely to close these questions.

## Baseline execution state

- `python -m ruff check src tests scripts`: passed at the exact base.
- Full base test run: 734 passed, 87 failed, 32 errors in 372.92 s.
- The failures/errors are bound to unavailable large external datasets,
  manifests, weights, and adjacent prompt artifacts in the compact isolated
  worktree. They are recorded as baseline environment failures and are not
  attributed to damage-response changes.
- New training performed: **NO**.
