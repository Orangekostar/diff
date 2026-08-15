# CPB V3 Final Gate Status

## Decision

The frozen decision tree was executed without changing cohorts, endpoints,
thresholds, seed panels, or inferential units after observing results.

```text
G1 scalar observability: FAIL (frozen)
G2 scalar incremental utility: FAIL (frozen)
P1 / G2b measured full-field utility: PASS
P2 simple privileged transfer: FAIL
P3-P7: NOT AUTHORIZED
```

The defensible result is therefore asymmetric: measured full-field C-scan
information improved held-out-domain CAI prediction, but the registered
surface-only student did not recover that advantage reliably. No positive MSPD
story, SiC/SiC promotion, calibration claim, or manuscript rewrite was made.

## Phase ledger

| Phase | Required output or action | Evidence | Status |
| --- | --- | --- | --- |
| P0 | Current evidence audit | `V3_P0_CURRENT_EVIDENCE.md` | COMPLETE |
| P0 | Raw full-field C-scan audit | `V3_P0_RAW_CSCAN_AUDIT.md` | COMPLETE |
| P0 | Fourteen-item G2b feasibility response | `V3_P0_G2B_FEASIBILITY.md` | COMPLETE |
| P1 | B-morph, B-frozen, B-learned, internal-only and shuffled controls under strict nested LODO | `../experiments/P1_full_field_oracle/` | COMPLETE |
| P1 | Registered gate, tables and vector figures | `V3_P1_FULL_FIELD_ORACLE.md`, `V3_P1_G2B_GO_NOGO.md`, `tables/V3_P1_*`, `figures/V3_P1_*` | PASS |
| P2 | Teacher, equal-capacity student, point, relational and predictive transfer, scalar and mismatch controls | `../experiments/P2_student_transfer/` | COMPLETE |
| P2 | Registered transfer gate, tables and vector figures | `V3_P2_DISTILLATION_BASELINES.md`, `V3_P2_GO_NOGO.md`, `tables/V3_P2_*`, `figures/V3_P2_*` | FAIL |
| P3-P6 | Advanced method, DG, UQ and SiC/SiC transfer | No authorized result directory | STOPPED |
| P7 | Manuscript rewrite | Frozen manuscript hashes unchanged | STOPPED |

## Frozen failures

G1 failed because the registered surface-to-scalar targets did not meet the
simultaneous cross-domain observability gate: area improved 6.2135% (3/7;
lower -0.164873), height 6.5989% (6/7; lower -0.052227), and width 7.0470%
(5/7; lower -0.131310).

G2 failed because scalar internal descriptors did not add registered CAI value:
measured scalars changed MAE by -0.5760% (4/6; lower -0.005070), strict-OOF
predicted scalars improved 1.3409% (3/6; lower -0.013738), and shuffled scalars
improved 0.0168% (2/6; lower -0.005755).

These outputs were not modified by V3.

## P1 / G2b result

On 276 specimens from six datasets, the selected measured full-field
representation reduced equal-domain CAI-ratio MAE from 0.188121 to 0.128489.
The registered effect was 0.059631 (31.6985%), 5/6 held-out datasets improved,
and the 99% familywise simultaneous interval was [0.007218, 0.148421]. All seven
registered conditions and the mismatch-reproduction screen passed.

- Scientific digest: `498c17a83c687d32eb504420ed5c8687be05f01f04506eec0d89a4887efabfd1`
- Package digest: `8c13793c064d78cb17b2201b1856db490d12ec564db810fb1a3968cd3199d297`
- Production/replay manifest SHA-256: `6dc7f86060e91c1fab885685521b0c0e1a3a1290c4b4d2147c61dd1ee94e4f25`
- Production and replay: 107 files each and byte-identical, including 90 checkpoints
- Evidence manifest SHA-256: `69f79a7d2ea92ea413bf96eab0fb755d5e12afaebb4b047ef6b8bb624cf61ff0`

## P2 result

The selected MSPD student reached equal-domain MAE 0.141926675030 versus
0.148362319222 for the equal-capacity D0 surface student. The relative
improvement was only 4.3378%, below the registered 10% threshold, and its
simultaneous lower bound was -0.010214260255. Only the 4/6 improved-domain
condition passed. The scalar-teacher and global, stratified, and random-teacher
control lower bounds were also non-positive. The seven-condition AND gate
therefore failed.

- Scientific digest: `fe650ce27c42a8e6f8233374a365b14d2140057a78d5106fe74c99817b0ef24c`
- Production manifest SHA-256: `84c9dbf45d5733bdad89159b9c62626f08c55aabf9cd86bc616deeb666bac0b3`
- Replay manifest SHA-256: `ce8e9f8e77b8a1b3bf056f9b3e30d1749a045fc10587ab924f83c38d0d7469a3`
- Production/replay: 307 files each; 305 registered non-timing files scientifically identical
- Evidence manifest SHA-256: `2663bec3fa1366940851998c417292a4e0d2f92676ab271da93d8926aeb85fa3`
- `p3_invoked=false`, `p2_authorized=false`, `stop_privileged_promotion=true`

## Reproducibility inventory

P1 saves the data and split manifests, 78 representation records, 180 tabular
inner states, 270 learned inner summaries, 90 training/checkpoint records, 9936
shuffle pairs, 8556 per-specimen predictions, 102 domain metrics, 17 aggregate
metrics, four bootstrap effects, the gate, config, source/code hashes, and
runtime authority.

P2 saves 11592 split rows, 37260 inner prediction rows, 144 training and
checkpoint records, 8280 control mappings, 5796 predictions, 36 domain metrics,
five effects and intervals, seven gate conditions, config, source/code hashes,
and runtime authority.

All publication tables are copied from validated machine-readable outputs. P1
and P2 figures are deterministic vector PDFs at 166 mm width, contain no raster
image objects, and have independent source/hash manifests.

- P1 figure manifest SHA-256: `33622b39611d29b6eef6c1b1cc5592d0819fd7157c3206f60b52d5f6e50dca45`
- P2 figure manifest SHA-256: `9b9f364b31a12de51a793c09d1bd733feddabce62f3c572487b4986ae9d03111`
- P2 domain figure SHA-256: `bfd49871d2551ea7e86803c000b0b065ee086335ae852839ab36c0e3aee59b73`
- P2 gate figure SHA-256: `24e236eb7ad5c2064bf933271cdbb88d56cfcf70ddd727d6cf94e137bd35d043`

## Executed-source boundary

The exact `hasebe_cai.py` bytes registered by P1 are archived at
`../source_snapshots/hasebe_cai__sha256_23f277dfca786f60372fa8ba6102702200897e96c612fe5f8dc82e4c93e9fa5a.py`.
Both formal P1 packages pass full validation when this registered snapshot is
used for that code record. The live module differs only by explicit workbook
closure. Reading all three registered workbooks through both versions produced
identical LVI records, size records, included CAI records, and excluded records.

## Reviewer attack closure

| # | Attack | Outcome |
| ---: | --- | --- |
| 1 | Full-field C-scan utility under LODO | P1 PASS |
| 2 | Dataset-identity explanation | Equal-domain endpoint and strict outer LODO enforced |
| 3 | Shuffled C-scan reproduction | Did not reproduce the P1 gain |
| 4 | Scalar versus spatial information | Scalar failed; selected full field passed |
| 5 | MSPD versus equal-capacity surface student | P2 FAIL; no positive claim |
| 6 | Shuffled-teacher falsification | P2 control intervals crossed zero |
| 7 | Held-out-domain preprocessing leakage | Fold-local selection, PCA and learned fitting enforced |
| 8 | Hidden specimen leakage | Manifest identity and nested split authorities validated |
| 9 | Multi-domain consistency | P1 improved 5/6; P2 D0 comparison improved 4/6 but failed the AND gate |
| 10 | Main effect at least 10% | P1 31.70%; P2 only 4.34% and failed |
| 11 | Simultaneous lower bound positive | P1 positive; P2 negative and failed |
| 12 | Mechanics-sufficient versus full-latent imitation | Not promoted after P2 failure |
| 13 | Uncertainty tracks real error | P5 not authorized |
| 14 | Unsupported-case rejection | P5 not authorized |
| 15 | CFRP and SiC/SiC separation | SiC/SiC promotion not run |
| 16 | Target SiC/SiC exploratory boundary | P6 not authorized |
| 17 | Simple-model comparators | Ridge, engineered, frozen and learned baselines retained |
| 18 | Novelty beyond stacking | No novelty claim promoted after P2 failure |
| 19 | Headline claims use measured endpoints | P1/P2 use the registered CAI ratio endpoint |
| 20 | Stop after a failed gate | P3-P7 stopped without protocol manipulation |

## Frozen manuscript boundary

The four pre-existing manuscript files remain byte-identical:

- `mechanics_equivalent_damage.tex`: `a920e3118504071f47d7ebc1791444213fe1ea3572256a5c85e64758119841d5`
- `functional_recoverability.tex`: `6b37ea0f28647a1ab66356227f0789c071608bfd9452cce7780cd6b14f072b7e`
- `cpb_cfrp_observability.tex`: `6f5ab93d82cfc2a768278798ed0947ceb08a799f0e1259b66e9f384d1eedfe3b`
- `cpb_cfrp_observability_supplement.tex`: `3b58852be1c0b53b8853730eff7645a8e875e05c6e8951c00461b9d1d2013d6b`

## Verification status

The frozen tree passed the final verification suite:

- repository-wide `pytest -q -W error -x`: 2096 passed in 3259.54 s;
- the resource-lifetime regression file: 29 passed in 69.17 s;
- scoped Ruff for V3 and every touched regression surface: no findings;
- `python -m compileall -q src scripts`: exit 0;
- P1 production and replay deep package validation: PASS against the exact
  registered `hasebe_cai.py` source snapshot, with identical scientific and
  package digests;
- P2 production and replay deep package validation: PASS on the registered CUDA
  runtime;
- combined P1/P2 vector-figure package validation: PASS for both manifests and
  every registered result-source hash.

The P1 production and replay directories contain 107 files each and are
byte-identical. The P2 production and replay directories contain 307 files
each; both independently validate and carry the same registered scientific
digest. No P3-P7 result directory exists, `p3_invoked` is false, and the four
frozen manuscript hashes above remain unchanged.
