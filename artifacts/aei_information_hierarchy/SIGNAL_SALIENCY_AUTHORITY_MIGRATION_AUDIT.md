# Signal-Saliency Authority Migration Audit

## Pre-migration authority

- Base commit: `35248f17f603e94962dc19e939162e9ef4eee5f2`
- Old canonical claim count: 39
- Old canonical SHA-256:
  `f0d2615637a6470744f275a2ac6e1c5e7aff110ca7e31cb323793c29405be4e6`
- New training or endpoint recomputation: no

Reconstruction/field-content rows requiring explicit disposition:

| Old claim | Old main role | Migration disposition |
|---|---|---|
| `U3_RECONSTRUCTION_ORACLE` | Figure 2/main comparison | retain canonical evidence; demote to supplement-only legacy technical evidence |
| `U4_ORACLE_CAI_SPECIFICITY` | Figure 2/main comparison | retain canonical evidence; demote to supplement-only legacy cross-objective evidence |
| `U4_ORACLE_IMAGE_SPECIFICITY` | Figure 2/main comparison | retain canonical evidence; demote to supplement-only legacy normalized-RGB-MSE evidence |
| `U4_LEARNED_SPECIFICITY_BOUNDARY` | supplement boundary | retain supplement-only legacy learned-mask boundary |
| `O3_REAL_MINUS_RECONSTRUCTION` | Figure 4/main control | remove from main figure/story; retain supplement-only legacy technical control |

## Frozen appearance authority

The registered score is the mean element-wise absolute RGB deviation of newly
revealed native-raster values from the specimen-specific full-image border
median, divided by 255. It reads no CAI outcome, but full candidate values make
it retrospective and non-deployable.

| Authority | SHA-256 |
|---|---|
| `src/cmc_bbdm/mva/appearance_value.py` | `a10ee487e0b33d28599166dbe200f0c1cf2cd0211dbe7b146a2f0aadc877d9cc` |
| `src/cmc_bbdm/mva/oracle_execution.py` | `472edf5a6d9ad1cf4e44eb0d1380a043d54ce8ab743bb60612c6c0aa7dc8f931` |
| `docs/MVA_A0_A3_PROTOCOL.md` | `6b1e6f91329d80196ac67cef78c596da9ca2ad9cfc83700b500809e79ab77ec0` |
| `results/mva/a2_oracle_value/REPORT.md` | `6c92fcff56c893a30c1e6c6a763a85562bb10e6044b2403b712a4070410f6b65` |
| `results/mva/a2_oracle_value/bootstrap.csv` | `a6abce9d9e3647d0668854f2772614bb5b940d5c2bf6355e12de293e487d765d` |
| `results/mva/a2_oracle_value/domain_metrics.csv` | `de31087b353f71dd42855e62a97b783c093ff0f197ac5eaf979b1f970b44127f` |
| `results/mva/a2_oracle_value/map_similarity.csv` | `e161c2043269456aa2e7321bf676ac534cc9fa01c7e0e22213781f32b2980998` |
| `results/mva/a2_oracle_value/oracle_values.parquet` | `6b289f2f6f74ac75dde47ea7cbfefcda1c49f025e74227bfb34ef269182ff963` |

Registered evidence:

- appearance-minus-mechanical AUEBC: `0.007080059382261465`
- synchronized 95% domain-bootstrap interval:
  `[0.004799356600193281, 0.00974029297002471]`
- direction: mechanical favored in `6/6` held-out domains
- 276-map mean mechanical/appearance Spearman: `0.02221200907923673`
- 276-map mean top-decile overlap: `0.20031055900621111`
- representative `c8-2` initial map: 64 unique cells per method, identical
  registered domain/grid, no duplicates or missing cells

The comparator and H3 were fixed in `docs/MVA_A0_A3_PROTOCOL.md` before formal
A1/A2 execution. The A2 result commit predates the P7 frozen outer endpoint,
so the new claims are preregistered pre-P7 evidence, not posthoc diagnostics.

## Planned old-to-new main-story mapping

| Main-story concept removed | Main-story authority added |
|---|---|
| `U3_RECONSTRUCTION_ORACLE` | `U3_CAI_VS_APPEARANCE_SALIENCY_AUEBC` |
| `U4_ORACLE_CAI_SPECIFICITY` | `U4_CAI_SALIENCY_MAP_SPEARMAN` |
| `U4_ORACLE_IMAGE_SPECIFICITY` | `U4_CAI_SALIENCY_TOP10_OVERLAP` |
| `O3_REAL_MINUS_RECONSTRUCTION` in Figure 4 | no saliency replacement; preserve history and shuffled adverse controls |

## Post-migration authority

- New canonical claim count: 42
- New canonical SHA-256:
  `59ce986b56961370dcee5772e199f2d897bc1bcdc04bacae4f5b772af31a5408`
- Claims added from existing frozen evidence:
  - `U3_CAI_VS_APPEARANCE_SALIENCY_AUEBC`
  - `U4_CAI_SALIENCY_MAP_SPEARMAN`
  - `U4_CAI_SALIENCY_TOP10_OVERLAP`
- New training, tuning, bootstrap, endpoint selection, or result recomputation:
  no

| Added claim | Source artifact | Source SHA-256 |
|---|---|---|
| `U3_CAI_VS_APPEARANCE_SALIENCY_AUEBC` | `results/mva/a2_oracle_value/bootstrap.csv` | `a6abce9d9e3647d0668854f2772614bb5b940d5c2bf6355e12de293e487d765d` |
| `U4_CAI_SALIENCY_MAP_SPEARMAN` | `results/mva/a2_oracle_value/map_similarity.csv` | `e161c2043269456aa2e7321bf676ac534cc9fa01c7e0e22213781f32b2980998` |
| `U4_CAI_SALIENCY_TOP10_OVERLAP` | `results/mva/a2_oracle_value/map_similarity.csv` | `e161c2043269456aa2e7321bf676ac534cc9fa01c7e0e22213781f32b2980998` |

Visibility migration:

| Claim | Old visibility | New visibility |
|---|---|---|
| `U3_RECONSTRUCTION_ORACLE` | `MAIN_HEADLINE` | `SUPPLEMENT_ONLY` |
| `U4_ORACLE_CAI_SPECIFICITY` | `MAIN_HEADLINE` | `SUPPLEMENT_ONLY` |
| `U4_ORACLE_IMAGE_SPECIFICITY` | `MAIN_HEADLINE` | `SUPPLEMENT_ONLY` |
| `U4_LEARNED_SPECIFICITY_BOUNDARY` | `SUPPLEMENT_ONLY` | `SUPPLEMENT_ONLY` |
| `O3_REAL_MINUS_RECONSTRUCTION` | `MAIN_SUPPORT` | `SUPPLEMENT_ONLY` |
| `U3_CAI_VS_APPEARANCE_SALIENCY_AUEBC` | not present | `MAIN_HEADLINE` |
| `U4_CAI_SALIENCY_MAP_SPEARMAN` | not present | `MAIN_SUPPORT` |
| `U4_CAI_SALIENCY_TOP10_OVERLAP` | not present | `MAIN_SUPPORT` |

The final visibility partition is 10 main headlines, 16 main-support claims,
one main system diagnostic, and 15 supplement-only claims. Main Figure 2 uses
true `appearance_oracle` values; no reconstruction result was renamed.
Main Figure 4 omits the reconstruction-derived control while its exact adverse
result remains in the supplement.

Frozen-path verification command:

```bash
git diff --name-only 35248f17f603e94962dc19e939162e9ef4eee5f2 -- \
  results/p1_full_field_oracle results/p5_sparse_scan results/mvd \
  results/mavis results/mavis_science_closure results/mva \
  artifacts/mavis artifacts/mavis_science_closure \
  artifacts/mvd_authority artifacts/mavis_authority
```

Result: empty. No frozen scientific result or authority path changed. The
appearance evidence is not posthoc: its metric and H3 contrast were registered
in `docs/MVA_A0_A3_PROTOCOL.md` before formal A1/A2 execution, and the frozen
A2 evidence predates the P7 outer endpoint.
