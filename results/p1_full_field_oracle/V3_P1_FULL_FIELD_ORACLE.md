# V3 P1 Full-Field Oracle

## Decision

The preregistered P1/G2b experiment **PASSED**. On the frozen 276-specimen, six-dataset cohort, the selected measured full-field representation reduced equal-domain CAI-ratio MAE from 0.188121 for metadata plus surface information to 0.128489. The reference-minus-candidate improvement was 0.059631 (31.6985%), with 5/6 held-out datasets improved and a familywise simultaneous interval [0.007218, 0.148421].

This supports a bounded claim: under the registered cohort and strict nested leave-one-dataset-out protocol, measured spatial C-scan information carried cross-dataset CAI signal that the three scalar descriptors did not preserve. It does not establish causal sufficiency, universal transfer, specimen-specific internal reconstruction, or surface observability of the full internal field.

## Frozen context

- G1 scalar observability failed: area 6.2135% (3/7, lower -0.164873), height 6.5989% (6/7, lower -0.052227), and width 7.0470% (5/7, lower -0.131310).
- G2 scalar utility failed: measured scalar descriptors changed MAE by -0.5760% (4/6, lower -0.005070), strict-OOF scalar predictions improved 1.3409% (3/6, lower -0.013738), and shuffled scalars improved 0.0168% (2/6, lower -0.005755).
- P1 tested the different hypothesis that scalarization had removed spatial information relevant to CAI.

## Equal-domain MAE

| Method | Equal-domain MAE |
| --- | --- |
| A_surface | 0.188121 |
| B_scalar | 0.189204 |
| B_morph | 0.203890 |
| B_frozen | 0.128489 |
| B_combined | 0.149047 |
| B_field_selected | 0.128489 |
| B_learned | 0.154897 |
| B_learned_zero_field | 0.154791 |
| I_morph | 0.144009 |
| I_frozen | 0.089636 |
| I_combined | 0.126334 |
| I_field_selected | 0.089636 |
| I_learned | 0.122905 |
| global_shuffle | 0.190790 |
| stratified_shuffle | 0.191482 |
| global_shuffle_cnn | 0.160137 |
| stratified_shuffle_cnn | 0.160818 |

## Confirmatory domain MAE

| Dataset | A_surface | B_scalar | B_field_selected | global_shuffle | stratified_shuffle |
| --- | --- | --- | --- | --- | --- |
| 74t7kcdgkr | 0.140261 | 0.139438 | 0.058737 | 0.142559 | 0.141465 |
| cgtnjyggtm | 0.122152 | 0.121567 | 0.124699 | 0.135539 | 0.130985 |
| w68dtmpfyf | 0.124114 | 0.126236 | 0.105902 | 0.121796 | 0.125797 |
| xcmzfsbd9t | 0.100653 | 0.099153 | 0.090589 | 0.123094 | 0.118416 |
| yfxyg8jm46 | 0.163381 | 0.162458 | 0.127022 | 0.156068 | 0.164836 |
| ykhs7s2dck | 0.478165 | 0.486375 | 0.263986 | 0.465683 | 0.467395 |

## Fold-local field selection

| Outer dataset | Selected field | PCA dimension | Inner MAE |
| --- | --- | --- | --- |
| 74t7kcdgkr | B_frozen | 16 | 0.173297 |
| cgtnjyggtm | B_frozen | 32 | 0.132630 |
| w68dtmpfyf | B_frozen | 8 | 0.114823 |
| xcmzfsbd9t | B_frozen | 32 | 0.133927 |
| yfxyg8jm46 | B_frozen | 8 | 0.109759 |
| ykhs7s2dck | B_frozen | 8 | 0.103726 |

`B_frozen` was selected independently in every outer fold; PCA dimension remained fold-local. This selection result was obtained before constructing mismatch controls.

## Registered effects

| Effect | Delta | Relative improvement | Improved domains | Simultaneous interval |
| --- | --- | --- | --- | --- |
| A_vs_field | 0.059631 | 31.6985% | 5/6 | [0.007218, 0.148421] |
| scalar_vs_field | 0.060715 | 32.0897% | 5/6 | [0.006639, 0.153643] |
| global_shuffle_vs_field | 0.062300 | 32.6540% | 6/6 | [0.016401, 0.141955] |
| stratified_shuffle_vs_field | 0.062993 | 32.8976% | 6/6 | [0.015734, 0.143069] |

## Negative controls

The noninferential reproduction screen passed because neither tabular mismatch control reproduced the A-to-field gain: global shuffle was -1.4188% versus A with 3/6 improved, and stratified shuffle was -1.7870% with 1/6 improved. Adverse and null outcomes are retained in the exported tables.

## Reproducibility

- Production and independent replay trees are byte-identical.
- Result manifest SHA-256: `6dc7f86060e91c1fab885685521b0c0e1a3a1290c4b4d2147c61dd1ee94e4f25`.
- Run summary SHA-256: `ff7d859bc8e192c340f80bde574b110c0d5718185ec1f8ada6ee6569b4c041fe`.
- Package digest: `8c13793c064d78cb17b2201b1856db490d12ec564db810fb1a3968cd3199d297`.
- Scientific digest: `498c17a83c687d32eb504420ed5c8687be05f01f04506eec0d89a4887efabfd1`.
- Config SHA-256: `42b13f714a0ebd11d820e879808825ca5a7f326d01baacc4e088f67102ff2f12`.
- The 90 production and replay checkpoint files are byte-identical.

All numeric statements above were independently recomputed from `predictions.csv`; serialized domain metrics, effects, bootstrap endpoints, screen status, and the seven-condition AND decision were then checked against that recomputation.
