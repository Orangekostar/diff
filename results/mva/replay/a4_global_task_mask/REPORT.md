# MVA A4 Global Task-Aware Static Acquisition

## Scope

This retrospective six-domain evaluation compares three fixed source-only acquisition rankings with the registered A2 controls. Each outer-domain ranking excludes all target-domain images, CAI targets, metadata, and fitted target-domain predictors.

## Registered decisions

- Global-mask decision: `MVA_A4_GLOBAL_NO_GO`
- A5 authorization decision: `MVA_A5_AUTHORIZED`
- Relative global-to-oracle AUEBC gap: 39.766%

The two decisions are independent and use the frozen preregistered gates.

## Synchronized domain-bootstrap effects

| Contrast | Point | 95% interval | Improved domains |
|---|---:|---:|---:|
| Uniform - global mechanical | -0.000275 | [-0.001181, 0.000532] | 4/6 |
| Global reconstruction - global mechanical | -0.000316 | [-0.001297, 0.000638] | 4/6 |
| Global appearance - global mechanical | -0.000137 | [-0.001562, 0.001218] | 4/6 |
| Global mechanical - mechanical oracle | 0.007014 | [0.004716, 0.009568] | 6/6 |

Positive values favor the second method named in each contrast.

## Equal-domain budget metrics

| Method | AUEBC | B2.5 | B5 | B7.5 |
|---|---:|---:|---:|---:|
| Uniform | 0.017363 | 18.750% | 18.750% | 6.250% |
| Global appearance | 0.017502 | 18.750% | 18.750% | 9.375% |
| Global reconstruction | 0.017322 | 18.750% | 18.750% | 9.375% |
| Global mechanical | 0.017639 | 18.750% | 18.750% | 6.250% |
| Mechanical oracle | 0.010625 | 6.250% | 3.125% | 3.125% |
| Random median | 0.017619 | not reached | 18.750% | 6.250% |

## Ranking stability diagnostics

| Source objective | Mean top-10 overlap | Minimum Spearman | Mean RBO |
|---|---:|---:|---:|
| Global appearance | 0.7810 | 0.8395 | 0.8142 |
| Global reconstruction | 0.9905 | 0.9885 | 0.9614 |
| Global mechanical | 0.6381 | 0.6468 | 0.6302 |

## Interpretation boundary

The findings describe retrospective simulation under the frozen interpolation, encoder, CAI estimators, and domain roster. RGB reconstruction fidelity is reported as a mechanism diagnostic, not as evidence of mechanical validity. No adaptive policy is trained or evaluated in A4.
