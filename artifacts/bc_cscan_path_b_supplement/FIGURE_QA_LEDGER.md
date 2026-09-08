# Figure QA Ledger

| Issue | Artifact | Severity | Fix | Status |
|---|---|---|---|---|
| Nonblank raster and vector export | All figures | High | Checked five PNG/PDF pairs; each PDF has one page and each PNG has nonzero grayscale variance | Passed |
| Labels, legends, error bars, and panel overlap | Figures 1--5 | High | Inspected rendered PNGs at final aspect ratios | Passed |
| Metric direction and numeric source | Figures 1--4 | High | Cross-checked against `per_episode_metrics.csv`, `paired_effects.csv`, `per_domain_effects.csv`, `ablation_effects.csv`, and `risk_coverage.csv` | Passed |
| Representative selection and STOP prefixes | Figure 5 | High | Six hash-selected domains; 24 actual true-break episodes match cached prefixes, with zero post-STOP world steps | Passed |
| Color/grayscale identity | All figures | Medium | Okabe-Ito/neutral colors are backed by markers, line styles, and hatches | Passed |
