# Closest-work comparison and reading boundary

| Work | Task endpoint | Information available to decision | Feedback and learning | Cost / termination | Verified evidence and distinction |
|---|---|---|---|---|---|
| Mack et al. (2026) | CAI strength / impact energy regression | C-scan damage image input | ResNet18 regression described; acquisition feedback unverified | Acquisition-cost protocol unverified | Institutional abstract only; predictor input established, no module-level absence claim |
| Fuentes et al. (2020) | Damage indication / component damage probability | Measurements collected at selected locations | Sequential novelty-index field and Bayesian optimization | Reduces observation count; exact termination unverified | Institutional abstract only; differs in endpoint, not a reproduced baseline |
| Shim et al. (2018) | Cost-sensitive classification | Acquired feature set | Joint classifier/acquirer with set encoding | Acquire or stop/predict; specified feature costs | PDF pp1–4, Section 3; differs from fixed-cap frozen-regressor evaluation |
| Janisch et al. (2019) | Costly-feature classification | Currently acquired features | Deep RL; optional external classifier described | Feature requests or classification actions | Author PDF pp1–3, Problem definition; no claim of CAI evaluation |
| Covert et al. (2023) | Dynamic predictive feature selection | Observed feature subset | Greedy CMI / amortized optimization | Fixed feature count, uniform costs in formulation | PDF Sec2–4; squared-loss regression conditions do not establish this loss's optimality |
| This study | CAI trajectory and endpoint error | Surface, numerical prior, acquired internal descriptors | Frozen regressor with learned acquisition actor | Native-pixel cap; whole-cell affordability termination | Local model/policy/metric code and saved selected-VALID trajectories |

No source is marked FULLTEXT_READ when only its abstract or selected sections were read. Three algorithmic neighbours were read at the sections supporting the comparisons; unavailable engineering-neighbour details remain unverified. No cross-study accuracy leaderboard is constructed.
