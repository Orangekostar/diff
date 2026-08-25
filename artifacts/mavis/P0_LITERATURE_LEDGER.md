# MAVIS P0 Literature Ledger

Verified on 2026-08-25 from publisher or institutional primary records. This
ledger constrains novelty language; it is not a systematic review.

| Work | Primary record | Re-verified fact | MAVIS boundary |
|---|---|---|---|
| Fuentes et al. (2020), DOI `10.1016/j.ymssp.2020.106897` | [Strathprints](https://strathprints.strath.ac.uk/72351/) | Robotic NDT on a carbon-fibre wing panel sequentially selects the next physical inspection point after observations. | Do not claim first adaptive or sequential ultrasonic inspection. MAVIS targets mechanics-aware state-dependent measurement utility. |
| Cantero-Chinchilla et al. (2020), DOI `10.1016/j.ymssp.2019.106377` | [Mechanical Systems and Signal Processing](https://www.sciencedirect.com/science/article/pii/S0888327019305989) | Ultrasonic guided-wave sensor number and location are optimized under an information/cost trade-off. | Do not claim first ultrasonic VoI or cost-aware ultrasonic placement. Distinguish static system configuration from evolving specimen-specific state. |
| Memarzadeh and Pozzi (2016), DOI `10.1016/j.ress.2016.05.014` | [Reliability Engineering & System Safety](https://www.sciencedirect.com/science/article/pii/S0951832016300771) | Sequential inspection is formulated through belief updates; VoI depends on information already available. | Conditional VoI is established background, not the novelty by itself. |
| Blumberg, Slator and Alexander (ICLR 2024) | [OpenReview](https://openreview.net/forum?id=MloaGA6WwX), [UCL Discovery](https://discovery.ucl.ac.uk/id/eprint/10195823/) | Dense multi-channel acquisitions supervise task-specific sparse channel selection for a downstream task. | Task-driven acquisition is prior art. MAVIS adds evolving partial NDE state and mechanics-specific closed-loop valuation. |
| Ji et al. (2026), DOI `10.1016/j.ultras.2026.107972` | [Ultrasonics](https://www.sciencedirect.com/science/article/pii/S0041624X26000247) | Adaptive sampling of Lamb wavefields in composite laminates uses an STMAE with reconstruction as the objective. | Directly compare reconstruction-driven and mechanics-driven acquisition. |
| Mack et al. (2026), DOI `10.1016/j.aei.2026.104518` | [University of Akron](https://ideaexchange.uakron.edu/university_research/55/), [Advanced Engineering Informatics](https://www.sciencedirect.com/science/article/pii/S1474034626002107) | Experimental C-scan morphology supports impact-energy and compression-after-impact prediction. | Full-scan prediction is motivation, not MAVIS; the method must operate causally on partial measurements. |

## Allowed positioning

Existing adaptive ultrasound largely targets localization or reconstruction;
existing ultrasonic VoI includes static sensor configuration; task-driven sparse
design exists. MAVIS tests whether an evolving, specimen-specific partial
ultrasonic state can support conditional mechanical valuation and closed-loop
acquisition under exact raster cost.

## Prohibited claims

- first adaptive or sequential ultrasonic inspection;
- first ultrasonic VoI or cost-aware ultrasonic placement;
- first task-driven experimental design;
- real scanner-time reduction, industrial deployment, or external
  generalization without new evidence.
