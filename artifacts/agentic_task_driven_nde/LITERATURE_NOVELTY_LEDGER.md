# Literature and Novelty Ledger

Date searched: 2026-08-31 UTC
Mode: standard
Purpose: bound the agentic task-driven NDE claim before visual-model work
Source policy: primary proceedings, publisher, institutional, project, arXiv,
and official-code sources; MDPI sources excluded

## Safe public queries

- `autonomous ultrasonic inspection Bayesian optimisation robust outlier analysis`
- `task-driven experimental design multichannel imaging TADRED ICLR 2024`
- `multimodal region-aware localizer ultrasonic wavefield maps MoRAL`
- `ultrasound vision language action embodied scanning US-VLA`
- `DriveAgent-R1 active perception tool calling ICLR 2026`
- `ActiveVLA active perception CVPR 2026`
- `human robot collaborative visual inspection large language models`
- `surface profile C-scan low velocity impact deep learning 2026`

No private manuscript text, result value, specimen identity, or local path was
used in a public query.

## Closest-work ledger

| Work | Modality and task | Action/acquisition | Active | Task-conditioned | Hardware vs simulation | Code | Boundary for this route |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [Fuentes et al., MSSP 145 (2020), 106897](https://strathprints.strath.ac.uk/72351/) | Ultrasonic signals; detect anomalous/damaged regions | Sequential Bayesian-optimization scan locations | Yes | Damage/novelty objective | Robotic ultrasonic composite-panel data | No official code found in the checked primary record | Already establishes autonomous sequential ultrasonic inspection; forbids a first-autonomous-inspection claim |
| [Blumberg et al., ICLR 2024, TADRED](https://iclr.cc/virtual/2024/poster/18810) | MRI, hyperspectral remote sensing, physiology; downstream image-analysis tasks | Select a fixed subset of measurement channels | Yes, design selection | Yes | Retrospective multichannel data | [Apache-2.0 code](https://github.com/sbb-gh/experimental-design-multichannel) | Already establishes task-driven measurement design; it is not a spatial closed-loop ultrasound policy |
| [MoRAL, ESWA 321 (2026), 132306](https://www.sciencedirect.com/science/article/pii/S0957417426012194) | Ultrasonic guided-wave/SLDV maps plus language; damage localization/reporting | Region localization, not sequential measurement control | No | Damage detection/reporting | Experimental/simulation-augmented wavefield maps | No verified official code found | Already establishes multimodal-LLM ultrasonic localization/reporting; forbids a first-MLLM-ultrasonic-NDT claim |
| [US-VLA, arXiv:2608.16074](https://arxiv.org/abs/2608.16074) | RGB cameras, ultrasound feedback, language, robot state; standard-plane scanning | Continuous sequential probe controls | Yes | Clinical target-plane language | Real expert trajectories; repository omits hardware deployment loop | [Apache-2.0 code](https://github.com/VMVLab/US-VLA) | Already establishes ultrasound VLA and language-conditioned ultrasound action; medical environment is not reusable here |
| [DriveAgent-R1, ICLR 2026](https://openreview.net/forum?id=r2g8TV4nJy) | Driving images/text; high-level behavior planning | Visual tools followed by meta-actions | Yes | Navigation/behavior context | Offline driving benchmarks | [Training code, no identified license](https://github.com/wczheng/DriveAgent-R1) | Already establishes VLM tool-based active perception; only the typed-tool boundary is conceptually relevant |
| [ActiveVLA, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_ActiveVLA_Injecting_Active_Perception_into_Vision-Language-Action_Models_for_Precise_3D_CVPR_2026_paper.html) | Multi-view RGB/3D and language; precise manipulation | Active viewpoint selection and 3D zoom | Yes | Manipulation instruction | Benchmarks and real-robot evaluation | [Release still pending; no identified license](https://github.com/ZhenyangLiu/ActiveVLA-Injecting-Active-Perception-into-VLA) | Already establishes coarse-to-fine VLA active perception; no code or claim import is authorized |
| [Tasneem and Pieters, RCIM 98 (2026), 103154](https://doi.org/10.1016/j.rcim.2025.103154) | Depth camera, speech/text, local LLM; industrial visual inspection | LLM path plan with deterministic ROS execution | Yes at planning level | Natural-language inspection request | Simulation and real industrial use cases | [Apache-2.0 code](https://github.com/CuriousLad1000/RoboSpection) | Already establishes high-level LLM inspection planning with low-level control separation |
| [Hasebe et al., Data in Brief 43 (2022), 108462](https://pmc.ncbi.nlm.nih.gov/articles/PMC9294053/) | Impacted-surface RGB/profile and ultrasonic C-scan | None | No | Impact/damage description | Paired laboratory CFRP specimens | Seven Mendeley datasets, CC BY article | Supplies the present raw authority but does not publish a surface-to-C-scan coordinate transform |
| [Deep learning framework for damage prediction in low-velocity impact, ESWA (2026), 131810](https://www.sciencedirect.com/science/article/pii/S0957417426007232) | Hasebe surface deformation/profile to C-scan image and impactor classification | None | No | Internal-damage prediction | Retrospective public CFRP data | Unknown from the accessible primary page | Direct surface-to-C-scan prediction is already covered; the retained question is task-value observability under strict acquisition gates |

## Evidence-backed differentiation

Established work covers autonomous ultrasonic point selection, task-driven
channel design, multimodal ultrasonic damage localization, ultrasound VLA,
VLM active perception, and LLM-guided robotic inspection. The proposed route
therefore cannot claim any of those concepts as first.

The narrower empirical question remains open from the checked sources: whether
a released pre-acquisition impacted-surface image improves held-out-domain
ranking of a frozen downstream CAI value-of-measurement target, under the exact
same legal 8x8 replay actions and normalized native-raster cost. This is a
question to test, not an authorized novelty claim. A missing cross-instrument
registration or a negative P1 result removes even that claim route.

## Source unknowns

- The checked Hasebe article documents 80 x 80 mm specimens, a VR-5000 surface
  system, and a 75 x 75 mm C-scan, but no common coordinate frame, orientation,
  offset, or specimen-level transform.
- MoRAL and the 2026 surface-to-C-scan paper had no verified official code link
  on the accessible primary pages.
- DriveAgent-R1 and ActiveVLA had no identified top-level license at search
  time; their public availability does not authorize copying.
- Search coverage cannot prove absence of unpublished or unindexed work.

Novelty status: SEARCHED_NOT_ASSUMED

Direct novelty claim authorized: NO

Empirical route retained: YES, subject to P0 and P1 gates.
