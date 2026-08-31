# Literature and Novelty Ledger

Verified against primary publisher, proceedings, institutional, or author-code
sources on 2026-08-31.

| Work | Verified source | Prior capability relevant to G0 | Consequence |
|---|---|---|---|
| Fuentes et al. (2020), *Autonomous ultrasonic inspection using Bayesian optimisation and robust outlier analysis*, MSSP 145:106897, DOI 10.1016/j.ymssp.2020.106897 | [Elsevier article](https://www.sciencedirect.com/science/article/pii/S0888327020302831) | GP/Bayesian-optimization next-location selection for autonomous ultrasonic NDT | Never claim first autonomous or adaptive ultrasonic inspection. Add a Fuentes-like GP/BO comparator in G2, not G0. |
| Hollinger et al. (2013), *Active planning for underwater inspection and the benefit of adaptivity*, IJRR 32(1), DOI 10.1177/0278364912467485 | [SAGE article](https://journals.sagepub.com/doi/10.1177/0278364912467485) | Adaptive inspection and information/cost planning | Adaptivity and inspection planning are established concepts. |
| Kurniawati (2022), *Partially Observable Markov Decision Processes and Robotics*, ARCRAS 5:253-277, DOI 10.1146/annurev-control-042920-092451 | [Annual Reviews](https://www.annualreviews.org/content/journals/10.1146/annurev-control-042920-092451) | Robotics planning under partial observability | Supports the belief/action framing, not a novelty claim. |
| Blumberg, Slator, and Alexander (2024), TADRED, ICLR 2024 | [ICLR proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/aec2dfc4f5e19acf05c15587c889dbc4-Paper-Conference.pdf), [author code](https://github.com/sbb-gh/tadred) | Downstream-task-driven experimental design and feature/channel selection | Task-driven acquisition is established outside this NDE setting. |
| Hu et al. (2025), *Real-World Reinforcement Learning of Active Perception Behaviors*, NeurIPS 2025 / AAWR | [NeurIPS proceedings](https://papers.neurips.cc/paper_files/paper/2025/hash/aa1b1a959c80086cba61d0fd66de412f-Abstract-Conference.html) | Privileged sensing can teach partially observable active-perception behavior | Conceptual precedent for teacher-only full C-scan/CAI privilege. G0 does not run AAWR/RL. |
| DriveAgent-R1 (2026), ICLR 2026 | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cbb776e737ec3ea5925887f8740c68b4-Abstract-Conference.html) | Active perception, tools, and high-level planning under uncertainty | VLM/tool orchestration is not the G0 scientific core. |
| ActiveVLA (2026), CVPR 2026 | [CVF proceedings](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_ActiveVLA_Injecting_Active_Perception_into_Vision-Language-Action_Models_for_Precise_3D_CVPR_2026_paper.html) | Critical-region localization and coarse-to-fine active perception | Coarse-to-fine active perception is established; G0 tests whether NDE supports it. |
| Tasneem and Pieters (2026), *Human-robot collaborative visual inspection with Large Language Models*, RCIM 98:103154, DOI 10.1016/j.rcim.2025.103154 | [Tampere research portal](https://researchportal.tuni.fi/fi/publications/humanrobot-collaborative-visual-inspection-with-large-language-mo/) | High-level LLM planning with deterministic low-level execution | LLM planning is deferred until deterministic inspection evidence exists. |
| Huang and Zou (2026), *Act or ask*, AEI 72:104454, DOI 10.1016/j.aei.2026.104454 | [Elsevier article](https://www.sciencedirect.com/science/article/abs/pii/S1474034626001461) | VLM action selection with confidence-guided deferral | Confidence/deferral is prior art; G0 STOP is an evidence sufficiency audit. |
| US-VLA (2026) | [Author repository](https://github.com/VMVLab/US-VLA) | Author-reported semantic goal + RGB + live ultrasound feedback to sequential actions | ACM MM 2026 acceptance is author-reported and was not independently present in ACM proceedings at audit time. Do not use it for priority claims. The repository reports no physical deployment loop. |

## Candidate contribution boundary

Only if the registered gates pass, later work may claim evidence for the combined
setting of zero-measurement initiation, structured belief revision,
task-conditioned acquisition, adaptive stopping, and ultrasonic NDE. G0 makes
no “first” claim. A negative oracle result is a route-stopping scientific result,
not a result to rescue by changing tasks or gates.
