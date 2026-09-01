# G1 Literature and Method Ledger

Frozen on 2026-09-01 before G1 production training. Sources are used only as
conceptual precedents. No external implementation is copied.

| Work | Primary source | G1 boundary | Code/license finding |
|---|---|---|---|
| Hu et al., *Real-World Reinforcement Learning of Active Perception Behaviors*, NeurIPS 2025 (AAWR) | [Proceedings](https://papers.neurips.cc/paper_files/paper/2025/hash/aa1b1a959c80086cba61d0fd66de412f-Abstract-Conference.html), DOI `10.52202/085713-3910`; [official repository](https://github.com/penn-pal-lab/aawr) | Supports a partially observed actor trained with a privileged critic. G1 AAWR is conditional, source-only, and may not be enabled after target results. | GitHub reported no declared repository license and the license endpoint returned 404 on 2026-09-01. Code copying is forbidden. |
| Ross, Gordon, and Bagnell, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*, AISTATS 2011 (DAgger) | [PMLR 15:627-635](https://proceedings.mlr.press/v15/ross11a.html) | Supports querying a teacher on policy-induced states. G1 relabeling is limited to causal outer-source worlds and deterministic state caps. | Method is implemented independently from the paper; no source is copied. |
| Peng et al., *Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning* | [arXiv:1910.00177](https://arxiv.org/abs/1910.00177); [official repository](https://github.com/xbpeng/awr) | Supports exponentiated advantage-weighted policy extraction. G1 uses this only inside the conditionally authorized privileged-critic stage. | Official repository declares MIT; G1 still implements from the paper without copying. |
| *DriveAgent-R1: Advancing VLM-based Autonomous Driving with Active Perception and Hybrid Thinking*, ICLR 2026 | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cbb776e737ec3ea5925887f8740c68b4-Abstract-Conference.html); [official project](https://tsinghua-mars-lab.github.io/DriveAgent-R1/) | Conceptual precedent for learning skills before adaptive mode switching. It does not justify a VLM for the 276-specimen G1 study. | The public project repository reported no declared license on 2026-09-01. No code is reused. |
| *ActiveVLA: Injecting Active Perception into Vision-Language-Action Models for Precise 3D Robotic Manipulation*, CVPR 2026 | [CVF proceedings](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_ActiveVLA_Injecting_Active_Perception_into_Vision-Language-Action_Models_for_Precise_3D_CVPR_2026_paper.html); [official repository](https://github.com/ZhenyangLiu/ActiveVLA-Injecting-Active-Perception-into-VLA) | Establishes critical-region localization and coarse-to-fine active refinement as contemporary problems. G1 instead tests a small structured observable policy. | Repository stated code/data were not released and exposed no declared license on 2026-09-01. No code is reused. |
| Fuentes et al., *Autonomous ultrasonic inspection using Bayesian optimisation and robust outlier analysis*, MSSP 145 (2020) 106897 | [DOI](https://doi.org/10.1016/j.ymssp.2020.106897) | Prior art for sequential ultrasonic measurement with GP/BO. A Fuentes-like baseline is reserved as mandatory for the later broad benchmark; it is optional diagnostic scope in G1. | No authoritative reusable implementation was identified during the frozen search. |
| *Experimental Design for Multi-Channel Imaging via Task-Driven Feature Selection*, ICLR 2024 (TADRED) | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/aec2dfc4f5e19acf05c15587c889dbc4-Abstract-Conference.html); [official repository](https://github.com/sbb-gh/experimental-design-multichannel) | Supports downstream-task-aware sensing rather than generic reconstruction quality. G1 retains task-specific FIELD and CAI objectives. | Official repository declares Apache-2.0. No code is copied. |
| *US-VLA: An Ultrasound Vision-Language-Action Model for Embodied Abdominal Ultrasound Scanning*, 2026 | [arXiv:2608.16074](https://arxiv.org/abs/2608.16074); [official repository](https://github.com/VMVLab/US-VLA) | Contemporary semantic-goal-to-sequential-probe precedent. It does not establish observability for the Hasebe policy or authorize VLM escalation. | Official repository declares Apache-2.0. No code is copied. |

## Novelty Boundary

G1 makes no “first” claim. Its scoped question is whether task-conditioned
FOCUS/BROADEN/REFINE decisions and conservative STOP decisions can be distilled
from privileged complete inspections into a policy that receives only surface
hypothesis, causally acquired ultrasound, task identity, scanner geometry, and
measurement state. A negative observability result is a valid endpoint.

## Implementation Rule

All G1 code is a repository-native implementation. AAWR-style training remains
`CONDITIONAL` until source-only validation authorizes it; the absence of a
declared AAWR repository license independently forbids copying that source.
