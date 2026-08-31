# External Repository Audit

Date checked: 2026-08-31 UTC
Scope: architecture and dependency boundary before P1
Copied external code: NO

All commit identities below were read from the repository `HEAD` with
`git ls-remote`. License status was checked through the repository metadata and
top-level tree at the same time. A repository being public does not authorize
copying code when no license is present.

| Repository | Current HEAD | License | Files inspected | Use in this route | Code copied |
| --- | --- | --- | --- | --- | --- |
| [DriveAgent-R1](https://github.com/wczheng/DriveAgent-R1) | `a1abd75fe80cc6868f74f308f71042f7f0d054a8` | No top-level license identified | `tool-rl/src/r1-v/src/open_r1/tools/tool_registry.py` (`779c88ae...`), `tools/utils.py` (`e140c699...`), `trainer/vllm_grpo_trainer_modified.py` (`949b6c6d...`) | Conceptual tool registry, validation, and multi-round parsing reference for conditional P4 only | NO |
| [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) | `96588727e44c78b25ba03ea03b8e12f7e64fd0da` | Apache-2.0 | `README.md`, `LICENSE` (`c71d239d...`) | Possible frozen zero/few-shot legal-grid diagnostic in P1, only after P0 GO and protocol freeze | NO |
| [US-VLA](https://github.com/VMVLab/US-VLA) | `dde576ae9992d33231f3de925ff44a1fc818ab00` | Apache-2.0 | `src/openpi/models/ultrasound_fusion.py` (`b0d271e5...`), `README.md`, `LICENSE` | Conceptual cross-attention reference for conditional P3/P4; medical USFM, JAX/OpenPI, and continuous robot actions are out of scope | NO |
| [ActiveVLA](https://github.com/ZhenyangLiu/ActiveVLA-Injecting-Active-Perception-into-VLA) | `4450b1188a24f3f1d1d3fb3e47d226d150b9e470` | No top-level license identified | `README.md` (`dd2f6ffe...`) | Conceptual coarse-to-fine active-perception reference only | NO |
| [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | `856dde20aee659246248e20734ef9ba5214f5e44` | Apache-2.0 | `README.md`, `LICENSE` (`b403c98e...`) | Optional specimen-boundary diagnostic only; no mechanical target role | NO |
| [SAM 2](https://github.com/facebookresearch/sam2) | `2b90b9f5ceec907a1c18123530e92e794ad901a4` | Apache-2.0 | `README.md`, `LICENSE` (`c71d239d...`) | Optional prompted-region diagnostic only; no mechanical target role | NO |
| [RoboSpection](https://github.com/CuriousLad1000/RoboSpection) | `cacd70e8aa734383b6046ec0818d9af6d06bfcbf` | Apache-2.0 | `README.md` (`0f8f11e0...`), `LICENSE`, `src/` listing | Conceptual high-level planner / deterministic low-level executor boundary; no ROS or hardware dependency | NO |
| [TADRED](https://github.com/sbb-gh/experimental-design-multichannel) | `7cf5ab47e2504da7bf7075a40f852fe8be8b951f` | Apache-2.0 | `README.md` (`97c36c7b...`), `LICENSE`, top-level package listing | Prior-art boundary for task-driven experimental design; existing spatial engine already supplies the needed action semantics | NO |

## Current-state findings

- DriveAgent-R1 now exposes training code, but its evaluation scripts, datasets,
  and trained weights remain listed as unreleased. Its absent top-level license
  makes all code reuse unauthorized.
- ActiveVLA still marks training, inference, models, evaluation, and robot code
  as pending. Its absent top-level license independently prevents code reuse.
- US-VLA now has an Apache-2.0 implementation. Its medical standard-plane task,
  USFM encoder, JAX/OpenPI stack, and continuous probe controls do not match the
  frozen 64-cell offline replay environment.
- GroundingDINO and SAM 2 could only assist source-side geometry. Their outputs
  cannot establish surface-to-C-scan correspondence or mechanical value.
- No external repository supplies the missing Hasebe cross-instrument spatial
  transform. Importing one would not resolve P0 authority.

## Decision

P0-P3 will have no external runtime repository dependency and no copied code.
If P4 is authorized later, its typed registry and parser will be implemented
locally from the frozen tool schema. DriveAgent-R1 remains a cited architecture
reference only unless a future audit finds an explicit compatible license.
