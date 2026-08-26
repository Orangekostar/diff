# P0 Repository and Evidence Audit

Audit date: 2026-08-26

## Repository state

- Repository: `/home/ww/diff`
- Isolated worktree: `/home/ww/diff/.worktrees/aei-information-hierarchy`
- Branch: `aei-information-hierarchy`
- Audit base: `c2eab6eac79dd3fbb9ecb0d19f98923e515e762b`
- Relation to registered base: exact match to `c2eab6e`
- Pre-audit worktree state: clean
- Remote verification: `origin/main` and `origin/mavis-science-closure` both
  resolve to `c2eab6eac79dd3fbb9ecb0d19f98923e515e762b`.

The tracked science-closure package includes the M0/M1 decision records, core
CSV files, and the complete P9-P16 replay package. No upload gap was found.

## Evidence inventory

| Namespace | Tracked files | Approximate size | Paper role |
|---|---:|---:|---|
| `artifacts/mavis/` | 8 | 40 KiB | Frozen evidence and manuscript map |
| `artifacts/mavis_science_closure/` | 13 | 88 KiB | Closure claims, audits, and core metrics |
| `results/mavis/` | 497 | 1.1 GiB | P1-P7 acquisition evidence |
| `results/mavis_science_closure/` | 175 | 138 MiB | P9-P16 formal and replay evidence |
| `results/mvd/` | 69 | 27 MiB | One-shot headroom and static observability |
| `results/mva/` | 177 | 131 MiB | Oracle value and learned-mask evidence |
| `results/p1_full_field_oracle/` | 13 | 1.1 MiB | Registered scalar/full-field comparison |
| `paper_v3/` | 27 | 46 MiB | Configurations and frozen P1 prediction payload |
| `src/cmc_bbdm/mavis/` | 107 | n/a | Reusable acquisition and closure code |
| `src/cmc_bbdm/cpb_v3/` | 2 | n/a | Registered full-field feature semantics |

Counts are an inventory check, not a scientific completeness claim.

## Authority hierarchy

1. Registered P1 report and frozen machine-readable P1 outputs control the
   scalar-versus-spatial confirmatory claim.
2. Frozen MVA and MVD formal outputs control retrospective usefulness and
   pre-inspection observability claims.
3. Frozen MAVIS P7 outputs control the deployable closed-loop boundary.
4. P9-P16 science-closure outputs control conditional value, matched state
   controls, attribution, planning, task specificity, predictor dependence,
   and feedback claims.
5. Historical manuscript maps are discovery aids. They do not override a
   registered comparison or its method semantics.

## Frozen boundaries

The following namespaces are immutable evidence inputs for Paper 1:

- `results/p1_full_field_oracle/`
- `analysis_tables/`
- `results/mva/`
- `results/mvd/`
- `results/mavis/`
- `results/mavis_science_closure/`
- `artifacts/mavis/`
- `artifacts/mavis_science_closure/`
- `artifacts/mvd_authority/`
- `artifacts/mavis_authority/`

The frozen P7 tree state remains
`931dc86c26caf1c7246709c4706a7cd0428e3a1533b6ff1ad3c2ad8f9517d1e4`.
Paper-specific derived files will be written only under:

- `artifacts/aei_information_hierarchy/`
- `results/aei_information_hierarchy/`
- `paper_aei_information_hierarchy/`

## Reusable implementation map

| File | Symbol | Paper role |
|---|---|---|
| `src/cmc_bbdm/mavis/reveal.py` | `reveal_uniform_scout`, `reveal_action` | causal measurement reveal semantics |
| `src/cmc_bbdm/mavis/state_bank.py` | `materialize_action_plan` | legal action/state histories |
| `src/cmc_bbdm/mavis/state_encoder.py` | `build_mris_input`, `MRISStateEncoder` | partial-state representation description |
| `src/cmc_bbdm/mavis/mechanics_head.py` | `MRISMechanicsModel` | downstream mechanics estimator |
| `src/cmc_bbdm/mavis/teacher.py` | `fit_strict_oof_teacher`, `label_teacher_state` | strict-OOF conditional value |
| `src/cmc_bbdm/mavis/dynamic_voi.py` | `DynamicActionScorer`, `conditional_teacher_value` | conditional action valuation |
| `src/cmc_bbdm/mavis/dynamic_training.py` | `fit_inner_dynamic_voi`, `fit_final_dynamic_voi` | source-only training protocol description |
| `src/cmc_bbdm/mavis/rollout.py` | `rollout_scout_and_focus_curve` | acquisition trajectories |
| `src/cmc_bbdm/mavis/policy.py` | `select_cost_aware_action` | deployable action selection |
| `src/cmc_bbdm/mavis/dynamic_metrics.py` | `evaluate_dynamic_scores`, `bootstrap_dynamic_contrasts` | valuation metrics |
| `src/cmc_bbdm/mavis/closed_loop_metrics.py` | `evaluate_closed_loop_predictions`, `bootstrap_closed_loop_contrasts` | actionability metrics |
| `src/cmc_bbdm/mavis/task_specificity.py` | evaluation/bootstrap helpers | reconstruction/mechanics specificity |

New code is limited to paper evidence aggregation, figures, tables, package
generation, and provenance validation. No new model fitting or training is
required.

## Baseline verification

| Check | Result |
|---|---|
| `PYTHONPATH=src python -m pytest -q tests/test_mavis_*.py` | 155 passed |
| `PYTHONPATH=src python -m pytest -q tests/test_mvd_*.py` | 29 passed |
| Full historical MVA suite in the complete authority repository | 126 passed |

The MVA suite requires the complete authority repository at
`/home/ww/paper3/cmc_damage_inference`; this is recorded as an environment
dependency rather than represented as a reduced-worktree test.

## P0 decision

Paper 1 can proceed without new model training. The defensible contribution is
the Useful-Observable-Actionable information hierarchy and its empirical
boundaries, not a claim of closed-loop superiority.
