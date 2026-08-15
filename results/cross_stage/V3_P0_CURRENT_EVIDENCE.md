# V3 P0 Current Evidence

## Scope

This audit freezes the evidence available before any V3 model result is inspected. The
authoritative optimization prompt is `docs/CPB_V3_PRIVILEGED_NDE_CODEX_PROMPT.md`, SHA-256
`6b784772ee18f65b8d13e0f41d835e084760c496264dbe42d9a0c0c242719e11`.

V3 is isolated under `paper_v3/`. No V2 result, protected manuscript, or raw source is modified.
The project directory is not a Git worktree, so V3 provenance must bind source bytes and runtime
state directly rather than claim a commit identity.

## Frozen G1 evidence

The registered G1 gate requires all three conditions for each target: at least 10% relative MAE
improvement, improvement in at least five of seven released datasets, and a simultaneous
confidence lower bound above zero.

| Physical C-scan target | Relative improvement | Improved datasets | Simultaneous lower bound | Decision |
|---|---:|---:|---:|---|
| Projected damage area (mm^2) | 6.2135% | 3/7 | -0.164873 | FAIL |
| Damage height (mm) | 6.5989% | 6/7 | -0.052227 | FAIL |
| Damage width (mm) | 7.0470% | 5/7 | -0.131310 | FAIL |

Authoritative table: `paper_v2/artifacts/tables/P1_g1.csv`, SHA-256
`4125373bb37382750a8800090218477de34ac239b58e1cb51935b97abfbd4bf8`.

Interpretation is frozen: the scalar targets show weak and domain-dependent surface
observability, not a stable confirmatory gain. This does not establish that surface evidence
contains no internal information.

## Frozen G2 evidence

The registered G2 gate requires at least 10% raw equal-dataset CAI-ratio MAE improvement,
improvement in at least four of six released datasets, and a simultaneous lower bound above
zero. Baseline A contains 13 registered specimen/impact/acquisition variables and 21 surface
statistics.

| Pathway | Equal-dataset raw CAI-ratio MAE | Relative change vs A | Improved datasets | Simultaneous lower bound | Decision |
|---|---:|---:|---:|---:|---|
| A: metadata + surface | 0.188121 | reference | - | - | reference |
| B: A + measured scalar internal | 0.189204 | -0.5760% | 4/6 | -0.005070 | FAIL |
| C: A + strict-OOF predicted scalar internal | 0.185598 | +1.3409% | 3/6 | -0.013738 | FAIL |
| D: A + train-deranged scalar internal | 0.188089 | +0.0168% | 2/6 | -0.005755 | FAIL |

C versus D has simultaneous paired lower bound `-0.018558`. The measured scalar oracle is worse
than A, so the scalar oracle-gap denominator is non-positive and no scalar utility-recovery claim
is defined.

Authoritative metrics: `paper_v2/experiments/P2_mechanical_utility/metrics.json`, SHA-256
`c0032524bcd7a67266b05b845ef0706fa0980aca361f0c75c3fbc50ac9e1c293`.

## Frozen cohort and response

- Surface + measured C-scan: 292 specimens across seven released datasets.
- Surface + measured C-scan + numeric CAI: 276 specimens across six released datasets.
- Primary response: published damaged-to-intact CAI strength ratio, unit `1`.
- Primary metric: raw equal-dataset MAE.
- Inferential unit: held-out dataset for confirmatory G2b effects; specimens remain paired within
  each held-out dataset.
- Outer protocol: one complete released dataset held out at a time.

The 276-specimen G2b cohort is fixed before any V3 prediction is generated. Its counts are
`74t7kcdgkr=45`, `cgtnjyggtm=49`, `w68dtmpfyf=43`, `xcmzfsbd9t=59`,
`yfxyg8jm46=42`, and `ykhs7s2dck=38`.

Five additional raw C-scan crops have numeric CAI but lack an accepted entry in the frozen
physical-measurement registry. They define an explicit 281-specimen alternative inventory, not an
evaluated sensitivity or the confirmatory denominator. Keeping the 276-specimen primary cohort
preserves exact pairing with B-scalar and the frozen A comparison rather than selecting a cohort
after observing V3 errors.

## Scientific boundary before V3 P1

The only authorized next test is whether measured full-field C-scan morphology adds transferable
CAI information beyond A under strict LODO. G1/G2 are not rerun, relabeled, or threshold-adjusted.
MSPD, advanced distillation, manuscript rewriting, diffusion, Simformer, flow matching, and large
transformers remain unauthorized until the full-field G2b gate is known.
