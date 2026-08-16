# Suggested ChatGPT analysis prompt

Analyze this repository as a registered cross-dataset experiment, not as a
model leaderboard. Cite exact files and numeric values for every conclusion.
Separate registered evidence from post-hoc explanation.

Please answer:

1. Why did G1 fail even though some scalar targets improved in five or six
   datasets? Explain the simultaneous interval and domain-consistency gates.
2. Why did measured scalar descriptors fail G2, and what does that imply for
   strict-OOF predicted scalars?
3. Why can P1 and P3 pass while G1 and G2 fail? Distinguish scalarization loss,
   spatial organization, observability, and predictive utility.
4. Why did P2 and P4 fail despite using learned or privileged representations?
   Compare their candidates with the frozen simple baselines and controls.
5. Why did P5 pass? Quantify the 25% sparse-scan retention and explain why this
   does not imply that a learned reconstruction is necessary.
6. Analyze P6 in detail. Compare diffusion with bilinear, PCHIP, and the
   deterministic U-Net using mechanical MAE, retention, image L1/SSIM/PSNR,
   improved-domain counts, and simultaneous intervals. Explain why exact
   measured-point preservation did not rescue the diffusion method.
7. Treat `analysis_tables/p6_posthoc_uncertainty_diagnostics.csv` only as
   descriptive. Does the eight-draw spread track held-out error, and what does
   its domain pattern suggest without proving?
8. Why did P7 show negative rather than positive surface complementarity at
   both retained sparse densities? Separate direct evidence from hypotheses
   involving domain shift, redundant/noisy features, model selection, and
   sample size.
9. State the strongest defensible cross-stage conclusion and the minimum next
   experiment that would distinguish representation failure from lack of
   recoverable information.

Do not reinterpret a failed gate as a pass. Do not infer causal sufficiency,
unique damage reconstruction, uncertainty calibration, or material transfer
unless a registered result directly supports it. The file
`results/cross_stage/V3_FINAL_GATE_STATUS.md` is a historical P2-time snapshot;
use `V3_EXTENDED_GATE_STATUS.md` for the current P3-P7 state.
