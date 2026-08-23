# MVA A0-A3 Oracle Headroom Report

Terminal decision: `MVA_ORACLE_GO`.

This is a retrospective diagnostic upper bound on the normalized raster observation grid. Ground-truth CAI is used only to label the mechanical oracle. The result does not establish physical scanner pitch, inspection-time reduction, or a deployable acquisition policy.

## Required Questions

1. Mechanical CAI oracle versus uniform: yes under H1; 12.5% relative MAE improvement is 35.43% with 6/6 domains in the same direction.
2. Mechanical CAI oracle versus reconstruction oracle: yes under the synchronized domain bootstrap; reconstruction-minus-mechanical AUEBC is 0.00657452 (95% interval [0.00477228, 0.00863115]).
3. Mechanical CAI oracle versus the frozen appearance heuristic: yes under the synchronized domain bootstrap; appearance-minus-mechanical AUEBC is 0.00708006 (95% interval [0.00479936, 0.00974029]).
4. Budgets with lower mechanical-oracle MAE than uniform: 6.250% (17.94% relative MAE), 9.375% (33.14% relative MAE), 12.500% (35.43% relative MAE), 18.750% (44.67% relative MAE), 25.000% (49.43% relative MAE).
5. Domain-level AUEBC headroom occurs in 6/6 domains: 74t7kcdgkr, cgtnjyggtm, w68dtmpfyf, xcmzfsbd9t, yfxyg8jm46, ykhs7s2dck.
6. Largest specimen-level uniform-minus-mechanical AUEBC reductions: c24-18 (0.02999605), c24-14 (0.02904897), q24-22 (0.02661218), c24-25 (0.02551297), q16-26 (0.02487887).
7. Initial mechanical versus reconstruction maps: mean Pearson 0.1553, Spearman 0.0419, top-10% overlap 0.3302, RBO 0.3352.
8. Initial mechanical versus appearance maps: mean Pearson 0.0472, Spearman 0.0222, top-10% overlap 0.2003, RBO 0.2269; this quantifies whether the mechanical map is a copy rather than assuming it is distinct.
9. B5 is 6.250% for the mechanical oracle and 18.750% for uniform; simulated measurement reduction versus uniform is 66.67%.
10. Policy-learning headroom is supported by H4; relative AUEBC headroom is 38.81% and B5 saving against the stronger fixed/random reference is 66.67%.

## Gate

- H1: True; relative improvement 0.354316; improved domains 6/6.
- H2: True; bootstrap lower 0.00477228.
- H3: True; bootstrap lower 0.00479936.
- H4: True; relative AUEBC improvement 0.388110; B5 saving 0.6666666666666667.

## Initial-Ranking Stability

- bicubic_ridge10: top-1 0.413, top-10% overlap 0.533, Spearman 0.545, RBO 0.538.
- bilinear_ridge1: top-1 0.779, top-10% overlap 0.868, Spearman 0.909, RBO 0.864.
- bilinear_ridge100: top-1 0.641, top-10% overlap 0.756, Spearman 0.710, RBO 0.753.
- nearest_ridge10: top-1 0.337, top-10% overlap 0.463, Spearman 0.386, RBO 0.467.

The 50% uniform and 100% FULL points are report-only anchors and are excluded from AUEBC. A4-A7 were not implemented or executed in this package.
