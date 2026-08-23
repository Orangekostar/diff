# MVA A0-A3 Completion Audit

Date: 2026-08-23

Terminal decision: `MVA_ORACLE_GO`.

## Scope compliance

- A0 reproduced the 276-specimen frozen P1 baseline within the registered
  `1e-12` tolerance and verified the P5 25% reconstruction endpoint for all
  276 specimens.
- A1 validated nested cell masks, exact measured-value restoration, monotone
  budgets, exact all-level-1 P5 equivalence, and the sparse-state simulator.
- A2 evaluated six held-out domains with strict fold-local preprocessing and
  predictors, 100 registered random seeds, three oracle controls, and the
  mechanical oracle.
- A3 recomputed curves, AUEBC, sufficiency budgets, synchronized bootstrap
  intervals, and all four preregistered gates from published raw tables.
- A4-A7 were not implemented or executed.

## Formal evidence

- H1 passed: 35.43% relative MAE improvement over uniform at 12.5%, with
  improvement in 6/6 held-out domains.
- H2 passed: reconstruction-minus-mechanical AUEBC is 0.00657452 with 95%
  interval [0.00477228, 0.00863115].
- H3 passed: appearance-minus-mechanical AUEBC is 0.00708006 with 95% interval
  [0.00479936, 0.00974029].
- H4 passed: relative AUEBC improvement is 38.81%; mechanical `B_5%` is 6.25%
  versus 18.75% for the stronger fixed/random reference, a 66.67% saving.
- Published tables contain 172224 state rows, 1232679 trajectory rows, and
  1652664 candidate oracle-value rows.
- The shared 100000 x 6 bootstrap index digest is
  `732abea9fa90893d53c6a0620c020a2072cd97a2b7d8e48e16aa8feb10c5564c`.

## Interpretation boundary

This package establishes retrospective oracle headroom on a normalized raster
observation simulation. It does not establish physical scanner pitch,
inspection-time reduction, an online adaptive scanner, or deployable policy
performance. True target CAI and unobserved candidate RGB values are used only
to define the diagnostic oracle.

## Artifact checks

- Formal shard validation passed for all six main domains and all six low
  domains.
- Four nonselecting stability variants passed for every specimen; reconstruction
  equivalence deltas were exactly zero.
- Six publication figures were rendered as PNG and SVG with exported source
  data and passed manual layout and nonblank-pixel inspection.
- The final manifest and independent replay verification are recorded in the
  published result tree.
