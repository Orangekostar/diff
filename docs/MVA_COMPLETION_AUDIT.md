# MVA A0-A5 Completion Audit

Date: 2026-08-23

Terminal decisions:

- A3: `MVA_ORACLE_GO`.
- A4 fixed global mask: `MVA_A4_GLOBAL_NO_GO`.
- A4 adaptive-policy authorization: `MVA_A5_AUTHORIZED`.
- A5 imitation policy: `MVA_A5_POLICY_NO_GO`.
- A6: `MVA_A6_NOT_AUTHORIZED`; A6 and A7 were not implemented or executed.

## Scope compliance

- A0 reproduced the 276-specimen frozen P1 baseline within the registered
  `1e-12` tolerance and verified the P5 25% reconstruction endpoint.
- A1 validated nested masks, measured-value restoration, monotone budgets,
  P5 equivalence, and the sparse-state simulator.
- A2 evaluated six held-out domains with strict fold-local fitting, 100 random
  seeds, three oracle controls, and the mechanical oracle.
- A3 recomputed curves, AUEBC, sufficiency budgets, synchronized intervals,
  and the four preregistered oracle-headroom gates.
- A4 trained source-only fixed rankings for every outer domain, audited all
  teacher fits, and evaluated the unchanged rankings on held-out targets.
- A5 regenerated outer-safe teacher trajectories, trained one fixed 41,617-
  parameter imitation policy per outer domain, and audited every target action.

## Formal evidence

- A3 mechanical-oracle AUEBC was 0.01062453 versus 0.01719904 for the
  reconstruction oracle; all four A3 gates passed.
- A4 global mechanical AUEBC was 0.01763889 versus 0.01736346 for uniform.
  Its paired effect interval crossed zero, so the fixed mask failed.
- A4 retained a 39.766% global-to-oracle relative AUEBC gap with a positive
  lower bound in 6/6 domains, authorizing A5.
- A5 policy AUEBC was 0.01709223. Global-minus-policy was 0.00054666 with 95%
  interval [-0.00079259, 0.00209711], and uniform-minus-policy was 0.00027123
  with interval [-0.00192366, 0.00260569]; both improved only 3/6 domains.
- A5 closed 7.793% of the registered oracle gap, below the 20% gate. Its
  secondary `B_5%` was 6.25%, versus 18.75% for global mechanical and uniform.
- A5 published 1380 teacher-index rows, 390 fit-audit rows, 300 training rows,
  6624 checkpoint states, and 33049 target actions.

## Interpretation boundary

All acquisition results are retrospective normalized-raster simulations. True
target CAI and unobserved candidate RGB values appear only in diagnostic oracle
construction and evaluation. The package does not establish physical scanner
pitch, inspection-time reduction, prospective adaptive control, or transferable
laminate conditioning.

## Artifact checks

- The shared 100000 x 6 bootstrap digest is
  `732abea9fa90893d53c6a0620c020a2072cd97a2b7d8e48e16aa8feb10c5564c`.
- A4 formal and replay packages are byte-identical: output-tree digest
  `2ed736568419ff549981430b202889d3f5b89e0971a8298ae9e125ddf9681e78`;
  manifest digest
  `bc2556041d5de9d42522830784ea328aed5a3857eec5e36d464f8237145a5ec5`.
- A5 formal and replay packages are byte-identical: output-tree digest
  `3712e6ebe09517565204a15b034cd3eb03da1fe3737a9a56e62cb8548f9fcbe8`;
  manifest digest
  `c59db1608d6f180587bb24b994bef2f1049b576e072cc4edbe0113f92407dd73`.
- A5 independent validation recomputed raw rosters, leak barriers, action
  digests, curves, metrics, intervals, gates, model packages, and checksums.
- A5 figures were rendered in PNG and editable SVG with 357 source-data rows;
  manual layout, XML, and nonblank-image checks passed.
