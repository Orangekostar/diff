# MAVIS P12 Representation-Valuation-Planning Attribution

Status: `COMPLETE`.

Rows A-E change only the registered diagnostic component shown in `substitution_matrix.csv`. B and D intentionally coincide: both are the frozen retrospective conditional mechanical-value trajectory under the current greedy planner. C is a causal two-step beam over the unchanged learned P2/P3 models. E is a retrospective joint downstream-error portfolio over the two pre-registered true-value set trajectories; it is a limited near-oracle, not an unrestricted set oracle.

Valuation substitution improvement (A-B): `0.0000497867`.  Learned planning substitution improvement (A-C): `0.0000031643`.  True-value planning substitution improvement (D-E): `0.0001117124`.  Total limited-oracle improvement (A-E): `0.0001614992`.

Paired 95% intervals for substitution-minus-reference AUEBC are B-A `[-8.104897453356152e-05, -1.8446727599749598e-05]`, C-A `[-6.250270954010764e-06, -1.4114836609307647e-07]`, E-D `[-0.00012479264383388855, -9.898242657418798e-05]`, and E-A `[-0.00018778812503263552, -0.0001352116944961955]`.

Lower CAI AUEBC is better. Positive improvement values favor the substituted row. Equal-domain aggregation follows specimen-first AUEBC. Rows B, D, and E use retrospective target outcomes and are explicitly non-deployable. P10 did not support adding optional row F: observed UT content did not improve over positions-only, so no new representation was trained. No P7 checkpoint or artifact was modified.
