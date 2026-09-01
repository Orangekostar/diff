# G0 Zero-State Audit

Status: `ZERO_STATE_AUDIT_PASSED` for the formal and byte-identical replay
packages.

The existing MVA state cannot express zero ultrasound because it validates only
levels 0, 1, and 2 and constructs `(0,)*64`. The new state adds level -1 without
changing `mva/measurement_state.py` or any frozen MVA/MAVIS source.

Across all six authorized shape/budget families, the union of individually
acquired 64 level-0 cell lattices is byte-identical to the old MVA initial mask.
Exact family counts are 1764, 3600, 1848, 3720, 7056, and 14161. The complete
grid and mask hashes and canonical incremental-cost histograms are recorded in
`G0_PREFLIGHT_20_QUESTION_AUDIT.md`; the formal package materializes one row per
authorized specimen in `zero_state_audit.csv`.

Formal evidence covers all 276 authorized specimens in six Hasebe domains:

- every zero state has no positions or values, zero exact cost, and zero budget;
- all `-1->0` unions match the old scout mask and exact budget;
- all level-1 states match the old MVA level-1 mask;
- all level-2 states equal the full native raster;
- every transition reveals only new positions and preserves prior values;
- target unmeasured-pixel changes cannot affect generalized reconstruction;
- formal and replay `zero_state_audit.csv` files are byte-identical.

The formal zero-state table has SHA-256
`2445b4015ce43438f01ba3290cead4c2ad239078baba48733467658f75207aa2`.
The complete formal and replay packages both have SHA-256
`429f829b60bc9f520a41814ae2b6d34d05ef07cdfa188f39e2d9dbac93c45eca`.
