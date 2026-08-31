# G0 Zero-State Audit

Status before implementation: geometry and compatibility preregistration passed;
formal causal-world replay pending.

The existing MVA state cannot express zero ultrasound because it validates only
levels 0, 1, and 2 and constructs `(0,)*64`. The new state adds level -1 without
changing `mva/measurement_state.py` or any frozen MVA/MAVIS source.

Across all six authorized shape/budget families, the union of individually
acquired 64 level-0 cell lattices is byte-identical to the old MVA initial mask.
Exact family counts are 1764, 3600, 1848, 3720, 7056, and 14161. The complete
grid and mask hashes and canonical incremental-cost histograms are recorded in
`G0_PREFLIGHT_20_QUESTION_AUDIT.md`; the formal run will materialize one row per
family/action in `zero_state_audit.csv`.

Formal acceptance still requires:

- zero state has no positions, no values, zero exact cost, and zero budget;
- every transition reveals only new positions and preserves prior values;
- all `-1->0` states match the old scout mask and budget;
- all level 1 states match the old MVA level-1 mask;
- all level 2 states equal the full native raster;
- target unmeasured-pixel changes cannot affect generalized reconstruction;
- formal and replay zero-state artifacts are byte-identical.
