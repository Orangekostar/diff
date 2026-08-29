# N4 Spatial Closed-Loop Probe

Gate: `END_TO_END_NO_GO`.

Frozen static reference AUEBC: `0.1249920401`. Spatial candidate AUEBC: `0.1249907885`. The registered static-minus-candidate contrast is `0.0000012516` with paired 95% CI `[-0.0065366288, 0.0063906428]` and favorable direction in `3/6` held-out domains. Positive values favor the candidate.

The frozen current learned implementation AUEBC is `0.1250531822`. The existing uniform scout, 8x8 action grid, acquisition levels, exact native-raster cost, candidate generation, reveal action, direct-cost-aware objective, rollout, CAI metrics, and bootstrap are unchanged. No target outcome was used for training or selection.
