# Methods facts

- Cohort: VALID only; 50 physical specimens, 48 capture groups, six domains. No TEST inference or label join.
- Track A: 500 archived rows across six methods. Random has five repeats; all other methods have one. Repeats are averaged within specimen before domain-equal aggregation.
- Track B: the 16-cell terminal set from each proposed-policy trajectory is held fixed. Native, reverse, and five preset SHA256 permutations are scored without actor decisions.
- Clock: Track A preserves archived float64 costs. Track B recomputes cumulative native-cell pixels divided by native image pixels, with float32 cost only at predictor input.
- Predictor: frozen MEAN_SC, selected update 1750. Every original and reordered prefix is evaluated through one common engine and a fully bound cache key.
- Primary estimand: permuted five-repeat mean area minus native area. Secondary estimand: reverse area minus native area. Positive differences favor native.
- Aggregation: area metrics use repeat mean, within-domain mean, then six-domain equal mean. Endpoint MAE/RMSE average losses, never predictions.
- Uncertainty: 5,000 fixed capture-group-within-domain paired bootstrap draws; no fallback bootstrap.
- Quality comparison: first queue-level MAE crossing on separate event-union grids; recrossings, unreached targets, and negative savings remain explicit.
- Completion is technical. Scientific effect size or confidence-interval sign is not a gate.
