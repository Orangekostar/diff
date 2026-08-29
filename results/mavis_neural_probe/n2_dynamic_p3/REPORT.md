# N2 Spatial Dynamic-Value Probe

Gate: `VALUE_NO_GO`.

DeepSets real next-action regret: `0.0122191085`. Spatial real next-action regret: `0.0122161595`. The registered DeepSets-minus-Spatial regret contrast is `0.0000029490` with paired 95% CI `[-0.0001244385, 0.0001298122]` and favorable direction in `2/6` held-out domains. Positive values favor Spatial.

The DynamicActionScorer, candidate descriptors, loss weights, teacher, metrics, and bootstrap are unchanged. Only the legal 64-D P2 embedding provider is spatial. Selection and fitting remain source-only; target teacher values are retrospective evaluation data.
