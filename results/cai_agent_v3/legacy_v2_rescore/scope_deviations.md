# Legacy v2 Rescore Scope Deviations

- Status: `LEGACY_RESCORE_COMPLETE`.
- Source models and trajectories were not rerun or modified.
- The corrected area uses a left-constant state trajectory with a last-state tail; the legacy area used trapezoidal integration.
- Each seed/run is scored before seed losses are averaged by physical specimen. Stored ensemble-prediction rows remain `ENSEMBLED_PREDICTIONS_UNPRICED` and are not treated as a paid run.
- The original stored v2 target is retained. Its maximum absolute difference from the direct author-workbook MPa extraction is `1.29934353e-05` MPa, attributable to the stored float32 feature-bank value.
- `LEARNED_STATIC` is not reinterpreted as a true static policy: it consumed surface-image features. Its reporting alias is `V2_SURFACE_DEPENDENT_LEARNED_CONTROL`.
- Corrected evaluation does not retroactively correct the v2 Actor training objective, checkpoint selection, or batch-termination behavior.
- Capture groups come only from P0R specimen/source-image identities; no independence relation was inferred from predictions or labels.
