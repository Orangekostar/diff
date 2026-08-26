# P15 Downstream-Learner Value-Stability Audit

Ridge and Huber retain similar value structure (mean Spearman 0.762), but the less accurate shallow MLP (OOF MAE 0.151) does not reproduce it (Ridge-MLP Spearman 0.116). Across the predeclared family, the defensible term is downstream-predictor-conditioned task value.

The audit uses the same 276-specimen cohort, six LODO outer splits, fold-local PCA dimension, 64-action initial state bank, measurement reveal, and exact action cost for Ridge, Huber, and a 16-unit shallow MLP. The emitted Ridge map reproduces the issued formal oracle values exactly (maximum delta 0); a fresh candidate-bank Ridge inference differs by at most 1.49e-07 because the frozen image states were encoded in a separate GPU pass.

Mechanical oracle utility is retrospective and non-deployable. The analysis does not establish an intrinsic or universal physical value map, and coordinate summaries do not identify a failure mechanism. P7 remains unchanged.
