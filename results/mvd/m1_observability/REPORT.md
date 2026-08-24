# MVD M1 Mechanical-Value Observability

Decision: `MVD_OBSERVABILITY_NO_GO`. M2/M3 remain locked.

The selected source-CV O2 scorer has equal-domain Spearman `-0.0196`; its synchronized 95% interval is `[-0.0591, 0.0195]`. O2-minus-global NDCG@10 is `0.0210` with lower bound `-0.0240` and improves only `3/6` domains.

Candidate-only versus global-plus-candidate: candidate-only MLP Spearman is `0.0169`, while selected O2 is `-0.0196`; global is `-0.0150` and observed uncertainty is `-0.0072`. None provides stable continuous value prediction or reliable top-set ranking. O2 Regret@1 is `0.0200` and its mean exact-budget set regret is `0.0817`, versus global `0.0799` and random `0.0798`. Global-minus-O2 and random-minus-O2 regret effects do not have positive lower bounds.

The frozen non-selection CAI diagnostic captures `-0.018` of the one-shot oracle advantage and improves `2/6` domains. This diagnostic cannot override the failed observability gate. No larger network, Transformer, GNN, RL, diffusion, M2, or M3 was run.

External feasibility: Imperial RSS has `0` exact paired C-scan+CAI specimens (`7` potential links remain unresolved); Imperial Interlock has `10` exact pairs and is a small pilot. No audited dataset is sufficient by itself for formal statistical external replication. TU Delft has `3` specimens and remains case-level `MICRO_CASE_VALIDATION_ONLY`. Cranfield raw PA recovers a discrete indexed spatial measurement grid and normalized 8x8 mapping, but not authoritative physical spacing or scanner-time reduction.
