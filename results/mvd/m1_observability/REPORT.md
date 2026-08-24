# MVD M1 Mechanical-Value Observability

Decision: `MVD_OBSERVABILITY_NO_GO`. M2/M3 remain locked.

The selected source-CV O2 scorer has equal-domain Spearman `-0.0196`; its synchronized 95% interval is `[-0.0591, 0.0195]`. O2-minus-global NDCG@10 is `0.0210` with lower bound `-0.0240` and improves only `3/6` domains.

Candidate-only MLP Spearman is `0.0169`, global is `-0.0150`, and observed uncertainty is `-0.0072`. Global-minus-O2 and random-minus-O2 budget-regret effects do not have positive lower bounds. The evidence therefore shows neither stable continuous value prediction nor reliable top-set ranking from deployable coarse observations.

The frozen non-selection CAI diagnostic captures `-0.018` of the one-shot oracle advantage and improves `2/6` domains. This diagnostic cannot override the failed observability gate. No larger network, Transformer, GNN, RL, diffusion, M2, or M3 was run.
