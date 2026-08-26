# MAVIS P13 Set-Level Planning Diagnosis

Status: `COMPLETE`.

At the frozen 6.25% checkpoint, each method selects exactly two legal actions. The exact and retrospective rows enumerate every feasible ordered pair in the target-safe reachable pool: the initial learned top-8 plus any action reached by the registered greedy/beam plans. Final selection quality is evaluated with the true joint downstream CAI-error change of the complete set, never by summing point values. Beam widths 2 and 4 and lookahead width 8 were predeclared; no target outcome selected a width.

Current-greedy mean joint utility is `0.0000206755` with planning regret `0.0001206542`. The retrospective reachable-pool near-oracle utility is `0.0001413297`.

Interpretation: The bounded set-planning gap is supported because the paired current-greedy regret interval is strictly positive.

This is a bounded diagnostic, not a deployable policy or an unrestricted oracle. It does not tune the P7 checkpoint and does not establish scanner-time reduction.
