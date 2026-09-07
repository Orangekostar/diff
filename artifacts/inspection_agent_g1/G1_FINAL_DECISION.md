# G1 Final Decision

Final status: `G1_POLICY_OBSERVABILITY_NO_GO`

G2 authorized: **NO**

Both deployable action-policy components fail the preregistered held-out-domain
gate. FIELD is worse than its strongest fixed reference in all six domains
(effect -0.0000727815142626863, 95% CI
[-0.00008689672989010225, -0.00006182037360547485], gap closure
-0.8846013469833925). CAI is worse overall and improves only two domains
(effect -0.0010582009874103384, 95% CI
[-0.0020696313885150072, -0.000059502896739864196], gap closure
-0.0508944066361726).

FIELD task-token controls are positive, but CAI controls are inconclusive;
surface controls are inconclusive for both tasks; and STOP is unauthorized in
all folds. Therefore neither a task-conditioned GO, active-policy GO, FIELD-only
GO, nor CAI-only GO is issued.

The scientific interpretation is bounded: the G0 privileged opportunity is not
sufficiently observable from the currently available deployment evidence under
this preregistered policy class. This result does not authorize post-target model
rescue, VLM escalation, or claims of scanner-time saving. It does not negate the
G0 privileged opportunity; it identifies observability, rather than model size,
as the unresolved scientific problem.

Evidence integrity:

- All six outer selections and target trajectories were frozen before hidden
  target truth was opened.
- Target outcomes were used only for evaluation, bootstrap inference, and gates.
- Formal and replay packages are byte-identical.
- Package SHA-256:
  `f444187542bd583dad5f689a87fc1f725940b81f9596eb1a8768e3de9699bf14`.
- Artifact-manifest SHA-256:
  `994fe76bf20e8d5e7bb07e1b2b431bba0916ce62727ab5b27df81692f4804929`.
- Output-tree SHA-256:
  `0deec49686793289afa45680b0390581a4020b4615ed86fcb418a8a29419852b`.
