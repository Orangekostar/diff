# G0 Autonomous Inspection Opportunity Decision

Final registered status: `G0_ACTIVE_INSPECTION_OPPORTUNITY_GO`.

| Component | Gate |
|---|---|
| Zero-state compatibility and causal reveal | PASS |
| Metadata-free CAI assessor authorization | PASS |
| Initialization headroom | FAIL |
| FIELD hierarchical headroom | PASS |
| CAI hierarchical headroom | PASS |
| FIELD/CAI task-conditioning headroom | PASS |
| FIELD stopping headroom | PASS |
| CAI stopping headroom | PASS |

Privileged adaptive acquisition materially outperforms fixed acquisition for the
engineering tasks, task identity changes useful evidence, and specimen-specific
stopping contains substantial measurement-budget headroom. The initialization
oracle improves capture AUC but misses the preregistered 10% magnitude gate, so
the strongest task-conditioned status is not issued.

G1 is authorized to learn structured FOCUS/BROADEN/REFINE/STOP decisions from
policy-visible state while retaining a simple geometry-spread initialization.
G2 must compare the learned closed loop against fixed protocols, a Fuentes-like
GP/BO baseline, and the clearly privileged historical MAVIS upper bound. G3 may
test learned task conditioning because G0-C passed. G4 may add a VLM/tool
interface only after the deterministic planner is established.

No planner, VLM, LLM, or reinforcement-learning policy was trained in G0. Full
C-scans and true CAI remain teacher/evaluation privilege. The result is not a
deployment-readiness claim and makes no priority or "first" claim.

Formal and independent replay packages are byte-identical:

- status: `G0_ACTIVE_INSPECTION_OPPORTUNITY_GO`;
- manifest SHA-256: `a85a62f14bd05d69c684deab1673e01a1a84d7ebf9c3e7805760c2898eacc179`;
- output-tree SHA-256: `e1441d847eaf187eb98de7eb84e93b708225924b5263d99560354120a7f30b0a`;
- complete package SHA-256: `429f829b60bc9f520a41814ae2b6d34d05ef07cdfa188f39e2d9dbac93c45eca`.
