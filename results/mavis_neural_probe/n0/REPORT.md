# N0 Integration Report

Decision: `N1_AUTHORIZED`

No training or scientific endpoint computation was performed. The repository,
frozen evidence, legal-state inputs, current architectures, checkpoint formats,
fold isolation, rollout protocol, metrics, compute capacity, and new namespace
were audited against the registered neural-probe protocol.

Key evidence:

- base commit and remote authority: `9794d53a9549f2e3501fe482e8db8735f468ba20`;
- canonical SHA-256: `f0d2615637a6470744f275a2ac6e1c5e7aff110ca7e31cb323793c29405be4e6`;
- frozen config SHA-256: `e99b47e161663fdaefe28719d16321a010f95b4ad8cf8f506a6e18d1d7f57b9d`;
- frozen feature-bank SHA-256: `280c608d43be164cce8617aea1cc24bf3152d537c94a48d40b61ea15085d6467`;
- baseline software regression: 30 tests passed;
- new candidate P2 parameters: 27,617 versus 27,777 for DeepSets P2;
- legal spatial input: existing 64x6 tokens plus mask, context, and cost only.

Full evidence is recorded in
`artifacts/mavis_neural_probe/N0_INTEGRATION_AUDIT.md` and the fixed decisions in
`artifacts/mavis_neural_probe/NEURAL_PROBE_PROTOCOL.md`.
