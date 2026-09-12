# CAI Agent v3 Results and Claim Boundaries

## Status

- Implementation: complete; execution stopped at the W2 resource gate.
- Protocol: `NONCONFORMANT_W2_CHECKPOINT_SELECTION_PRECISION`.
- Scientific evidence: W0 remains valid; W2-W4 values are invalidated diagnostics; no three-seed or TEST claim.
- Engineering status: `ENGINEERING_ACCURACY_CRITERION_NOT_SPECIFIED`.

## Executed results

| Stage | Result | Evidence boundary |
|---|---|---|
| Corrected v2 rescore | Early VLM -0.4161 MPa, CI [-1.6645, 0.8974]; feedback +0.1468, CI [-0.3343, 0.6140]; fixed baseline +0.8584, CI [-1.4455, 3.0866] | All three 98.3333% intervals cross zero: `NOT_SUPPORTED` |
| Cohort | 276 physical specimens, 259 capture groups; split 161/50/65 over six domains | Reused internal cohort, not untouched external confirmation |
| Predictor snapshots | Provisional retained-snapshot winner `MEAN_SC`; full-input MAE 41.1409 MPa, RMSE 53.1760, R2 0.7187; zero-input MAE 58.2584; prefix area 48.3126 | Checkpoint update identity is unverified under exact cost; not a valid `P_all` |
| Frozen VLM | 211 TRAIN/VALID specimens requested; 205 valid, 6 unavailable; 167 cumulative v3 calls including 12 failed calls | Surface priority cue, not CAI information-gain ground truth |
| W3 retained diagnostic | Main 44.6276; `GEOMETRY_SPREAD` 46.9671; open-loop 44.8529; no-VLM early 49.8969; main early 48.9080 | Invalidated by upstream W2 checkpoint-selection failure; no policy evidence |
| Structure retained diagnostic | `VLM_MEAN_FEEDBACK` 44.2955 vs spatial 44.6276 | Invalidated; even diagnostically it does not favor the spatial structure |
| GDFS retained diagnostic | Hard VALID area 46.3356; frozen predictor hashes unchanged | Adapter mechanism check remains reproducible; performance evidence is invalidated |
| Seed expansion / TEST | Upstream not ready and `RESOURCE_LIMITED`; TEST VLM calls 0; TEST labels not accessed | No formal three-seed effect, bootstrap interval or TEST performance claim |

## Allowed claims

1. The left-step metric correction changes neither of the three old v2 conclusions: all remain unsupported.
2. The full 276-specimen cohort, source-derived capture groups, frozen features, and real frozen-VLM interface were implemented and audited.
3. The retained W2-W4 artifacts can be used only to reproduce and diagnose the invalidated execution; their numerical ordering is not scientific evidence.
4. The grouped-Concrete implementation leaves predictor parameters unchanged and evaluates hard acquisition, but its observed performance is invalidated upstream.

## Prohibited claims

- Do not call `MEAN_SC` a valid common predictor or claim that W3 passed its continuation gate.
- Do not cite W3/GDFS retained values as positive, negative, comparative, or statistically confirmed evidence.
- Do not claim external generalization, deployment readiness or engineering accuracy.
- Do not claim that the spatial Actor is superior to the mean-feedback structure.
- Do not claim GDFS is an exact reproduction of Covert et al. or superior to the main policy.
- Do not treat VLM regions as CAI ground truth or the reused TEST role as untouched confirmation.
