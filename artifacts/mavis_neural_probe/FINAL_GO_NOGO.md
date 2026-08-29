# Spatial Neural Probe Final GO/NO-GO

## Final Decision

`NEURAL_PROBE_NO_GO`

The fixed Spatial CNN candidate does not satisfy the registered evidence chain.
The integration gate passes, but N1 does not establish a representation gain;
N2, N3, and N4 also fail their registered gates. No AEI manuscript change and
no N5 candidate-local readout are authorized.

## Registered Questions

| Question | Answer | Evidence |
| --- | --- | --- |
| Q1. Does the Spatial CNN improve P2 partial-state representation? | No | DeepSets minus Spatial AUEBC is `-0.0003210289`, 95% CI `[-0.0067107306, 0.0060540032]`, favorable in `3/6` domains. Gate: `REPRESENTATION_NO_GO`. |
| Q2. Does an improvement propagate to P3 dynamic value estimation? | No | DeepSets minus Spatial next-action regret is `0.0000029490`, 95% CI `[-0.0001244385, 0.0001298122]`, favorable in `2/6` domains. Gate: `VALUE_NO_GO`. |
| Q3. Is an improvement attributable to specimen-specific measured content rather than acquired position/history? | No | The pre-registered `CLEAN_NONPRIV = {uniform, random}` analysis does not support both real-versus-positions/history and real-versus-shuffled contrasts at both P2 and P3. Gate: `CONTENT_NO_GO`. |
| Q4. Does an improvement propagate to end-to-end CAI AUEBC? | No | Static-reference minus Spatial-candidate AUEBC is `0.0000012516`, 95% CI `[-0.0065366288, 0.0063906428]`, favorable in `3/6` domains. Gate: `END_TO_END_NO_GO`. |
| Q5. Is the evidence sufficient to change the current AEI paper claim? | No | The registered strong evidence chain is not met. Protected paper sources and canonical claims remain unchanged. |

Lower AUEBC and regret are better. The positive N2 and N4 point estimates are
too small and too uncertain to constitute registered improvements; neither is
reported as superiority.

## Stage Evidence

| Stage | Primary comparison | Point | Paired 95% CI | Favorable domains | Gate | Output manifest SHA-256 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| N0 | Integration and leakage audit | all required checks pass | n/a | n/a | `N1_AUTHORIZED` | `e81818487e93a455c7257fffc3f79661a43d94a1930d082fcbf931016858801a` |
| N1 | DeepSets real AUEBC minus Spatial real AUEBC | `-0.0003210289` | `[-0.0067107306, 0.0060540032]` | `3/6` | `REPRESENTATION_NO_GO` | `e504642131a738172e649bb36e2975b142de45ce09f6cfa00ab9e2f55540d570` |
| N2 | DeepSets regret minus Spatial regret | `0.0000029490` | `[-0.0001244385, 0.0001298122]` | `2/6` | `VALUE_NO_GO` | `d5f2a0b14bae320f67745ccb61cc15d5144fab29a47639343b86de467939cfa4` |
| N3 | Real content versus registered controls | mixed/adverse | see package | mixed | `CONTENT_NO_GO` | `724faa848377e7206f222d86364ee3efd0e043d5da40b3d4e1abf72173edc5f0` |
| N4 | Static-reference AUEBC minus Spatial-candidate AUEBC | `0.0000012516` | `[-0.0065366288, 0.0063906428]` | `3/6` | `END_TO_END_NO_GO` | `270959dcc00f5354fb03e5d3e19f85cee34a64f63ae52b2168f9d86056227037` |

N1 real AUEBC is `0.1250432019` for DeepSets and `0.1253642308`
for Spatial. N2 real next-action regret is `0.0122191085` for DeepSets
and `0.0122161595` for Spatial. N4 AUEBC is `0.1249920401` for the
frozen static reference, `0.1249907885` for the Spatial candidate, and
`0.1250531822` for the frozen learned implementation.

## Content Attribution

The headline clean, non-privileged Spatial contrasts are:

| Layer | Control minus real | Point | 95% CI | Favorable domains |
| --- | --- | ---: | ---: | ---: |
| P2 | positions/history | `-0.0030174325` | `[-0.0122995304, 0.0064825516]` | `2/6` |
| P2 | shuffled | `-0.0015067303` | `[-0.0096817885, 0.0062555787]` | `3/6` |
| P3 | positions/history | `-0.0002325869` | `[-0.0004068854, -0.0000641154]` | `2/6` |
| P3 | shuffled | `-0.0001872229` | `[-0.0003397380, -0.0000272111]` | `2/6` |

Positive values favor real measured content under the unified reporting sign.
These results do not support specimen-specific content gain.

## Failure Localization

The registered localization is `C: representation no gain`. The implementation
failure hypothesis is not supported: legal-state, fold-isolation, deterministic
training, checkpoint, artifact, rollout, and exact-cost gates pass. Because N1
fails, the protocol forbids escalation directly to a Transformer, GNN, larger
network, or N5. Downstream N2-N4 no-go results are observations, not evidence
authorizing a different causal failure label.

## Integration And Leakage Gate

- The Spatial encoder consumes only registered `64 x 6` partial-state tokens,
  the `64`-cell acquisition mask, context, and exact cost features.
- N1 and N2 each contain `144` source-only model-selection audit rows; the held-out
  outer domain appears in zero fit-domain rosters and every
  `target_data_used_for_selection` value is false.
- N4 records `276` specimens, six domains, and six checkpoints. Each of
  `spatial_probe`, `mvd_m1_o2`, and `mavis_full` has exactly `1,656` aligned
  prediction rows.
- The Spatial rollout contains `8,450` causal action rows. Exact costs increase
  monotonically, the cost chain has zero discontinuities, and all rows use
  feedback through the existing `mavis_causal_rollout`.
- State hashes have zero discontinuities within a checkpoint. The registered
  rollout rematerializes state at each of `1,335` checkpoint boundaries while
  preserving the exact cost and action history.
- No target CAI, future C-scan, unrevealed measurement, full-field image, or
  target-domain selection input is available to deployed scoring.

## Scientific Integrity

- Base commit: `9794d53a9549f2e3501fe482e8db8735f468ba20`.
- Registered neural-probe config SHA-256:
  `26299353b1870081660af452cb7e8ca3ba0b4fe87666f26c3922ef1a9ef7fc66`.
- Canonical metrics SHA-256:
  `f0d2615637a6470744f275a2ac6e1c5e7aff110ca7e31cb323793c29405be4e6`.
- Frozen MAVIS config SHA-256:
  `e99b47e161663fdaefe28719d16321a010f95b4ad8cf8f506a6e18d1d7f57b9d`.
- Frozen P2, P3, P7, and science-closure Git trees remain exactly
  `f73038ab616710c3953af82569241819fafb96d7`,
  `36866c1f4f351ef9d0d3915a24c71dade6807b65`,
  `b7fb24ff2d808db6fd8ec4f6571daef55016b96c`, and
  `f57c92f5b629745746decd6e005d7b7b152b491a`.
- The protected result-root diff from the base commit is empty.
- The protected paper-source and canonical-claim diff from the base commit is
  empty.
- Exploratory Spatial P2/P3 models were trained under the fixed protocol. No
  frozen model was retrained and no historical endpoint was recomputed or
  reselected.

## Verification Evidence

- `python -m ruff check src/cmc_bbdm/mavis tests`: passed.
- Full MAVIS regression: `295 passed in 132.31s`.
- Full MVD regression: `29 passed in 52.96s`.
- AEI paper smoke/contracts: `122 passed in 85.47s`.
- Target isolation and deterministic rerun checks: `6 passed in 14.29s`.
- N0 checksum ledger: all five entries passed `sha256sum -c`.
- N1-N4 artifact manifests: verified, covering `158`, `156`, `6`, and `12`
  scientific files, respectively.
- Frozen P2/P3/P7 package verifiers: passed, covering `160`, `154`, and `29`
  scientific files, respectively.
- `git diff --check`: passed with this audit file present.

## Authorized Next State

- Preserve the current AEI framework/calibration claim and all adverse evidence.
- Do not create `PAPER_INTEGRATION_RECOMMENDATION.md`.
- Do not modify the manuscript, canonical metrics, figures, or tables.
- Do not proceed to N5 or enlarge the neural architecture under this protocol.
