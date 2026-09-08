# BC C-scan Path-B Supplement Results and Claim Boundaries

## Evidence status

This retrospective supplement is complete under `PROXY_LEGACY`. Independently reviewed C-scan references remain 0/60 for the reused pilot cohort and 0/24 for the prepared confirmation cohort. All `REVIEWED_V1` effects therefore remain null. The scientific status is `BC_PLANNING_SUPPORTED_STOP_NOT_SUPPORTED_PROXY_ONLY`.

## Planner-only result

The three-seed BC mean AUSC is 0.726878 for LOCATE and 0.508254 for CHARACTERIZE. The corresponding P4 values are 0.618495 and 0.462514; P8 values are 0.629819 and 0.467930.

Domain-balanced paired effects, with seeds averaged within physical specimen, are:

| Task | Comparison | BC minus rule | 95% CI | 97.5% CI | Positive domains |
|---|---|---:|---:|---:|---:|
| LOCATE | BC vs P4 | 0.108383 | [0.031492, 0.189060] | [0.020994, 0.200318] | 6/6 |
| LOCATE | BC vs P8 | 0.097059 | [0.021459, 0.175436] | [0.012096, 0.187249] | 6/6 |
| CHARACTERIZE | BC vs P4 | 0.045740 | [0.025064, 0.066330] | [0.021526, 0.069452] | 5/6 |
| CHARACTERIZE | BC vs P8 | 0.040324 | [0.018870, 0.061449] | [0.015609, 0.064537] | 5/6 |

Finite-seed variation is material. Against P8, seed-specific LOCATE effects are 0.180136, 0.088574, and 0.022467; CHARACTERIZE effects are 0.080889, 0.048166, and -0.008084. The algorithm-level interval is conditional on these three trained seeds; seeds are not extra physical specimens.

## Actor-input ablations

At seed 1, full BC minus `BC_NO_VLM` is 0.080795 for LOCATE and 0.058830 for CHARACTERIZE. The 97.5% intervals are [0.030208, 0.135362] and [0.003782, 0.118155].

Full BC minus `BC_NO_US_FEEDBACK` is 0.459773 for LOCATE and 0.441817 for CHARACTERIZE. The 97.5% intervals are [0.380911, 0.536593] and [0.343865, 0.543677].

These results isolate inputs available to the Actor. The common Reader and STOP still use their registered visible inputs; the ablations do not establish a whole-system no-VLM or no-ultrasound comparison.

## STOP and Path-B result

The frozen old STOP head had no qualifying VALID threshold, so the allowed one-time same-architecture conditional refit was used. The refitted 13,697-parameter head used 1,152 TRAIN fit states and 384 internal-validation states, stopped after 2,500 optimizer steps, and locked threshold 0.99 for both tasks before TEST.

On TEST with S_BC_CAL:

| Task | Planner | Completion | False-stop episode rate | Exhaustion | Failure-penalized cost |
|---|---|---:|---:|---:|---:|
| LOCATE | P8 | 0.958333 | 0.041667 | 0.000000 | 0.687637 |
| LOCATE | BC three-seed mean | 0.958333 | 0.041667 | 0.000000 | 0.686657 |
| CHARACTERIZE | P8 | 0.875000 | 0.000000 | 0.125000 | 0.813523 |
| CHARACTERIZE | BC three-seed mean | 0.902778 | 0.069444 | 0.027778 | 0.792728 |

The fixed 97.5% familywise Path-B effects are:

| Task | BC minus P8 completion | Completion CI | P8 minus BC cost | Cost CI | Joint decision |
|---|---:|---:|---:|---:|---|
| LOCATE | 0.000000 | [-0.055556, 0.069444] | 0.000979 | [-0.035969, 0.041334] | Not supported |
| CHARACTERIZE | 0.027778 | [-0.069444, 0.152778] | 0.020795 | [0.005284, 0.036780] | Not supported |

LOCATE misses both the -0.05 completion lower-bound margin and positive cost lower bound. CHARACTERIZE has a positive cost interval but misses the completion noninferiority margin. Neither task therefore satisfies the predeclared joint Path-B standard.

S_BC_CAL also does not dominate S_RULE. It raises LOCATE completion but stops later, increasing failure-penalized cost for both BC and P8; CHARACTERIZE STOP increments are imprecise. The old S_OLD_090 results remain historical controls rather than evidence for the refitted head.

## True-break and resources

Actual true-break execution was run for BC seed 1 and P8, both tasks, on one hash-selected TEST specimen per domain: 24 episodes in total. Every terminal state and action prefix matched its cached full trajectory; post-STOP world steps were zero.

The corresponding execution measurements are:

| Task | Planner | Episodes | Terminal cost | Normalized route | Route turns | Planner time (s) | STOP time (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| LOCATE | BC seed 1 | 6 | 0.655031 | 1.199953 | 9964.0 | 0.462715 | 0.226905 |
| LOCATE | P8 | 6 | 0.634941 | 1.184429 | 9860.0 | 0.121406 | 0.165516 |
| CHARACTERIZE | BC seed 1 | 6 | 0.727395 | 1.315306 | 10943.3 | 0.466701 | 0.207869 |
| CHARACTERIZE | P8 | 6 | 0.764199 | 1.371073 | 11374.7 | 0.125594 | 0.237861 |

Times are accumulated process measurements for these runs, not hardware scan time. Normalized route is the registered image-plane proxy and may exceed one. The original-sample VLM calls were served from the frozen cache; zero new calls must not be interpreted as zero deployment cost.

Total new optimizer updates were 7,500/20,000. Total supplement world transitions were 105,208/180,000. The reused original 60 specimens required zero new VLM calls. The prepared confirmation cohort used zero world transitions and zero VLM calls because independently reviewed references are pending.

## Permitted claims

- The BC planning signal is positive relative to registered P4 and P8 rules under the same-Reader proxy evaluation.
- Surface cues and new ultrasound content provide positive seed-1 Actor-input increments under the proxy evaluation.
- The first-stop execution implementation agrees with cached prefix evaluation on the six-domain audit sample.

## Claims not supported

- No reviewed-reference effect or formal damage-ground-truth claim.
- No Path-B autonomous improvement claim for either task.
- No population safety guarantee from observed false-stop rates.
- No hardware scan-time, force-control, coupling, or external-material generalization claim.
- No claim that the three trained seeds characterize all optimization randomness.
