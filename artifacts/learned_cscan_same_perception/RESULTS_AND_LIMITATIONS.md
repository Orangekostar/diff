# Learned C-scan Same-Perception Pilot: Results and Limitations

## Outcome

The pilot completed real training and frozen TEST outcome evaluation, but it does not support the main learned-planning hypothesis. All outcome labels are `SAME_READER_SELF_CONSISTENCY` proxies because none of the 60 references has independent review. Formal effects and formal confidence intervals are therefore null.

| Status | Result |
|---|---|
| Execution | `TRAINED_AND_EVALUATED` |
| Reference | `PROXY_ONLY` |
| Reader v2 | `FUNCTIONAL` (`72/72` nondegenerate TRAIN/VALID full-input task readouts) |
| Proxy planner effect | `NOT_SUPPORTED` |
| Formal planner effect | `INCONCLUSIVE` |
| Learned STOP | VALID-authorized at 0.90; TEST effect `INCONCLUSIVE` |
| VLM increment | `NOT_TESTED` because the VALID planning signal was negative |

## Data and perception

The frozen 60-specimen roster contains six domains and is split by physical specimen into 24 TRAIN, 12 VALID, and 24 TEST cases. The surface backend is `Qwen/Qwen2.5-VL-7B-Instruct` revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`. The task-independent surface cache contains one percept per specimen. Cumulative deployment inference used 62 calls: 60 initial specimen calls and two one-shot format repairs. The 24-call diagnostic allowance was exhausted during schema debugging; blank-surface and display-number permutation diagnostics were not run and are not claimed. The corrected full-input Reader v2 diagnostic contains 72/72 nondegenerate TRAIN/VALID task readouts.

## Training

The actor has 329,505 trainable parameters and the STOP head has 13,697. The base TRAIN bank contains 192 states and 916 candidate suffixes. One aggregation round added 96 states and 541 candidate suffixes. Training-side construction used 179,206 logical transitions; the entire pilot, including full VALID and TEST rollouts, used 285,190 logical transitions against the 400,000 cap.

| Model | Optimizer steps | Initial loss | Final loss | Best internal-valid loss |
|---|---:|---:|---:|---:|
| `L_BC` | 1,250 | 4.049588 | 0.453555 | 2.512921 |
| `L_CTG_0` | 3,750 | 1.562553 | 1.526504 | 1.556876 |
| `L_CTG_1` | 1,250 | 1.615381 | 1.596169 | 1.618729 |
| `S_LEARN` | 3,750 | 0.720053 | 0.280019 | 0.331856 |

The original BC bank encoded only the chosen action, which made its one-element softmax loss identically zero. Before the final training run and before VALID/TEST access, the labels were deterministically expanded to a one-hot target over every legal cell using the recorded behavior action and stored legal mask. The CTG targets and candidate suffixes were not recomputed.

## VALID selection

VALID selected `R_BALANCED_P8` at mean `AUSC_any=0.623371` and `L_CTG_1` at `0.386862`; the learned-minus-rule signal was `-0.236508`. Consequently, seeds 2/3, `L_NO_VLM`, and `L_OPEN_INIT` were not run under the preregistered resource gate.

`L_BC` reached `0.752262` overall on VALID, compared with `0.623371` for the selected rule. This shows that the network can learn useful action structure, but the cost-to-go supervision used here did not improve on behavior cloning.

The learned STOP threshold 0.90 was authorized on VALID. Its state-level false-stop rate was 97/5,528 = 0.01755 with Wilson 95% interval [0.01441, 0.02136]. These states are correlated within 12 physical specimens, so this interval is descriptive and is not a safety guarantee.

## TEST planner-only results

| Method | LOCATE AUSC | LOCATE C@SR=.8/.9 | CHARACTERIZE AUSC | CHARACTERIZE C@SR=.8/.9 |
|---|---:|---:|---:|---:|
| `R_BALANCED` | 0.629819 | 0.8465 / 0.8699 | 0.467930 | 0.8888 / 0.8935 |
| `L_BC` | 0.809955 | 0.4089 / 0.4706 | 0.548819 | 0.7354 / 0.8895 |
| `L_CTG` | 0.413091 | 0.8602 / 0.9651 | 0.309835 | 0.9070 / 0.9192 |

All methods reached proxy success at full input, so the difference is ordering efficiency rather than endpoint readout availability.

| Task | Comparison | Mean AUSC difference | 95% paired bootstrap CI | Physical n | Domains |
|---|---|---:|---:|---:|---:|
| LOCATE | `L_CTG - R_BALANCED` | -0.216727 | [-0.337556, -0.089456] | 24 | 6 |
| CHARACTERIZE | `L_CTG - R_BALANCED` | -0.158095 | [-0.219955, -0.098379] | 24 | 6 |
| LOCATE | `L_CTG - L_BC` | -0.396863 | [-0.479054, -0.305049] | 24 | 6 |
| CHARACTERIZE | `L_CTG - L_BC` | -0.238984 | [-0.305228, -0.174322] | 24 | 6 |

The CTG-minus-rule direction was negative in five of six domains for both tasks. The sole positive domain was `cgtnjyggtm` (+0.274334 LOCATE, +0.049408 CHARACTERIZE). Domain-level CTG-minus-rule differences were:

| Domain | LOCATE | CHARACTERIZE |
|---|---:|---:|
| `74t7kcdgkr` | -0.359896 | -0.293725 |
| `cgtnjyggtm` | +0.274334 | +0.049408 |
| `w68dtmpfyf` | -0.460088 | -0.247878 |
| `xcmzfsbd9t` | -0.216404 | -0.226764 |
| `yfxyg8jm46` | -0.204028 | -0.162871 |
| `ykhs7s2dck` | -0.334282 | -0.066741 |

## STOP outcomes

With `S_LEARN`, `L_CTG` completed 37.5% of LOCATE and 62.5% of CHARACTERIZE TEST episodes. It exhausted the resource endpoint in 62.5% and 37.5%, respectively, with no recorded false stops. `R_BALANCED + S_LEARN` completed 58.3%/66.7%, falsely stopped 29.2%/8.3%, and exhausted 12.5%/25.0%. The threshold was VALID-authorized, but TEST completion is insufficient to claim a positive STOP increment.

With `S_rule`, `L_CTG` completed 41.7% of LOCATE and 100% of CHARACTERIZE episodes; 58.3% of LOCATE episodes were false stops under the proxy. `R_BALANCED + S_rule` completed 87.5% for both tasks and falsely stopped 12.5% for each. STOP outcomes therefore do not rescue the CTG planning result.

| Planner / task / stop | Completion | False stop | Exhaustion | Mean stopped cost | Autonomous AUSC | Failure-penalized cost |
|---|---:|---:|---:|---:|---:|---:|
| `R_BALANCED` LOCATE / `S_rule` | 0.8750 | 0.1250 | 0.0000 | 0.2905 | 0.6162 | 0.3838 |
| `R_BALANCED` LOCATE / `S_learn` | 0.5833 | 0.2917 | 0.1250 | 0.1245 | 0.4882 | 0.5118 |
| `R_BALANCED` CHARACTERIZE / `S_rule` | 0.8750 | 0.1250 | 0.0000 | 0.7104 | 0.2240 | 0.7760 |
| `R_BALANCED` CHARACTERIZE / `S_learn` | 0.6667 | 0.0833 | 0.2500 | 0.7842 | 0.1292 | 0.8708 |
| `L_CTG` LOCATE / `S_rule` | 0.4167 | 0.5833 | 0.0000 | 0.5537 | 0.1408 | 0.8592 |
| `L_CTG` LOCATE / `S_learn` | 0.3750 | 0.0000 | 0.6250 | 0.9923 | 0.0029 | 0.9971 |
| `L_CTG` CHARACTERIZE / `S_rule` | 1.0000 | 0.0000 | 0.0000 | 0.9627 | 0.0373 | 0.9627 |
| `L_CTG` CHARACTERIZE / `S_learn` | 0.6250 | 0.0000 | 0.3750 | 0.9000 | 0.0625 | 0.9375 |

The final audit replayed one TEST specimen per domain for both tasks. All 2,316 rows matched the frozen TEST values across 26 deterministic fields. The replay record adds top-five learned logits, evidence deltas, and normalized route start/end positions without changing the original metrics.

## Interpretation and limits

Within the frozen common-perception interface, the cost-sensitive `L_CTG` agent is less effective than the selected human-designed rule on both proxy tasks. Behavior cloning is stronger than both on mean TEST AUSC, so the observed failure is specific to the queried cost-to-go learning route rather than proof that learned C-scan planning is impossible.

This conclusion is limited to a retrospective reused cohort, same-reader proxy labels, one seed after a negative VALID gate, and normalized image-plane acquisition/route costs. The first W1 export also computed non-outcome Reader v2 diagnostics on TEST before the VALID lock; those rows were not used for fitting or selection, and the corrected artifact is TRAIN/VALID-only. There is no independently reviewed damage reference, prospective scanner execution, physical travel-time model, causal damage validation, new-material transfer test, or statistical basis for a formal detection claim. Independent review can rescore the frozen TEST reports, but changing training labels or reader calibration requires a new versioned run.
