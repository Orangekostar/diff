# Damage-to-Failure Response Literature Novelty Ledger

Audit date: 2026-08-30  
Search boundary: public bibliographic metadata, publisher records, author-hosted
manuscripts, and official data-repository records available on the audit date.  
Source policy: primary records only for retained evidence. Search-result wording
was used only to locate a primary record. Missing details are recorded as
`UNKNOWN_FROM_ACCESSIBLE_PRIMARY_SOURCE`; they are not inferred.

## Preregistered search families

The same-day search covered these eight public topic families:

1. pre-CAI C-scan or ultrasonic damage to a full compression-response curve;
2. post-impact NDE to CAI load-displacement or stress-strain response;
3. composite image or microstructure to an entire stress-strain curve;
4. CAI failure response monitored by AE, IRT, DIC, or strain gauges;
5. progressive finite-element impact-to-CAI response simulation;
6. machine-learning prediction of CAI strength and damage descriptors;
7. theory-guided or transfer-learning reconstruction of composite response;
8. experimental cross-domain prediction of post-damage mechanical curves.

DOIs and normalized titles were used for deduplication. The search establishes
the closest accessible work; it does not establish absence from the literature.

## Retained primary works

| Work | Evidence type and material | Input | Output / endpoint | Sample count | Split protocol | Full response? | Pre-CAI-only input? | Exact boundary relative to this route |
|---|---|---|---|---:|---|---|---|---|
| [Mack et al., AEI 72, 104518 (2026)](https://doi.org/10.1016/j.aei.2026.104518) | Experimental composite C-scan | Full and damage-focused C-scan images | Impact energy and ultimate CAI strength scalars; SHAP interpretation | 1,428 augmented images; unique specimens `UNKNOWN_FROM_ACCESSIBLE_PRIMARY_SOURCE` | `UNKNOWN_FROM_ACCESSIBLE_PRIMARY_SOURCE` | No | Yes | Closest direct data route, but it predicts scalars rather than subsequent compression-response descriptors or curves. C-scan-to-CAI-strength is not a novelty claim here. |
| [Hasebe et al., Composites Part A 189, 108560 (2025)](https://doi.org/10.1016/j.compositesa.2024.108560) | Experimental CFRP | Post-impact surface profiles | Ultimate CAI strength scalar | Exact modeling subset `UNKNOWN_FROM_ACCESSIBLE_PRIMARY_SOURCE` | 75/25 holdout and four-fold cross-validation are reported | No | Yes | Surface-profile-to-CAI-strength is established; the present question concerns information beyond the strength scalar under strict held-out-domain evaluation. |
| [Yang et al., Materials & Design 189, 108509 (2020)](https://doi.org/10.1016/j.matdes.2020.108509) | Finite-element composite microstructures | Binary microstructure images | 61-point stress-strain curves | 100,000 simulated microstructures | 95/5 train/test | Yes | Not a post-impact CAI setting | Establishes image-to-full-curve prediction in composites, but not experimental pre-CAI damage observations or cross-domain CAI response. |
| [Liu et al., Composites Part A 188, 108574 (2025)](https://doi.org/10.1016/j.compositesa.2024.108574) | Simulated unidirectional composite response, validated against experiments | Fiber-misalignment micrographs | Longitudinal compressive response curves | More than 15,000 simulated curves; validation against 83 experiments | `UNKNOWN_FROM_ACCESSIBLE_PRIMARY_SOURCE` | Yes | Not a post-impact CAI setting | Establishes micrograph-to-compressive-response learning, but not post-impact NDE-to-CAI response or six-domain transfer. |
| [Cai et al., Polymer Composites 47, 11808-11825 (2026)](https://doi.org/10.1002/pc.71057) | Experimental carbon-fiber composite CAI monitoring | AE, IRT, DIC, load-displacement, and fracture morphology acquired during/after CAI | Failure-mechanism and damage-tolerance correlation | `UNKNOWN_FROM_ACCESSIBLE_PRIMARY_SOURCE` | Not a predictive split; exact specimen design `UNKNOWN_FROM_ACCESSIBLE_PRIMARY_SOURCE` | Observes response | No | Supplies mechanism evidence using measurements acquired during CAI; it does not test prediction from pre-CAI observations alone. |
| [Yang et al., IJMS 307, 110888 (2025)](https://doi.org/10.1016/j.ijmecsci.2025.110888) | Progressive finite-element CFRP impact-to-CAI simulation with experimental comparisons | Simulated impact damage state | CAI force-displacement/failure response and strength | Four impact-energy scenarios, at least three experimental specimens per scenario | Not an ML split | Yes | Simulation starts from impact state | It is a progressive simulation framework, not direct experimental NDE-to-response learning or strict cross-domain validation. |
| [Mezeix et al., Journal of Composite Materials (2026)](https://doi.org/10.1177/00219983261464288) | CFRP, machine learning plus classical laminate theory | Laminate/impact descriptors | Delamination, indentation, perforation, and CAI-strength scalars | 500 samples | `UNKNOWN_FROM_ACCESSIBLE_PRIMARY_SOURCE` | No | Yes for its reported scalar predictors | Shows that expanding the list of damage and strength scalars is already crowded; it does not reconstruct response shape. |
| [Du et al., Composites Part B 325, 113996 (2026)](https://doi.org/10.1016/j.compositesb.2026.113996) | Experimental CFRP acoustic emission | Separate AE streams from impact and the subsequent CAI test | Real-time CAI factor | `UNKNOWN_FROM_ACCESSIBLE_PRIMARY_SOURCE` | Transfer-learning protocol; exact partition `UNKNOWN_FROM_ACCESSIBLE_PRIMARY_SOURCE` | Time-varying factor, not a full mechanical curve | No | It uses answer-side CAI-stage sensing, which is forbidden in the proposed pre-CAI-only input boundary. |
| [Lu et al., Composites Science and Technology 213, 108952 (2021)](https://doi.org/10.1016/j.compscitech.2021.108952) | Thermoplastic composite experiments plus finite elements | Pre-CAI X-ray CT damage region | Experiment-calibrated CAI behavior and strength | `UNKNOWN_FROM_ACCESSIBLE_PRIMARY_SOURCE` | Not an ML split | Yes, through simulation | Partly; CAI tests calibrate damaged stiffness | This is the closest full-response conceptual precedent. The remaining distinction is direct pre-CAI observation-to-response inference, strict domain holdout, and no CAI-stage input; it does not authorize a first-ever claim. |
| [Zobeiry et al., Composite Structures 246, 112407 (2020)](https://doi.org/10.1016/j.compstruct.2020.112407) | Theory-guided finite elements and neural networks for quasi-isotropic laminates | Macroscopic load-displacement observations | Damage/strain-softening characterization | `UNKNOWN_FROM_ACCESSIBLE_PRIMARY_SOURCE` | `UNKNOWN_FROM_ACCESSIBLE_PRIMARY_SOURCE` | Uses a response record | No | Learns damage-model properties from the mechanical response itself, rather than predicting a later response from pre-test NDE. |
| [Zhang et al., Materials & Design 218, 110700 (2022)](https://doi.org/10.1016/j.matdes.2022.110700) | Additively manufactured polymer composites | Material/process descriptors and source-domain curves | Flexural stress-strain curves under transfer learning | 162 source records, 27 target records, 3 target tests | Source/target transfer protocol | Yes | Not post-impact CAI | Demonstrates cross-material curve transfer, but not experimental post-impact observation-to-CAI response. |
| [Wu et al., Composites Science and Technology 282, 111673 (2026)](https://doi.org/10.1016/j.compscitech.2026.111673) | Simulated CFRP laminates with random gaps | Defect/laminate representation | Stress-strain curves via PCA and Transformer | `UNKNOWN_FROM_ACCESSIBLE_PRIMARY_SOURCE` | Five-fold cross-validation | Yes | Not post-impact CAI | Establishes another simulated composite full-curve route; it does not resolve experimental pre-CAI NDE transfer. |
| [Hasebe et al., Data in Brief 60, 111509 (2025)](https://doi.org/10.1016/j.dib.2025.111509) | Official experimental CFRP dataset authority | Impact, geometry, NDE, and CAI records | Raw CAI traces and published CAI strength | 446 CAI tests: 26 intact and 420 LVI-damaged | No predictive split | Provides full raw measurements | Contains pre-CAI records and post-CAI outputs | This record authorizes data identity and measurement semantics, not a novelty claim. |

## Screened but not retained as direct novelty evidence

| Screened family | Disposition |
|---|---|
| Scalar impact-damage or CAI-strength surrogates beyond the retained Mack, Hasebe, and Mezeix records | Topic overlap only; no stronger full-response/pre-CAI match was supported by an accessible primary record. |
| In-situ DIC/AE/ultrasonic CAI onset and failure-monitoring studies | Mechanistically relevant but use information acquired during CAI, so they cannot support the deployable pre-CAI input route. |
| Fatigue progression and remaining-life work | Different loading history and endpoint. |
| Generic simulated impact-damage surrogates | No experimental pre-CAI observation-to-subsequent-response protocol. |
| Publisher results excluded by the operator source policy, including MDPI search hits | Excluded from retained evidence and not used to support any claim. |
| Reviews, aggregators, and citation-index pages | Used at most for discovery; not accepted as evidence. |

## Bounded interpretation

The closest accessible boundaries are: Mack supplies direct pre-CAI C-scan but
only scalar endpoints; Lu supplies pre-CAI CT and a simulated full CAI response
but depends on experiment-calibrated mechanics; Du tracks response using
CAI-stage AE; and Yang, Liu, Zhang, and Wu establish full-curve learning outside
the proposed experimental post-impact CAI setting. No title/abstract search can
prove that the exact route is absent, so no direct novelty claim is authorized.

Novelty status: SEARCHED_NOT_ASSUMED  
Direct novelty claim authorized: NO  
P0/P1 scientific question retained: YES  
Reason: closest work determines differentiation, while empirical gates determine support.
