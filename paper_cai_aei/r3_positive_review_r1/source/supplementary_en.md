<!-- block: S001 -->

# Supplementary material

<!-- block: S002 -->

This supplement accompanies the revised manuscript on region selection and acquisition order for CAI assessment. All numerical results, intervals, trajectories and diagnostic arrays were already saved. This revision performs deterministic source binding, formatting and plotting only. No optimizer update, research-model forward, new bootstrap draw, seed, reserved-partition evaluation or checkpoint selection was performed.

<!-- block: S003 -->

# S1. Data identity, partitions and evaluation units

<!-- block: S004 -->

The six paired-image collections and strength labels are the existing references [1,2,9–15] in the main article. A specimen record binds its laminate configuration, specimen ID, registered image pair and CAI label. Capture groups were formed before splitting from shared source/crop identities, then retained within partitions. The split contains 276 physical specimens (161 training, 50 development validation and 65 reserved), with 152/48/59 capture groups. The reserved partition was not evaluated. The same 50 development specimens selected checkpoints and estimated effects; all intervals are conditional on that selection.

<!-- block: S005 -->

**Table S1.** The complete 50-specimen evaluation cohort. Specimens are identified by layer count, layup and specimen number; dimensions and area clocks remain in cohort_manifest.csv. The full 276-row partition metadata are provided as S1_all_partition_metadata.csv. Reading split labels is not reserved-set evaluation.

<!-- block: S006 -->

| Specimen | Capture group | CAI (MPa) |
| --- | --- | --- |
| 8-layer cross-ply / c8-12 | cg_306e4a499acc46c1 | 197.97145080566406 |
| 8-layer cross-ply / c8-13 | cg_f6941b7d17a6b65f | 266.5553894042969 |
| 8-layer cross-ply / c8-16 | cg_f6941b7d17a6b65f | 264.1634521484375 |
| 8-layer cross-ply / c8-2 | cg_8ffe4fd6a878d7f3 | 163.86158752441406 |
| 8-layer cross-ply / c8-21 | cg_49d6f1237b716b24 | 263.10400390625 |
| 8-layer cross-ply / c8-28 | cg_a983245f10699099 | 282.8260803222656 |
| 8-layer cross-ply / c8-3 | cg_8ffe4fd6a878d7f3 | 213.09927368164062 |
| 8-layer cross-ply / c8-39 | cg_ff43edde8af2ccdb | 168.51828002929688 |
| 8-layer cross-ply / c8-43 | cg_a37dc8a00cc334eb | 199.83413696289062 |
| 24-layer quasi-isotropic / q24-11 | cg_02fd3fc8caefe377 | 300.3553771972656 |
| 24-layer quasi-isotropic / q24-22 | cg_54890e18ff2b2669 | 415.8692932128906 |
| 24-layer quasi-isotropic / q24-24 | cg_39bf4ddc679c9a42 | 428.6219482421875 |
| 24-layer quasi-isotropic / q24-30 | cg_d3f7feef05d5a66a | 283.3366394042969 |
| 24-layer quasi-isotropic / q24-44 | cg_9ba3cd5fb00ea91f | 109.75041198730469 |
| 24-layer quasi-isotropic / q24-47 | cg_523711040788bde2 | 119.86738586425781 |
| 24-layer quasi-isotropic / q24-48 | cg_5549ad4618fa66a9 | 136.6800079345703 |
| 24-layer quasi-isotropic / q24-6 | cg_344fc945dcb88757 | 348.28472900390625 |
| 24-layer quasi-isotropic / q24-9 | cg_2cc9cac1c5baaf14 | 262.2107238769531 |
| 16-layer quasi-isotropic / q16-14 | cg_73ef27ab04a735de | 404.2454528808594 |
| 16-layer quasi-isotropic / q16-2 | cg_ff2d01cab2de3ddc | 304.9983215332031 |
| 16-layer quasi-isotropic / q16-29 | cg_94a031a5fc453cba | 338.54833984375 |
| 16-layer quasi-isotropic / q16-3 | cg_3361913654df8b27 | 352.2940673828125 |
| 16-layer quasi-isotropic / q16-33 | cg_a37eab4f5b7beb12 | 257.2702331542969 |
| 16-layer quasi-isotropic / q16-38 | cg_a052ba6f5200ee4f | 344.6358337402344 |
| 16-layer quasi-isotropic / q16-43 | cg_6853c8e74a68016b | 302.281982421875 |
| 16-layer quasi-isotropic / q16-6 | cg_ce4d6a48a6968060 | 418.267333984375 |
| 24-layer cross-ply / c24-37 | cg_a01963fde121eda0 | 381.1910705566406 |
| 24-layer cross-ply / c24-39 | cg_d72ecf3171227da2 | 246.54722595214844 |
| 24-layer cross-ply / c24-42 | cg_a5c5defb812088ae | 301.55389404296875 |
| 24-layer cross-ply / c24-47 | cg_954e3ed61c8db0a4 | 480.5938415527344 |
| 24-layer cross-ply / c24-50 | cg_59e95410b683945f | 292.1376647949219 |
| 24-layer cross-ply / c24-51 | cg_7e1c8f7865d73e78 | 263.2923583984375 |
| 24-layer cross-ply / c24-52 | cg_b98b3f9ca1bfb997 | 252.29762268066406 |
| 24-layer cross-ply / c24-56 | cg_06ec34425b30c2e6 | 230.2566680908203 |
| 24-layer cross-ply / c24-9t | cg_0d1994d0164384d4 | 278.2536315917969 |
| 16-layer cross-ply / c16-12 | cg_5a80e5179694ff19 | 268.03814697265625 |
| 16-layer cross-ply / c16-14 | cg_efd8928c6d17de81 | 508.2288818359375 |
| 16-layer cross-ply / c16-16 | cg_cd055af98e2ea119 | 499.259765625 |
| 16-layer cross-ply / c16-17 | cg_f64812b6759e7d2b | 512.5420532226562 |
| 16-layer cross-ply / c16-24 | cg_354aea2e84731e86 | 523.44140625 |
| 16-layer cross-ply / c16-39 | cg_514ca9d914207c05 | 234.76339721679688 |
| 16-layer cross-ply / c16-42 | cg_cb9c6f640f08aee7 | 322.6684265136719 |
| 16-layer cross-ply / c16-5 | cg_7f9fe4a18a151cd4 | 270.7127685546875 |
| 8-layer quasi-isotropic / q8-15 | cg_60c6812942ba2332 | 260.08502197265625 |
| 8-layer quasi-isotropic / q8-2 | cg_93e604617899fa5f | 219.15264892578125 |
| 8-layer quasi-isotropic / q8-22 | cg_ca08b9786e924e66 | 235.01319885253906 |
| 8-layer quasi-isotropic / q8-33 | cg_a841812423416444 | 215.7657928466797 |
| 8-layer quasi-isotropic / q8-34 | cg_f28b9720dbc7e026 | 235.4805145263672 |
| 8-layer quasi-isotropic / q8-37 | cg_a064df3d047f4519 | 248.33407592773438 |
| 8-layer quasi-isotropic / q8-8 | cg_8fd87d72a8d71b83 | 184.39810180664062 |

<!-- block: S007 -->

Random contributes five saved trajectories per specimen; the other five methods contribute one each. Thus 500 archived trajectories and 7,849 acquisition events are repeated observations of these 50 specimens, not additional independent samples. Fixed-final-set analysis adds seven conditions per specimen (original, reverse, five permutations), giving 350 conditions on the same cohort. Paired intervals resample capture groups within six source groups, carrying each group’s specimens together.

<!-- block: S008 -->

# S2. Model identity, representation and learning protocol

<!-- block: S009 -->

The frozen actor is SpatialCAIActor with use_vlm=False and use_feedback=True. The method key is `NO_VLM_SPATIAL_FEEDBACK`; its seed is 2026091302, selected update 1000 after 1250 completed updates. The frozen readout is MeanSCPredictor (MEAN_SC), selected at update 1750 after 2000 updates. The actor contains 378,978 parameters including its training value head. The model-identity lock records the checkpoint and state-dictionary identities and whether local checkpoint bytes could be rehashed. A manifest-only checkpoint verification is not represented as a new model load.

<!-- block: S010 -->

**Table S2.** Frozen architecture and observable-input contract.

<!-- block: S011 -->

| Component | Specification |
| --- | --- |
| Local encoder | Frozen ResNet18; independently crop each cell before encoding; 512 features per region |
| Grid / native area | 8 × 8 row-major, indices 0–63; edges rint(linspace(0, native dimension, 9)); integer pixels |
| Surface registration | Frozen orientation; clockwise 90° once; maximum edge 1024 before local extraction |
| Actor local projections | Surface 512 → 48; visible/learned-unknown internal 512 → 48 |
| Actor regional input (102) | 48 surface + 48 internal + 2 coordinates + 1 measured + 1 ordinal/64 + 2 zero VLM placeholders |
| Actor global input (6) | Acquired area, remaining area, standardized CAI, measured-cell fraction, 2 zero VLM placeholders |
| Actor contextualizer | 64 regional + 1 global tokens; width 128; 2 layers; 4 heads; feed-forward 256; dropout 0 |
| Actor heads | Concatenate local/global tokens → 128 → 1 score; separate global value head during training |
| Readout local cell | Two 512 → 64 affine projections; concatenate with mask and coordinates (131 inputs); 131 → 64 → 64 GELU |
| Readout pooling / output | Mean of all 64 cells + mean of measured cells + acquired area (129 inputs); 129 → 64 → 1; training-only target scaling |
| Readout missing input | Internal descriptor is zero-masked BEFORE affine projection, retaining bias; measured-cell mean is zero at empty mask |
| Readout order dependence | Set-based; no ordinal or history input. Same final set and area imply the same terminal input. |

<!-- block: S012 -->

The actor uses a learned unknown internal descriptor and an explicit measurement indicator. It never treats an unmeasured internal cell as known image evidence. Its two reserved regional channels and two reserved global channels are constant zero in this model. The predictor instead masks raw internal descriptors before its affine projection, so an unmeasured cell still contributes the projection bias to the all-cell summary. These are different missing-input mechanisms. Surface descriptors are available initially; measured CAI enters training costs or retrospective metrics only, never an execution decision.

<!-- block: S013 -->

Predictor mask training uses the frozen mixture: 10% empty masks; 60% masks with 1–16 cells drawn from random or fixed geometric orders; 20% random masks with 17–48 cells; 10% full masks. The standardized residual is fitted with Huber loss. Three capture-group cross-fitted training predictors provide policy cost: a training specimen is scored only by the predictor excluding its fold. The common evaluation predictor is fitted on the training partition. This description documents the completed training; the revision does not execute it.

<!-- block: S014 -->

Policy optimization used AdamW with learning rate $3 \times 10^{-4}$, weight decay $10^{-4}$, batch size 16 and gradient-norm limit 1. Training sampled feasible categorical actions. Evaluation used deterministic feasible-score argmax. Candidate checkpoints were compared every 250 updates on development process A; an earlier candidate was replaced only for an improvement exceeding $10^{-12}$. The following scores are the existing job history, not a new candidate comparison.

<!-- block: S015 -->

**Table S3.** Archived actor-selection history.

<!-- block: S016 -->

| Update | Development A (MPa) |
| --- | --- |
| 250 | 47.276 |
| 500 | 46.461 |
| 750 | 44.911 |
| 1000 | 43.597 |
| 1250 | 44.451 |

<!-- block: S017 -->

## S2.1. Derivation and cost-minimization sign

<!-- block: S018 -->

All 13 original numbered display equations remain in the main article; equation_map.csv records their locations. To derive Eqs. (9–10), substitute the signed error identity below into the interval sum in Eq. (7). Each completed event contributes to every later interval, with total length B − c_j including final carry. Unequal cell areas and negative decreases are retained.

<!-- block: S019 -->

$$e_t=e_0-\sum_{j=1}^{t}\Delta e_j,\qquad \sum_{k=j}^{T-1}(c_{k+1}-c_k)+(B-c_T)=B-c_j.$$

<!-- block: S020 -->

$$A(B)=e_0-\sum_{j=1}^{T}(1-c_j/B)\Delta e_j,\qquad J=(1+\lambda_T)e_0-\sum_{j=1}^{T}(1-c_j/B+\lambda_T)\Delta e_j.$$

<!-- block: S021 -->

The trajectory baseline e₀ is constant for a fixed specimen and predictor. An earlier favorable event has more remaining integration length, whereas an adverse event contributes negatively. At c_j = B the process contribution vanishes and the terminal weight λT = 0.25 remains. Equation (11) divides remaining process-plus-terminal cost by the training target scale. Equation (12) has a positive log-probability sign because it minimizes cost: above-baseline sampled cost reduces that action’s probability under gradient descent. The advantage is stopped for actor differentiation; actor terms sum decisions within an episode and average episodes, while squared-value errors and entropy average within episodes before batch aggregation. Total loss adds 0.5 times value loss and subtracts entropy weight, linearly reduced from 0.01 to zero.

<!-- block: S022 -->

# S3. Caps, acquired area, intervals and quality targets

<!-- block: S023 -->

A requested reporting cap queries the latest completed prediction at or below that cap on the single path planned for total B = 0.25. No policy is replanned for a smaller cap. Predictions are right-continuous step functions; the initial or final estimate is carried until the next event or total cap. MAE and RMSE pool physical specimens; Random averages absolute and squared losses within specimen before pooling. Process A and event-stage quantities average repeats within specimen, then specimens within source group, then the six source groups equally. These weights differ because development group sizes are 9,9,8,9,8,7.

<!-- block: S024 -->

**Table S4.** All five-cap results and actual acquired fractions (MAE in MPa). Full-precision RMSE and R² remain in same_cap_metrics.csv.

<!-- block: S025 -->

| Method | Cap | MAE | Mean area | Min area | Max area |
| --- | --- | --- | --- | --- | --- |
| Center-first | 0.0 | 58.551 | 0.000000 | 0.000000 | 0.000000 |
| Center-first | 0.0625 | 47.885 | 0.062350 | 0.061399 | 0.062407 |
| Center-first | 0.125 | 48.827 | 0.111012 | 0.109491 | 0.125000 |
| Center-first | 0.1875 | 49.094 | 0.172513 | 0.171900 | 0.187121 |
| Center-first | 0.25 | 48.027 | 0.234759 | 0.234676 | 0.235577 |
| Geometry-spread | 0.0 | 58.551 | 0.000000 | 0.000000 | 0.000000 |
| Geometry-spread | 0.0625 | 50.221 | 0.062018 | 0.061399 | 0.062130 |
| Geometry-spread | 0.125 | 46.189 | 0.124548 | 0.123164 | 0.124629 |
| Geometry-spread | 0.1875 | 46.555 | 0.187336 | 0.186034 | 0.187407 |
| Geometry-spread | 0.25 | 46.910 | 0.249970 | 0.249269 | 0.250000 |
| Serpentine | 0.0 | 58.551 | 0.000000 | 0.000000 | 0.000000 |
| Serpentine | 0.0625 | 49.731 | 0.062379 | 0.062130 | 0.062407 |
| Serpentine | 0.125 | 48.773 | 0.124592 | 0.124260 | 0.124629 |
| Serpentine | 0.1875 | 48.343 | 0.186805 | 0.186391 | 0.186851 |
| Serpentine | 0.25 | 48.013 | 0.249184 | 0.248521 | 0.249258 |
| Random | 0.0 | 58.551 | 0.000000 | 0.000000 | 0.000000 |
| Random | 0.0625 | 53.156 | 0.053603 | 0.046897 | 0.062500 |
| Random | 0.125 | 49.153 | 0.118768 | 0.109307 | 0.125000 |
| Random | 0.1875 | 48.108 | 0.186646 | 0.171598 | 0.187500 |
| Random | 0.25 | 46.886 | 0.246943 | 0.234492 | 0.250000 |
| Learned-static | 0.0 | 58.551 | 0.000000 | 0.000000 | 0.000000 |
| Learned-static | 0.0625 | 46.465 | 0.062401 | 0.062130 | 0.062500 |
| Learned-static | 0.125 | 48.064 | 0.124803 | 0.124260 | 0.125000 |
| Learned-static | 0.1875 | 46.784 | 0.187400 | 0.187130 | 0.187500 |
| Learned-static | 0.25 | 46.993 | 0.235297 | 0.234674 | 0.249991 |
| Proposed | 0.0 | 58.551 | 0.000000 | 0.000000 | 0.000000 |
| Proposed | 0.0625 | 43.324 | 0.057700 | 0.046897 | 0.062500 |
| Proposed | 0.125 | 43.164 | 0.119824 | 0.109120 | 0.124998 |
| Proposed | 0.1875 | 42.928 | 0.182555 | 0.171415 | 0.187407 |
| Proposed | 0.25 | 42.385 | 0.247174 | 0.234492 | 0.250000 |

<!-- block: S026 -->

**Table S5.** Saved paired differences, proposed minus control. Negative error differences favor proposed; these are existing 95% intervals, conditional on selection.

<!-- block: S027 -->

| Control | Metric | Estimate | 95% low | 95% high | Improved n |
| --- | --- | --- | --- | --- | --- |
| Center | A | -4.878 | -9.109 | -0.764 | 29 |
| Center | early_A | -3.050 | -6.935 | 1.000 | 34 |
| Center | MAE | -5.642 | -10.912 | -1.012 | 27 |
| Geometry | A | -3.714 | -7.283 | 0.337 | 33 |
| Geometry | early_A | -3.109 | -7.690 | 1.943 | 27 |
| Geometry | MAE | -4.525 | -8.276 | -0.670 | 29 |
| Serpentine | A | -5.314 | -9.973 | 0.296 | 33 |
| Serpentine | early_A | -3.399 | -8.155 | 2.096 | 28 |
| Serpentine | MAE | -5.628 | -10.306 | -0.477 | 34 |
| Random | A | -5.997 | -9.134 | -2.601 | 35 |
| Random | early_A | -5.346 | -8.934 | -1.375 | 33 |
| Random | MAE | -4.502 | -7.953 | -0.950 | 34 |
| Static | A | -4.380 | -8.405 | -0.199 | 25 |
| Static | early_A | -4.175 | -8.881 | 0.490 | 29 |
| Static | MAE | -4.608 | -8.628 | -0.656 | 33 |
| FULL_SCAN | MAE | 0.695 | -4.009 | 5.327 | 27 |

<!-- block: S028 -->

The percentile intervals use the existing 5,000 paired capture-group draws within source groups. Neither checkpoint selection nor multiple comparisons is adjusted. No interval is newly generated. The partial-minus-full MAE contrast of +0.695 MPa [−4.009, 5.327] is a difference estimate; no prespecified equivalence/noninferiority margin was tested. The area-cost axis measures supplied image pixels rather than measured scanner time.

<!-- block: S029 -->

**Table S6a.** Five-cap first-attainment caps for every preset target, including the full-input anchor. NR = not reached by 0.25; 0 = initially met; † = later recrossing. Relative saving is undefined if either method is NR or control cost is zero. All statuses and savings are preserved in matched_quality.csv.

<!-- block: S030 -->

| Target | Center-first | Geometry-spread | Serpentine | Random | Learned-static | Proposed |
| --- | --- | --- | --- | --- | --- | --- |
| 41 | NR | NR | NR | NR | NR | NR |
| 42 | NR | NR | NR | NR | NR | NR |
| 43 | NR | NR | NR | NR | NR | 0.187500 |
| 44 | NR | NR | NR | NR | NR | 0.062500 |
| 45 | NR | NR | NR | NR | NR | 0.062500 |
| 46 | NR | NR | NR | NR | NR | 0.062500 |
| 47 | NR | 0.125000 | NR | 0.250000 | 0.062500† | 0.062500 |
| 48 | 0.062500† | 0.125000 | NR | 0.250000 | 0.062500† | 0.062500 |
| 49 | 0.062500† | 0.125000 | 0.125000 | 0.187500 | 0.062500 | 0.062500 |
| 50 | 0.062500 | 0.125000 | 0.062500 | 0.125000 | 0.062500 | 0.062500 |
| 51 | 0.062500 | 0.062500 | 0.062500 | 0.125000 | 0.062500 | 0.062500 |
| 52 | 0.062500 | 0.062500 | 0.062500 | 0.125000 | 0.062500 | 0.062500 |
| 53 | 0.062500 | 0.062500 | 0.062500 | 0.125000 | 0.062500 | 0.062500 |
| 54 | 0.062500 | 0.062500 | 0.062500 | 0.062500 | 0.062500 | 0.062500 |
| 55 | 0.062500 | 0.062500 | 0.062500 | 0.062500 | 0.062500 | 0.062500 |
| 56 | 0.062500 | 0.062500 | 0.062500 | 0.062500 | 0.062500 | 0.062500 |
| 57 | 0.062500 | 0.062500 | 0.062500 | 0.062500 | 0.062500 | 0.062500 |
| 58 | 0.062500 | 0.062500 | 0.062500 | 0.062500 | 0.062500 | 0.062500 |
| 59 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 60 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 61 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 41.690 | NR | NR | NR | NR | NR | NR |

<!-- block: S031 -->

**Table S6b.** Event-union first-attainment caps for every preset target, including the full-input anchor. NR = not reached by 0.25; 0 = initially met; † = later recrossing. Relative saving is undefined if either method is NR or control cost is zero. All statuses and savings are preserved in matched_quality.csv.

<!-- block: S032 -->

| Target | Center-first | Geometry-spread | Serpentine | Random | Learned-static | Proposed |
| --- | --- | --- | --- | --- | --- | --- |
| 41 | NR | NR | NR | NR | NR | NR |
| 42 | NR | NR | NR | NR | NR | 0.078844† |
| 43 | NR | NR | NR | NR | NR | 0.062591† |
| 44 | NR | NR | NR | NR | NR | 0.062222 |
| 45 | NR | NR | NR | NR | NR | 0.046897† |
| 46 | NR | 0.109120† | NR | NR | 0.062407† | 0.031203† |
| 47 | NR | 0.093426 | NR | 0.249630 | 0.062407† | 0.031203† |
| 48 | 0.062407† | 0.077732 | NR | 0.217830 | 0.062407† | 0.031203 |
| 49 | 0.046897† | 0.077732 | 0.109120† | 0.125001 | 0.062130 | 0.015715 |
| 50 | 0.046897 | 0.062038† | 0.062407 | 0.109304 | 0.046713 | 0.015715 |
| 51 | 0.046897 | 0.030700† | 0.046713† | 0.093610 | 0.046713 | 0.015715 |
| 52 | 0.031065 | 0.015509 | 0.015509† | 0.077732† | 0.046713 | 0.015694 |
| 53 | 0.015694 | 0.015509 | 0.015509 | 0.047337† | 0.046713 | 0.015694 |
| 54 | 0.015694 | 0.015509 | 0.015509 | 0.031431 | 0.031203 | 0.015694 |
| 55 | 0.015694 | 0.015509 | 0.015509 | 0.031203 | 0.031203 | 0.015509 |
| 56 | 0.015694 | 0.015509 | 0.015509 | 0.031019 | 0.015902 | 0.015509 |
| 57 | 0.015350 | 0.015350 | 0.015350 | 0.015715 | 0.015715 | 0.015509 |
| 58 | 0.015350 | 0.015350 | 0.015350 | 0.015694 | 0.015694 | 0.015350 |
| 59 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 60 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 61 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 41.690 | NR | NR | NR | NR | NR | NR |

<!-- block: S033 -->

At q = 47 MPa, five-cap attainment is 0.0625 for proposed and learned-static, 0.125 for geometry and 0.25 for Random. The 50% reduction applies to geometry on this grid only. At q = 42 MPa the proposed event curve first crosses at 0.078844 and later recrosses; the five-cap grid is NR. The 41.690 MPa full-input anchor is not attained by 0.25. These are cohort first crossings, not an execution-time specimen-level stopping rule or engineering certification.

<!-- block: S034 -->

# S4. Event contributions and fixed-set order interventions

<!-- block: S035 -->

Stage 1–4 completion intervals are (0,.0625], (.0625,.125], (.125,.1875] and (.1875,.25]. A raw decrease is previous absolute error minus updated absolute error. Its weighted contribution multiplies that decrease by (1 − completion fraction / B). Both retain adverse events. The sum of weighted decreases equals initial error minus process A. Archived methods use their original float64 area clock; fixed-set order intervention uses integer cumulative native pixels divided by total crop pixels. The two identity families remain separate; small native-replay differences from archived predictions are not silently mixed.

<!-- block: S036 -->

**Table S7.** All saved paired stage contrasts (MPa), proposed minus control; positive favors proposed. No direction change is applied to this already correct source table.

<!-- block: S037 -->

| Control | Stage | Type | Gain | 95% low | 95% high |
| --- | --- | --- | --- | --- | --- |
| Center-first | 1 | weighted | 4.176 | -1.056 | 9.054 |
| Center-first | 1 | raw | 4.552 | -1.246 | 10.032 |
| Center-first | 2 | weighted | 0.572 | -2.116 | 3.205 |
| Center-first | 2 | raw | 0.727 | -3.500 | 4.848 |
| Center-first | 3 | weighted | 0.161 | -0.705 | 1.027 |
| Center-first | 3 | raw | 0.457 | -1.753 | 2.608 |
| Center-first | 4 | weighted | -0.031 | -0.244 | 0.194 |
| Center-first | 4 | raw | -0.590 | -2.359 | 1.054 |
| Geometry-spread | 1 | weighted | 5.926 | -0.117 | 11.095 |
| Geometry-spread | 1 | raw | 6.865 | 0.362 | 12.502 |
| Geometry-spread | 2 | weighted | -2.475 | -6.110 | 1.112 |
| Geometry-spread | 2 | raw | -4.084 | -9.387 | 1.182 |
| Geometry-spread | 3 | weighted | 0.199 | -0.619 | 0.942 |
| Geometry-spread | 3 | raw | 0.635 | -1.626 | 2.672 |
| Geometry-spread | 4 | weighted | 0.063 | -0.083 | 0.217 |
| Geometry-spread | 4 | raw | 0.842 | -0.597 | 2.243 |
| Serpentine | 1 | weighted | 5.562 | -0.805 | 11.071 |
| Serpentine | 1 | raw | 6.282 | -0.580 | 12.239 |
| Serpentine | 2 | weighted | -0.340 | -2.653 | 2.159 |
| Serpentine | 2 | raw | -1.004 | -4.757 | 2.916 |
| Serpentine | 3 | weighted | -0.032 | -0.886 | 0.830 |
| Serpentine | 3 | raw | -0.051 | -2.323 | 2.196 |
| Serpentine | 4 | weighted | 0.125 | -0.046 | 0.305 |
| Serpentine | 4 | raw | 0.116 | -1.293 | 1.664 |
| Random | 1 | weighted | 8.617 | 4.194 | 12.678 |
| Random | 1 | raw | 9.707 | 4.862 | 14.131 |
| Random | 2 | weighted | -2.314 | -4.549 | 0.070 |
| Random | 2 | raw | -3.998 | -7.409 | -0.326 |
| Random | 3 | weighted | -0.286 | -0.954 | 0.347 |
| Random | 3 | raw | -0.724 | -2.547 | 1.021 |
| Random | 4 | weighted | -0.019 | -0.191 | 0.165 |
| Random | 4 | raw | -0.713 | -2.124 | 0.736 |
| Learned-static | 1 | weighted | 3.341 | -1.651 | 8.050 |
| Learned-static | 1 | raw | 3.063 | -2.381 | 8.149 |
| Learned-static | 2 | weighted | 1.254 | -0.919 | 3.350 |
| Learned-static | 2 | raw | 1.597 | -1.752 | 4.860 |
| Learned-static | 3 | weighted | -0.365 | -1.225 | 0.505 |
| Learned-static | 3 | raw | -0.971 | -3.312 | 1.405 |
| Learned-static | 4 | weighted | 0.150 | -0.035 | 0.330 |
| Learned-static | 4 | raw | 0.694 | -0.874 | 2.160 |

<!-- block: S038 -->

**Table S8.** Normalized fixed-set contrasts. N = native/original; C = comparator. Error rows are comparator minus native. Contribution rows are native minus comparator. All positive values favor native under these distinct definitions.

<!-- block: S039 -->

| Contrast | Metric | Direction | Estimate | 95% low | 95% high |
| --- | --- | --- | --- | --- | --- |
| Primary permutation | A | C−N error | 1.486 | -0.658 | 3.384 |
| Primary permutation | Early A | C−N error | 1.628 | -2.190 | 5.180 |
| Primary permutation | J | C−N error | 1.486 | -0.658 | 3.384 |
| Primary permutation | Final error | C−N error | 0.000 | 0.000 | 0.000 |
| Primary permutation | Weighted S1 | N−C contribution | 3.074 | -0.578 | 6.502 |
| Primary permutation | Weighted S2 | N−C contribution | -1.255 | -3.355 | 0.899 |
| Primary permutation | Weighted S3 | N−C contribution | -0.325 | -0.940 | 0.250 |
| Primary permutation | Weighted S4 | N−C contribution | -0.008 | -0.167 | 0.167 |
| Secondary reverse | A | C−N error | 3.929 | 0.145 | 7.405 |
| Secondary reverse | Early A | C−N error | 3.328 | -1.527 | 7.979 |
| Secondary reverse | J | C−N error | 3.929 | 0.145 | 7.405 |
| Secondary reverse | Final error | C−N error | 0.000 | 0.000 | 0.000 |
| Secondary reverse | Weighted S1 | N−C contribution | 6.348 | 0.695 | 11.784 |
| Secondary reverse | Weighted S2 | N−C contribution | -1.742 | -4.553 | 1.118 |
| Secondary reverse | Weighted S3 | N−C contribution | -0.291 | -1.385 | 0.844 |
| Secondary reverse | Weighted S4 | N−C contribution | -0.387 | -0.744 | -0.011 |

<!-- block: S040 -->

The legacy order_paired_contrasts.csv contains eight weighted-stage rows labeled comparator-minus-native. Only those contribution rows are negated in the derived normalized table, with CI [lo,hi] transformed to [−hi,−lo]. Eight error/objective rows remain unchanged. source_* columns preserve the original values and source Git blob 23a65f8d481797298b25bb228d1b2fd91181c65f. Applying normalization twice is identical to applying it once. The four normalized stage differences sum to the corresponding A difference within $10^{-8}$ MPa. Because final error is invariant within each fixed set, J − A differences also close. DIRECTION_CORRECTION.json records residuals.

<!-- block: S041 -->

The original-order intervention reproduced the archived predictions within the saved maximum deviation 6.104 × $10^{-5}$ MPa. All 350 terminal-set checks preserved the same terminal input and prediction. This is expected from a set-based predictor and validates the intervention; it is not an endpoint accuracy gain caused by order. The primary permutation A interval [−0.658,3.384] crosses zero, whereas the secondary reverse interval [0.145,7.405] is positive. Both early-A intervals cross zero. No component-specific causal claim or additive percentage decomposition follows.

<!-- block: S042 -->

# S5. Six fixed cases and saved information-use diagnostics

<!-- block: S043 -->

The six cases were fixed before this revision and are all retained, including adverse residuals. Initial preferences are feasible-action softmax probabilities, not regional damage probabilities. The attribution maps are already saved L1 sensitivity of the selected action’s log probability to local surface descriptors, including the path through the current strength prediction. Attention maps are the saved global-query-to-regional attention among fused tokens. These three quantities have different meanings and units; none is a regional mechanical ground truth. The displays below use only the initial, NO_VLM model’s saved arrays. No attribution or attention was regenerated.

<!-- block: S044 -->

**Table S9.** All six cases; CAI values and errors in MPa.

<!-- block: S045 -->

| Specimen | First cell | Actions | CAI target | Initial | Final | Final abs. error |
| ------------------------------ | -------- | -------- | --------- | --------- | --------- | ---------- |
| 8-layer cross-ply / c8-16 | 62 | 16 | 264.163 | 301.366 | 232.095 | 32.068 |
| 24-layer quasi-isotropic / q24-48 | 31 | 16 | 136.680 | 302.099 | 258.932 | 122.252 |
| 16-layer quasi-isotropic / q16-29 | 14 | 15 | 338.548 | 326.638 | 335.890 | 2.658 |
| 24-layer cross-ply / c24-9t | 27 | 16 | 278.254 | 225.969 | 233.148 | 45.105 |
| 16-layer cross-ply / c16-24 | 63 | 15 | 523.441 | 388.144 | 406.717 | 116.725 |
| 8-layer quasi-isotropic / q8-22 | 63 | 16 | 235.013 | 245.533 | 244.583 | 9.570 |

<!-- block: S046 -->

Nine of 94 recorded decisions departed from the highest initially scored cell still feasible on the actual path: c8-16 (2), q16-29 (2), c16-24 (4), q8-22 (1); q24-48 and c24-9t retained their initial feasible ranking. The comparison removes measured and unaffordable cells at every actual state. This describes specimen-specific initialization and selective reordering in six cases only. Changing internal evidence, history, progress and prediction are coupled, so the departures do not isolate internal-feedback causality.

<!-- block: S047 -->

## S5.1. 8-layer cross-ply / c8-16

<!-- block: S048 -->

![Saved diagnostic plate 1](figures/S5_1.png){width=6.5in}

<!-- block: S049 -->

**Figure S5.1.** Initial NO_VLM state and saved CAI trajectory. (a) Feasible-action preference; (b) surface L1 sensitivity for the selected-action query (first saved target); (c) first-layer global-query regional attention (first saved layer); (d) saved CAI predictions with measured strength dashed. Maps retain all 64 cells and use independent ranges. Values and target/layer indices are in figure_data/cases.json; selected cell 62. Final absolute error: 32.068 MPa.

<!-- block: S050 -->

## S5.2. 24-layer quasi-isotropic / q24-48

<!-- block: S051 -->

![Saved diagnostic plate 2](figures/S5_2.png){width=6.5in}

<!-- block: S052 -->

**Figure S5.2.** Initial NO_VLM state and saved CAI trajectory. (a) Feasible-action preference; (b) surface L1 sensitivity for the selected-action query (first saved target); (c) first-layer global-query regional attention (first saved layer); (d) saved CAI predictions with measured strength dashed. Maps retain all 64 cells and use independent ranges. Values and target/layer indices are in figure_data/cases.json; selected cell 31. Final absolute error: 122.252 MPa.

<!-- block: S053 -->

## S5.3. 16-layer quasi-isotropic / q16-29

<!-- block: S054 -->

![Saved diagnostic plate 3](figures/S5_3.png){width=6.5in}

<!-- block: S055 -->

**Figure S5.3.** Initial NO_VLM state and saved CAI trajectory. (a) Feasible-action preference; (b) surface L1 sensitivity for the selected-action query (first saved target); (c) first-layer global-query regional attention (first saved layer); (d) saved CAI predictions with measured strength dashed. Maps retain all 64 cells and use independent ranges. Values and target/layer indices are in figure_data/cases.json; selected cell 14. Final absolute error: 2.658 MPa.

<!-- block: S056 -->

## S5.4. 24-layer cross-ply / c24-9t

<!-- block: S057 -->

![Saved diagnostic plate 4](figures/S5_4.png){width=6.5in}

<!-- block: S058 -->

**Figure S5.4.** Initial NO_VLM state and saved CAI trajectory. (a) Feasible-action preference; (b) surface L1 sensitivity for the selected-action query (first saved target); (c) first-layer global-query regional attention (first saved layer); (d) saved CAI predictions with measured strength dashed. Maps retain all 64 cells and use independent ranges. Values and target/layer indices are in figure_data/cases.json; selected cell 27. Final absolute error: 45.105 MPa.

<!-- block: S059 -->

## S5.5. 16-layer cross-ply / c16-24

<!-- block: S060 -->

![Saved diagnostic plate 5](figures/S5_5.png){width=6.5in}

<!-- block: S061 -->

**Figure S5.5.** Initial NO_VLM state and saved CAI trajectory. (a) Feasible-action preference; (b) surface L1 sensitivity for the selected-action query (first saved target); (c) first-layer global-query regional attention (first saved layer); (d) saved CAI predictions with measured strength dashed. Maps retain all 64 cells and use independent ranges. Values and target/layer indices are in figure_data/cases.json; selected cell 63. Final absolute error: 116.725 MPa.

<!-- block: S062 -->

## S5.6. 8-layer quasi-isotropic / q8-22

<!-- block: S063 -->

![Saved diagnostic plate 6](figures/S5_6.png){width=6.5in}

<!-- block: S064 -->

**Figure S5.6.** Initial NO_VLM state and saved CAI trajectory. (a) Feasible-action preference; (b) surface L1 sensitivity for the selected-action query (first saved target); (c) first-layer global-query regional attention (first saved layer); (d) saved CAI predictions with measured strength dashed. Maps retain all 64 cells and use independent ranges. Values and target/layer indices are in figure_data/cases.json; selected cell 63. Final absolute error: 9.570 MPa.

<!-- block: S065 -->

## S5.7. Real-image provenance and reproducible displays

<!-- block: S066 -->

Main Fig. 6 uses the c8-16 (8-layer cross-ply) and q24-48 (24-layer quasi-isotropic) image pairs from the corresponding Mendeley Data version 1 collections (Hasebe, Higuchi, Yokozeki and Takeda, 2022; CC BY 4.0). The candidate manifest resolves each surface and registered C-scan path with SHA256 and native dimensions. Surface orientation follows the frozen clockwise-90° transform and maximum-edge-1024 display resize. The registered C-scan pixels are unchanged; overlays use rint native grid edges and the saved action ordinals. Grey overlays mark unavailable cells. The full-image backdrop is explicitly retrospective and was never supplied to the actor. assets.json records source URLs, hashes, dimensions, transformations and attribution. No damage outline, scale calibration or physical length measurement is invented.

<!-- block: S067 -->

The delivered scripts bind, prepare, render, build and verify this frozen-evidence package. Machine-readable trajectory ledgers provide all 500 archived paths and all 350 fixed-set conditions, without duplicating weights or feature shards. The eight document outputs provide English and aligned bilingual main text and supplement. Reference numbering is inherited from the main article; this supplement introduces no new references. Author names, affiliations, funding, competing interests and final journal submission metadata require author confirmation before submission; the package is a revision delivery, not a journal submission.
