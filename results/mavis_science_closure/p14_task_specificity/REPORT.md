# MAVIS P14 Task Specificity

Status: `COMPLETE`.

The same 276-specimen cohort, six held-out domains, exact-cost checkpoint roster, frozen CAI predictors, and historical `normalized_rgb_mse` metric are used for all four acquisition strategies.

Oracle CAI contrast (reconstruction minus mechanics) is `0.0486228431` with paired 95% interval `[0.045273264544767, 0.052045350460869]`. Oracle image contrast (mechanics minus reconstruction) is `0.0005502549` with interval `[0.000500623349499, 0.000606291994363]`.

Interpretation: Oracle acquisitions show task specificity: the mechanical oracle improves CAI while the reconstruction oracle improves reconstruction. The source-trained global mechanics mask does not reproduce this separation and is worse on both aggregate objectives than the global reconstruction mask.

Spatial comparisons describe only grid-action overlap, distance, radial allocation, and refinement level. They do not assign physical failure mechanisms to selected regions. P7 remains unchanged.
