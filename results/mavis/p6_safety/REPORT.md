# MAVIS P6 Source-only Confidence Fallback

Status: `COMPLETE`. Baseline and confidence threshold are selected separately for each outer target using only double-held-out source validation curves. The first post-scout decision confidence routes a physical specimen to MAVIS or the selected robust baseline. Outer-target outcomes are not used. Calibration uses nested pre-aggregation P3 models so it remains independent of source on-policy training outcomes.
