# G0-A Initialization Opportunity

Status: `INITIALIZATION_HEADROOM_NO_GO`.

All methods start from zero ultrasound and end at the exact old-scout union.
Capture AUC is measured against privileged internal-signal saliency, not damage
ground truth.

| Initialization method | Equal-domain capture AUC |
|---|---:|
| ORACLE_DISCOVERY | 0.009807225289 |
| SURFACE_FOCUS | 0.009145856231 |
| CENTER_FIRST | 0.009081234859 |
| ZERO_UNIFORM | 0.009067012945 |
| RANDOM | 0.009053158504 |

ORACLE_DISCOVERY exceeds ZERO_UNIFORM by `0.000740212343696`, with synchronized
95% CI `[0.000622894718166, 0.000867124273196]` and 6/6 improving domains. The
relative AUC improvement is only `8.163795%`, and the registered capture-budget
reduction is `0`. Both are below the alternative 10% magnitude requirements, so
the gate fails despite a statistically positive effect.

The frozen surface/internal strata contain 1 `AGREE`, 114 `PARTIAL`, and 161
`MISLEADING` specimens. This reinforces the historical P1 lesson: surface
appearance is a provisional hypothesis, not mechanical truth. G1 should retain
a simple geometry-spread start rather than claim learned initialization value.

Authority: `initialization_curves.csv` SHA-256
`519b96c0cf525cda32ff54e91792524c77fcc858e3fa33b10ee292d3be8141e6`.
