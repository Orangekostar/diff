# N3 Content Attribution

Gate: `CONTENT_NO_GO`.

Positive `control_minus_real` values mean measured real content is better. The headline stratum is the pre-registered `CLEAN_NONPRIV = {uniform, random}`; the complete frozen bank is retained in every output table.

| Stage | Model | Control | Point | 95% CI | Favorable domains |
|---|---|---|---:|---:|---:|
| p2 | deepsets | positions_only | -0.0178299307 | [-0.0262319396, -0.0095974273] | 1/6 |
| p2 | deepsets | shuffled | 0.0040866987 | [-0.0028401923, 0.0110914055] | 3/6 |
| p2 | spatial | positions_only | -0.0030174325 | [-0.0122995304, 0.0064825516] | 2/6 |
| p2 | spatial | shuffled | -0.0015067303 | [-0.0096817885, 0.0062555787] | 3/6 |
| p3 | deepsets | positions_only | 0.0002307719 | [-0.0001353438, 0.0006052474] | 4/6 |
| p3 | deepsets | shuffled | -0.0002466121 | [-0.0004547463, -0.0000493227] | 2/6 |
| p3 | spatial | positions_only | -0.0002325869 | [-0.0004068854, -0.0000641154] | 2/6 |
| p3 | spatial | shuffled | -0.0001872229 | [-0.0003397380, -0.0000272111] | 2/6 |

P2 retains AUEBC and P3 retains next-action regret. The unified sign is an explicit exploratory report conversion only; the registered metric implementations, state rows, controls, and bootstrap units are unchanged. The overall gate conservatively requires both clean P2 and clean P3 layers to support both headline controls.
