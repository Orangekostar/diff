# P2 Damage-to-Response Decision

Status: `MACK_EXTENSION_NO_GO`

- Base SHA: `3951f71f28b6efdf8c74eea0fe274b2a78a9cd57`
- P0 authority: `9d44ead975119db2181a91efbf14b74165671a9d25b7b576d90f6e104757a633` / `P0_GO`
- P1 authority: `37da95962395a0915f586820ab03f06d8d859856e8637d975bc302b1d555ebc7` / `P1_GO`
- Cohort: 276/276, 6/6 domains
- Primary reference: `F2`
- Primary candidates: `F3`, `F4`
- Required relative improvement: at least 10%
- Required improved domains: at least 4/6
- Required familywise lower bound: strictly above zero

- `extension_peak_mm` / `F3` vs `F2`: `FAIL`; relative equal-domain MAE improvement is below 10%; familywise bootstrap lower bound is not positive.
- `extension_peak_mm` / `F4` vs `F2`: `FAIL`; relative equal-domain MAE improvement is below 10%; fewer than four held-out domains improve; familywise bootstrap lower bound is not positive.
- `slope_u20_u60_mpa_per_mm` / `F3` vs `F2`: `FAIL`; relative equal-domain MAE improvement is below 10%.
- `slope_u20_u60_mpa_per_mm` / `F4` vs `F2`: `FAIL`; relative equal-domain MAE improvement is below 10%; fewer than four held-out domains improve; familywise bootstrap lower bound is not positive.
- `normalized_prepeak_auc` / `F3` vs `F2`: `FAIL`; relative equal-domain MAE improvement is below 10%; fewer than four held-out domains improve; familywise bootstrap lower bound is not positive.
- `normalized_prepeak_auc` / `F4` vs `F2`: `FAIL`; relative equal-domain MAE improvement is below 10%; fewer than four held-out domains improve; familywise bootstrap lower bound is not positive.

- Passing contrasts: []
- P3-P5: `NOT_RUN_NOT_AUTHORIZED`
- New paper route: `NOT_AUTHORIZED`
- Evidence: `results/damage_to_failure_response/p2_response_baselines/`
