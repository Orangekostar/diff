# P1 Response Richness Decision

Status: `P1_GO`

- Base SHA: `3951f71f28b6efdf8c74eea0fe274b2a78a9cd57`
- P0 authority: `9d44ead975119db2181a91efbf14b74165671a9d25b7b576d90f6e104757a633` / `P0_GO`
- Valid responses: 276/276 across 6/6 domains
- Passing endpoints: extension_peak_mm, slope_u20_u60_mpa_per_mm, normalized_prepeak_auc
- Strength-only gate metrics:
  - `extension_peak_mm`: pooled R2 `0.68150593455684616`
  - `slope_u20_u60_mpa_per_mm`: pooled R2 `-1.2037088518000298`
  - `normalized_prepeak_auc`: pooled R2 `0.24905174995994184`
- Extraction repeat: byte-identical
- Artifact package replay: required and verified by the publishing command
- P2: `AUTHORIZED_NOT_RUN`
- P3-P5: `NOT_RUN_NOT_AUTHORIZED`
- Evidence: `results/damage_to_failure_response/p1_response_richness/`
- Manual specimen/result selection: NO
- Hyperparameter search: NO
