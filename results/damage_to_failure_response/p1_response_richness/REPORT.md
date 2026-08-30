# P1 CAI Response Richness Audit

Status: `P1_GO`

## Authority and scope

- Exact Git base: `3951f71f28b6efdf8c74eea0fe274b2a78a9cd57`
- P0 summary SHA-256: `9d44ead975119db2181a91efbf14b74165671a9d25b7b576d90f6e104757a633` (`P0_GO`)
- Primary cohort: 276/276 valid, with all six domains.
- Response axis: stress-extension only; strain remains `STRAIN_UNIT_UNRESOLVED`.
- New damage-to-response model training: NO.
- Fixed redundancy reference fits: 36; no search.

## Endpoint decisions

- `extension_peak_mm`: range `1.4367192500000001`, strength-only pooled R2 `0.68150593455684616`, strength+design pooled R2 `0.93330884482835463`, gate `PASS`.
- `slope_u20_u60_mpa_per_mm`: range `555.16808668673559`, strength-only pooled R2 `-1.2037088518000298`, strength+design pooled R2 `0.7430793890508709`, gate `PASS`.
- `normalized_prepeak_auc`: range `0.34018986134233753`, strength-only pooled R2 `0.24905174995994184`, strength+design pooled R2 `0.60850340322399155`, gate `PASS`.

## Decision

- Passing endpoints: ['extension_peak_mm', 'slope_u20_u60_mpa_per_mm', 'normalized_prepeak_auc']
- Artifact/extraction replay byte-identical: YES
- P2 status: `AUTHORIZED_NOT_RUN`
- P3-P5: `NOT_RUN_NOT_AUTHORIZED`
