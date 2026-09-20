# Execution Review

The C release satisfies Q1-Q7 and is ready for the bounded Q8 Git delivery. Scientific effect direction was not used as a completion gate.

- Q1-Q2: 211 terminal TRAIN/VALID priors, six exact pilot reuses, no TEST access, per-case attempt cap preserved.
- Q3-Q4: 15 candidate weights, 750 candidate trajectories, three selected actors and 150 selected trajectories. All disk-reload smoke checks passed.
- Q5: 500 frozen controls plus 150 new C rows form the 650-row primary matrix; historical A remains separate.
- Q6: nine-method same-cost, paired, A/C, equal-quality, timing, domain and three-case evidence were recomputed.
- Q7: six-section C manuscript, HTML, TeX, main PDF and SI PDF were rebuilt and visually checked.
- Q8: pending publish-stage result commit and push.

Actor selection:
- `VLM_SPATIAL_FEEDBACK`: seed 2026091301, selected update 250, logical 1250, charged upper bound 1250.
- `VLM_SPATIAL_OPEN_LOOP`: seed 2026091303, selected update 1250, logical 1250, charged upper bound 1250.
- `VLM_MEAN_FEEDBACK`: seed 2026091305, selected update 250, logical 1250, charged upper bound 1250.

Remaining author inputs are names, affiliations, funding, declarations, final availability wording and submission approval. These do not block the scientific author-review draft.
