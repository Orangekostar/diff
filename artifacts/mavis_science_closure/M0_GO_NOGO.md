# MVD M0 GO/NO-GO

Decision: `MVD_ONE_SHOT_STRONG_GO`.

M0 establishes retrospective one-shot mechanical-value headroom on the
normalized raster observation grid. It authorizes the minimum M1 observability
test. It does not authorize a deployable policy claim.

| Gate | Result | Decision |
|---|---:|---|
| One-shot mechanical oracle AUEBC | `0.0134579441` | Diagnostic upper bound |
| Uniform AUEBC | `0.0173634580` | One-shot oracle better in 6/6 domains |
| Uniform-minus-oracle effect | `0.0039055139`, 95% CI `[0.0028007401, 0.0050178614]` | PASS |
| Reconstruction AUEBC | `0.0171876604` | One-shot oracle better in 6/6 domains |
| Reconstruction-minus-oracle effect | `0.0037297163`, 95% CI `[0.0028380324, 0.0046883883]` | PASS |
| Sequential mechanical oracle AUEBC | `0.0106245258` | Retrospective reference |
| Headroom retention | `0.5682827637` | PASS |

The interaction audit also shows material non-additivity: additive point gains
cannot be treated as joint measurement-set utility. All costs use the frozen
exact unique-location fit/skip rule.

Authority: `results/mvd/m0_one_shot_oracle/summary.json` and
`results/mvd/m0_one_shot_oracle/REPORT.md`.

Allowed conclusion: retrospective mechanical value and one-shot acquisition
headroom exist. Forbidden conclusion: M0 is a deployable scanner policy or
demonstrates physical inspection-time reduction.
