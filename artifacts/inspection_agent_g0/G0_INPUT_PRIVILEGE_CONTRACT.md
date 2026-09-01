# G0 Input and Privilege Contract

## Primary scenario

A robot receives a previously unseen component, its impacted-surface RGB image,
an engineering task, and a known scanner action lattice. It initially has zero
ultrasound measurements.

## Policy-visible state

The primary policy interface may receive only:

- surface RGB and its verified content hash;
- task identifier;
- 8x8 measurement geometry and legal one-level actions;
- generalized levels in `{-1,0,1,2}`;
- positions and RGB values actually acquired so far;
- exact acquired count, normalized budget, remaining budget;
- transparent surface-hypothesis scores/cells;
- structured `InspectionBeliefRecord`.

No primary policy input contains a domain name or experimental specimen ID.
Random control identity is derived from the visible surface hash and the frozen
global seed.

## Forbidden primary inputs

The following must never appear in a policy-facing object, callable signature,
candidate descriptor, or belief record:

- true CAI or any post-CAI field;
- hidden full/future C-scan pixels;
- candidate measurement values before acquisition;
- oracle task values or oracle ranks;
- dataset/domain identity or experimental specimen ID;
- impact energy/history, impactor identity, laminate type, ply count;
- the old `metadata13 + profile_stats21` 34-D context;
- target-domain labels, target-derived normalization, or target-based selection.

## Privileged layers

| Layer | Permitted privilege | Prohibited use |
|---|---|---|
| `CausalInspectionWorld` | Private specimen lookup and `MAVISAuthority._reveal_values` for newly requested native positions | Returning full scan, true CAI, domain/specimen ID, future pixels, or teacher value |
| Source-prior fitting | Full C-scans from the five source domains | Any outer-target image/statistic |
| StateCAIAssessor fitting | Source-state reconstructions and source true CAI | Outer-target fit, normalization, tuning, or model selection |
| ORACLE_DISCOVERY | Hidden target full C-scan to score incremental internal-signal mass | Calling the oracle deployable or exposing scores to observation |
| ORACLE_FIELD | Hidden target full C-scan to score reconstruction improvement | Passing future pixels through policy APIs |
| ORACLE_CAI | Hidden target true CAI after assessor authorization | Running before the assessor gate or exposing CAI to policy state |
| Evaluation/statistics | Full target scan, target CAI, method outcomes after trajectories freeze | Changing policies, gates, thresholds, or task definitions |

## Historical baseline exception

`FIXED_UNIFORM_THEN_MAVIS` replays the frozen `mavis_full` trajectory, whose old
model used the 34-D context. It is explicitly labeled
`METADATA_AUGMENTED_UPPER_BOUND`; it is not evidence that the primary
unknown-component interface is deployable.

## Enforcement

Typed observations omit privileged fields. World transitions reveal only the
set difference between candidate and current native masks and verify that all
old position/value pairs remain byte-identical. Tests use sentinel full scans,
CAI values, IDs, and future pixels to reject leakage. Each outer assessor records
fit specimen/state IDs and domains, and the target domain must be absent.
