# G1 Deployment Geometry Decision

Status: `G1_DEPLOYMENT_GEOMETRY_GO`  
Decision date: 2026-09-01  
Authority: `results/inspection_agent/g0/authorized_roster.csv` at base
`7a10cd425de582fa158bf6639285731ccd8ff7a7`

## Privilege Audit

The authorized 276-specimen roster contains three visible native geometries. The
same geometry maps to both G0 nominal grid presets, so the mapping is case B and
cannot be recovered without hidden domain membership.

| Native shape | G0 nominal budget | Domain(s) | Count |
|---|---:|---|---:|
| 338 x 340 | 0.015625 | `xcmzfsbd9t` | 10 |
| 338 x 340 | 0.03125 | `74t7kcdgkr` | 7 |
| 338 x 352 | 0.015625 | `xcmzfsbd9t` | 10 |
| 338 x 352 | 0.03125 | `74t7kcdgkr` | 9 |
| 674 x 675 | 0.015625 | `cgtnjyggtm,w68dtmpfyf,xcmzfsbd9t,yfxyg8jm46,ykhs7s2dck` | 211 |
| 674 x 675 | 0.03125 | `74t7kcdgkr` | 29 |

Each registered candidate `{0.015625, 0.03125, 0.0625}` successfully builds the
nested endpoint-preserving acquisition grid for every shape. The frozen choice is
the smallest universal valid nominal budget:

```text
deployment_initial_nominal_budget = 0.015625
rule = build_acquisition_grid(native_height, native_width, initial_budget=0.015625)
```

The actor sees normalized native height/width/aspect but no domain or preset ID.

## Same-Geometry Bridge

Changing the `74t7kcdgkr` grid means old G0 numerical curves cannot enter G1 gap
closure. For every outer fold and method, G1 recomputes fixed baselines, learned
policies, FIELD/CAI oracles, costs, checkpoints, and losses under the universal
grid, K=8 warm start, endpoint 0.25, and common evaluators. Historical G0 values
remain contextual evidence only.

## Primary Warm Start

The first eight cells of `mva.oracle.uniform_cell_order()` are frozen as:

```text
0, 63, 7, 56, 27, 38, 3, 24
```

| Native shape | Exact unique pixels after K=8 | Native pixels | Exact budget | Remaining BROADEN | Immediate REFINE |
|---|---:|---:|---:|---:|---:|
| 338 x 340 | 294 | 114920 | 0.0025583014270797078 | 56 | 8 |
| 338 x 352 | 305 | 118976 | 0.0025635422270037654 | 56 | 8 |
| 674 x 675 | 1128 | 454950 | 0.0024793933399274645 | 56 | 8 |

The warm start is geometry-only, lies far below 0.25 for every specimen, preserves
both broadening and refinement, and does not claim optimized initialization.
K=4 and K=16 are sensitivity protocols only and cannot replace K=8 after target
results.

## Complete-Scout Exclusion

With all 64 cells at level 0, the number of `-1->0` actions is zero. Therefore a
complete scout would mechanically eliminate FOCUS/BROADEN and invalidate the G1
action-policy question. It is not a candidate primary warm start.

## Gate Evidence

- 276/276 authorized specimens map to one of the three audited shapes.
- 3/3 shapes accept the selected universal grid.
- 276/276 K=8 states have valid exact cost below 0.25.
- Every K=8 state preserves 56 BROADEN and 8 immediate REFINE actions.
- No dataset/domain label participates in grid or warm-start construction.

Result: `G1_DEPLOYMENT_GEOMETRY_GO`.
