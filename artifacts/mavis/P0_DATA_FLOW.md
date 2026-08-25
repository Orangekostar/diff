# MAVIS P0 Data Flow and Privilege Boundaries

## Existing frozen flow

```text
paired public manifest
  -> hash-verified CscanRecord bytes
  -> 276 decoded RGB C-scans + metadata13 + CAI
  -> MVA initial sparse reconstruction and 64 counterfactual refinements
  -> full-scan-derived candidate embeddings/value labels
  -> MVD M0 oracle and M1 static student
  -> exact-cost curves, domain metrics, bootstrap, replay
```

The MVD candidate bank is a counterfactual cache. It is not a sequence of
measurements revealed by executing actions.

## MAVIS causal flow

```text
PrivilegedSpecimenAuthority(full_scan, source_true_cai, hashes)
             | reveal(action) only
             v
DeployableState(context, acquired_indices, acquired_rgb, remaining_budget)
             |
             v
MRIS encoder -> current CAI estimate + conditional action scores
             | legal action + exact cost guard
             v
reveal next authoritative measurements -> new DeployableState -> repeat
```

## Visibility contract

| Information | Target policy | Source teacher | Final evaluator |
|---|---:|---:|---:|
| Surface/context and registered deployable metadata | Yes | Yes | Yes |
| Acquired positions and their exact RGB values | Yes | Yes | Yes |
| Unacquired RGB values | No | Counterfactual labeling only | No during rollout |
| Full target C-scan | No | Never | Endpoint computation only after rollout |
| True source CAI | No for own prediction | Strict-OOF teacher utility only | Yes |
| True outer-target CAI | Never | Never | Metric computation only |
| Outer-target normalization/statistics | Never | Never | Reporting only |

## Important semantic correction

Historical MVD `initial_embeddings` are generated after a registered 1.5625% or
3.125% sparse scout. They are not zero-ultrasound states. MAVIS therefore has a
pre-scout context state and a post-scout measurement state; every comparison
must include scout cost.
