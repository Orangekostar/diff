# Inspection Agent G0 Opportunity Audit

Status: `G0_ACTIVE_INSPECTION_OPPORTUNITY_GO`

This package evaluates privileged opportunity from a strict zero-ultrasound state. 
It does not contain a learned planner and does not claim deployment readiness.

## CAI assessor gate

- Zero-state equal-domain MAE: 0.182643120536
- 25% equal-domain MAE: 0.100215374569
- Paired improvement 95% CI: [0.0696398759474, 0.0951896340019]
- Authorized: `True`

## Initialization

- Oracle-minus-uniform capture AUC: 0.000740212343696
- Relative improvement: 8.163795%
- Gate: `False`

## Hierarchical FIELD allocation

- Fixed-minus-oracle AUEBC: 9.06315357945e-05
- Relative improvement: 7.795044%
- Gate: `True`

## Historical MAVIS upper bound

- FIELD AUEBC: 0.00122603523702
- Privilege: `METADATA_AUGMENTED_UPPER_BOUND`
- Gate eligible: `False`

## FIELD stopping

- Mean normalized measurement saving: 48.241088%
- Task-loss ratio: 1.02285124094
- Gate: `True`

## Conditional CAI route

Status: `RUN_AUTHORIZED`

All contrasts use synchronized specimen-within-domain bootstrap and equal-domain aggregation.
Full C-scans and true CAI are evaluation/teacher privilege only.
