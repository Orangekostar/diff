# BC C-scan Path-B Inventory and Bindings

## Frozen identity

- Evidence base: `d8b5b090891fc030931c6dc81e3619a80966f739`
- Execution commit at audit: `abfad35dccb39887354464002fe10be9628deadb`
- Parent config SHA-256: `12268dcaf470f769c326007702f7b0b6f4a13cf326447eef652a6dda972b5662`
- BC seed-1 SHA-256: `c95185aec70515f2578d671e520ebe9b57e600e91e70e5f6752567ae2f87fbe5`
- Old STOP SHA-256: `c6af5c383ef4c586a4ac85c7dc9afbcd5bc71dc07558c692fb295d0f5386892d`
- Source bank SHA-256: `52167731b60bbd6db759a1e74a85bd0eb0bc8ea208bd51e981bff44d841281c7`
- Canonical base-BC content SHA-256: `9d73c3e27de235ffe8337738517aec48f3f4a3267eb5f687ef0f6eeaae3566d1`

## Cohort and fit split

- Reused cohort: 60 physical specimens across six domains; TRAIN/VALID/TEST = 24/12/24.
- BC bank: 192 rows; internal fit/validation = 144/48 rows from 18/6 specimens.
- Internal-validation keys: `74t7kcdgkr:c8-32, cgtnjyggtm:q24-4, w68dtmpfyf:q16-42, xcmzfsbd9t:c24-43, yfxyg8jm46:c16-42, ykhs7s2dck:q8-7`.
- Hasebe data: `/home/ww/paper3/cmc_damage_inference/data/public/hasebe`.
- Hasebe CAI data: `/home/ww/paper3/cmc_damage_inference/data/public/hasebe_cai`; no CAI training is performed.

## Cache and reference coverage

- Frozen surface-cache coverage by TRAIN/VALID/TEST: 24/12/24.
- Independently reviewed reference coverage by TRAIN/VALID/TEST: 0/0/0.
- The 276-row reference manifest remains algorithm-derived and unreviewed; formal effects are null.

## Replay sufficiency

- Historical trajectories contain actions, report hashes, rule STOP, learned STOP probability and eligibility, cumulative route cost, and turn counts.
- Predicted masks and support positions are not stored. Reviewed rescoring must reconstruct reports by replaying stored action prefixes; a report hash alone is not rescored.
- The source bank is read-only. Only its 192 `base_bc` rows are copied to the ignored destination `_work/base_bc.pt` cache.

## Frozen historical metric replay

| Task | Method | Metric | Mean over 24 TEST specimens |
|---|---|---|---:|
| LOCATE | R_BALANCED_P8 | planner_ausc | 0.6298188483710805 |
| LOCATE | R_BALANCED_P8 | rule_completion | 0.875 |
| LOCATE | R_BALANCED_P8 | rule_autonomous_ausc | 0.6162122172500377 |
| LOCATE | R_BALANCED_P8 | rule_failure_penalized_cost | 0.3837877827499623 |
| LOCATE | R_BALANCED_P8 | old_stop_completion | 0.5833333333333334 |
| LOCATE | R_BALANCED_P8 | old_stop_autonomous_ausc | 0.4881646378434759 |
| LOCATE | R_BALANCED_P8 | old_stop_failure_penalized_cost | 0.511835362156524 |
| LOCATE | L_BC | planner_ausc | 0.8099545385296553 |
| LOCATE | L_BC | rule_completion | 0.7083333333333334 |
| LOCATE | L_BC | rule_autonomous_ausc | 0.4679026083452393 |
| LOCATE | L_BC | rule_failure_penalized_cost | 0.5320973916547607 |
| LOCATE | L_BC | old_stop_completion | 0.625 |
| LOCATE | L_BC | old_stop_autonomous_ausc | 0.5042872640824722 |
| LOCATE | L_BC | old_stop_failure_penalized_cost | 0.4957127359175279 |
| CHARACTERIZE | R_BALANCED_P8 | planner_ausc | 0.467930098731157 |
| CHARACTERIZE | R_BALANCED_P8 | rule_completion | 0.875 |
| CHARACTERIZE | R_BALANCED_P8 | rule_autonomous_ausc | 0.2240150922375049 |
| CHARACTERIZE | R_BALANCED_P8 | rule_failure_penalized_cost | 0.7759849077624952 |
| CHARACTERIZE | R_BALANCED_P8 | old_stop_completion | 0.6666666666666666 |
| CHARACTERIZE | R_BALANCED_P8 | old_stop_autonomous_ausc | 0.1292325912828579 |
| CHARACTERIZE | R_BALANCED_P8 | old_stop_failure_penalized_cost | 0.8707674087171421 |
| CHARACTERIZE | L_BC | planner_ausc | 0.5488189459859376 |
| CHARACTERIZE | L_BC | rule_completion | 1 |
| CHARACTERIZE | L_BC | rule_autonomous_ausc | 0.2078612234264538 |
| CHARACTERIZE | L_BC | rule_failure_penalized_cost | 0.7921387765735463 |
| CHARACTERIZE | L_BC | old_stop_completion | 0.6666666666666666 |
| CHARACTERIZE | L_BC | old_stop_autonomous_ausc | 0.1304759036538059 |
| CHARACTERIZE | L_BC | old_stop_failure_penalized_cost | 0.8695240963461942 |

These values are same-Reader proxy evidence. This audit performs no fitting, no new TEST evaluation, and no VLM calls.
