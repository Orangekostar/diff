# Inputs from User

The same `finalize` command accepts any subset of the following real inputs. Omitted tracks remain `PENDING_USER_INPUT`.

## Reviewed References

A directory of one JSON file per specimen using the existing `registered_cscan` reference schema: `specimen_key`, `source_image_sha256`, `reference_type`, `review_state=reviewed`, `reviewer_alias`, `frame=registered_cscan`, `regions`, and `uncertain_regions`.

## Blind Report Reviews

A UTF-8 CSV with `report_id`, `reviewer_id`, and `decision`. Decision is one of `DELIVERABLE`, `NEEDS_FURTHER_INSPECTION`, or `UNABLE_TO_JUDGE`. Optional columns are `problem_type`, `review_basis`, and `blinding_condition`.

## Human Sessions

A UTF-8 CSV or JSON list containing `session_id`, `specimen_key`, `task`, `operator_id`, `geometry_version`, `reference_version`, `completion`, `false_stop`, `measurement_cost`, `route_cost`, `cost_unit`, and `route_cost_unit`. For process-level interpretation also provide `information_permission`, `report_production`, `action_sequence`, `first_stop_or_handoff`, and `final_report_id` or `final_report_sha256`. Use `FROZEN_NATIVE_8X8_THREE_LEVEL`, `NATIVE_RASTER_FRACTION`, `FROZEN_GRID_ROUTE_COST`, `MATCHED_TO_FROZEN_MODEL`, and `COMMON_READER_UNEDITED` only when those contracts are actually satisfied. Missing process fields remain explicit and restrict the result to a full-system endpoint comparison.

## Command

```bash
PYTHONPATH=src python scripts/analyze_bc_cscan_frozen_process.py finalize \
  --config paper_v3/configs/bc_cscan_frozen_process_analysis.yaml \
  --source-root /home/ww/paper3/cmc_damage_inference \
  --references <reference-directory> \
  --blind-reviews <blind-review.csv> \
  --human-sessions <human-sessions.csv-or-json>
```
