# Input and Code Bindings

## Repository and Matrix

- Base commit: `fe58de39298c412d0829580cfda40b7c7c4c53e9`
- Config: `paper_v3/configs/bc_cscan_frozen_process_analysis.yaml` (`6fc7d88e72bec9a37f3527a0a137b3407fb11803d5f90547183f56c82db4f5d7`)
- Matrix: 24 physical specimens, 6 domains, 7 methods, 2 tasks, 336 episodes, 64,848 state rows.
- State/action binding: row `j` is state/report `j`; its action produces row `j+1`; a STOP at `j` excludes that action.
- Recovery scope: Reader-only replay of stored actions; report digest is checked, packet hash is not claimed for the fast path.

## Frozen Input SHA-256

- `results/bc_cscan_path_b_supplement/cohort_and_reference_coverage.csv`: `2b5938e93d8df8b738a527dba17127bdcc8ab0876e03001e7de6eb736471193f`
- `results/bc_cscan_path_b_supplement/paired_effects.csv`: `40fbb128bfb80686bdf49032de2f33363ed2c969415ebdeacae96fb77b05bb86`
- `results/bc_cscan_path_b_supplement/per_episode_metrics.csv`: `1f8ce073d79c3d38965415f8f74ebfb1159289cf4529cb50073ff5ad62814638`
- `results/bc_cscan_path_b_supplement/report_manifest.json`: `67a7d54e14f0cfe6516391af0486a483c1d1d61289c2215b429703ec8609c057`
- `results/bc_cscan_path_b_supplement/risk_coverage.csv`: `d35a028f3e23b39710bd2772a69ae8df0c6301b0b826eccf331abaf1ba72c027`
- `results/bc_cscan_path_b_supplement/stop_calibration.json`: `281fca34979d0b482932e6d1b29c37faa4604b0ec7707cfa0921984ef3d22b02`
- `results/bc_cscan_path_b_supplement/summary.json`: `744d5cdc625561ee9994d43cf7e4a97080fb18a1581ff0f1fb3e34b6bf088a27`
- `results/bc_cscan_path_b_supplement/trajectories.parquet`: `5bf8e4ba677be063f3d434ce947c541e8dc213bf5d017c07b893d3d74a9b50c1`
- `paper_v3/configs/bc_cscan_path_b_supplement.yaml`: `1bf9cdf85083bfd6a71d04479d362cb3d8c4b0eaf07236e19ab0cd7c076cb0bd`
- `paper_v3/configs/learned_cscan_same_perception.yaml`: `12268dcaf470f769c326007702f7b0b6f4a13cf326447eef652a6dda972b5662`
- `results/learned_cscan_same_perception/surface_percepts.jsonl`: `85f400119093b362e2eb76489c9ae158ecdd453ce84ad04b6dfab7f04384693e`

## Trajectory Columns and Dtypes

- `schema_version`: `Int64`
- `dataset_id`: `String`
- `specimen_id`: `String`
- `specimen_key`: `String`
- `task`: `String`
- `method`: `String`
- `seed`: `Int64`
- `step`: `Int64`
- `cost`: `Float64`
- `success`: `Boolean`
- `task_loss`: `Float64`
- `iou`: `Float64`
- `recall`: `Float64`
- `relative_area_error`: `Float64`
- `report_sha256`: `String`
- `packet_sha256`: `String`
- `candidate_cell_count`: `Int64`
- `support_count`: `Int64`
- `evidence_candidate_cell_delta`: `Int64`
- `evidence_support_delta`: `Int64`
- `evidence_task_loss_delta`: `Float64`
- `action_cell`: `Int64`
- `action_from_level`: `Int64`
- `action_to_level`: `Int64`
- `candidate_score_kind`: `String`
- `top_candidate_cells`: `String`
- `top_candidate_scores`: `String`
- `selected_action_score`: `Float64`
- `rule_stop`: `Boolean`
- `learned_stop_probability`: `Float64`
- `learned_stop_eligible`: `Boolean`
- `cumulative_route_cost`: `Float64`
- `cumulative_route_turns`: `Int64`
- `route_start_row`: `Float64`
- `route_start_column`: `Float64`
- `route_end_row`: `Float64`
- `route_end_column`: `Float64`
- `action_inference_seconds`: `Float64`
- `evaluation_split`: `String`
- `reference_version`: `String`
- `stop_model`: `String`
- `calibrated_threshold`: `Float64`
- `calibrated_stop_trigger`: `Boolean`
- `trajectory_mode`: `String`

## Function Bindings

- `verify_frozen_input`: `hashlib.sha256` and the predeclared config identities.
- `analyze_trajectory_tables`: `exact_step_integral`, `make_domain_bootstrap_draws`, and `paired_domain_bootstrap`.
- `recover_spatial_analysis`: `_load_percepts`, `FrozenVisibleReportReader`, `_report_digest`, `_proxy_score`, `action_added_positions_from_mask`, and `apply_action` over stored actions only.
- `recover_reviewed_report_scores`: the same stored-action Reader path plus `adapt_task_report_v2` and `evaluate_task_report`; no Actor, STOP, or VLM inference.
- `finalize_frozen_evidence`: reviewed-score aggregation, physical-specimen/domain bootstrap, external file identity checks, blind-review pairing, and human-session comparability gates.
- `execute_frozen_analysis` / `execute_frozen_finalize`: existing atomic CSV, Parquet, JSON, text, and checksum writers under the new output roots only.
