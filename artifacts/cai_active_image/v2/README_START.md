# Start Here

Main method: `VLM_CAI_FEEDBACK_AGENT`, with frozen VLM-guided initialization and a feedback Actor that reselects after every acquired C-scan cell.

- Protocol: `paper_v3/configs/cai_active_image_v2.yaml`
- Results: `results/cai_active_image/v2/summary.json`
- Primary effects: `results/cai_active_image/v2/vlm_ablation_effects.csv`
- Absolute metrics: `results/cai_active_image/v2/absolute_cai_performance.csv`
- Handoff: `CODEX_HANDOFF_CAI_VLM_GUIDED_AGENT_V2.md`

Run a stage with `python scripts/run_cai_active_image_v2.py <stage> --source-root /path/to/cmc_damage_inference`.
