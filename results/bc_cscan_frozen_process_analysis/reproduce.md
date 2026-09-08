# Reproduce

```bash
PYTHONPATH=src python scripts/analyze_bc_cscan_frozen_process.py analyze \
  --config paper_v3/configs/bc_cscan_frozen_process_analysis.yaml \
  --source-root /home/ww/paper3/cmc_damage_inference

PYTHONPATH=src python scripts/analyze_bc_cscan_frozen_process.py finalize \
  --config paper_v3/configs/bc_cscan_frozen_process_analysis.yaml \
  --source-root /home/ww/paper3/cmc_damage_inference
```

Both commands are CPU-only. They perform no training, VLM call, Actor forward call, STOP forward call, threshold scan, or action selection.
