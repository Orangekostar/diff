# BC C-scan Path-B Supplement Reproduction

Run from the repository root with the evidence-base files and external data root available. Replace `<source-root>` with the root containing `data/public/hasebe`, `data/public/hasebe_cai`, and the frozen authority inputs.

```bash
PYTHONPATH=src python scripts/run_bc_cscan_supplement.py audit --config paper_v3/configs/bc_cscan_path_b_supplement.yaml --source-root <source-root>
PYTHONPATH=src python scripts/run_bc_cscan_supplement.py import-references --config paper_v3/configs/bc_cscan_path_b_supplement.yaml --source-root <source-root>
PYTHONPATH=src python scripts/run_bc_cscan_supplement.py analyze-existing --config paper_v3/configs/bc_cscan_path_b_supplement.yaml --source-root <source-root>
PYTHONPATH=src python scripts/run_bc_cscan_supplement.py train-replicas --config paper_v3/configs/bc_cscan_path_b_supplement.yaml --source-root <source-root>
PYTHONPATH=src python scripts/run_bc_cscan_supplement.py train-ablations --config paper_v3/configs/bc_cscan_path_b_supplement.yaml --source-root <source-root>
PYTHONPATH=src python scripts/run_bc_cscan_supplement.py calibrate-stop --config paper_v3/configs/bc_cscan_path_b_supplement.yaml --source-root <source-root>
PYTHONPATH=src python scripts/run_bc_cscan_supplement.py evaluate --split test --config paper_v3/configs/bc_cscan_path_b_supplement.yaml --source-root <source-root>
PYTHONPATH=src python scripts/run_bc_cscan_supplement.py rescore --references results/vlm_cscan_efficiency/annotation_queue --config paper_v3/configs/bc_cscan_path_b_supplement.yaml --source-root <source-root>
PYTHONPATH=src python scripts/run_bc_cscan_supplement.py prepare-confirm --config paper_v3/configs/bc_cscan_path_b_supplement.yaml --source-root <source-root>
PYTHONPATH=src python scripts/run_bc_cscan_supplement.py summarize --config paper_v3/configs/bc_cscan_path_b_supplement.yaml --source-root <source-root>
```

Existing checkpoints and completed TEST outputs are identity-checked and recovered instead of retrained or overwritten. The current `rescore` result remains header-only because no attributable reviewed reference is available. Confirmation evaluation is intentionally not run until all 24 confirmation references are independently reviewed.

Verification:

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_bc_cscan_supplement.py
PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  tests/test_learned_cscan_perception_readout.py \
  tests/test_learned_cscan_controls.py \
  tests/test_learned_cscan_rollouts_metrics.py \
  tests/test_learned_cscan_cli.py \
  tests/test_learned_cscan_runtime.py \
  tests/test_learned_cscan_training.py
python -m ruff check src/cmc_bbdm/learned_cscan/bc_supplement.py src/cmc_bbdm/learned_cscan/supplement_adapters.py src/cmc_bbdm/learned_cscan/episode_stop_calibration.py src/cmc_bbdm/learned_cscan/supplement_analysis.py src/cmc_bbdm/learned_cscan/supplement_reporting.py src/cmc_bbdm/learned_cscan/supplement_figures.py scripts/run_bc_cscan_supplement.py tests/test_bc_cscan_supplement.py
git diff --check
sha256sum -c results/bc_cscan_path_b_supplement/CHECKSUMS.sha256
```
