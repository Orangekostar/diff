# Reproduce pooled expert rescore

Run from the repository worktree:

```bash
PYTHONPATH=src python scripts/cscan_human_review.py export-references \
  --session /home/ww/diff/.worktrees/cscan-human-review-html/.local/cscan_human_review_html/reference_packets/reference-20260909070923.json \
  --packet /home/ww/diff/.worktrees/cscan-human-review-html/.local/cscan_human_review_html/reference_packets/reference_packet_001.json \
  --output-root .local/cscan_human_review_html/expert_reference_exports
PYTHONPATH=src python scripts/cscan_human_review.py export-references \
  --session /home/ww/diff/.worktrees/cscan-human-review-html/.local/cscan_human_review_html/reference_packets/reference-20260909073915.json \
  --packet /home/ww/diff/.worktrees/cscan-human-review-html/.local/cscan_human_review_html/reference_packets/reference_packet_002.json \
  --output-root .local/cscan_human_review_html/expert_reference_exports
PYTHONPATH=src python scripts/cscan_human_review.py export-references \
  --session /home/ww/diff/.worktrees/cscan-human-review-html/.local/cscan_human_review_html/reference_packets/reference-20260909081350.json \
  --packet /home/ww/diff/.worktrees/cscan-human-review-html/.local/cscan_human_review_html/reference_packets/reference_packet_003.json \
  --output-root .local/cscan_human_review_html/expert_reference_exports
PYTHONPATH=src python scripts/cscan_human_review.py export-references \
  --session /home/ww/diff/.worktrees/cscan-human-review-html/.local/cscan_human_review_html/reference_packets/reference-20260909082011.json \
  --packet /home/ww/diff/.worktrees/cscan-human-review-html/.local/cscan_human_review_html/reference_packets/reference_packet_004.json \
  --output-root .local/cscan_human_review_html/expert_reference_exports
PYTHONPATH=src python scripts/rescore_cscan_expert_pool.py prepare \
  --source-root /home/ww/paper3/cmc_damage_inference \
  --export-root .local/cscan_human_review_html/expert_reference_exports \
  --original-session-root /home/ww/diff/.worktrees/cscan-human-review-html/.local/cscan_human_review_html/reference_packets
PYTHONPATH=src python scripts/rescore_cscan_expert_pool.py run \
  --source-root /home/ww/paper3/cmc_damage_inference
PYTHONPATH=src python scripts/rescore_cscan_expert_pool.py summarize
```

`run` is the sole reviewed recovery command. `summarize` only reads completed reviewed outputs and must not replay recovery.
