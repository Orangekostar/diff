# Codex Handoff: CAI Agent v3 Controlled Reuse

## Repository

- Repository: `git@github.com:Orangekostar/diff.git`
- Branch: `research/cai-vlm-agent-v3-controlled-reuse`
- Base SHA: `4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d`
- Raw image execution root: `/home/ww/paper3/cmc_damage_inference`
- Delivery commit and remote SHA: reported in the final response after push, avoiding a self-referential commit.

## Outcome

`P_all=MEAN_SC` passed the predictor gate. The corrected seed-1 W3 pilot also passed its internal VALID continuation gate: `VLM_SPATIAL_FEEDBACK=44.627637 MPa`, `GEOMETRY_SPREAD=46.967150 MPa`, and `VLM_SPATIAL_OPEN_LOOP=44.852927 MPa`. The GDFS diagnostic completed at `46.335553 MPa` with unchanged predictor hashes. Seed expansion was not run because its 9,000 updates would exceed the registered 28,100 limit after conservatively accounting for invalidated work. TEST perception and scoring remained closed.

## Data and perception

- Cohort: 276 specimens and 259 source-derived capture groups over six domains.
- Group-isolated split: TRAIN 161/152 groups, VALID 50/48 groups, TEST 65/59 groups.
- Author workbook cross-check: 276/276 `CAI_STRENGTH` MPa matches.
- Feature bank: 60 v2 cell-token records reused after source checks; 216 newly encoded; six shards, each below 50 MiB.
- TEST targets are NaN in feature shards; full-image diagnostic tokens are not available to the Actor.
- Frozen Qwen perception: 205/211 valid fit results and 6 explicit terminal-unavailable fallbacks; 167 cumulative calls, including 12 failed schema calls.

## Integrity remediation

During W3 review, a float32 cumulative-cost round trip was shown to alter reachable hard-action legality for 18 TRAIN and 5 VALID specimens. The run was stopped; three completed checkpoints were invalidated and removed, and the interrupted static job is charged at an honest upper bound of 750 updates. Actor and GDFS hard legality now use float64 native-pixel fractions, and all corrected policy/GDFS episode routes replay exactly. See `results/cai_agent_v3/new_protocol/w3_cost_legality_audit.json`.

## Resource and stage gates

- Known optimizer updates: 21,514.
- Interrupted invalid static updates: unknown, bounded by 750 because the old loop lacked a durable per-update counter.
- Conservative used upper bound: 22,264 / 28,100; remaining lower bound: 5,836.
- Expansion requirement: 9,000; result: `RESOURCE_LIMITED`, 0 new updates.
- TEST VLM: `NOT_EXECUTED_POLICY_EXPANSION_NOT_LOCKED`, 0 calls.
- TEST scoring: `NOT_EXECUTED_POLICY_EXPANSION_NOT_LOCKED`, labels accessed=false.

## Key artifacts

- Legacy correction: `results/cai_agent_v3/legacy_v2_rescore/`
- Cohort/features/predictors/policies: `results/cai_agent_v3/new_protocol/`
- W3 gate: `results/cai_agent_v3/new_protocol/policy_pilot_gate.json`
- GDFS: `results/cai_agent_v3/new_protocol/gdfs_pilot.json`
- Resource gate: `results/cai_agent_v3/new_protocol/policy_expansion.json`
- Action trace: `results/cai_agent_v3/new_protocol/policy_valid_action_trace.csv`
- Three fixed-hash figures: `results/cai_agent_v3/new_protocol/figures/`
- Figure manifest: `results/cai_agent_v3/new_protocol/figure_manifest_valid.json`
- Reviews and boundaries: `artifacts/cai_agent_v3/REQUIREMENTS_REVIEW.md`, `requirements_review_final.json`, and `RESULTS_AND_CLAIM_BOUNDARIES.md`

## Actual execution commands

```text
python /home/ww/diff/docs/CODEX_HANDOFF_CAI_AGENT_V3/metric_reference.py --self-test
python scripts/run_cai_agent_v3.py --project-root . prepare
python scripts/run_cai_agent_v3.py --project-root . encode-features --source-root /home/ww/paper3/cmc_damage_inference --device cuda:1
python scripts/run_cai_agent_v3.py --project-root . precheck-predictors --device cuda:0
python scripts/run_cai_agent_v3.py --project-root . train-predictors --device cuda:0
python scripts/run_cai_agent_v3.py --project-root . train-oof --device cuda:0
python scripts/run_cai_agent_v3.py --project-root . refresh-cost-evaluation --device cpu
python scripts/run_cai_agent_v3.py --project-root . run-vlm --source-root /home/ww/paper3/cmc_damage_inference --scope fit
python scripts/run_cai_agent_v3.py --project-root . precheck-policies --device cuda:0
python scripts/run_cai_agent_v3.py --project-root . train-policy-pilots --device cuda:0
python scripts/run_cai_agent_v3.py --project-root . precheck-gdfs --device cuda:0
python scripts/run_cai_agent_v3.py --project-root . train-gdfs-pilot --device cuda:0
python scripts/run_cai_agent_v3.py --project-root . train-policy-expansion --device cpu
python scripts/run_cai_agent_v3.py --project-root . run-vlm --source-root /home/ww/paper3/cmc_damage_inference --scope test
python scripts/run_cai_agent_v3.py --project-root . evaluate-test --device cpu
python scripts/run_cai_agent_v3.py --project-root . export-diagnostics --source-root /home/ww/paper3/cmc_damage_inference --scope valid
```

GPU commands were physically restricted to one selected GPU with CPU thread variables capped at four. Commands above omit the repeated environment prefix for readability. Invalidated and failed attempts remain in `compute_ledger.jsonl`; they are not hidden from resource totals.

## Reproduction boundary

The repository includes code, configuration, light checkpoints, VLM cache, sharded feature bank and generated evidence. Raw source images are not duplicated. No legacy scientific result directory was modified. The final Git verification must compare local `HEAD`, upstream tracking SHA and `git ls-remote` for this branch; no PR, merge or force push is part of this handoff.
