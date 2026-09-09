# Codex Handoff: BC C-scan Expert Pooled Rescore

## Repository identity

- Repository: `git@github.com:Orangekostar/diff.git`
- Branch: `research/bc-cscan-expert-pooled-rescore`
- Worktree: `/home/ww/diff/.worktrees/bc-cscan-expert-pooled-rescore`
- Development base: `0e7f15b5c4cb59b05f19b2c598cdc9a9278cc15e`
- Frozen scientific evidence commit: `2102cc4a1726910931dfaaf20e29ad29a20eaf2e`
- Frozen configuration base identity: `fe58de39298c412d0829580cfda40b7c7c4c53e9`
- This file belongs to the commit returned by `git rev-parse HEAD`; the post-push terminal verification records the final local/upstream/remote SHA.

## Expert reference input

- Reference set: `EXPERT_POOL_V1`
- Scoring identity: `REVIEWED_02339eda86d486c3`
- Coverage: 24 unique TEST specimens, six domains with four specimens each.
- Sources: two experts (`ww`, `phl`) across four disjoint six-specimen sessions.
- Analysis grouping: `POOLED_NOT_BY_REVIEWER`; experts do not increase physical N.
- All 24 references contain a confirmed certain region; none contains an uncertain region.
- Pooled path: `results/bc_cscan_expert_pooled_rescore/v1/inputs/references/`
- Source/provenance: `results/bc_cscan_expert_pooled_rescore/v1/inputs/reference_provenance.csv`
- Original session paths and SHA-256 values are recorded in `inputs/reference_set_manifest.json`.
- The 24 pooled reference files preserve the exported JSON bytes, polygon coordinates, source identities, reference types, and reviewer aliases.

## Implementation

- `src/cmc_bbdm/learned_cscan/expert_pooled_rescore.py`: byte-preserving pooling, output-only config gate, inherited proxy dependencies, one-run wrapper, read-only reviewed summary, claims, acceptance checks, and figures.
- `scripts/rescore_cscan_expert_pool.py`: separate `prepare`, `run`, and `summarize` commands.
- `paper_v3/configs/bc_cscan_expert_pooled_rescore_v1.yaml`: exact copy of the frozen process config except output and artifact roots.
- `tests/test_cscan_expert_pooled_rescore.py`: focused reference, config, aggregation, claim, invariant, no-recovery-summary, and committed-output contracts.

The existing Reader, report adapter, reviewed evaluator, STOP event logic, cost definitions, bootstrap, and Path B decision function were reused without modification.

## Execution record

- Four `export-references` calls: one for each source session; 24 exported, zero rejected.
- `prepare`: two invocations. The second only relocated the export staging tree under the repository's ignored `.local/cscan_human_review_html/` path and refreshed provenance; pooled bytes and reference version did not change.
- Formal `run`: one invocation of `execute_frozen_finalize`; no failed recovery retry.
- `summarize`: six read-only invocations while result presentation, checksum coverage, and figure validation were finalized. The function contains no recovery/finalize call.
- Reviewed recovery: 288 trajectories, 55,584 report states, 55,296 stored action transitions, 48 full-input reports, and 576 episode/STOP rows.
- Cache hit: zero specimens/episodes; all 55,296 stored transitions were processed once in the formal run.
- Main four methods retain 187 actual STOP events and five no-STOP episodes across 192 task/specimen/method runs.
- Training updates: 0.
- New VLM calls: 0.
- Actor forward calls: 0.
- STOP forward calls: 0.
- World steps: 0.

## Main reviewed results

### Full-input Reader

| Task | Success | Mean IoU | Mean recall | Mean relative area error |
|---|---:|---:|---:|---:|
| LOCATE | 6/24 | 0.2610 | 0.2500 | 0.0000 |
| CHARACTERIZE | 0/24 | 0.2173 | 0.4734 | 2.7002 |

LOCATE recall and area-error fields retain their registered task-specific semantics. CHARACTERIZE's zero success is retained in all comparisons; no specimen was excluded.

### Planner AUSC: BC three-seed mean versus P8

| Task | P8 | BC three-seed | Difference | 97.5% CI | Status |
|---|---:|---:|---:|---:|---|
| LOCATE | 0.1912 | 0.2092 | +0.0180 | [-0.0083, 0.0457] | NOT_SUPPORTED |
| CHARACTERIZE | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | NOT_SUPPORTED |

The old proxy planner effects (+0.0971 LOCATE and +0.0403 CHARACTERIZE versus P8) do not transfer as supported reviewed claims. The reviewed LOCATE point estimate remains positive but the registered interval crosses zero; CHARACTERIZE has no successful reviewed state.

### Original calibrated STOP and Path B

| Task | Method | Completion | False stop | Exhausted | Failure cost | Prefix measurement cost |
|---|---|---:|---:|---:|---:|---:|
| LOCATE | P8 | 0.2500 | 0.7500 | 0.0000 | 0.9066 | 0.6726 |
| LOCATE | BC three-seed | 0.2361 | 0.7639 | 0.0000 | 0.9125 | 0.6712 |
| CHARACTERIZE | P8 | 0.0000 | 0.8750 | 0.1250 | 1.0000 | 0.8135 |
| CHARACTERIZE | BC three-seed | 0.0000 | 0.9722 | 0.0278 | 1.0000 | 0.7797 |

Path B is `NOT_SUPPORTED` for both tasks. Completion noninferiority passes, but the registered failure-cost-improvement condition does not pass. STOP positions, no-STOP identities, and actual prefix costs are unchanged from the frozen trajectories.

### Existing seed-1 Actor input ablations

| Task | Registered input | Difference | 97.5% CI | Status |
|---|---|---:|---:|---|
| LOCATE | Surface cues | +0.0107 | [-0.0071, 0.0283] | NOT_SUPPORTED |
| LOCATE | Ultrasound feedback content | +0.2013 | [0.0732, 0.3372] | SUPPORTED |
| CHARACTERIZE | Surface cues | 0.0000 | [0.0000, 0.0000] | NOT_SUPPORTED |
| CHARACTERIZE | Ultrasound feedback content | 0.0000 | [0.0000, 0.0000] | NOT_SUPPORTED |

These are seed-1 Actor input increments only. They do not establish whole-system absence effects or multi-seed ablations.

## Proxy/reviewed boundary

The frozen reports and actions are identical, but proxy and reviewed scoring differ in reference and protocol details. The main comparison is therefore described as a result under changed reference and evaluation protocol, not a causal effect of annotation alone. Exact differences are documented in `SCORING_PROTOCOL_NOTE.md` and the paired tables.

Legacy C1-C3 in `final_evidence_manifest.json` remain explicitly `PROXY_LEGACY`. They are not the new reviewed conclusion. The reviewed authority is:

1. `REVIEWED_RESULTS_AND_CLAIM_BOUNDARIES.md`
2. `REVIEWED_SUBMISSION_EVIDENCE_MATRIX.md`
3. `results/bc_cscan_expert_pooled_rescore/v1/reviewed_claims.json`

## Pending external evidence

- Blinded report review: `PENDING_INPUT`; no returned blind-review CSV was supplied.
- Human scan/planning comparison: `PENDING_INPUT`; no operator session file was supplied.
- These tracks were not fabricated, substituted with reference annotation, or awaited before completing the reviewed rescore.

## Verification evidence

Commands executed from the task worktree:

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  tests/test_cscan_expert_pooled_rescore.py \
  tests/test_bc_cscan_frozen_process_analysis.py \
  tests/test_cscan_human_review_tool.py
python -m ruff check \
  src/cmc_bbdm/learned_cscan/expert_pooled_rescore.py \
  scripts/rescore_cscan_expert_pool.py \
  tests/test_cscan_expert_pooled_rescore.py
git diff --check
```

- Focused tests: 58 passed.
- Acceptance manifest: `PASSED`, including 24 references, six domains, 48 full-input rows, 288 trajectories, 55,584 states, 55,296 transitions, 576 episode rows, 187/5 STOP coverage, exact report/STOP identity, zero model calls, and three nonblank figures.
- Frozen legacy result/artifact paths changed: none.
- Full-repository tests were not run, as required for this bounded rescore.

## Primary artifacts and hashes

- `results/bc_cscan_expert_pooled_rescore/v1/reviewed/report_scores.parquet`: `89edf03b0f1ade210a017e042c05ae6059c787078167917e35bcf17673602490`
- `results/bc_cscan_expert_pooled_rescore/v1/reviewed/per_episode_metrics.csv`: `5ade632f68bacada2547308f6f2a8bf5a78b5822de7af97d9714dcefc54cae20`
- `results/bc_cscan_expert_pooled_rescore/v1/reviewed/planner_effects.csv`: `a6cadb84abb43ca2e7d6339292e1426ffe2a21e27bf1ec8f5ce8c437c294e764`
- `results/bc_cscan_expert_pooled_rescore/v1/reviewed_claims.json`: `e2cc0a745aeb28836467b86a28370a641ae2fc09ff8d38d7889f5900f5d9e4eb`
- `results/bc_cscan_expert_pooled_rescore/v1/acceptance_results.json`: `7846866d68a01d71d704d812245ca2f5d8e1f270115995248375d4dd858c5234`
- `results/bc_cscan_expert_pooled_rescore/v1/CHECKSUMS.sha256`: `accb1e3e2f83d85d3f44a7d2a6a36faffcb7b057d888580dda42a2dbc38369c2`
- `REVIEWED_RESULTS_AND_CLAIM_BOUNDARIES.md`: `573e9c2afa88a03ec354124961891b6e8574013a728ca5c8f8e9fd42be91f864`

## GitHub delivery

Required command:

```bash
git push -u origin research/bc-cscan-expert-pooled-rescore
```

After push, verify `git rev-parse HEAD`, `git rev-parse '@{upstream}'`, and `git ls-remote origin refs/heads/research/bc-cscan-expert-pooled-rescore` are identical. No PR, merge, or force push is part of this task.
