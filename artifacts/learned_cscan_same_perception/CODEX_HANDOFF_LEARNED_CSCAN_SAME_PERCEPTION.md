# Codex Handoff: Learned C-scan Same-Perception Pilot

## Repository identity

- Repository: `git@github.com:Orangekostar/diff.git`
- Branch: `research/learned-cscan-same-perception`
- Base: `59a67511c0ee6395b68220c6a1644f40c383dfa2`
- Controlling prompt SHA-256: `2b9a1812b16c232564dfc9bde3847d2732495706e32532a4c82ed36b6418315d`
- Data root: `/home/ww/paper3/cmc_damage_inference`
- Final commit and verified remote SHA: reported in the completion response to avoid self-referential tracked metadata.

## Delivered behavior

The new `cmc_bbdm.learned_cscan` package implements the frozen shared-perception contract, strict action-free surface percept parsing/cache, Reader v2, rule controls, compact masked actor, cost-to-go labels, one TRAIN-only state-aggregation round, a separate learned STOP head, exact action-step metrics, physical-specimen bootstrap, and staged CLI gates.

The final run used 24 TRAIN, 12 VALID, and 24 TEST physical specimens across six domains. Surface perception completed for 60/60 specimens with Reader v2 nondegenerate on 120/120 full-input task readouts. No TEST result was opened before `validation_selection.json` was locked.

The final trained models are `L_BC`, `L_CTG_0`, `L_CTG_1`, and `S_LEARN`. VALID selected `L_CTG_1`, seed 1, and `R_BALANCED_P8`. The VALID learned-minus-rule AUSC signal was -0.236508, so seeds 2/3 and `L_NO_VLM` were correctly skipped. TEST contains 24 physical specimens, 336 episodes, and 64,848 trajectory rows.

## Scientific result

- LOCATE: `L_CTG - R_BALANCED = -0.216727`, 95% CI [-0.337556, -0.089456].
- CHARACTERIZE: `L_CTG - R_BALANCED = -0.158095`, 95% CI [-0.219955, -0.098379].
- `L_BC` TEST AUSC is 0.809955 LOCATE and 0.548819 CHARACTERIZE; `L_CTG` is 0.413091 and 0.309835.
- Proxy learned-planning effect: `NOT_SUPPORTED`.
- Formal planner effect: `INCONCLUSIVE`; reviewed reference coverage is zero.
- Learned STOP: VALID-authorized at 0.90, but TEST stop effect remains `INCONCLUSIVE`.
- VLM increment: `NOT_TESTED` under the negative-VALID gate.

No result supports claiming that cost-to-go learning outperforms the human-designed rule. The stronger BC diagnostic localizes the shortfall to the CTG supervision/training route, not to the mere use of a compact learned actor.

## Commands actually run

```bash
PYTHONPATH=src python scripts/run_learned_cscan.py prepare \
  --config paper_v3/configs/learned_cscan_same_perception.yaml \
  --source-root /home/ww/paper3/cmc_damage_inference
PYTHONPATH=src python scripts/run_learned_cscan.py perception \
  --config paper_v3/configs/learned_cscan_same_perception.yaml \
  --source-root /home/ww/paper3/cmc_damage_inference
PYTHONPATH=src python scripts/run_learned_cscan.py build-train-bank \
  --config paper_v3/configs/learned_cscan_same_perception.yaml \
  --source-root /home/ww/paper3/cmc_damage_inference
PYTHONPATH=src python scripts/run_learned_cscan.py train \
  --config paper_v3/configs/learned_cscan_same_perception.yaml \
  --source-root /home/ww/paper3/cmc_damage_inference
PYTHONPATH=src python scripts/run_learned_cscan.py validate \
  --config paper_v3/configs/learned_cscan_same_perception.yaml \
  --source-root /home/ww/paper3/cmc_damage_inference
PYTHONPATH=src python scripts/run_learned_cscan.py evaluate \
  --config paper_v3/configs/learned_cscan_same_perception.yaml \
  --source-root /home/ww/paper3/cmc_damage_inference --split test
PYTHONPATH=src python scripts/run_learned_cscan.py summarize \
  --config paper_v3/configs/learned_cscan_same_perception.yaml \
  --source-root /home/ww/paper3/cmc_damage_inference
```

VALID and TEST specimen episodes were dispatched to four independent processes to keep the full-resolution evaluation tractable. Output ordering follows the frozen split order; episode semantics and statistics are unchanged.

## Resource and training evidence

- Base bank: 192 states, 916 candidate suffixes.
- Aggregation: one round, 96 states, 541 candidate suffixes, 58,304 transitions.
- Training-side total: 179,206 logical transitions.
- Full pilot total: 285,190 logical transitions; cap 400,000.
- Surface calls: 60 initial specimen calls plus two one-shot repairs; 24 diagnostic calls.
- `L_BC`: 1,250 steps, loss 4.049588 to 0.453555.
- `L_CTG_0`: 3,750 steps, loss 1.562553 to 1.526504.
- `L_CTG_1`: 1,250 steps, loss 1.615381 to 1.596169.
- `S_LEARN`: 3,750 steps, loss 0.720053 to 0.280019.
- Peak recorded training GPU allocation: 110,337,536 bytes.

The BC target-space defect found before VALID/TEST was corrected deterministically from each stored legal mask and recorded behavior action. The expensive CTG suffix bank was reused unchanged. The final BC target is one-hot over all legal cells, and regression tests prevent a return to one-element softmax supervision.

## Generated artifacts

All required public result artifacts are under `results/learned_cscan_same_perception/`; `_work/training_bank.pt` and diagnostic scratch caches are regeneration-only and are not part of the checksum roster.

| Artifact | SHA-256 |
|---|---|
| `summary.json` | `10973ab457eb03671155d18f682e7e870920e2b479719fb6734606325e001edf` |
| `per_episode_metrics.csv` | `a0833519461189a544becf3273e8c442fd9dc22a5044580d1e3ca46be85da986` |
| `trajectories.parquet` | `12a389bb69cba09f01ed5299bd775b4e90487dbdc9b3ad2616731c9bea700b2e` |
| `comparisons.csv` | `c9d8f35ecea01af18a18e24415d7f7b87271b30f9dc4c2a77f9d9f456eb09fb3` |
| `failure_cases.csv` | `bc22612d31191921f3924397fc3ce5f59d7c4076b706e3b5f05762122e5661a4` |
| `model_manifest.json` | `e697464ce87a63cbf2f02091d351d1333c1db8e6e6cd77a22cb0ff3bdf9636f8` |
| `training_bank_manifest.json` | `e54c06142f3d19515ed2ea45688da01404b7fb774904ab2ea7cb64e8f916f8ef` |
| `validation_selection.json` | `3a5889e838645885583d9b5f1b0a4410e2b0f838129038b963b9fe02a54daad3` |
| `CHECKSUMS.sha256` | `d5ec6b0cb48ce8ac417d1ad05bd566850c32cd2386cc4232a94766c7b2b946d9` |

The model hashes are recorded in both `model_manifest.json` and `CHECKSUMS.sha256`. No source image, foundation-model weight, secret, or private communication is included.

## Validation and frozen boundaries

Final verification commands:

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_learned_cscan*.py
python -m ruff check src/cmc_bbdm/learned_cscan tests/test_learned_cscan*.py
git diff --check
git diff --name-only 59a67511c0ee6395b68220c6a1644f40c383dfa2 -- \
  src/cmc_bbdm/mva src/cmc_bbdm/mvd src/cmc_bbdm/mavis \
  src/cmc_bbdm/inspection_agent src/cmc_bbdm/inspection_agent_g1 \
  src/cmc_bbdm/vlm_cscan results/inspection_agent/g0 \
  results/inspection_agent/g1 results/vlm_cscan_efficiency \
  artifacts/vlm_cscan_efficiency
```

The frozen-science diff is empty. The VLM, historical G0/G1 models, CAI assessor, and historical result roots were not trained or modified.

## Reproduction and review resume

The staged commands above are idempotent where a final cache/TEST marker exists. To resume independent annotation, use the existing review package:

```bash
python scripts/run_vlm_cscan.py export-annotations
```

Follow `artifacts/vlm_cscan_efficiency/ANNOTATION_GUIDE.md`. Reviewed TEST references may rescore the frozen reports without retraining. If reviewed references alter TRAIN labels, thresholds, or Reader calibration, create a new versioned run rather than rewriting this result.

## GitHub synchronization

Delivery command:

```bash
git push -u origin research/learned-cscan-same-perception
```

After push, verify local, upstream, and `refs/heads/research/learned-cscan-same-perception` are identical and the worktree is clean. No PR, merge, or force push is part of this handoff.
