# Codex Handoff: Inspection Agent G1 Observable Policy

## Repository

- Repository: `git@github.com:Orangekostar/diff.git`
- Base SHA: `7a10cd425de582fa158bf6639285731ccd8ff7a7`
- Branch: `research/inspection-agent-g1-observable-policy`
- Worktree:
  `/home/ww/diff/.worktrees/inspection-agent-g1-observable-policy`
- Evidence commit: `9d483949e2f44be0e3d189454572e27ff5a5873c`
- Final documentation commit: the commit containing this handoff. Its enclosing
  Git SHA cannot be embedded in its own contents; the final local/upstream/remote
  SHA is reported by the post-push verification and final Codex response.
- Evidence-package push checkpoint: local = upstream = remote
  `9d483949e2f44be0e3d189454572e27ff5a5873c`; the branch was created without
  force push and the 132,516,962-byte LFS object upload completed.
- Final local/upstream/remote: verified equal after the documentation push; the
  exact enclosing commit SHA is reported in the final Codex response because a
  tracked file cannot contain the SHA of the commit that contains that file.
- Final clean-worktree check: required and verified after the documentation
  push; exact command output is reported in the final Codex response.

Meaningful commits from the G0 evidence base through the evidence commit:

```text
59510ef audit: freeze G1 evidence and deployment geometry
25ee3e7 feat: add fold-safe teacher crossfitting
d8d21c7 feat: add observable inspection policy state
8b6d68f feat: add structured action policy and legal masking
b0c5cd0 exp: add hard and soft privileged distillation
0568baf feat: add reproducible G1 source teacher banks
2e48aaf exp: add source-only policy relabeling
1339ab6 exp: add conditional privileged advantage training
9fb2449 feat: add conservative observable stopping
9e4a6b3 feat: add causal G1 rollout and target freeze
d67a9f7 feat: freeze G1 metrics statistics and gates
064691d feat: add deterministic G1 artifact replay
e2cfb40 feat: add fold-safe G1 policy training
cdedab4 feat: freeze source-only G1 model selection
c250920 feat: bind G1 protocol and teacher bank storage
86a182f feat: materialize fold-safe G1 source banks
cfca6c0 feat: add G1 source-bank CLI
38ceb94 feat: add resumable G1 teacher-bank batch build
66f8aed feat: fit final five-source G1 dependencies
13871c3 feat: add common-geometry G1 formal execution
eb078f8 fix: honor frozen G1 teacher-bank work path
7fc8fc8 feat: resume G1 teacher-bank batches by fold
c317b8f feat: add source-only staged G1 selection
5c30dd3 feat: persist resumable G1 outer selection
87e60c3 feat: add warm-started G1 evaluation oracle
a5d948c feat: add hash-bound G1 stop banks
8bed02c feat: materialize fold-safe G1 fixed endpoints
be6bf51 feat: materialize source-only G1 stop labels
491ea46 feat: train independent observable G1 stop heads
6f0d2c0 feat: persist G1 DAgger relabel banks
b8dea77 feat: relabel source-world G1 DAgger states
9431b29 feat: persist G1 source engineering bridges
4c1517a feat: materialize G1 source bridge curves
dcb821d feat: orchestrate G1 source bridge banks
c34871d refactor: centralize G1 curve replay validation
ed706e3 feat: persist learned G1 source policy curves
c4228d5 feat: evaluate G1 actors on source worlds
27ab4c6 feat: build G1 learned source curve banks
f139092 feat: select G1 actors by source engineering curves
653e936 perf: batch observable G1 policy rollouts
eba197a feat: persist cross-fitted G1 DAgger banks
cb13bdd feat: select G1 DAgger by source engineering evidence
6de9783 perf: checkpoint G1 source bridge builds
903e50b perf: batch observable G1 stop inference
9a1197a feat: evaluate G1 source stop validation curves
3756083 feat: materialize G1 source stop rollouts
0e28316 feat: select G1 stopping from source rollouts
77ef736 feat: freeze G1 target policy trajectories
76d4352 feat: evaluate frozen G1 target policies
6837173 feat: analyze formal G1 target evidence
ef31842 feat: assemble formal G1 evidence package
f0f9895 feat: freeze G1 source decision diagnostics
bc27a12 feat: report frozen G1 surface strata
bcecd7c exp: add conditional privileged advantage training
27e7e9a feat: bind G1 training routes to formal evidence
9d48394 exp: add formal G1 held-out-domain evaluation
```

## Frozen G0

- G0 status: `G0_ACTIVE_INSPECTION_OPPORTUNITY_GO`.
- G0 artifact manifest:
  `a85a62f14bd05d69c684deab1673e01a1a84d7ebf9c3e7805760c2898eacc179`.
- G0 checksum ledger:
  `30cf5b9c639b3017b90990a62d9ea6c162116cdf804bda80a30333f9bebe5e22`.
- G0 output-tree SHA:
  `e1441d847eaf187eb98de7eb84e93b708225924b5263d99560354120a7f30b0a`.
- G0 formal/replay package SHA:
  `429f829b60bc9f520a41814ae2b6d34d05ef07cdfa188f39e2d9dbac93c45eca`.
- G0 decision/protocol/config hashes: `41421833efe3a458c6f098a10dbd349d134519e5f21e134a422345585238f147`,
  `0d59ea13060d573f92d1cdb46e90abf4ecd55170f4a8f1487a7ff2a11dd65193`,
  and `cb056637f88294876c27a4ae4094b4802a125f43b2d625dae93b28accb084cbc`.
- Frozen-path diff against the base: empty.

## Literature And License Boundary

The frozen `LITERATURE_METHOD_LEDGER.md` records all sources and links. AAWR is
a conceptual privileged-critic precedent; its official repository exposed no
declared license, so no code was copied. DAgger and AWR were independently
implemented from their papers; the official AWR repository is MIT. DriveAgent-R1
and ActiveVLA are conceptual active-perception precedents with no usable declared
code license at audit time. Fuentes is sequential ultrasonic GP/BO prior art with
no authoritative reusable implementation found. TADRED and US-VLA repositories
declare Apache-2.0, but no code was copied. No VLM stage was authorized.

## Deployment Geometry And Warm Start

The same visible native geometry mapped to different G0 domain presets, so using
the old preset would leak domain identity. G1 resolved this case-B dependence
with the universal registered nominal grid 0.015625 and recomputed every bridge
curve. The deployment gate is `G1_DEPLOYMENT_GEOMETRY_GO`.

The exact K=8 cells are `[0,63,7,56,27,38,3,24]`. Exact warm-start budgets are
294/114920 = 0.0025583014270797078 for 338x340, 305/118976 =
0.0025635422270037654 for 338x352, and 1128/454950 =
0.0024793933399274645 for 674x675. Every state retains 56 BROADEN and eight
immediate REFINE actions. K4/K16 remain sensitivity settings only.

## Crossfit And Leakage Gate

Aliases: A=`74t7kcdgkr`, B=`cgtnjyggtm`, C=`w68dtmpfyf`, D=`xcmzfsbd9t`,
E=`yfxyg8jm46`, F=`ykhs7s2dck`. For every row, both prior and CAI assessor use
the same four-domain fit roster:

| Outer | Teacher-labeled source | Prior/assessor fit domains |
|---|---|---|
| A | B | C,D,E,F |
| A | C | B,D,E,F |
| A | D | B,C,E,F |
| A | E | B,C,D,F |
| A | F | B,C,D,E |
| B | A | C,D,E,F |
| B | C | A,D,E,F |
| B | D | A,C,E,F |
| B | E | A,C,D,F |
| B | F | A,C,D,E |
| C | A | B,D,E,F |
| C | B | A,D,E,F |
| C | D | A,B,E,F |
| C | E | A,B,D,F |
| C | F | A,B,D,E |
| D | A | B,C,E,F |
| D | B | A,C,E,F |
| D | C | A,B,E,F |
| D | E | A,B,C,F |
| D | F | A,B,C,E |
| E | A | B,C,D,F |
| E | B | A,C,D,F |
| E | C | A,B,D,F |
| E | D | A,B,C,F |
| E | F | A,B,C,D |
| F | A | B,C,D,E |
| F | B | A,C,D,E |
| F | C | A,B,D,E |
| F | D | A,B,C,E |
| F | E | A,B,C,D |

All 30 roster checks pass. Final deployment dependencies use all five non-target
domains and were frozen before target rollout. Every formal selection records
`target_outcomes_opened=false`; the package records
`target_outcomes_opened_during_selection=false`, `no_target_leakage=true`, and
six sealed target trajectories before evaluation-only truth access.

## Teacher Bank

Thirty base banks contain 46,920 records over 276 physical specimens, with
23,460 records per task. Source-domain row counts are 7,650/8,330/7,310/
10,030/7,140/6,460 in A-F order. Each physical specimen contributes 170 rows
across its five source roles; each outer/source specimen contributes 34.

State sources are WARM_START 2,760, ORACLE_CHECKPOINT 11,040, and 8,280 each
for UNIFORM_CONTINUE, SURFACE_FOCUS_CONTINUE, RANDOM_CONTINUE, and
ALTERNATE_BROADEN_REFINE. Candidate count min/median/max/mean is
1/62/64/46.35880221653879. Candidate occurrences are FOCUS 82,334, BROADEN
668,635, and REFINE 1,424,186. Sixty source-only DAgger banks add 88,320
actor-visited records. Selected base teacher decisions are FIELD
1,885/9,155/12,420 and CAI 1,942/10,387/11,131 for
FOCUS/BROADEN/REFINE; DAgger decisions are FIELD 6,069/14,482/16,121 and CAI
4,264/15,461/16,947. STOP banks contain 20,249 sufficient and 26,671
insufficient labels. Exact row-level and aggregate hashes, versions, and the
rematerialization command are in `G1_TEACHER_BANK_AUDIT.md`.

## Model Selection And Training Route

The candidate roster contained two sub-million-parameter actors, hard BC and
soft utility distillation, two CAI context modes, registered temperature/LR/WD
grids, DAgger 0/1/2, conditional AAWR, and seven STOP thresholds. All selection
was inner-source-only.

Every fold selected `SharedActionMLP` (260,098 parameters),
`SOFT_UTILITY_DISTILL`, tau 0.5, LR 0.0003, weight decay 0.0001, shared observable
CAI context, and the correct task token. DAgger round 2 was selected except for
outer E, which selected round 0. Source-only AAWR authorized CAI in five folds
but every evaluated AAWR candidate lost to the selected distillation actor;
outer E was `NOT_RUN_NOT_AUTHORIZED`. Thus all final actors are non-AAWR.
STOP was `STOP_NOT_AUTHORIZED` for both tasks in every fold.

Source-only diagnostic accuracy was 0.8068218988443118 at the high-level
decision, 0.13508592485972484 primitive top-1, and 0.303918434815895 top-5
utility recall; expected teacher regret was 0.00014463970670956552 and candidate
utility NDCG 0.8425698860949694. These diagnostics were not target gates.

## Held-Out G1 Results

| Component | Fixed AUEBC | Learned AUEBC | Oracle AUEBC | Fixed minus learned (95% CI) | Domains | Gap closure |
|---|---:|---:|---:|---:|---:|---:|
| FIELD | 0.002047837067539446 | 0.002120618581802132 | 0.0019655610066778156 | -0.0000727815142626863 [-0.00008689672989010225, -0.00006182037360547485] | 0/6 | -0.8846013469833925 |
| CAI | 0.02859496549734356 | 0.029653166484753902 | 0.007802877380960266 | -0.0010582009874103384 [-0.0020696313885150072, -0.000059502896739864196] | 2/6 | -0.0508944066361726 |

FIELD fixed references are SURFACE_FOCUS in all folds. CAI uses RANDOM in four
folds and CENTER_FIRST in two. Both policy components are NO_GO.

Task controls, expressed as comparator minus correct: FIELD WRONG_TASK
0.0000030030636740644057 (CI [0.0000012334469453508708,
0.000005033437921055967], 4/6) and NO_TASK 0.0000017949810500134293
([0.0000006563384341036675, 0.0000031700703012368995], 4/6) pass. CAI
WRONG_TASK 0.00006731435252771757 ([-0.00007233853622756369,
0.0002084059247615635], 4/6) and NO_TASK 0.00003107987501816518
([-0.00011143823251693787, 0.0001704391507672262], 4/6) do not. Combined status:
`G1_TASK_CONDITIONING_NO_GO`.

Surface controls, again comparator minus correct, are FIELD NO_SURFACE
0.00000045434448030215293 (CI [-0.000012668568159756931,
0.000009324295637885728], 4/6), FIELD SHUFFLED -0.000002043157793511139
([-0.000015982797216280143, 0.000010279149341085201], 3/6), CAI NO_SURFACE
-0.00076450645226755 ([-0.001571119192022546, 0.00005345423365951419],
3/6), and CAI SHUFFLED 0.000056822168960363406
([-0.0006233811088856258, 0.000720148044443176], 4/6). All CIs include zero.
The frozen G0 strata contain AGREE 1, PARTIAL 114, and MISLEADING 161; all
MISLEADING analyses remain diagnostic and inconclusive.

STOP selected no threshold in 0/6 domains for either task. Both policies use the
fixed endpoint, so saving is 0.0 (CI [0.0,0.0]), premature-stop rate 0.0, and
never-stopped fraction 1.0. FIELD task-loss ratio is 1.4281895236048958 and CAI
is 2.538297473468669. Statuses are `G1_FIELD_STOPPING_NO_GO` and
`G1_CAI_STOPPING_NO_GO`.

Final decision: `G1_POLICY_OBSERVABILITY_NO_GO`; G2 is not authorized. The G0
privileged opportunity is not sufficiently observable from current deployment
evidence under this registered policy class.

## Formal Package And Replay

- Formal/replay files: 19 each; corresponding bytes and file hashes match.
- `action_trajectories.parquet`: 2,760 records.
- `action_score_audit.parquet`: 301,525 records.
- `state_level_metrics.csv`: 150,624 records.
- `per_specimen_metrics.csv`: 6,072 records.
- `bootstrap.csv`: 100,000 synchronized physical-specimen replicates.
- Artifact-manifest SHA:
  `994fe76bf20e8d5e7bb07e1b2b431bba0916ce62727ab5b27df81692f4804929`.
- Output-tree SHA:
  `0deec49686793289afa45680b0390581a4020b4615ed86fcb418a8a29419852b`.
- Formal/replay package SHA:
  `f444187542bd583dad5f689a87fc1f725940b81f9596eb1a8768e3de9699bf14`.
- `validate` passed for both packages; `compare` returned
  `byte_identical=true`; both checksum ledgers passed all 18 entries.

## Tests And Reproduction Evidence

Environment: Python 3.13.13, NumPy 2.5.1, Polars 1.42.0, scikit-learn 1.9.0,
PyTorch 2.12.1+cu130, CUDA 13.0, cuDNN 9.2.0, NVIDIA A40. Training used AdamW
and the registered deterministic context: explicit NumPy PCG64 and Torch
CPU/CUDA seeds, deterministic Torch algorithms, cuDNN benchmark false, and TF32
false. The Python standard-library RNG is not used in G1, so its seed is
`NOT_APPLICABLE_NO_PYTHON_RANDOM_CALLS`.

```bash
python -m ruff check src/cmc_bbdm/inspection_agent_g1 \
  tests/test_inspection_agent_g1*.py scripts/run_inspection_agent_g1.py
# All checks passed.

PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  tests/test_inspection_agent*.py
# 278 passed in 319.55s (0:05:19)

python scripts/run_inspection_agent_g1.py validate \
  --config paper_v3/configs/inspection_agent_g1.yaml --project-root . \
  --path results/inspection_agent/g1
# passed: G1_POLICY_OBSERVABILITY_NO_GO

python scripts/run_inspection_agent_g1.py validate \
  --config paper_v3/configs/inspection_agent_g1.yaml --project-root . \
  --path results/inspection_agent/replay/g1
# passed: G1_POLICY_OBSERVABILITY_NO_GO

python scripts/run_inspection_agent_g1.py compare \
  --config paper_v3/configs/inspection_agent_g1.yaml --project-root . \
  --formal results/inspection_agent/g1 \
  --replay results/inspection_agent/replay/g1
# passed: byte_identical=true

git diff --check
# passed, no output

git lfs fsck
# Git LFS fsck OK
```

## Complete Changed-File Inventory

The complete base-to-evidence inventory is reproducible with:

```bash
git diff --name-only 7a10cd425de582fa158bf6639285731ccd8ff7a7..9d483949e2f44be0e3d189454572e27ff5a5873c
```

It contains `.gitattributes`, `.gitignore`, the five pre-result G1 audit/ledger
documents, G1 protocol/design/plan/config, `scripts/run_inspection_agent_g1.py`,
all 42 files in `src/cmc_bbdm/inspection_agent_g1/`, all 55
`tests/test_inspection_agent_g1*.py` files, and the exact 19-file formal and
19-file replay packages. This documentation commit additionally adds:

```text
artifacts/inspection_agent_g1/G1_TEACHER_BANK_AUDIT.md
artifacts/inspection_agent_g1/G1_MODEL_SELECTION_AUDIT.md
artifacts/inspection_agent_g1/G1_FIELD_POLICY_RESULT.md
artifacts/inspection_agent_g1/G1_CAI_POLICY_RESULT.md
artifacts/inspection_agent_g1/G1_TASK_CONDITIONING_RESULT.md
artifacts/inspection_agent_g1/G1_SURFACE_ROBUSTNESS_RESULT.md
artifacts/inspection_agent_g1/G1_STOPPING_RESULT.md
artifacts/inspection_agent_g1/G1_FINAL_DECISION.md
artifacts/inspection_agent_g1/CODEX_HANDOFF_INSPECTION_AGENT_G1.md
```

## GitHub Verification

```bash
git push -u origin research/inspection-agent-g1-observable-policy
git rev-parse HEAD
git rev-parse @{upstream}
git ls-remote origin refs/heads/research/inspection-agent-g1-observable-policy
git status --short
```

Evidence-package checkpoint and final verification values are filled only from
actual command output; no PR, merge, force push, or post-target rescue is used.
`action_score_audit.parquet` is stored through Git LFS because each identical
formal/replay object is 132,516,962 bytes; object SHA-256 is
`e1350fe8bd67e36d2b208cd36a9d8de751621e62117bb23733dc0166d0447d4e`.
