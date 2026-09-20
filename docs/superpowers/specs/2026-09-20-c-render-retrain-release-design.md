# C Render Retrain Release Design

## Authority

This design implements `CODEX_C_RENDER_RETRAIN_RELEASE_FINAL.md` from
`CAI_C_RENDER_RETRAIN_RELEASE_FINAL_PACKAGE.zip`. The standalone scope JSON is
the machine-readable authority. Frozen scientific values are copied without
reinterpretation.

## Outcome

The branch gains an isolated, resumable C=P0+R1 pipeline that:

1. prepares and verifies the 211 TRAIN/VALID specimens without accessing TEST;
2. reuses exact six-case pilot records and generates the remaining C priors;
3. retrains only the three VLM-dependent actors from their original seeds;
4. selects from all five 250-step candidates per actor and reloads winners;
5. combines 150 new selected episodes with 500 unchanged control episodes;
6. recomputes all C evidence, cases, tables, figures and local HTML;
7. builds an independent six-section C manuscript and two reviewed PDFs; and
8. verifies, commits and pushes the complete release to the existing branch.

Scientific performance does not gate completion. Missing inputs, unavailable
resources, exhausted recovery allowance or a failed build produce
`PARTIAL_EXECUTION`; old A values never fill missing C outputs.

## Architecture

`scripts/cai_c_retrain/cli.py` is the only public entry point. It loads the
scope, resolves the current repository root, validates immutable roots and
dispatches each phase. `all` starts phases through the interpreters recorded in
`runtime_lock.json`, so Qwen, Actor and report work never rely on an accidental
editable install.

The implementation lives under `scripts/cai_c_retrain/`:

- `context.py`: scope/path validation, hashes, atomic state, phase completion,
  append-only local and global resource records, subprocess dispatch.
- `prepare.py`: source/branch/runtime/resource checks, cohort lock, source
  bindings and C protocol lock.
- `vlm.py`: exact pilot renderer/parser contract, input manifests, six-case
  signature reuse, bounded generation/repair/interruption state machine and
  211-row actor feature export.
- `train.py`: explicit W2/C/output roots, three actor loops, 250-step atomic
  candidates and snapshots, one bounded resume, disk winner reload and smoke.
- `assemble.py`: immutable reuse checks, 500+150 assembly, row provenance and
  separate historical-A comparison.
- `evidence.py`: parameterized reuse of existing numeric functions, new
  bootstrap intervals, full reference, equal-quality/timing analyses, cases,
  figures and HTML.
- `paper.py`: copy source-only old manuscript, bind it to C evidence, rebuild
  MD/HTML/TeX/PDF and create visual-review inputs.
- `validate.py`: Q1-Q8 checks, release manifest, handoff and Git delivery
  evidence. Publish performs Git operations only and never starts research
  compute.

Old result, artifact and paper directories are immutable. Shared source
functions remain the authority for model construction, sampling, rollout loss,
evaluation, cost math and predictor loading. Any copied pilot renderer/parser
code is checked against all six known C image/signature records.

## State And Recovery

Every phase has a signature derived from the scope plus its immutable inputs.
A complete phase is reused only when its signature and output hashes match.
Changed inputs are an error, never an implicit cache hit.

VLM state is per specimen. Raw text and token IDs are atomically saved before
parsing. A stored terminal state is immutable. A STARTED call without raw data
conservatively consumes an attempt; each non-reused specimen has at most two
total attempts.

Actor state is per method and per 250-update segment. Update 0 and each closed
candidate include model, optimizer, Python/NumPy/Torch RNG, logical update,
selection state, signatures and run identity. A segment marker closes weights,
episodes and snapshot together. One incomplete segment may be charged and
replayed once; logical progress still stops at 1250.

## Verification

Development follows red-green-refactor with focused CPU tests. The real run
then supplies evidence unavailable to unit tests: 211 terminal records, 15
candidates, 750 candidate episodes, three disk-reload smokes, the 650-row
matrix, new tables/figures, PDF renders and remote SHA equality. The final
review maps each instruction requirement to one authoritative artifact or
fresh command result.
