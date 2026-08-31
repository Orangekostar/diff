# Inspection Agent G0 Implementation Plan

1. Commit the source-backed preflight audit, privilege map, literature ledger,
   frozen protocol, configuration, and design before target evaluation.
2. Write failing zero-state, transition, exact-cost, MVA-equivalence, and causal
   leakage tests; implement `contracts.py`, `state.py`, and `world.py`.
3. Write failing surface/prior/reconstruction tests; implement transparent
   saliency, source-only prior, generalized interpolation, and FIELD metrics.
4. Write failing state-bank and assessor tests; implement six fixed policies,
   equal specimen state rows, frozen encoder representation, LODO PCA/Ridge,
   exclusion audits, and the CAI authorization gate.
5. Write failing oracle/task-swap/stopping/statistics tests; implement privileged
   trajectories, fixed and historical baselines, exact checkpoint evaluation,
   synchronized bootstrap, and registered gates.
6. Write failing artifact/replay/frozen-path tests; implement the CLI, canonical
   package writer, manifest/checksums, and full replay.
7. Run focused tests, all `test_inspection_agent_*.py`, Ruff, then the existing
   agentic-NDE regression suite. Run the formal G0 once, freeze its package, run
   independent replay, and require byte identity.
8. Generate result audits and handoff from actual artifacts, run frozen-path Git
   diff gates and full verification, review the complete diff, commit all outcomes,
   push the isolated branch, and require local/upstream/remote SHA equality.

Validation commands are fixed to:

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_inspection_agent_*.py
PYTHONPATH=src python -m pytest -q -p no:cacheprovider $(rg --files tests | rg '/test_agentic_nde.*\.py$' | sort)
python -m ruff check src/cmc_bbdm/inspection_agent scripts/run_inspection_agent.py tests/test_inspection_agent_*.py
git diff --check
```

Formal and replay CLI commands will use the same frozen config, local research
root, external Hasebe project root, and frozen source project root; only the
output directory differs.
