# Reproduce

From the repository worktree, with the locked Python environment available:

```bash
python -m scripts.cai_order_mechanism.run all --root . --scope docs/cai/order_mechanism/ORDER_MECHANISM_SCOPE.json
```

The phase state hashes inputs and outputs. Completed phases are reused; the single permitted recovery was already recorded. The report reads cached tables only and never invokes a research model. For a focused local acceptance run:

```bash
pytest -q tests/test_cai_order_mechanism.py
python -m scripts.cai_order_mechanism.run verify --root . --scope docs/cai/order_mechanism/ORDER_MECHANISM_SCOPE.json
```
