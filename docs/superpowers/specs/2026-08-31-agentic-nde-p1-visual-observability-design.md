# Agentic NDE P1 Visual Observability Design

## Scope

Implement the already authorized P1 experiment without reopening P0R or any
frozen MVA/MVD/MAVIS result. The design is source-trained nested LODO visual
observability followed by target-only evaluation and exact-cost CAI curves.

## Boundaries

- `surface_cells.py` owns P0R inverse cell boxes and deterministic RGB crops.
- `surface_encoder.py` owns the frozen RGB ResNet18 transform and cache.
- `visual_observability.py` owns label-isolated examples, heads, controls,
  fusion, ranking metrics, bootstrap, and the P1 gate.
- `p1.py` owns strict config loading, phase ordering, outer workers, exact-cost
  CAI evaluation, package assembly, and replay.
- Existing MVA/MVD/MAVIS code and scientific result trees remain read-only.

## Phase barrier

For each outer domain: build label-free inputs; train/select on source labels;
fit on all five sources; generate and hash target scores; only then load target
mechanical values and CAI outcomes for evaluation. Tests must fail if target
labels enter a selector or fitter.

## Data flow

```text
P0R surfaces + inverse cell mapping -> frozen RGB embeddings
MVD candidate banks + A2 current prediction -> 521-D old state
five source-domain A2 value maps -> nested source selection/refit
outer label-free inputs -> frozen 64-cell scores
outer A2 values -> action metrics
frozen ranking + exact native cost -> CAI curves/AUEBC
specimen-first paired bootstrap -> four-way P1 decision
```

## Determinism

All source files, feature arrays, model states, score tables, controls, seeds,
runtime versions, and output files are hash-bound. Formal assembly is atomic.
Replay rebuilds into a separate directory and compares exact bytes.
