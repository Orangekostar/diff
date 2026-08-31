# Agentic NDE P1 Visual Observability Implementation Plan

> Execute only after the commit freezing the P1 protocol and YAML config.

1. Add strict config/authority tests, including source hashes, P0R roster,
   target filter, old-state dimensions, transform hash, controls, gates, and
   exact output roster.
2. Add `surface_cells.py` tests and implementation for inverse-P0R boxes,
   half-open integer crops, all D4 controls, shuffled donors, and Sattolo
   derangements.
3. Add `surface_encoder.py` tests and implementation for RGB preprocessing,
   frozen 512-D ResNet18 inference, feature-cache hashes, and replay.
4. Add `visual_observability.py` tests and implementation for isolated data
   roles, Ridge/MLP heads, nested source selection, fusion, controls, action
   metrics, specimen-first bootstrap, and the four-status gate.
5. Add `p1.py` tests and implementation for six outer workers, score freeze,
   target evaluation, existing exact-cost CAI evaluation, atomic package
   assembly, artifact reports, CLI commands, and replay.
6. Run unit and leakage tests before any formal target evaluation.
7. Materialize and independently replay the feature cache, then run the six
   formal outer folds and CAI workers using the frozen config.
8. Assemble the exact 15-file result package and two authority reports; replay
   to byte equality; issue the P1 status.
9. Follow the controlling prompt's conditional P2/P3/P4 decision tree without
   expanding an unauthorized route.
