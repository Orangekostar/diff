# G0 Repository Code Map

Audit base: `892d92ea4979d9ca8ceeafef3348cd43266ed1b8`.

| Role | Existing authority | Exact symbol / behavior | G0 disposition |
|---|---|---|---|
| Hidden C-scan and CAI | `src/cmc_bbdm/mavis/authority.py` | `MAVISAuthority`, `_reveal_values`, `source_teacher_view`, `evaluation_view` | Wrap; only `_reveal_values` is reachable by the causal world. Teacher/evaluation views remain outside policy APIs. |
| Old causal reveal | `src/cmc_bbdm/mavis/reveal.py:72-100` | `reveal_uniform_scout` calls MVA `initial_state` before any adaptive action | Reuse invariants, not the zero-start implementation. |
| Old state constraint | `src/cmc_bbdm/mva/measurement_state.py:37-53` | levels are only 0/1/2 and `initial_state=(0,)*64` | Frozen; replace with a new generalized state in `inspection_agent`. |
| Geometry | `src/cmc_bbdm/mva/acquisition_grid.py` | `build_acquisition_grid`, `AcquisitionGrid`, 8x8 cell lattices | Reuse unchanged. |
| Exact mask/cost pattern | `src/cmc_bbdm/mva/measurement_state.py:84-102,146-166` | union mask and candidate added cost | Reimplement for level -1 while preserving native unique-pixel semantics. |
| Uniform order | `src/cmc_bbdm/mva/oracle.py:24-45` | deterministic farthest-point `uniform_cell_order()` | Reuse for ZERO_UNIFORM and fixed staged order. |
| Old rollout | `src/cmc_bbdm/mavis/rollout.py` | `_rollout_curve` begins with `reveal_uniform_scout` | Historical baseline only. |
| Historical B4 actions | `results/mavis/p4_closed_loop/action_trajectories.parquet` | 276 specimens, `mavis_full`, exact per-action cost | Prepend zero-to-scout uniform actions; replay frozen post-scout actions. |
| Zero-summary precedent | `src/cmc_bbdm/mavis/state_encoder.py:238-301` | `summarize_mris_input` accepts zero acquired positions and reports zero acquisition fraction | Semantic precedent only; primary G0 omits its old context. |
| Unrealistic old context | `src/cmc_bbdm/mavis/contracts.py:72-97`; `state_encoder.py:304-323` | fixed 34-D context; indices decode laminate, ply count, impact energy | Excluded from primary G0; historical metadata-augmented baseline only. |
| Interpolation | `src/cmc_bbdm/mva/interpolation.py` | `_interpolate_rectilinear`, exact restoration, cell patch ownership | Reuse bilinear primitive from new source-safe generalized reconstructor. Frozen file unchanged. |
| FIELD loss | `src/cmc_bbdm/mva/reconstruction_value.py` | `normalized_rgb_mse` | Reuse unchanged. |
| Discovery semantics | `src/cmc_bbdm/mva/appearance_value.py` | RGB deviation from full-image border median | Extend from mean incremental value to cumulative saliency mass; evaluation only. |
| CAI encoder | `src/cmc_bbdm/mva/encoder_session.py`; `cpb_v3/embeddings.py` | frozen ImageNet ResNet18, deterministic 512-D output | Reuse with registered weights SHA256 `f37072...e07ec`. |
| CAI regression | `src/cmc_bbdm/mva/cai_evaluator.py` | fold-local PCA and Ridge | Reuse with three observable state scalars and state-unique sample IDs. No metadata13/profile_stats21. |
| Budget metrics | `src/cmc_bbdm/mva/budget_metrics.py` | AUEBC/sufficiency/saving semantics | Reuse concepts; add zero-inclusive G0 curves and exact G0 stopping reference. |
| Bootstrap precedent | `src/cmc_bbdm/mva/statistics.py` | domain-level resampling | New implementation resamples physical specimens within domain synchronously, then equal-weights domains. |
| Surface registration | `src/cmc_bbdm/agentic_nde/surface_cells.py` | `load_surface_cell_authority`, `crop_rgb_patch`, P0R boxes | Reuse unchanged; no registration reopening. |
| P0R authority | `results/agentic_task_driven_nde/p0r_author_registration/` | 276 authorized ROT90 records | Frozen input and replay gate. |

## New ownership boundary

Only `src/cmc_bbdm/inspection_agent/`, `scripts/run_inspection_agent.py`, the G0
config/protocol, new tests, and new G0 result/audit directories may be written.
All existing MVA/MVD/MAVIS and historical P0R/P1 science paths are immutable.

The causal policy surface will contain observation, belief, and legal action
types only. Privileged teachers live in oracle/evaluation orchestration and are
not accepted by deployable policy-facing functions.
