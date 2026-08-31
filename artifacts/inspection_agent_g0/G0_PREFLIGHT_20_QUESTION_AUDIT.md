# G0 Preflight: First 20 Questions

Audit date: 2026-08-31. Repository base and remote branch both resolve to
`892d92ea4979d9ca8ceeafef3348cd43266ed1b8`.

1. **Does the base resolve locally and remotely?** Yes. `git cat-file -t` reports
   a commit and `origin/research/agentic-task-driven-nde-author-registration`
   resolves to the exact 40-character base SHA. The isolated branch starts at it.

2. **Are historical P0R/P1 results byte-preserved?** Yes at preflight. The P0R
   and P1 Git trees are the base objects `ae4f06b83b4e04275142be9877bb58bcb1aa99f2`
   and `d4b436f7477fd4c7e58fafc0e6afee5abf8b9e7d`. Their registered checksum files
   hash to `470a1d...bb82` and `6d856b...f4fc`. Frozen-diff tests will enforce this.

3. **What forces old MAVIS to start after a complete level-0 scout?** MVA
   `initial_state()` returns `(0,)*64` (`measurement_state.py:50-53`). MAVIS
   `reveal_uniform_scout()` calls it (`reveal.py:72-96`), and `_rollout_curve()`
   invokes that scout before its candidate loop (`mavis/rollout.py`).

4. **What are the exact level-0 native masks?** The six resolution/budget
   families below are the complete authorized roster. SHA256 is over the native
   boolean mask bytes.

   | HxW | nominal scout | grid SHA256 | unique pixels | mask SHA256 |
   |---|---:|---|---:|---|
   | 338x340 | 0.015625 | `9f9ebc641c05f42ba2f509e907182fbe078c4a4a0f620818c85feb329e67e279` | 1764 | `2df0af4d909f10269b701e96072c4992ca209a6ade3c3443138617eb149e2097` |
   | 338x340 | 0.03125 | `542da1516accc65115fd5ba5bc7bd312767f625f72aca314102e86766d31dba2` | 3600 | `451af5cccdac1af84553bc3bf7450e7ca8aff9da0ffa01fb04944b6df5c679e7` |
   | 338x352 | 0.015625 | `1e6158e14643bff066811b375a2b20ab054ab072c48745b6b47db729ef0f81f0` | 1848 | `4e583afa8f8d18cf3af1ed577314495fea905145c310b87397da96755a893b20` |
   | 338x352 | 0.03125 | `96ceb3a4450c6eee2a5a25dd5ede5f02eb3ac38704783aeb0f158786ddc2f025` | 3720 | `bcb5430bc71842d782e85a93cedda48902a47e66ec92140454d6820fd0c5f255` |
   | 674x675 | 0.015625 | `9b013907df821ccaa85bcaa795167259a3371598ae902970065976d0b1ac5fed` | 7056 | `dd972c4094dfdeed10315aad235910507fc4abc74f40bc4e8d5ac6176e215213` |
   | 674x675 | 0.03125 | `1aa0a256e20a9a1c60ddb41e007c20d1ef49a29ffff570622990ff4ba42637b2` | 14161 | `ba8f65d486cf296169949b84be557a782c869595139b958467dd98a1e8724edd` |

5. **Can 64 level-0 lattices be acquired individually without changing their
   union?** Yes. For every family, sequential union of the 64 cell level-0
   lattices is byte-identical to MVA `measurement_mask(initial_state(grid))`.

6. **What cost does each `-1->0` add with shared boundaries?** Cost is computed
   against the current union. In canonical `uniform_cell_order()`, the complete
   per-family histograms are:

   - 338x340/.015625: `16x3,20x17,24x12,25x7,29x1,34x2,35x5,36x12,40x3,42x1,47x1`.
   - 338x340/.03125: `25x2,35x7,36x1,40x4,42x1,45x2,47x2,49x5,52x1,53x2,54x3,56x7,63x9,64x4,71x2,79x5,80x1,81x6`.
   - 338x352/.015625: `15x3,16x6,18x1,19x2,20x5,24x5,25x4,28x2,29x6,30x11,34x2,35x1,40x3,41x3,42x9,47x1`.
   - 338x352/.03125: `30x2,35x4,40x1,41x1,42x3,45x1,47x1,48x3,49x6,52x1,53x1,54x2,55x1,56x7,62x1,63x10,64x4,71x2,72x1,79x5,80x1,81x6`.
   - 674x675/.015625: `81x3,90x18,99x5,100x3,109x2,110x1,117x4,119x2,120x7,121x7,129x1,130x4,132x1,142x1,143x3,156x1,169x1`.
   - 674x675/.03125: `110x2,121x1,143x1,150x4,160x2,165x2,176x2,180x1,185x1,186x2,187x5,191x2,221x3,225x5,240x7,255x6,256x4,271x2,287x5,288x1,289x6`.

7. **Does all `-1->0` recover the old scout?** Yes. Histogram sums are exactly
   1764, 3600, 1848, 3720, 7056, and 14161, matching the masks in question 4.

8. **Can `MAVISAuthority` reveal only requested zero-start pixels?** Yes.
   `_reveal_values(specimen_id, positions)` validates and returns only indexed
   positions (`mavis/authority.py:227-240`). The new world will call it only for
   candidate-mask minus current-mask positions and will not expose authority views.

9. **What state summary already accepts zero measurements?** Static `MRISInput`
   plus `summarize_mris_input()` supports an empty position array; its token mask
   remains false and acquisition fraction is zero (`state_encoder.py:238-300`).

10. **Which old context features are unrealistic?** The old 34 dimensions are
    `metadata13 + profile_stats21`. They encode energy/thickness, ply count,
    laminate, impactor type, total energy, dimensions, and full surface-profile
    statistics. `state_encoder.py:304-323` explicitly recovers laminate, ply
    count, and impact energy from context indices.

11. **How does primary G0 exclude those features?** Policy observations contain
    no context vector. The CAI assessor accepts only a reconstruction embedding
    and three observable acquisition scalars. The only old-context path is the
    explicitly labeled historical B4 baseline.

12. **How are unmeasured cells reconstructed without target leakage?** Each
    outer fold fits one constant RGB prior by equal-weighting border medians from
    the five source domains. The generalized reconstruction function accepts the
    prior and acquired positions/values, but has no full-target-scan argument.

13. **Which frozen encoder is used?** The registered ImageNet1K V1 ResNet18
    post-pooling 512-D encoder via `MVAEncoderSession`, with weight SHA256
    `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.

14. **How is the state bank label-independent?** Six fixed policies use geometry,
    a frozen surface-only score, or a specimen-seeded random permutation. No
    policy reads CAI or C-scan outcomes. Each specimen contributes exactly three
    snapshots per policy, 18 total, so row count cannot alter specimen weight.

15. **Does CAI quality improve with real ultrasound evidence?** The new
    metadata-free zero-start claim is not established and is not assumed.
    Historical post-scout P4 evidence shows only plausibility: aggregate CAI MAE
    decreases slightly with additional measurements (for example frozen uniform
    reaches about 0.1238 at 25%). Formal G0 must compare zero and fixed 25% states
    under the new LODO assessor. Until its full gate passes, ORACLE_CAI remains
    unauthorized.

16. **What defines internal-signal discovery?** Per-pixel saliency is the sum
    across RGB channels of absolute deviation from the hidden full-C-scan border
    median. Cumulative acquired saliency divided by total full-field saliency is
    `internal_signal_saliency`. It is an evaluation diagnostic, not damage truth.

17. **What reproduces the old fixed-scout workflow?** B4 first performs all 64
    `-1->0` actions in frozen uniform order, then replays the specimen's frozen
    P4 `mavis_full` 0/1/2 actions. Logged old exact costs must match the generalized
    state after the scout prefix.

18. **What task-swap test is decisive?** At common exact-budget checkpoints,
    evaluate CAI-oracle states with FIELD MSE and FIELD-oracle states with CAI
    absolute error. For each task, same-specimen correct-minus-wrong advantage
    needs a positive synchronized-bootstrap lower CI and at least four improving
    domains. Different action sequences alone are insufficient.

19. **What defines stopping sufficiency?** The earliest state with task loss no
    more than `1.05 *` the same specimen's source-selected strongest fixed
    nonprivileged 25% reference loss. Saving is `1 - adaptive_budget/reference_budget`.

20. **What authorizes G1 or forces STOP?** Task-conditioned G1 requires assessor
    authorization, hierarchical headroom, both task-swap gates, and stopping
    headroom on at least one engineering task. Adaptive headroom without task
    swap permits only a task-agnostic active-inspection route. FIELD-only headroom
    permits FIELD G1. No material privileged FIELD/CAI headroom produces
    `G0_NO_AGENTIC_HEADROOM_NO_GO` and terminates planner learning. Assessor
    failure produces `G0_CAI_ASSESSOR_NO_GO` for CAI planning while FIELD may proceed.

All answers above precede implementation and target evaluation. Question 15 is
therefore a locked authorization test, not a positive result claim.
