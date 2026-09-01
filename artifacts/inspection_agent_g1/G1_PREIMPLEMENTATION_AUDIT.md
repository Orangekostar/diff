# G1 Preimplementation Audit

Audit date: 2026-09-01  
Repository: `git@github.com:Orangekostar/diff.git`  
Branch: `research/inspection-agent-g1-observable-policy`  
Base and parent evidence commit: `7a10cd425de582fa158bf6639285731ccd8ff7a7`

This audit was completed before production training or formal target evaluation.

1. **Does the base resolve locally, upstream, and remotely?** Yes. Local
   `git cat-file -t` returns `commit`; local
   `origin/research/agentic-nde-inspection-reasoning-g0` and remote
   `refs/heads/research/agentic-nde-inspection-reasoning-g0` both resolve to
   `7a10cd425de582fa158bf6639285731ccd8ff7a7`.

2. **Are G0 formal/replay packages byte-preserved?** Yes. Formal and replay
   manifests are both `a85a62f14bd05d69c684deab1673e01a1a84d7ebf9c3e7805760c2898eacc179`;
   both checksum ledgers are
   `30cf5b9c639b3017b90990a62d9ea6c162116cdf804bda80a30333f9bebe5e22`.
   The G0 validator/comparator reports `byte_identical=true`, package/replay SHA
   `429f829b60bc9f520a41814ae2b6d34d05ef07cdfa188f39e2d9dbac93c45eca`,
   manifest SHA `a85a62f...`, and tree SHA
   `e1441d847eaf187eb98de7eb84e93b708225924b5263d99560354120a7f30b0a`.

3. **Does `InspectionObservation` exclude privilege?** Yes.
   `inspection_agent/contracts.py:InspectionObservation` contains surface RGB/hash,
   task, native geometry/count, grid/state identity, acquired positions and values,
   exact acquired count, endpoint, and action history. It has no true CAI,
   full/future C-scan, dataset/domain ID, specimen ID, oracle score, or true loss.

4. **Which functions may expose privilege?** The actor must never reach
   `MAVISAuthority.source_teacher_view()`, `evaluation_view()`,
   `policy_context()`, `_reveal_values()`, `_record`, or `_specimens`;
   `choose_field_action()`, `choose_cai_action()`, source STOP-label generation,
   and evaluation loss/oracle functions are also teacher/evaluation only.
   `_reveal_values()` remains reachable only through
   `CausalInspectionWorld.step()/replay()` and returns only requested positions.

5. **What fields exist in `OracleCandidateScore`?** Exactly `action`,
   `exact_added_cost`, `raw_value`, `objective_value`, `task_loss_after`, and
   `candidate_state_sha256` in `inspection_agent/oracle.py`.

6. **Are all candidate utilities retained?** Yes. FIELD and CAI selection return
   an `OracleSelection` whose `candidates` tuple contains every legal fitting
   one-level candidate, not only the selected action. G1 adds deterministic
   `decision_type` when materializing records.

7. **Can `choose_field_action()` query replayed observable states?** Yes. It takes
   an issued observation/state plus teacher-only full scan and source prior, so a
   source world can replay any legal action history and query that state.

8. **Can `choose_cai_action()` query replayed observable states?** Yes. It likewise
   accepts an arbitrary issued observation with teacher-only view, cross-fitted
   prior, assessor, and encoder.

9. **Which G0 CAI assessor served each domain?** The assessor whose
   `outer_domain` equals the evaluated G0 domain, fit on the other five domains:

   | G0 evaluated domain | G0 assessor fit domains |
   |---|---|
   | `74t7kcdgkr` | `cgtnjyggtm,w68dtmpfyf,xcmzfsbd9t,yfxyg8jm46,ykhs7s2dck` |
   | `cgtnjyggtm` | `74t7kcdgkr,w68dtmpfyf,xcmzfsbd9t,yfxyg8jm46,ykhs7s2dck` |
   | `w68dtmpfyf` | `74t7kcdgkr,cgtnjyggtm,xcmzfsbd9t,yfxyg8jm46,ykhs7s2dck` |
   | `xcmzfsbd9t` | `74t7kcdgkr,cgtnjyggtm,w68dtmpfyf,yfxyg8jm46,ykhs7s2dck` |
   | `yfxyg8jm46` | `74t7kcdgkr,cgtnjyggtm,w68dtmpfyf,xcmzfsbd9t,ykhs7s2dck` |
   | `ykhs7s2dck` | `74t7kcdgkr,cgtnjyggtm,w68dtmpfyf,xcmzfsbd9t,yfxyg8jm46` |

10. **Would direct G0 CAI-teacher reuse leak the new outer target?** Yes. A G0
    assessor labeled for source domain `D_s` excludes `D_s` but generally includes
    the new G1 outer target `D_o`. G1 rematerializes labels using dependencies that
    exclude both `D_o` and `D_s`.

11. **Would direct G0 FIELD-trajectory reuse leak through the prior?** Yes. The G0
    prior for `D_s` likewise includes all domains except `D_s`, hence generally
    includes `D_o`. G1 rematerializes FIELD states/labels with a dual-exclusion
    prior.

12. **Which existing state-bank states are label-independent?** G0's zero anchor
    and all geometry-, surface-, or seeded-random continuation states are generated
    without task labels. Their semantics may be reused, but their stored trajectories
    are not G1 training data because geometry and fold dependencies change.

13. **How many states require no privileged trajectory selection?** Thirteen per
    specimen/task: one state immediately after K=8 plus four continuation policies
    at three action-count fractions. Four cross-fitted oracle checkpoints bring the
    recommended pre-DAgger maximum to 17; exact duplicates are collapsed and counted.

14. **Which cell-wise observable features are available?** Cell row/column,
    surface score/top-eight membership, current level, observed unique-pixel
    fraction, observed RGB mean/std, observed RGB deviation from the source prior,
    observed-value mask, and previous-action-on-cell flag. Unmeasured RGB statistics
    are exactly zero. Interpolated unseen pixels never enter cell tokens.

15. **How is exact candidate cost computed?** From the set difference produced by
    `action_added_positions_from_mask()` or equivalently the count delta from
    `candidate_budget_record()`, divided by `native_height*native_width`. Shared
    boundary pixels are counted once.

16. **Can the same visible geometry have different G0 presets?** Yes. All three
    observed shapes do: `(338,340)` has 10 specimens at `0.015625` and 7 at
    `0.03125`; `(338,352)` has 10 and 9; `(674,675)` has 211 and 29.

17. **Is the G0 domain budget recoverable from scanner geometry alone?** No. The
    mapping is many-to-many, so using it would reveal hidden domain membership.

18. **What universal rule removes that dependence?** Build every G1 grid from
    native geometry with registered nominal grid `0.015625`, the smallest of
    `{0.015625,0.03125,0.0625}` valid for all three authorized geometries. All G1
    fixed/learned/oracle curves are recomputed under this bridge.

19. **Why not a complete level-0 scout?** At all 64 cells at level 0 no state level
    remains `-1`; therefore no `-1->0` action and hence no FOCUS or BROADEN action
    exists. That would remove the central BROADEN-versus-REFINE choice.

20. **How many BROADEN actions remain after K=8?** Exactly 56 `-1->0` actions;
    the eight initialized cells also supply eight legal `0->1` refinement actions.

21. **Is K=8 exact cost acceptable for every geometry?** Yes. Under `0.015625`,
    `(338,340)` uses 294/114920 = `0.0025583014270797078`, `(338,352)` uses
    305/118976 = `0.0025635422270037654`, and `(674,675)` uses 1128/454950 =
    `0.0024793933399274645`, all well below 0.25.

22. **How are normalization/PCA/model selection target-free?** In each outer fold,
    feature statistics, any PCA, every candidate fit, and all selection scores are
    computed only within the five source domains. Inner validation excludes its
    held-out source domain from fitting. After selection, final statistics and
    models are refit on all five outer sources and frozen before target rollout.

23. **How are CAI teacher labels cross-fitted?** For a source-labeled domain `D_s`
    under outer target `D_o`, `StateCAIAssessor_{D_s}` is fit on the four domains
    excluding both. Its fit rows are only the 13 label-independent post-K8 states;
    oracle-selected states are forbidden from assessor fitting.

24. **How are source priors cross-fitted?** The matching
    `SourceBackgroundPrior_{D_s}` uses equal-domain border medians from exactly the
    same four-domain roster and records its roster/hash. Final target rollout uses
    a prior fit on all five outer sources.

25. **How are STOP labels kept nondeployable?** On source data only, the fixed
    reference is selected using the other four domains; privileged current and
    reference true losses generate `current <= 1.05*reference`. Those values live
    only in `privileged_teacher`. The STOP network consumes only the observable
    policy state, and its conservative threshold is selected on source validation.

26. **Which target quantities open only after trajectory freeze?** Hidden full
    target C-scan, true target CAI, true task losses, target teacher/oracle utilities,
    fixed/learned/oracle target metrics, bootstrap effects, and final gates. A
    target-access guard requires the sealed trajectory/scores/state-hash ledger.

27. **Which files separate actor and teacher inputs?** `features.py` constructs
    `G1PolicyState` solely from `InspectionObservation`; `teacher.py` owns privileged
    queries; `teacher_bank.py` serializes distinct `policy_visible`,
    `privileged_teacher`, and `integrity` sections; `policy_training.py`,
    `utility_distillation.py`, and the actor branch of `privileged_awr.py` accept only
    extracted policy tensors. Privileged-column sentinels reject mixed input.

28. **How is DAgger source-only?** `dagger.py` receives an explicit outer split and
    rejects any world/record whose domain equals the outer target. It rolls only the
    five outer-source worlds with cross-fitted teachers, caps 16 deterministic
    quantile states per specimen/task/iteration, and records the domain roster.

29. **What is selected only on inner sources?** Architecture, hard versus soft
    objective, teacher temperature, CAI-context mode, learning rate, weight decay,
    DAgger iteration 0/1/2, AAWR authorization and its hyperparameters, STOP
    threshold, fixed comparator, and all surface/task variants. K=8, the split,
    task definitions, bootstrap seed, gates, and candidate roster are preregistered
    and are not selected.

30. **Which result authorizes G2?** Any of
    `G1_TASK_CONDITIONED_POLICY_GO`, `G1_ACTIVE_POLICY_GO`,
    `G1_CAI_ONLY_POLICY_GO`, or `G1_FIELD_ONLY_POLICY_GO`. Neither
    `G1_POLICY_OBSERVABILITY_NO_GO` nor `G1_DEPLOYMENT_BRIDGE_NO_GO` authorizes G2.

## Gate

`G1_DEPLOYMENT_GEOMETRY_GO`. Production implementation/training may begin only
after this audit, the binding/split contracts, deployment decision, protocol, and
frozen configuration are committed together.
