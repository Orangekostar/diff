# G1 Privilege and Split Contract

Frozen before production training. This contract is fail-closed: missing domain
identity, missing freeze identity, or a mixed actor/teacher record is an error.

## Domain Roster

Aliases below are documentation shorthand only; model features never contain them.

| Alias | Dataset ID | Specimens |
|---|---|---:|
| A | `74t7kcdgkr` | 45 |
| B | `cgtnjyggtm` | 49 |
| C | `w68dtmpfyf` | 43 |
| D | `xcmzfsbd9t` | 59 |
| E | `yfxyg8jm46` | 42 |
| F | `ykhs7s2dck` | 38 |

For every row below, both the source background prior and StateCAIAssessor use
exactly the four `Fit` domains. Neither the outer target nor labeled domain may
appear in fit rows, normalization, PCA, tuning, or teacher dependency hashes.

| Outer | Labeled source | Fit |
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

Final target dependencies are refit on all five non-outer domains after all
inner-source selections are complete. They are hashed and frozen before target
rollout.

## Access Matrix

| Component | May read | Must reject |
|---|---|---|
| Actor feature builder | `InspectionObservation`, `SurfaceHypothesis`, source-safe reconstruction, observable CAI estimate, native geometry | authority/view objects, dataset/specimen ID, full/future scan, true CAI/loss, oracle utility |
| Actor models | fixed actor tensors, legal mask | raw mappings with privileged keys, teacher tensors, target identity |
| Causal world | private specimen key and `_reveal_values` for requested positions | returning identity, future values, full scan, true CAI |
| Crossfit prior fit | source teacher views from exactly the registered four/five fit domains | outer target or labeled-source row in four-domain fits |
| Crossfit assessor fit | label-independent source rows and source true CAI from registered fit domains | target/labeled domain, oracle-selected state, unequal specimen weighting |
| FIELD/CAI teacher | source full scan/true CAI, crossfit dependencies, current issued observation | outer target during training, actor API exposure |
| STOP teacher | source current/reference true losses | true loss in STOP-network input or threshold tuning on target |
| DAgger | causal outer-source worlds and source teachers | outer-target world, target state, target teacher query |
| Privileged critic | source full-scan embedding, source true CAI/loss | privileged actor input, target transition |
| Target rollout | frozen source dependencies/policy/threshold, causal observations | hidden target truth, target oracle, target metric feedback before seal |
| Target evaluation | sealed target trajectory/scores/state/value hashes, evaluation view | changing any model, threshold, route, K, objective, or gate |

## Teacher-Bank Physical Separation

Every logical record has three namespaces:

```text
policy_visible:
  state tensors and legal mask only
privileged_teacher:
  source-only candidate utilities, losses, selected action, STOP label
integrity:
  outer/labeled/fit roster identities, hashes, state source, specimen key
```

The persisted table may flatten these with prefixes, but tensor extraction accepts
only `policy_visible.*`. Supplying any `privileged_teacher.*`, `true_cai`,
`full_scan`, `dataset_id`, `specimen_id`, `oracle_value`, `future_measurement`, or
`true_task_loss` to the feature/model path raises an exception.

## Target Freeze State Machine

```text
DEPENDENCIES_FROZEN
  -> TARGET_ROLLOUT_OPEN          (causal reveal only)
  -> TARGET_TRAJECTORY_SEALED     (actions, scores, STOP scores, state/value hashes)
  -> TARGET_EVALUATION_OPEN       (hidden full scan/true CAI/oracle allowed)
  -> TARGET_EVALUATION_SEALED
```

Skipping or reversing a transition is forbidden. The seal binds outer selection,
dependency/model/threshold hashes, variant, task, specimen, trajectory, action
scores, STOP scores, acquired positions/values, and state hashes. An intentional
pre-seal target truth request must fail in tests.

## Source-Only Selection

Within each outer fold, leave one of the five source domains out for inner
validation. Fit normalization/PCA/models on the other four, score only the inner
held-out source, aggregate inner domains equally, select by registered priority,
then refit the selection on all five sources. DAgger and conditional AAWR operate
only in those source worlds. `outer_selection.json` is sealed before target truth
access.

## Prohibited Reuse

Frozen G0 teacher trajectories are evidence, not G1 training rows. Their G0
assessors/priors generally include the new outer target. G1 must rematerialize all
teacher states and labels under the deployment grid and the table above.
