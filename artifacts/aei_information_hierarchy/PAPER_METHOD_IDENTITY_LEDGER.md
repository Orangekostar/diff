# Paper Method Identity Ledger

This ledger is the naming authority for the manuscript, supplementary
material, figures, tables, validation, and submission package. It separates the
paper-level scientific method from one concrete implementation and from all
references, oracles, and source controls.

| Entity | Scientific role | Main-method status | Main-text naming |
|---|---|---|---|
| Task-Relevant Information Acquisition | paper-level proposed framework | PRIMARY METHOD | proposed framework / approach |
| Information Characterization | Part-I primary module | PRIMARY MODULE | information characterization |
| State-Conditioned Task-Oriented Acquisition | Part-II primary module | PRIMARY MODULE | state-conditioned acquisition |
| MAVIS | codebase closed-loop implementation | IMPLEMENTATION ONLY | state-conditioned learned implementation |
| mvd_m1_o2 | static deployable reference/comparator | REFERENCE | static reference |
| mechanical oracle | retrospective task-value opportunity analysis | ORACLE | mechanical oracle |
| appearance oracle | preregistered task-agnostic signal-saliency comparator | ORACLE | task-agnostic C-scan saliency reference |
| reconstruction oracle | legacy retrospective objective comparator | LEGACY ORACLE | legacy reconstruction oracle (supplement only) |
| acquired-position/history | source control | CONTROL | acquired-position/history control |
| reconstruction | legacy source control | LEGACY CONTROL | legacy reconstruction-derived control (supplement only) |
| shuffled content | source control | CONTROL | shuffled-content control |

## Main-text rules

- The proposed paper-level method is **Task-Relevant Information Acquisition**.
- Part I asks what information matters; Part II uses partial state to value and
  realize measurements under cost.
- `MAVIS` does not appear in the title, abstract, Introduction, contributions,
  headings, figures, tables, or Conclusions. The supplement records that code
  identity once for reproducibility.
- `mvd_m1_o2` is an internal static reference, not a published competing
  method. Main text calls it the `static reference`.
- AUEBC 0.125053 belongs to the supervised state-conditioned learned
  implementation, not to the complete proposed framework.
- Retrospective oracles characterize opportunity; controls identify which
  state signals contribute. Neither category is the proposed method.
- The appearance oracle is the preregistered task-agnostic comparator; its score
  uses newly revealed RGB versus the specimen border median, uses no CAI labels,
  and is retrospective/nondeployable.
- Frozen artifact identifiers retain the term `reconstruction`. They are legacy
  supplement evidence and must never be renamed as appearance saliency.

## System diagnostic rule

`A4_BASELINE_MINUS_MAVIS` is one direction-preserving system-level diagnostic
in Section 5.2.3 and the subordinate calibration panel in Figure 4(d). It is
absent from the abstract, Introduction, contributions, main tables, and
Conclusions. Its complete interval and domain directions remain in the
supplement and machine-readable evidence.
