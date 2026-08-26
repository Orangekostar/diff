# AEI Information Hierarchy Paper Design

## Objective

Create a reproducible Advanced Engineering Informatics manuscript package that
argues one engineering-information thesis: Useful information is not
necessarily Observable from the available state, and Observable value is not
necessarily Actionable through a learned acquisition policy.

The paper is a mechanism-and-boundary account. It is not a new training study,
a generic model paper, or a closed-loop superiority paper.

## Scientific architecture

The evidence is organized into three separately testable layers:

1. Useful: spatial internal morphology improves CAI prediction over the matched
   scalar representation; sparse acquisition retains most of that gain;
   retrospective oracle designs reveal unequal and task-specific value.
2. Observable: static pre-inspection scores do not recover strict-OOF action
   value; conditional value evolves; real partial content does not beat matched
   geometry/reconstruction controls; the dynamic advantage is narrow.
3. Actionable: retrospective valuation and set-planning substitutions expose
   separable gaps, while feedback and the frozen learned policy do not improve
   the strongest deployable endpoint.

Every empirical statement is bound to a machine-readable source, SHA-256,
cohort, contrast direction, statistical unit, interval, domain count, and
oracle/deployable status.

## Immutable boundary

Historical P1, MVA, MVD, MAVIS, and science-closure namespaces are read-only.
The paper uses only the new artifact, result, and manuscript namespaces. P7
tree immutability is checked against
`931dc86c26caf1c7246709c4706a7cd0428e3a1533b6ff1ad3c2ad8f9517d1e4`.

No result is recomputed by refitting a learner. Paper aggregation reads frozen
CSV, JSON, Parquet, and Markdown authorities and regenerates only derived paper
data, figures, tables, and provenance reports.

## Implementation architecture

Add a private package under `src/cmc_bbdm/aei_information_hierarchy/`:

- `authority.py`: strict schema, source hashes, canonical metric rows, and
  contrast semantics;
- `figures.py`: deterministic four-figure rendering from paper source tables;
- `tables.py`: deterministic protocol and evidence-hierarchy tables;
- `validation.py`: numeric provenance lint, six-section lint, forbidden-claim
  lint, figure/table contract checks, and frozen-tree verification;
- `package.py`: one command to rebuild the derived paper package and checksums.

The package does not expose a new public research API. All source paths are
repository-relative and every generated payload records its input hashes.

## Statistical contract

- Physical specimen is the resampling unit.
- Held-out experimental domain is the external evaluation unit.
- Aggregate results reduce within specimen and then weight six domains equally.
- Paired intervals use synchronized specimen bootstrap within domain.
- State/action rows are never treated as independent replicates.
- Lower MAE, AUEBC, and regret are better; all contrast names state subtraction
  direction.
- Oracle and substitution results are retrospective and non-deployable.

## Visual system

The visual system uses a restrained, color-blind-safe palette with three stable
semantic roles: Useful, Observable, and Actionable. Adverse controls use neutral
gray or an explicit warning color rather than being hidden.

1. Figure 1 is the conceptual information hierarchy and validation gates.
2. Figure 2 combines scalar/full-field usefulness, sparse retention, and oracle
   task specificity.
3. Figure 3 combines static observability, evolving teacher value, and matched
   real/geometry/reconstruction controls.
4. Figure 4 combines valuation/planning substitutions, feedback, and the frozen
   actionability boundary.

Each figure is produced as SVG, PDF, and 300 dpi PNG and has a machine-readable
source CSV, visual contract, caption, and checksum entry.

## Table system

Exactly two main tables are generated in CSV and LaTeX form:

1. Multi-domain case study and protocol.
2. Information hierarchy evidence table.

No numeric cell is manually duplicated from prose; both tables are generated
from canonical authority.

## Manuscript system

The manuscript uses exactly six numbered top-level sections fixed by the master
prompt. The abstract and Results are drafted only after canonical evidence is
generated. Internal stage labels are removed from prose and replaced with
scientific descriptions.

The paper makes no claim of scanner-time reduction, industrial deployment,
external empirical generalization, universal value maps, causal failure
mechanisms, or learned-policy superiority.

## Acceptance contract

Completion requires deterministic package replay, exact source hashes, all
paper-specific tests, MAVIS/MVD/MVA regressions, Ruff, `git diff --check`, LaTeX
compile when the toolchain is present, figure render QA, numeric provenance
lint, six-section lint, forbidden-claim lint, and frozen-tree verification.
