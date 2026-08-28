# Independent AEI Review: Main-Method Reframe

## 1. Report Metadata

Review date: 2026-08-28 UTC
Target venue/year/track: *Advanced Engineering Informatics*, 2026, original research article
Paper title: Task-Relevant Ultrasonic Information Acquisition for Impacted Composites: From Spatial Information to State-Conditioned Sensing
Input materials reviewed: main TeX/PDF, supplement, four figures, two tables,
39-claim authority, visibility and chronology ledgers, source hashes,
machine-readable supplementary evidence, tests, and deterministic package
Search basis: current official AEI scope; publisher/proceedings/institutional
records for closest work; public-keyword search only
Report file: `ccfa-review-reports/2026-08-28-aei-main-method-reframe-review.md`
Reviewer mode: independent simulated journal review, not an official AEI review

## 2. Desk Rejection Assessment

- Paper length: uncertain. The 17-page preprint is coherent, but the live
  journal-specific Guide for Authors could not be fetched because ScienceDirect
  returned HTTP 403.
- Topic compatibility: pass. The method supports a knowledge-intensive
  artifacts-centered engineering decision and joins information representation,
  valuation, and acquisition.
- Minimum quality: pass. Problem, method, experimental protocol, evidence,
  limitations, references, and supplement are present.
- Policy/anonymity/compliance: conditional pass. The PDF is anonymous and uses
  `elsarticle`; live Editorial Manager requirements still need confirmation.
- Prompt injection and hidden manipulation detection: pass. No hidden reviewer
  instruction, prompt injection, or manipulation text was found.
- Ethics and reviewability: pass with author completion. Public data and AI-use
  disclosure are stated; author, funding, conflict, CRediT, archive, and license
  metadata remain to be supplied.

Desk rejection risk: low. The main residual desk risk is live submission-form
compliance, not scientific reviewability.

## 3. Paper Summary And Contribution Map

The paper studies how ultrasonic C-scan measurements should be selected when
the downstream engineering task is CAI assessment rather than complete field
reconstruction. Task-Relevant Information Acquisition has two parts. Information
Characterization tests spatial enrichment, sparse recoverability, task
conditioning, and state/predictor dependence. State-Conditioned Task-Oriented
Acquisition then estimates marginal value from a legal partial state, uses
matched controls to identify signal sources, and realizes exact-cost measurement
sets. Evidence comes from 276 specimens in six experimental domains under
nested leave-one-domain-out evaluation. The manuscript explicitly separates
retrospective oracles, source controls, component diagnostics, and one learned
implementation from the proposed paper-level framework.

Claimed gap: completed-scan prediction, reconstruction-driven sampling, and
ultrasound value-of-information work do not jointly establish which measurements
serve CAI loss as evidence accumulates and how that value becomes a legal set.

Evidence package: matched scalar/spatial comparisons, a registered 25% sparse
condition, task oracles, state/predictor analyses, dynamic/static valuation,
matched information-source controls, component substitutions, bounded planning,
one system diagnostic, confidence intervals, domain directions, and deterministic
artifacts.

Stated limitations: digital rather than physical acquisition cost; fixed action
hierarchy, predictors, and representations; one six-domain data program; no
causal failure-mechanism identification; no universal transfer claim.

## 4. Search And Related-Work Basis

Queries used: public combinations of task-driven acquisition, adaptive
ultrasonic inspection, composite damage, value of information, CAI prediction,
and state-conditioned sensing. No private manuscript passage was submitted.

Sources searched: Elsevier/ScienceDirect, SAGE, OpenReview, institutional
repositories, DOI records, PubMed, and the frozen literature ledger.

Closest works found:

- Fuentes et al. (2020): sequential robotic ultrasonic inspection, but a
  damage/novelty objective rather than CAI utility.
- Cantero-Chinchilla et al. (2020): cost-aware ultrasonic sensor configuration
  through value of information, but static configuration rather than an evolving
  partial C-scan state.
- Memarzadeh and Pozzi (2016): belief-state-dependent sequential inspection
  value, but no ultrasonic/CAI realization.
- Blumberg et al. (2024): task-driven imaging design, primarily a static global
  channel subset rather than specimen-conditioned spatial acquisition.
- Ji et al. (2026): adaptive composite wavefield sampling for reconstruction,
  not downstream CAI loss.
- Mack et al. (2026): full-C-scan CAI prediction, with no sparse or sequential
  acquisition decision.
- Nguyen et al. (2025): limited-data ultrasonic segmentation using augmentation
  and synthetic examples; it is not evidence of incomplete spatial sampling.

Unverified risk: the search found no single work combining all of CAI utility,
legal partial state, state-conditioned marginal value, matched source controls,
and exact-cost set realization. This is not an exhaustive novelty proof.

Source-quality screening: pass. Scoring relies on primary publisher,
proceedings, DOI, or institutional records rather than aggregation sites.

## 5. Expected Review Outcome

Expected outcome: major revision / borderline positive.
Main accept signal: a coherent engineering-informatics framework with unusually
strong evidence provenance and honest separation of opportunity, valuation,
attribution, planning, and implementation.
Main reject signal: the learned end-to-end implementation does not beat the
static reference, while matched content controls often favor the controls; the
strongest empirical support is therefore for characterization and diagnostic
decomposition rather than deployable system superiority.
Confidence: 5/5.

## 6. Strengths And Weaknesses

Strengths:

- The method identity is recoverable after one pass. The main method, its two
  parts, the MAVIS implementation, and the static reference are no longer
  conflated (Abstract; Introduction; Supplement S1).
- The causal acquisition contract is precise. Legal state, forbidden future
  content, exact unique-location cost, action hierarchy, and outer-domain
  separation are explicit (Section 3; Section 4.2).
- Claims remain direction-preserving. Adverse content controls, negative A3,
  and the static-favorable A4 endpoint are retained rather than hidden
  (Section 5.2; Supplement S4-S5).
- The evidence chain is broader than a single accuracy result: it tests
  representation, task, state, predictor, source attribution, valuation,
  planning, and the final implementation boundary.
- Auditability is strong: 39 canonical claims, complete visibility mapping,
  immutable source hashes, deterministic figures/tables/ZIP, and public data.

Weakness 1: the deployable learned endpoint remains negative.
Evidence basis: Section 5.2.3 reports AUEBC 0.125053 for the learned
implementation and 0.124992 for the static reference.
Reviewer deduction: the paper establishes a framework and diagnostic headroom,
not a superior deployed acquisition policy.
Required fix: retain the current bounded claim; a stronger system-performance
claim would require a new, preregistered end-to-end result.

Weakness 2: measured-content attribution is not favorable under the registered
controls.
Evidence basis: Section 5.2.2 reports positive real-minus-position,
real-minus-reconstruction, and real-minus-shuffled contrasts, with real content
favorable in one of six domains.
Reviewer deduction: the dynamic benefit may be carried substantially by
geometry/history/state structure rather than specimen-specific content.
Required fix: make that distinction prominent in interpretation and avoid
describing the current implementation as content-driven.

Weakness 3: external and operational validation are absent.
Evidence basis: Section 5.3 limits cost to native-raster coverage and states
that travel, coupling, settling, path planning, hardware timing, and external
transfer are unavailable.
Reviewer deduction: generality and industrial value are plausible but not yet
empirically established.
Required fix: retain this as an accepted limitation or add an independent
hardware/external study in future work.

Weakness 4: uncertainty rests on six outer domains from one data program.
Evidence basis: Section 4.3 uses equal-domain aggregation and direction counts;
the six domains contain 38-59 specimens each.
Reviewer deduction: specimen-level evidence is substantial, but domain-level
generalization remains statistically narrow.
Required fix: do not broaden the transfer claim; an independent domain family
would be needed to materially raise evidence breadth.

## 7. Potentially Missing Related Work

No verified omitted work collapses the stated novelty. The most material
positioning risk is not a missing citation but the need to keep the novelty
delta narrow: integration of CAI-oriented information characterization,
legal-state valuation, source controls, and set realization. Existing work
already establishes adaptive ultrasound, ultrasound VoI, task-driven imaging,
and full-C-scan CAI prediction separately.

## 8. Claim-Evidence Audit

| Claim | Where stated | Evidence provided | Strength | Reviewer deduction | Required fix |
| --- | --- | --- | --- | --- | --- |
| Spatial internal information adds CAI signal | Abstract; Sec. 5.1.1 | 32.1% matched error reduction; familywise CI; 5/6 domains | strong | Directly supported | none |
| 25% sparse condition retains most field gain | Abstract; Sec. 5.1.1 | 89.9% retention and positive sparse/full gap | strong | Digital sampling claim only | retain no scanner-time wording |
| Task changes preferred locations | Sec. 5.1.2 | Cross-objective oracles under common cost | adequate | Retrospective opportunity, not learned policy | retain oracle label |
| Value changes with state | Abstract; Sec. 5.1.3 | 70.4% best-action turnover plus rank/top-k evidence | adequate | Teacher-based diagnostic | retain legal-state boundary |
| Dynamic valuation beats static next-action scoring | Abstract; Sec. 5.2.1 | Negative regret contrast; CI; 5/6 domains | strong | Supports local valuation | do not imply end-to-end win |
| Measured content drives the gain | Not claimed strongly | Adverse matched controls | weak/adverse | Current evidence restricts content attribution | preserve cautious wording |
| Bounded set realization matters | Sec. 5.2.2-5.2.3 | Component substitutions and positive greedy/beam regret | adequate | Retrospective bounded-pool evidence | do not generalize horizon |
| Learned system beats static reference | Explicitly not claimed | A4 favors static reference | adverse | Correctly presented as boundary | none |

## 9. Experiment / Benchmark / Reproducibility Audit

Baselines and controls are unusually well decomposed, including scalar/full,
sparse/full, uniform, reconstruction, static, position/history, reconstruction,
shuffled content, component substitutions, and bounded joint-set references.
The main missing baseline is not obvious from the current literature; the
decisive gap is an externally validated end-to-end deployment result.

The evaluation preserves outer domains and treats specimens as the first
statistical unit. Confidence intervals and domain directions are visible.
Six domains limit domain-level calibration, but the manuscript does not conceal
that limit. Reproducibility is strong at the artifact level: source data are
public, values are canonicalized, source hashes are recorded, and the source
archive replays deterministically. Repository/archive identity and final license
are still absent for anonymous review.

## 10. Multi-Reviewer Panel

Reviewer: Best-justified reviewer
Expertise: engineering informatics
Likely score: 7/10
Confidence: 4/5
Main positive signal: a traceable framework that turns information value into
an engineering acquisition contract.
Main negative signal: end-to-end gain is not demonstrated.
Score-change condition: independent operational validation would raise the score.

Reviewer: Critical reviewer
Expertise: adaptive sensing
Likely score: 5/10
Confidence: 5/5
Main positive signal: honest negative controls.
Main negative signal: static wins the final endpoint and content controls are
adverse.
Score-change condition: a preregistered learned policy beating static would
remove the central concern.

Reviewer: Method/soundness reviewer
Expertise: causal evaluation and domain generalization
Likely score: 7/10
Confidence: 5/5
Main positive signal: strict legal-state and held-out-domain boundaries.
Main negative signal: six outer domains constrain uncertainty calibration.
Score-change condition: an independent domain family would raise confidence.

Reviewer: Evidence/experiment reviewer
Expertise: empirical ML and NDE
Likely score: 6/10
Confidence: 5/5
Main positive signal: extensive matched controls and direction-preserving
reporting.
Main negative signal: diagnostic breadth exceeds end-to-end performance depth.
Score-change condition: a real acquisition study or external endpoint would
raise evidence and significance.

Reviewer: Novelty/positioning reviewer
Expertise: task-driven imaging and ultrasound
Likely score: 6/10
Confidence: 4/5
Main positive signal: the joint operational sequence is distinct from each
closest work.
Main negative signal: each ingredient exists separately.
Score-change condition: keep novelty on the integration/evidence contract, not
first adaptive ultrasound or first VoI.

Reviewer: Writing/clarity reviewer
Expertise: technical communication
Likely score: 8/10
Confidence: 5/5
Main positive signal: stable identity and six-stage narrative.
Main negative signal: the framework/implementation boundary still requires
careful reading around the final diagnostic.
Score-change condition: no major rewrite required; preserve the current A4
boundary wording.

Reviewer: Ethics/reproducibility reviewer
Expertise: artifact and responsible-research review
Likely score: 8/10
Confidence: 5/5
Main positive signal: public data, AI disclosure, source hashes, and deterministic
package.
Main negative signal: archive identity and license are deferred.
Score-change condition: supply final archive DOI, license, and author statements.

Reviewer: Domain application reviewer
Expertise: composite NDE
Likely score: 6/10
Confidence: 4/5
Main positive signal: CAI is an explicit downstream engineering endpoint.
Main negative signal: raster coverage is not scanner time or path cost.
Score-change condition: hardware timing/path evaluation would raise practical validity.

Reviewer: Evidence/ablation reviewer
Expertise: component evaluation
Likely score: 7/10
Confidence: 5/5
Main positive signal: substitutions isolate valuation and planning headroom.
Main negative signal: substitutions are retrospective and bounded.
Score-change condition: deployable ablations under the same outer protocol
would raise the score.

Reviewer: Novice advocate
Expertise: broad technical reader
Likely score: 7/10
Confidence: 4/5
Main positive signal: figures and headings make the progression recoverable.
Main negative signal: AUEBC, oracle, teacher, and legal state still demand a
careful protocol read.
Score-change condition: retain Table 1 and the framework figure as first-pass aids.

Agreement: the framework identity is clear and the auditability is strong.
Disagreement: whether a framework paper merits acceptance without a favorable
learned system endpoint.
Decisive positive axis: engineering-informatics formulation plus rigorous
evidence decomposition.
Decisive negative axis: no deployable/static superiority and limited measured-
content attribution.
Unresolved evidence: external/hardware transfer and operational cost.
AC stance: major revision / borderline positive, with no fatal integrity flaw.

## 11. Concerns Table

| ID | Severity | Concern | Evidence basis | Affected criterion | Fix class | Required action | Owner skill | Score-change condition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | major | Learned endpoint favors static | Sec. 5.2.3, A4 | evidence/significance | requires-new-result | retain bounded claim or add preregistered end-to-end result | experiment designer | favorable robust endpoint |
| C2 | major | Content controls are adverse | Sec. 5.2.2, O3/O4 | soundness/evidence | claim-qualification | keep benefit attributed to state conditioning, not measured content | paper writer | favorable matched-content evidence |
| C3 | moderate | No external/hardware validation | Sec. 5.3 limitations | significance | accepted-limitation | preserve scope; future independent study | experiment designer | external operational replication |
| C4 | moderate | Six-domain uncertainty breadth | Sec. 4.3 | evidence | accepted-limitation | retain domain directions and narrow transfer claim | experiment designer | independent domain family |
| C5 | minor | Archive/license metadata deferred | Data/code statement | reproducibility | reproducibility | add DOI, repository, and license after review policy permits | submission checker | complete public archive |
| C6 | minor | Live AEI form not verified | submission audit | compliance | LaTeX/format | confirm current Editorial Manager requirements | submission checker | portal checklist complete |

## 12. AC / Meta-Review

The panel agrees that the reframe succeeds scientifically: the proposed method
is Task-Relevant Information Acquisition, while MAVIS is one implementation and
the static reference is not a competing named method. No reviewer identified a
fatal correctness or integrity defect. The central discussion axis is evidence
scope. A favorable local dynamic-valuation result coexists with adverse
measured-content controls and a static-favorable final implementation endpoint.
That pattern supports a useful framework and diagnostic decomposition but not a
claim of deployed policy superiority. The likely editorial decision is major
revision if the journal requires stronger end-to-end validation, or borderline
positive if the framework/evidence contribution is accepted on its bounded terms.

## 13. Quantitative Scores

| Dimension | Score (1-5) | Confidence (1-5) | Evidence basis | Deduction / score-change condition |
|:---|:---:|:---:|:---|:---|
| Novelty | 4 | 4 | Sec. 2.3 and framework; closest-work records | Components are known separately; exhaustive search or direct overlap could lower it |
| Soundness | 4 | 5 | Sec. 3-4 causal/outer protocol | Six domains and retrospective stages; independent replication would raise it |
| Evidence | 3 | 5 | Sec. 5 and supplement | Final learned endpoint/content controls are adverse; favorable end-to-end evidence would raise it |
| Significance | 4 | 4 | CAI task and AEI scope alignment | Operational cost/external transfer remain untested |
| Clarity | 4 | 5 | Abstract, six-stage structure, Figs. 1-4 | Framework/implementation boundary still needs careful reading |
| Reproducibility | 4 | 5 | public data, hashes, deterministic package | Archive DOI/license are pending |
| Ethics / Limitations | 5 | 5 | Sec. 5.3, data/code and AI disclosures | author governance fields remain submission tasks |

**Overall:** 6/10 | **Scholarly Confidence:** 5/5
**Recommendation:** borderline / major revision
**Verdict:** a preregistered favorable end-to-end policy result or independent
operational validation would raise the score by one; evidence that the legal
state leaks held-out outcomes would be fatal, but the current audit found none.

Quality: 4/5
Clarity: 4/5
Significance: 4/5
Originality: 4/5
Soundness: 4/5
Evidence: 3/5
Reproducibility: 4/5
Ethics / Limitations: 5/5
Overall: 6/10
Confidence: 5/5

## 14. Questions For Authors

1. Which legal-state features account for the dynamic-versus-static gain when
   specimen-specific measured content does not beat the matched controls?
2. What aspect of the framework is expected to transfer across scanners when
   only native-raster digital cost is currently observed?
3. Will the final archive expose the code and exact commands needed to reproduce
   all 39 canonical claims, not only the four figures and two tables?

## 15. Score Revision Criteria

Raising the score would require a favorable preregistered end-to-end policy
comparison, independent external/hardware validation, or a stronger causal
account of the legal-state signal. Lowering the score would be triggered by
leakage, frozen-artifact inconsistency, or direct prior work collapsing the
joint contribution; none was found. External data and physical acquisition-cost
evidence are unlikely to change through writing alone.

## 16. Action Plan And CCFA Handoffs

Priority: P0
Action: keep A4 and adverse content-control directions explicit.
Owner skill: ccf-integrity-auditor
Input needed: canonical authority and supplement
Expected output: direction-preserving final manuscript
Handoff required: no

Priority: P1
Action: complete author, conflict, CRediT, archive, and license metadata and
confirm the live AEI portal.
Owner skill: ccf-submission-checker
Input needed: author-governance information
Expected output: upload-ready submission fields
Handoff required: yes

Priority: future work
Action: design independent operational validation only if a stronger deployment
claim is desired.
Owner skill: ccf-experiment-designer
Input needed: scanner access, physical path/timing costs, independent specimens
Expected output: preregistered external/hardware evidence
Handoff required: yes

Checks run: full paper/MAVIS/MVD/MVA tests, Ruff, three LaTeX builds, final-log
scan, font audit, claim/numeric/frozen-path validation, deterministic replay,
DOI/context checks, and official AEI scope check.
Checks skipped: live Editorial Manager form and inaccessible journal-specific
Guide for Authors.
Unresolved risks: deployable/static superiority, measured-content attribution,
external transfer, physical acquisition cost, and author-supplied submission metadata.
