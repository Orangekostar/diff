# 1. Report Metadata

Review date: 2026-08-26
Target venue/year/track: Advanced Engineering Informatics, original research article
Paper title: From Useful to Actionable Information: A Task-Relevant Information Hierarchy for Ultrasonic Inspection of Impacted Composites
Input materials reviewed: `main.tex`, compiled 27-page PDF, 15-entry bibliography, four figures, two tables, 39-row canonical claim authority, supplement, tests, and deterministic source package
Search basis: public-safe keyword searches; Elsevier/ScienceDirect, OpenReview, publisher DOI pages, and institutional records
Report file: this file
Reviewer mode: standard full scientific, writing, format, and reproducibility review

# 2. Desk Rejection Assessment

- Paper length: **pass/uncertain policy**. The 27-page single-column preprint is reviewable; AEI uses Your Paper Your Way and no current official initial-submission page cap was found.
- Topic compatibility: **pass**. The hierarchy concerns representation and use of engineering knowledge for a CAI decision-support task, rather than a generic neural-network application.
- Minimum quality: **pass**. All six sections, formal definitions, protocol, controls, results, limitations, references, and supplement are present.
- Policy/anonymity/compliance: **partial**. The review PDF is identity-neutral and contains the required AI-use statement; names, corresponding-author details, competing interests, CRediT roles, and title-page metadata require author completion.
- Prompt injection and hidden manipulation detection: **pass**. No reviewer-directed or model-directed hidden instruction was found in visible text, comments, captions, or supplement.
- Ethics and reviewability: **pass/partial**. The study uses public materials data and no human/animal subjects. A repository license is not present and must be selected by the authors before public artifact release.

Desk rejection risk: low.
Reason: the manuscript is complete and in scope; the remaining compliance items are author metadata rather than scientific defects.
Can be fixed before review: yes, except external empirical breadth, which is an accepted limitation requiring new data.

# 3. Paper Summary And Contribution Map

The manuscript argues that measurement information must be evaluated at three non-equivalent layers: usefulness to a fixed downstream engineering task, observability of conditional value from a legal partial-inspection state, and actionability through a bounded sensing policy. It operationalizes the hierarchy on 276 impacted CFRP specimens from six held-out experimental domains. The evidence chain combines matched scalar/spatial prediction, sparse retention, task-specific retrospective oracles, conditional-value evolution, matched state controls, component substitutions, bounded planning, feedback, and a frozen deployable endpoint. The main limitation is explicit: task-useful spatial information exists, but specimen-specific content observability and end-to-end policy superiority are not established.

Claimed problem: deciding which ultrasonic measurements matter for CAI assessment.
Claimed gap: existing C-scan prediction, adaptive ultrasound, ultrasound VoI, and task-driven design do not establish the implication Useful -> Observable -> Actionable.
Contribution map: conceptual hierarchy; executable information boundary; whole-domain evidence chain; negative-control and deployment boundary.
Evidence package: 39 hash-bound claims, synchronized specimen bootstrap, equal-domain aggregation, four main figures, two main tables, and 18 supplemental machine-readable files.
Stated limitations: one paired public data program, native-raster rather than scanner-time cost, predictor-specific utility, and fixed representation/scorer/planner implementations.

# 4. Search And Related-Work Basis

Queries used: adaptive ultrasonic inspection composites; ultrasonic value of information; task-driven experimental design imaging; C-scan CAI prediction AEI.
Sources searched: Elsevier/ScienceDirect, OpenReview, institutional publisher records, and DOI landing pages.
Closest works found: Fuentes et al. (adaptive ultrasonic inspection), Cantero-Chinchilla et al. (ultrasound VoI), Memarzadeh and Pozzi (sequential VoI), Blumberg et al. (task-driven imaging design), Ji et al. (adaptive composite wavefield reconstruction), and Mack et al. (C-scan to CAI prediction).
Unverified related-work risks: no comprehensive systematic review was attempted; recent composite NDE reviews are contextual rather than a direct hierarchy overlap.
Source-quality screening status: closest-work claims were checked against primary publisher, proceedings, or institutional records.

# 5. Expected Review Outcome

Expected outcome: weak accept / borderline positive.
Main accept signal: a coherent, falsifiable engineering-information hierarchy supported by unusually transparent positive and adverse controls.
Main reject signal: a skeptical reviewer may view the hierarchy as a principled reframing of one data program rather than a broadly validated informatics contribution.
Confidence: 5/5 because the full manuscript, artifacts, source hashes, compiled output, and current related-work basis were inspectable.

# 6. Strengths And Weaknesses

Strengths:

- The matched B-family scalar-to-spatial result is separated correctly from the independent I-family sensitivity estimator.
- All central adverse results remain visible in the main paper: position/reconstruction controls, shuffled content, adverse feedback, and endpoint non-superiority.
- The statistical unit is explicit and appropriate: specimens within domains, followed by equal weighting of six held-out domains.
- Figures 1--4 and Table 2 make the central argument recoverable without internal project-stage labels.
- The limitations section prevents oracle, raster-cost, predictor, and fixed-implementation evidence from migrating into stronger claims.

Weakness: empirical breadth is limited to six related domains from one paired public data program.
Evidence basis: Section 5.4 explicitly states this boundary and the external audit contains no formal replication.
Reviewer deduction: conceptual transfer beyond this program remains plausible rather than empirically shown.
Required fix: keep the limitation visible; an actual score increase on this axis requires a new external paired C-scan/CAI cohort.

Weakness: Table 2 is dense at `\scriptsize`.
Evidence basis: the 13-row, eight-column table needs two preprint pages.
Reviewer deduction: detailed cells require close reading, although no clipping or overflow remains.
Required fix: retain the two-page longtable for review and provide the CSV source, as the current package does.

Weakness: submission metadata and artifact licensing are incomplete.
Evidence basis: author details are withheld and the repository has no tracked license.
Reviewer deduction: not a scientific-score issue, but a final-submission and public-release blocker.
Required fix: authors must provide metadata, declarations, CRediT roles, and select a license.

High-impact paragraph review:

| ID | Current role | Reviewer takeaway | Main problem | Concrete edit | Severity |
| --- | --- | --- | --- | --- | --- |
| Abstract-P1 | task, hierarchy, evidence, boundary | useful information exists but is not automatically actionable | none material | keep | low |
| Intro-P1 | engineering hook | C-scan information must be valued against a CAI decision | none material | keep | low |
| Intro-P3 | hierarchy insight | the three layers are non-equivalent | slightly definitional | keep because it anchors Fig. 1 | low |
| Intro-P5 | contribution display | contribution is hierarchy plus controlled evidence, not a winning policy | four contributions are dense | retain; each maps to a results block | low |
| Results-RQ1 | usefulness evidence | spatial/sparse information is useful and predictor-conditioned | none material | keep | low |
| Results-RQ2 | observability evidence | target evolution exceeds demonstrated content observability | many adverse controls | keep controls in main; Table 2 aids scanning | low |
| Results-RQ3 | actionability evidence | diagnostic headroom does not yield a better frozen policy | none material | keep | low |
| Conclusion-P1--P3 | direct RQ closure | Yes / partial / no under current implementation | none material | keep | low |

# 7. Potentially Missing Related Work

Work: broad ultrasonic NDE reviews for impacted composites.
Status: searched.
Why relevant: could add domain context on inspection limitations.
Overlap: contextual only; they do not address task-conditioned measurement value.
Needed comparison: optional one-sentence context, not required for the novelty claim.

Work: adaptive multi-aperture phased-array inspection for inspection speed.
Status: searched.
Why relevant: reinforces that adaptive ultrasound and acquisition-efficiency work predates this manuscript.
Overlap: acquisition optimization, but not CAI-conditioned usefulness/observability/actionability.
Needed comparison: optional; Fuentes and Ji already cover the decisive prior-art boundary.

# 8. Claim-Evidence Audit

| Claim | Where stated | Evidence provided | Strength | Reviewer deduction | Required fix |
| --- | --- | --- | --- | --- | --- |
| Spatial morphology adds CAI-relevant information | Abstract; Sections 5.1 and 6 | matched B scalar/field contrast, CI, 5/6 domains | strong | none | retain estimator distinction |
| Sparse acquisition retains most full-field gain | Abstract; Section 5.1 | 89.9% retention and positive residual full-field gap | strong | digital cost only | retain no scanner-time wording |
| Oracle design is task-specific | Section 5.1 | CAI and image-error cross-objective oracle contrasts | adequate/strong | oracle-only evidence | keep non-deployable qualifier |
| Conditional value evolves | Abstract; Section 5.2 | turnover, rank, top-k, and opportunity at registered checkpoint | strong descriptive | teacher property only | keep scorer distinction |
| Content observability is not established | Section 5.2 | position, reconstruction, static, and shuffled controls | strong adverse evidence | no universal impossibility claim | no change |
| Retrospective valuation/planning headroom exists | Section 5.3 | substitution matrix and two-action planning diagnostic | adequate | bounded and privileged | keep scope |
| Current policy is not superior | Abstract; Sections 5.3 and 6 | adverse feedback and frozen deployable endpoint | strong boundary | does not prove all adaptive policies fail | no change |

# 9. Experiment / Benchmark / Reproducibility Audit

- Baselines: matched scalar/surface, uniform, random, reconstruction, static, shuffled, and strongest deployable endpoint are present.
- Ablations/controls: representation, valuation, planning, feedback, task objective, and downstream predictor are separated through registered controls.
- Datasets: 276 physical specimens, six domain-level holdouts; no independent external performance cohort.
- Metrics: CAI-ratio MAE, exact-cost AUEBC, image MSE, rank metrics, turnover, and set regret are defined with directions.
- Statistical rigor: nested LODO, synchronized within-domain specimen bootstrap, familywise P1 interval, and equal-domain aggregation are explicit.
- Robustness/failure cases: negative content, shuffled, feedback, and final endpoint results are central rather than hidden.
- Implementation details: legal state, reveal, DeepSets encoder, teacher, scorer, rollout, checkpoints, and exact cost are described; full configs remain in the source repository.
- Artifacts: deterministic source ZIP, checksums, canonical claim authority, figure/table sources, and supplemental raw files are present.
- Limitations: external replication, physical scanner time, causal damage interpretation, and implementation generality are appropriately bounded.

# 10. Multi-Reviewer Panel

Reviewer: Best-justified reviewer
Expertise: engineering informatics
Likely score: 8/10
Confidence: 5/5
Main positive signal: the hierarchy converts a failed implication into transferable validation logic.
Main negative signal: evidence remains one composite inspection program.
Evidence basis: Sections 3 and 5.4; Table 2.
Score-change condition: external paired replication would strengthen significance.

Reviewer: Critical reviewer
Expertise: adaptive sensing
Likely score: 5/10
Confidence: 4/5
Main positive signal: controls are unusually complete.
Main negative signal: no deployable policy improvement and limited external breadth.
Evidence basis: Section 5.3 endpoint and Section 5.4 limitations.
Score-change condition: a new external or deployment-facing result would be required.

Reviewer: Method / soundness reviewer
Expertise: sequential decision systems
Likely score: 7/10
Confidence: 5/5
Main positive signal: privileged and deployable information are cleanly separated.
Main negative signal: planning evidence is explicitly bounded to a two-action reachable pool.
Evidence basis: Sections 3.3--3.5 and 5.3.
Score-change condition: no quick change; arbitrary-horizon claims are already excluded.

Reviewer: Evidence / experiment reviewer
Expertise: empirical ML and statistics
Likely score: 7/10
Confidence: 5/5
Main positive signal: nested domain holdout and synchronized contrasts match the claims.
Main negative signal: only six domain clusters limit domain-level inference.
Evidence basis: Section 4.3--4.5.
Score-change condition: more independent domains would raise evidence breadth.

Reviewer: Novelty / positioning reviewer
Expertise: task-driven experimental design
Likely score: 7/10
Confidence: 4/5
Main positive signal: prior adaptive ultrasound, VoI, and task-driven design are explicitly disclaimed as novelty.
Main negative signal: novelty is conceptual/evidential rather than algorithmic.
Evidence basis: Section 2.
Score-change condition: preserve the exact distinction from TADRED and ultrasound VoI.

Reviewer: Writing / clarity reviewer
Expertise: technical communication
Likely score: 8/10
Confidence: 5/5
Main positive signal: the three-layer story and direct RQ answers are recoverable in one pass.
Main negative signal: Table 2 is dense.
Evidence basis: Fig. 1, Section 5 answer paragraphs, Table 2.
Score-change condition: no material rewrite needed.

Reviewer: Ethics / reproducibility reviewer
Expertise: research artifacts
Likely score: 6/10
Confidence: 5/5
Main positive signal: public data, exact hashes, deterministic package, and AI-use statement are present.
Main negative signal: repository license and author declarations remain unresolved.
Evidence basis: package README and submission audit.
Score-change condition: add author-approved license and declarations.

Reviewer: Domain application reviewer
Expertise: composite NDE and CAI
Likely score: 7/10
Confidence: 4/5
Main positive signal: CAI task, native raster, and scanner-time boundary are physically sensible.
Main negative signal: no physical scan-path timing or independent scanner campaign.
Evidence basis: Sections 2.2, 4.5, and 5.4.
Score-change condition: requires new experimental acquisition data.

Reviewer: Evidence/ablation reviewer
Expertise: controlled model evaluation
Likely score: 8/10
Confidence: 5/5
Main positive signal: predictor, objective, state content, valuation, planning, and feedback are separately tested.
Main negative signal: the fixed failed representation is not exhaustively varied.
Evidence basis: Sections 5.1--5.3.
Score-change condition: no change needed because the paper avoids universal architecture claims.

Reviewer: Reproducibility reviewer
Expertise: computational artifacts
Likely score: 7/10
Confidence: 5/5
Main positive signal: 39 claims and all deliverables are hash-bound and tested.
Main negative signal: public archival DOI and license are pending.
Evidence basis: canonical CSV, manifests, deterministic ZIP.
Score-change condition: archive the release and provide the DOI/license.

Reviewer: Novice advocate
Expertise: cross-disciplinary AEI reader
Likely score: 8/10
Confidence: 5/5
Main positive signal: Fig. 1, formal definitions, and direct answers make the negative endpoint interpretable.
Main negative signal: implementation vocabulary is dense in Section 3.5.
Evidence basis: Sections 3 and 5.
Score-change condition: current proportional implementation detail is acceptable.

Agreement: sound information boundaries, transparent controls, and strong writing.
Disagreement: whether a boundary study without a winning policy is sufficiently significant.
Decisive positive axis: operational hierarchy plus matched multi-domain evidence.
Decisive negative axis: external empirical breadth and conceptual rather than algorithmic novelty.
Unresolved evidence: independent paired C-scan/CAI replication.
AC stance: weak accept if AEI values the mechanism-and-boundary contribution; otherwise borderline.

# 11. Concerns Table

| ID | Severity | Concern | Evidence basis | Affected criterion | Fix class | Required action | Owner skill | Score-change condition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | major | one paired data program | Section 5.4; S18 | evidence/significance | experiment | retain limitation; future external cohort | experiment design | new paired external evidence |
| C2 | moderate | conceptual contribution may be read as taxonomy | Sections 2.4 and 3; Fig. 1 | novelty | writing | keep operational tests and non-equivalence explicit | paper writer | novelty remains tied to executable evidence |
| C3 | moderate | author compliance metadata absent | front matter and audit | policy | ethics/limitations | authors add title-page metadata, conflicts, CRediT | submission checker | compliance pass |
| C4 | moderate | no repository license or archival DOI | package audit | reproducibility | reproducibility | author selects license and archives release | submission checker | reproducibility 4/5 |
| C5 | minor | Table 2 is dense | compiled pages 21--22 | clarity | writing | retain CSV and two-page longtable | visual composer | no clipping or unreadable scaling |

# 12. AC / Meta-Review

Reviewer consensus favors the manuscript's soundness and transparency. The main disagreement is significance: the paper deliberately does not claim a superior acquisition policy, so its value depends on accepting an evidence hierarchy and a well-supported negative boundary as an AEI contribution. The decisive acceptance axis is that the hierarchy is executable and each implication is tested with matched controls. The decisive rejection axis is limited empirical breadth. The AC stance is weak accept / borderline positive, with no fatal technical or integrity concern found.

# 13. Quantitative Scores

## Scorecard

| Dimension | Score (1-5) | Confidence (1-5) | Evidence basis | Deduction / score-change condition |
|:---|:---:|:---:|:---|:---|
| Novelty | 4 | 4 | Sections 2.4 and 3 | conceptual novelty; retain exact prior-art separation |
| Soundness | 4 | 5 | Sections 3--4; canonical authority | bounded diagnostics are correctly scoped |
| Evidence | 4 | 5 | Section 5; supplement | external cohort would raise breadth |
| Significance | 4 | 4 | Abstract; Section 5.4 | depends on valuing a boundary study |
| Clarity | 4 | 5 | Fig. 1; direct RQ answers | dense Table 2 is the main deduction |
| Reproducibility | 4 | 5 | manifests, tests, deterministic package | license and archival DOI pending |
| Ethics / Limitations | 3 | 5 | Section 5.4; declarations | authors must complete conflicts/CRediT/license |

**Overall:** 7/10 | **Scholarly Confidence:** 5/5

**Recommendation:** weak accept
**Verdict:** completion of metadata/licensing removes compliance risk; a one-point scientific increase requires independent paired-domain evidence.

| Change | Condition | Likely affected dimensions | Expected movement |
| --- | --- | --- | --- |
| Raise score | independent paired C-scan/CAI replication supports hierarchy boundaries | evidence, significance | +1 overall |
| Lower score | audit reveals outer-domain leakage or B/I estimator conflation | soundness, evidence | fatal / -2 or more |
| No quick change | physical scanner-time or deployment validation | significance, domain validity | unlikely before submission |

# 14. Questions For Authors

1. What author names, affiliations, corresponding-author contact, competing-interest statement, and CRediT roles should replace the review placeholders?
2. Which code/data license and archival DOI will govern the public reproducibility release?
3. Does AEI Editorial Manager request a separate anonymous manuscript for this submission instance, or should the author block be restored in the initial PDF?

# 15. Score Revision Criteria

Raising the score would require: independent paired-domain replication or equivalent external evidence.
Lowering the score would be triggered by: any mismatch between canonical metrics and frozen source artifacts, hidden outer-domain use, or removal of adverse controls.
Concerns unlikely to change before submission: physical scanner-time validation and external empirical breadth.

# 16. Action Plan And CCFA Handoffs

Priority: P0
Action: complete author metadata, conflicts, CRediT, license, and archival identifier.
Owner skill: submission checker
Input needed: author-approved declarations and release choice
Expected output: externally submittable title page and artifact record
Handoff required: yes, because only authors can supply these facts.

Priority: P1
Action: preserve all main adverse controls and the B/I estimator distinction through final typesetting.
Owner skill: integrity auditor
Input needed: final compiled PDF
Expected output: zero claim/evidence drift
Handoff required: no.

Priority: P2
Action: retain external replication and scanner-time limits as accepted limitations.
Owner skill: paper writer
Input needed: none
Expected output: current bounded wording
Handoff required: no.

Checks run: full source review, related-work search, 70 paper tests at review time, LaTeX/BibTeX builds, font/metadata inspection, numeric provenance lint, frozen-path diff, and render QA.
Checks skipped: real Editorial Manager upload, author identity/declaration verification, license selection, and external experimental replication.
Unresolved risks: author-side compliance inputs and lack of external paired performance data.
