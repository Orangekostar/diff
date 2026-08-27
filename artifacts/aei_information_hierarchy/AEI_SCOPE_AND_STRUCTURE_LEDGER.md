# AEI Scope and Structure Ledger

Audit date: 2026-08-26

## Official scope fit

Official source: Elsevier, *Advanced Engineering Informatics*,
<https://shop.elsevier.com/journals/advanced-engineering-informatics/1474-0346>.

The official description defines the journal around support for
knowledge-intensive engineering activities and places joint emphasis on
knowledge and engineering applications. It asks for engineering relevance,
more reliable or creative decision support, explicit support for
knowledge-intensive tasks, and vigorous qualitative and quantitative
validation. It also warns that a generic computing method or an established
method transferred to a new subdomain is insufficient without noteworthy new
power, generality, or scalability.

Paper 1 is therefore framed as an engineering-information contribution:

- knowledge object: task-relevant measurement value under partial evidence;
- engineering task: CAI-oriented decision support after composite impact;
- generalizable construct: progress from information characterization to
  evidence-calibrated decision realization rather than treating information as
  intrinsically valuable;
- validation: 276 physical specimens, six held-out experimental domains,
  registered nested LODO comparisons, synchronized specimen bootstrap, matched
  controls, and explicit calibration boundaries;
- boundary: no generic model novelty, external deployment, or scanner-time
  claim.

## Recent AEI structure benchmark

`VERIFIED` means the publisher page exposed the named heading or an explicit
section-organization sentence. `PARTIAL` means only the listed headings were
visible. `STRUCTURE_NOT_VERIFIED` means the publisher page did not expose the
top-level section list; no heading was inferred from the abstract.

| # | Recent AEI research paper | Year | Actual top-level headings accessible from publisher | Related Work separate? | Case Study / Experimental Design separate? | Results and Discussion merged? | Status |
|---:|---|---:|---|---|---|---|---|
| 1 | Mack et al., *Deep learning for predicting impact energy and compression after impact strength of composite materials using C-scan images* | 2026 | `STRUCTURE_NOT_VERIFIED` | not verified | not verified | not verified | STRUCTURE_NOT_VERIFIED |
| 2 | Ezekiel et al., *Physics-guided generative surrogate modeling for full-field plasticity prediction in Al/SiC nanocomposites* | 2026 | Introduction; Methodology; Results and discussion; Conclusions | no separate heading visible | Methodology is separate | yes | VERIFIED |
| 3 | Giretti et al., *Knowledge design in complex domains* | 2026 | `STRUCTURE_NOT_VERIFIED` | not verified | not verified | not verified | STRUCTURE_NOT_VERIFIED |
| 4 | *Graph attention networks enhanced predictive modeling for penetration-explosion damage in concrete structures* | 2026 | `STRUCTURE_NOT_VERIFIED` | not verified | not verified | not verified | STRUCTURE_NOT_VERIFIED |
| 5 | Hussain et al., *Multi-model structure-agnostic framework for enhanced materials discovery in engineering informatics* | 2026 | Introduction | not verified | not verified | not verified | PARTIAL |
| 6 | *Integrating context awareness and knowledge graphs for enhanced knowledge recommendation in manufacturing process planning* | 2026 | Introduction; Knowledge-supported process planning; Knowledge graph-based recommendation for process planning; Context-aware GAN-based knowledge recommendation; Case study; Conclusion | related background is separate in substance | yes, Case study | not verified | PARTIAL |
| 7 | An et al., *Intelligent detection method for debonding and voids in concrete-filled steel/aluminum tubular structures based on impact acoustics and unsupervised learning* | 2026 | Introduction; Related work; Proposed intelligent detection method for debonding and voids; Experimental validation of proposed method; Automatic crawling and tapping robot and its validation on a real bridge; Model performance comparison; Conclusions | yes | validation and real-bridge sections are separate | no merged heading exposed | VERIFIED |
| 8 | Yuan et al., *Semantic-driven spatial fusion for noise-resilient distance measurement in autonomous inspection of insulators* | 2026 | Introduction; Methodology; Conclusion | no separate heading exposed | not verified | not verified | PARTIAL |
| 9 | *Multi-LLM-based augmentation and synthetic data generation of construction schedules and task descriptions with SLM-as-a-judge assessment* | 2026 | Dataset preparation and schema; LLM performance for construction schedule augmentation; Conclusion | not verified | evaluation is separate in substance | not verified | PARTIAL |
| 10 | Zheng et al., *FMANet: Fused mamba attention model with multi-type preprocessing for simulated crack-contaminated complex environments* | 2026 | Introduction; the publisher-exposed organization sentence verifies separate method, data/training, experiments, and conclusion sections, but their exact heading strings were not all exposed | no separate Related Work section stated | experiment section separate | no | PARTIAL |

Publisher links and search notes are recorded in
`docs/literature-search-20260826-aei-paper1/`.

## Structure decision for this paper

The benchmark supports three applicable patterns:

1. AEI papers commonly separate the engineering-information method or
   framework from validation.
2. Application-heavy papers may separate related research and a case study or
   experimental validation when the problem boundary matters.
3. Results and discussion may be merged when the contribution is an evidence
   argument rather than a sequence of unrelated experiments.

Paper 1 adopts exactly six top-level sections:

1. Introduction
2. Related Research and Problem Formulation
3. Task-Relevant Information Acquisition Framework
4. Multi-Domain CFRP Case Study and Experimental Design
5. Experimental Results and Discussion
6. Conclusions

This gives the knowledge construct its own framework section, isolates protocol
and case-study validity, and keeps Information Characterization and
Evidence-Calibrated Decision Realization in one integrated Results and
Discussion section. Usefulness, task-value observability, and actionability are
retained as validation criteria rather than separate primary narratives.
