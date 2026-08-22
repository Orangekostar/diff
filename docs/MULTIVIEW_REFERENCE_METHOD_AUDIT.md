# Multi-View Reference Method Audit

Audit date: 2026-08-21
Mode: narrow implementation audit
Source policy: primary papers, official proceedings/publisher records, OpenReview,
and author repositories only

## Cooperative Learning

- Paper: Daisy Yi Ding et al., "Cooperative learning for multiview analysis,"
  PNAS 119(38), 2022, DOI `10.1073/pnas.2202113119`.
- Paper record: <https://doi.org/10.1073/pnas.2202113119>
- Official code: <https://github.com/dingdaisy/cooperative-learning>
- Audited repository revision: `04c1cbb1a6d79acf4ab0c092392f35dbbe6fb8fc`.
- Borrowed: prediction loss plus cross-view agreement, validation selection of
  agreement strength, augmented design matrices, modular view predictors, and
  the explicit three-view pair construction.
- Not borrowed: its datasets, feature extractors, lasso-specific scientific
  claims, or random-fold evaluation.

The official three-view code augments the target design with three pairwise
zero-target blocks scaled by the square root of agreement strength. This project
retains that algebraic mechanism but uses domain-held-out selection and separate
C-scan feature paths.

## GMvR

- Paper: Saiji Fu et al., "A generalized framework for multi-view regression
  with diverse extensions," Neurocomputing 670 (2026), 131556.
- Publisher record: <https://doi.org/10.1016/j.neucom.2025.131556>
- Borrowed: the explicit separation of prediction, consistency, and
  complementarity, plus view-specific weights as predictive contributions.
- Not borrowed: its application datasets, kernel derivation, or benchmark
  claims.

The local E3 model is an intentionally small instantiation: cooperative experts
provide the consistency mechanism, while non-negative source-OOF-selected view
weights provide view-specific contributions. Complementarity is not defined by
forcing feature vectors apart.

## Deep Mutual Learning

- Paper: Ying Zhang et al., "Deep Mutual Learning," CVPR 2018.
- Proceedings: <https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.html>
- Official code: <https://github.com/YingZhangDUT/Deep-Mutual-Learning>
- Borrowed: independently parameterized peers can improve through simultaneous
  prediction-level teaching without a fixed teacher.
- Not borrowed: categorical KL divergence, TensorFlow classification models,
  or image-recognition training protocol.

The regression adaptation uses squared or Huber target loss and continuous
pairwise prediction disagreement. No classification KL term appears in code.

## DiCoM

- Paper record: "Diverse and Consistent Multi-view Networks for Semi-supervised
  Regression," OpenReview ICLR 2022 submission.
- Source: <https://openreview.net/forum?id=J9_7t9m8xRj>
- Borrowed: accurate ensembles need both useful peers and non-collapsed
  prediction diversity; agreement alone can be harmful.
- Not borrowed: semi-supervised assumptions, unlabeled-data loss, multi-network
  image architecture, or its reported datasets.

The project monitors prediction variance and residual correlation as consistency
strength increases. Greater agreement without lower outer MAE is recorded as
collapse and cannot pass a gate.

## Implementation Decision

The closest reusable mechanism is Cooperative Learning's source-validated
agreement penalty. GMvR supplies E3's conceptual decomposition; Deep Mutual
Learning supports peer-level rather than feature-level coupling; DiCoM supplies
the collapse warning. The local novelty claim is restricted to mechanics-validated
C-scan views, the observed failure of feature invariance, strict cross-domain CAI,
and engineering reliability analysis.
