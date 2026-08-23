# MVA Reference Method Audit

Date: 2026-08-23
Status: completed before A0-A3 implementation

No external implementation is copied or executed. The audit extracts method
principles only; project code is written against the frozen local P5/P7 APIs.

## R1: AdaSTMAE

- Paper: Ji et al., "Adaptive sampling for efficient Lamb wavefield
  reconstruction in composite laminates with Spatial-Temporal Masked
  AutoEncoder," *Ultrasonics*, 2026, DOI
  <https://doi.org/10.1016/j.ultras.2026.107972>.
- Public code: no authoritative repository was located in the audit.
- Relevant principle: allocate a sparse spatial budget adaptively and retain
  global exploration while refining informative regions.
- Boundary: AdaSTMAE optimizes Lamb-wavefield reconstruction and damage-region
  error. MVA instead evaluates downstream CAI absolute-error reduction on
  released raster C-scan images. It does not reuse STMAE, Bayesian optimization,
  SLDV timing claims, or wavefield acquisition semantics.

## R2: TACKLE

- Paper: Wu et al., "Learning Task-Specific Strategies for Accelerated MRI,"
  *IEEE Transactions on Computational Imaging*, 2024, DOI
  <https://doi.org/10.1109/TCI.2024.3410521>.
- Repository: <https://github.com/zihuiwu/TACKLE>.
- License audit: the repository had no detected license file on 2026-08-23;
  therefore no code is reused.
- Relevant principle: sampler, information retriever, and downstream predictor
  should be evaluated jointly under the task loss; reconstruction-only
  acquisition can be suboptimal. Its two-stage reconstruction pretraining then
  task fine-tuning motivates separating the current oracle diagnostic from any
  later deployable acquisition learner.
- Boundary: MVA does not reuse MRI k-space operators, VarNet, scanner sequences,
  or end-to-end sampler code.

## R3: LOUPE

- Papers: Bahadir et al., "Learning-based Optimization of the Under-sampling
  Pattern in MRI," IPMI 2019, <https://arxiv.org/abs/1901.01960>; and
  "Deep-learning-based Optimization of the Under-sampling Pattern in MRI,"
  *IEEE TCI* 2020, <https://arxiv.org/abs/1907.11374>.
- Repository: <https://github.com/cagladbahadir/LOUPE>.
- License audit: the repository had no detected license file on 2026-08-23;
  therefore no code is reused.
- Relevant principle for a later A4 only: a population-level probabilistic mask
  can be budget-normalized and converted to a deterministic inference mask.
- Boundary: A0-A3 implement no differentiable mask, Bernoulli estimator, STE,
  or U-Net reconstruction objective.

## R4: EDDI

- Paper: Ma et al., "EDDI: Efficient Dynamic Discovery of High-Value
  Information with Partial VAE," ICML 2019,
  <https://proceedings.mlr.press/v97/ma19c.html>.
- Repository: <https://github.com/microsoft/EDDI>.
- License: Microsoft Research License Terms, non-commercial research use and no
  source redistribution. No code is reused.
- Relevant principle: acquisition value is defined relative to information
  about target variables, not generic missing-data recovery.
- Boundary: MVA uses the registered CAI predictor and retrospective true-target
  oracle; it does not introduce a Partial VAE.

## R5: Active feature acquisition and oracle imitation

- He et al., "Imitation Learning by Coaching," NeurIPS 2012,
  <https://proceedings.neurips.cc/paper/2012/hash/2dffbc474aa176b6dc957938c15d0c8b-Abstract.html>.
- Valancius et al., "Acquisition Conditioned Oracle for Nongreedy Active
  Feature Acquisition," ICML 2024,
  <https://proceedings.mlr.press/v235/valancius24a.html>.
- Li and Oliva, "Active Feature Acquisition with Generative Surrogate Models,"
  ICML 2021, <https://proceedings.mlr.press/v139/li21p.html>.
- Relevant principle: full training information can define an oracle order, but
  a deployable policy must observe only its current state. A cheating oracle is
  a diagnostic teacher, not a deployable method.
- Boundary: A0-A3 stop before imitation. If A3 passes, outer-fold source
  trajectories must be regenerated without the outer domain before any A4/A5
  training; the global cross-fitted A2 trajectories are not policy-training
  authority.

## R6: Active MRI acquisition

- Pineda et al., "Active MR k-space Sampling with Reinforcement Learning,"
  MICCAI 2020, <https://arxiv.org/abs/2007.10469>.
- Repository: <https://github.com/facebookresearch/active-mri-acquisition>, MIT
  license; no code is reused.
- Relevant principle: sequential acquisition can be formulated as a partially
  observed budgeted decision process.
- Boundary: 276 specimens do not support a first-stage PPO, DQN, Decision
  Transformer, or other RL search. A0-A3 use deterministic retrospective
  oracles and fixed controls only.

## R7: Task-based sensing and quantization

- Neuhaus et al., "Task-Based Analog-to-Digital Converters,"
  <https://arxiv.org/abs/2009.14088>.
- Shlezinger et al., "Deep Task-Based Analog-to-Digital Conversion,"
  <https://arxiv.org/abs/2201.12634>.
- Relevant principle: the acquisition system should minimize loss on the
  downstream variable rather than reconstruction distortion of the observed
  signal. This supports comparing CAI error and image error under the same
  measurement budget.
- Boundary: the current data cannot validate an analog front end, ADC rate, or
  physical scanner design.

## R8: Lamination parameters

- Albazzan et al., "Efficient design optimization of nonconventional laminated
  composites using lamination parameters: A state of the art," *Composite
  Structures* 209 (2019), DOI
  <https://doi.org/10.1016/j.compstruct.2018.10.095>.
- Lamination parameters compactly represent laminate stiffness effects without
  retaining a variable-length stacking sequence, but the inverse mapping to a
  manufacturable sequence is non-unique and ply-level failure criteria may
  require the original sequence.
- Boundary: laminate conditioning is locked until A5. A0-A3 neither infer
  missing sequences nor compute lamination parameters.

## MVA differentiation

The closest task-level precedent is TACKLE, and the closest ultrasonic
acquisition precedent is AdaSTMAE. MVA's testable difference is the combination
of (i) retrospective cell-wise acquisition on ultrasonic C-scan screenshots,
(ii) CAI absolute-error reduction as an oracle label, (iii) strict cross-domain
OOF prediction, and (iv) direct comparison against reconstruction and visual
appearance value. The current experiment can establish oracle headroom only;
it cannot establish a deployable policy or physical scan-time reduction.
