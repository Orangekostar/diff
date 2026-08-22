# MSSS Reference Method Audit

Date: 2026-08-22
Status: frozen before implementation

## MFC-MIL

Primary sources: [ICLR 2025 paper](https://openreview.net/pdf?id=6xrDPHhwD3)
and [official implementation](https://github.com/WissingChen/MFC-MIL), audited
at commit `145494ec0f221c8b1cc32843efeafb28426634c0`.

MSSS borrows the separation of multi-scale spatial analysis, frequency-domain
structure analysis, and intervention-separated ablation. It does not borrow the
MIL architecture, pathology task, causal-memory module, or learned scale
fusion. Repository inspection found a 1-D Haar helper in
`modules/fre_domain.py`, but its forward path uses a Hilbert transform and the
DWT call is commented out. It is therefore conceptual evidence, not a DWT
implementation authority for this project.

## WaveRNet

Primary source: [official repository](https://github.com/Chanchan-Wang/WaveRNet),
audited at commit `3d127dc670ea33e41eda97eeca512b1a83c4ec11`.

MSSS borrows the leave-one-domain-out evaluation perspective and the principle
of examining separate candidate frequency structures. It does not borrow the
retinal segmentation network, domain adapters, domain tokens, test-time domain
weighting, or ensemble decoder. The repository's `SimpleWaveletTransform` is a
learned pair of convolution branches rather than a discrete wavelet transform.
Consequently MSSS uses PyWavelets for auditable DWT reconstruction and does not
claim code-level DWT reuse from WaveRNet.

## FreqGRL

Primary source: [Pattern Recognition publisher record](https://www.sciencedirect.com/science/article/pii/S0031320326007910),
DOI `10.1016/j.patcog.2026.113826`.

MSSS borrows only the methodological requirement that frequency-band transfer
must be diagnosed on the task and under domain shift. FreqGRL's task-specific
low-frequency representation, high-frequency enhancement, and fusion findings
are not treated as CAI facts. MSSS does not assume that low frequency transfers
better, that high frequency is nuisance, or that either band is mechanical.

## Adopted Boundary

The implementation is therefore limited to deterministic scale interventions,
frozen predictors, task-measured CAI effects, and source-only transfer tests.
No reference architecture, learned scale mixer, segmentation loss, or external
task conclusion enters the registered MSSS gate.
