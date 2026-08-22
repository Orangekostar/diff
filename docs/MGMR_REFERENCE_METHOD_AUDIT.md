# MGMR Reference Method Audit

Date: 2026-08-22
Status: completed before M0 implementation

No external repository code was executed or copied. Repositories were inspected
read-only at fixed commits.

## MFC-MIL

- Paper: ICLR 2025, "Multi-level Feature Guided Multi-instance Learning for
  WSI Classification".
- Repository: <https://github.com/WissingChen/MFC-MIL>
- Audited commit: `145494ec0f221c8b1cc32843efeafb28426634c0`.
- Files inspected: `modules/multi_level.py`, `modules/fre_domain.py`,
  `modules/causal.py`, and `models/abmil.py`.

The reusable idea is explicit separation of representation levels/frequency
components followed by lightweight fusion and intervention ablations. MGMR does
not borrow pathology MIL, bag semantics, causal memory, or attention pooling.
MFC-MIL's frequency code is not a reference implementation for MGMR's 2D
feature-map DWT: its DWT helper is one-dimensional across a feature axis, and a
`MultiLevelFuse` path refers to undefined `self.ll` and `self.hl` attributes.

## WaveRNet

- Paper: <https://arxiv.org/abs/2601.05942>
- Repository: <https://github.com/Chanchan-Wang/WaveRNet>
- Audited commit: `3d127dc670ea33e41eda97eeca512b1a83c4ec11`.
- Files inspected: `models/waverNet.py` and `models/sdm.py`.

WaveRNet places its frequency block after an image encoder feature map. MGMR
borrows only that placement. `SimpleWaveletTransform` is a learned pair of 3x3
convolution branches with 1x1 fusion and a residual connection; it is not a
mathematical discrete wavelet transform. MGMR therefore uses PyWavelets and
tests reconstruction, band orientation, borders, dtype, size, and determinism.

## AEI graph references

- "Design information-assisted graph neural network for modeling central air
  conditioning systems", AEI 2024, DOI
  <https://doi.org/10.1016/j.aei.2024.102379>.
- "Physics-guided graph convolutional network for damage severity and zone
  identification in industrial composites", AEI 2025, DOI
  <https://doi.org/10.1016/j.aei.2025.103701>.

The transferable principle is that graph topology should come from an audited
engineering structure and should be compared with a fully connected graph. The
HVAC graph obtains topology from design information; the composite paper
compares fully connected, actuator-clustered, and shared-path graphs. Neither
topology is task authority for C-scan CAI, so MGMR will not copy either graph.
Any M1 graph must be derived from verified C-scan spatial geometry and laminate
metadata, and remains prohibited until M0 passes.

## Dataset source used for laminate audit

The public data article is available at
<https://pmc.ncbi.nlm.nih.gov/articles/PMC9294053/>. Its Table 1 defines the
cross-ply and quasi-isotropic sequence notation and reports an approximate ply
thickness of 0.1875 mm. The later public CAI data article at
<https://pmc.ncbi.nlm.nih.gov/articles/PMC11999467/> lists all C8/C16/C24 and
Q8/Q16/Q24 sequences and average laminate thicknesses. These sources support
metadata authority only; they do not establish MGMR effectiveness.
