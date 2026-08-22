# MGMR Repository Audit

Date: 2026-08-22
Status: completed before M0 implementation

## 1. Exact baseline

The authoritative baseline is P1 `I_frozen`: metadata13 plus fold-local PCA of
the frozen 512-dimensional ResNet18 embedding, followed by fold-local mean
imputation, standardization, and Ridge regression with alpha 10. The six outer
PCA dimensions are `(8, 32, 8, 8, 8, 8)` in frozen domain order.

`reproduce_internal_only_baseline` rebuilds every outer prediction and compares
it with `paper_v3/experiments/P1_full_field_oracle/predictions.csv`. The exact
equal-domain MAE is `0.08963580465761432`; maximum prediction and target errors
must be at most `1e-12`. On this audit, `tests/test_aei_baseline_gate.py` passed
in 215.90 seconds.

## 2. Frozen encoder authority

- Architecture: torchvision ResNet18 ImageNet1K V1, classification head removed.
- Weight: `paper_v3/assets/resnet18-f37072fd.pth`.
- SHA-256: `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
- Input: grayscale `0.299R + 0.587G + 0.114B`, replicated to three channels.
- Resize: complete field to 224x224, bilinear, antialias enabled.
- Normalization: ImageNet mean and standard deviation.

## 3. Crop and spatial geometry

The 276 RGB crops have native `(width, height)` values `(675, 674)`, `(352,
338)`, or `(340, 338)`. All map to a verified 75x75 mm scan field. The full crop
is resized without content cropping.

The crop preserves the registered scan field and pixel-axis orientation, but no
source provides a specimen-specific impact coordinate inside that crop. The
existing `(37.5, 37.5) mm` center is a geometric convention, not an independently
measured impact center. Impact-center recovery is therefore not authoritative.

For a 224x224 input, the audited frozen map shapes are:

| Layer | Shape |
|---|---|
| layer1 | `N x 64 x 56 x 56` |
| layer2 | `N x 128 x 28 x 28` |
| layer3 | `N x 256 x 14 x 14` |
| layer4 | `N x 512 x 7 x 7` |

## 4. Direction and augmentation

The frozen preprocessing performs no rotation, flip, random crop, or other
augmentation. It preserves array row/column order through grayscale conversion
and resize. It does not, however, rotate scans into a common laminate material
axis. The manifest records scan direction, but that is not sufficient to claim
specimen-level fiber-axis registration.

## 5. Specimen and laminate mapping

The issued V3 authority cross-checks specimen ID, dataset ID, crop hash,
metadata13, ply count, and layup family. Counts are:

| Dataset | Ply count | Layup | Specimens |
|---|---:|---|---:|
| `74t7kcdgkr` | 8 | cross-ply | 45 |
| `cgtnjyggtm` | 24 | quasi-isotropic | 49 |
| `w68dtmpfyf` | 16 | quasi-isotropic | 43 |
| `xcmzfsbd9t` | 24 | cross-ply | 59 |
| `yfxyg8jm46` | 16 | cross-ply | 42 |
| `ykhs7s2dck` | 8 | quasi-isotropic | 38 |

Published source documents define C8/C16/C24 as `[0/90]_(2s/4s/6s)` and
Q8/Q16/Q24 as `[45/0/-45/90]_(s/2s/3s)`. The workbooks identify each specimen's
layup code and ply count; the CAI specimen-size workbook supplies its recorded
thickness. A future `LaminateAuthority` can therefore expand and hash-bind a
sequence per specimen. The network must consume that authority and must not
infer laminate state from dataset ID.

## 6. Existing wavelet behavior

`src/cmc_bbdm/msss/wavelet_scale.py` applies PyWavelets `wavedec2` to raw RGB
images with `periodization` borders. The primary `low_only` mode retains only
`cA_L` and zeros all detail coefficients. The exploratory
`low_plus_boundary_details` mode also retains `coefficients[1]`, namely the
coarsest `(cH_L, cV_L, cD_L)` detail tuple. It does not perform the new one-level
DWT on frozen feature maps.

## 7. Reusable controls and split authorities

P3 provides deterministic `patch_shuffle_rgb` with an 8x8 primary grid. It
permutes only equal-shaped native patches and records the specimen-derived seed,
mapping digest, and source/output hashes. MGMR can call it unchanged before
feature extraction.

The V3 data authority provides six-domain nested LODO. MSSS provides issued
leave-one-ply-count-out and leave-one-layup-family-out group tasks and verifies
that specimen groups do not cross fit/query boundaries. M0 uses only nested
LODO; the structural splits remain conditional on an M0 pass.

## 8. Previous outer-test exposure

All six domains have appeared in P1, P3, P5, P6, invariance, multi-view, and
MSSS result review. Source-only selection inside each new outer fold prevents
within-run target leakage, but it cannot undo this historical human and protocol
exposure. MGMR M0 is therefore a checksum-bound post-hoc follow-up on a frozen
cohort. Any positive claim requires later validation on genuinely new laminate
configurations.
