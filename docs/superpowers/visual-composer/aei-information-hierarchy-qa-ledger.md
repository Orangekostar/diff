# AEI Information Hierarchy Render QA Ledger

| Issue | Artifact | Page/section | Severity | Fix | Owner | Status |
|---|---|---|---|---|---|---|
| Evidence-lane text crossed box boundaries | Figure 1 | Section 3 | High | Wrapped lane descriptions and re-spaced title/detail rows | primary agent | closed |
| Four-stage WHY flow was incomplete | Figure 1 | Section 3 | High | Rebuilt as complete field, exact-cost limited sensing, task value, and state-conditioned loop | primary agent | closed |
| Effect annotations overlapped axes and bars | Figure 2 | Section 5 | High | Reserved internal annotation space and separated boundary notes | primary agent | closed |
| Former Figure 5 duplicated task-priority evidence | Figure 2 | Section 5 | High | Merged its registered real-state CAI/field-content/difference maps into panels (d--f) | primary agent | closed |
| Embedded controls occluded ticks and labels | Figure 3 | Section 5 | High | Rebuilt panels (a)/(c) with non-overlapping subaxes | primary agent | closed |
| Long adjacent panel titles collided | Figures 2-3 | Section 5 | High | Wrapped compact titles and added renderer-level bounding-box regression tests | primary agent | closed |
| Long planning labels crossed panel boundaries | Figure 4 | Section 5 | High | Shortened display labels while preserving full source labels | primary agent | closed |
| Image aspect repair triggered fixed-limit warnings | Figures 2-3/S1 | Section 5/supplement | Medium | Preserved equal data aspect with automatic data limits and warning regression test | primary agent | closed |
| Figure source/output hashes absent | Figures 1-4 | Section 3/5 | Medium | Maintained deterministic source/output checksum manifest | primary agent | closed |
| Vector text and font embedding | Figures 1-4 | Section 3/5 | Medium | Confirm editable SVG text and embedded Unicode TrueType PDF subsets | primary agent | closed |
| Manuscript-width recheck | Figures 1-4 | Section 3/5 | Medium | Inspect figures after LaTeX placement | primary agent | closed |
| Main evidence table duplicated the visual narrative | Table 2 | Section 5 | High | Retired it from manuscript/materialized assets/package; retained generator provenance only | primary agent | closed |
| Integrated manuscript table recheck | Table 1 | Section 4 | Medium | Compile and inspect with the journal manuscript class | primary agent | closed |

Checks to close: clipping, overlap, font embedding, editable SVG text, 300 dpi
PNG metadata, grayscale distinction, exact numbers/directions, caption presence,
source-data traceability, float order, and table width. PNG and vector visual
inspection used the generated artifacts at original resolution.

Current closeout scope: four main figures, one main table, one supplementary
gallery, 29 aspect-preserving panel PNGs, editable vector output, deterministic
checksums, grayscale-redundant encodings, exact signed effects, caption/source
traceability, manuscript float order, and retired-asset absence. Final test and
LaTeX build counts are recorded in the consolidation handoff.

Final automated QA: source preflight reported 18 PASS, 3 reviewed format WARN,
and 0 FAIL. The WARNs reflect the intentional PDF/SVG/300-dpi-PNG contract (no
TIFF, no 600-dpi raster) and a width expressed through a constant rather than a
literal. Four multi-panel alignment reports passed 26 comparisons with no WARN
or exemptions; Figure 1 is a single unframed panel. Five PDF text audits found
no glyph below 5 pt. Collision audit found 0 FAIL: Figures 1, 3, 4, and S1
passed directly; Figure 2 retained three reviewed WARNs where numeric direct
labels intentionally touch their own point markers. Original-size overlays and
the forced final LaTeX placements were inspected and accepted.
