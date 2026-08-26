# AEI Information Hierarchy Render QA Ledger

| Issue | Artifact | Page/section | Severity | Fix | Owner | Status |
|---|---|---|---|---|---|---|
| Evidence-lane text crossed box boundaries | Figure 1 | Section 3 | High | Wrapped lane descriptions and re-spaced title/detail rows | primary agent | closed |
| Effect annotations overlapped axes and bars | Figure 2 | Section 5 | High | Reserved internal annotation space and separated boundary notes | primary agent | closed |
| Embedded controls occluded ticks and labels | Figure 3 | Section 5 | High | Rebuilt panels (a)/(c) with non-overlapping subaxes | primary agent | closed |
| Long planning labels crossed panel boundaries | Figure 4 | Section 5 | High | Shortened display labels while preserving full source labels | primary agent | closed |
| Figure source/output hashes absent | Figures 1-4 | Section 3/5 | Medium | Added deterministic 20-row `FIGURE_CHECKSUMS.csv` | primary agent | closed |
| Vector text and font embedding | Figures 1-4 | Section 3/5 | Medium | Confirm editable SVG text and embedded Unicode TrueType PDF subsets | primary agent | closed |
| Manuscript-width recheck | Figures 1-4 | Section 3/5 | Medium | Inspect figures after LaTeX placement | primary agent | open |
| Dense hierarchy table repeated group labels | Table 2 | Section 5 | Medium | Show layer/question once per group and add inter-group spacing | primary agent | closed |
| Standalone table compile and render | Tables 1-2 | Section 4/5 | High | Compile a two-page booktabs/tabularx harness and inspect both pages | primary agent | closed |
| Integrated manuscript table recheck | Tables 1-2 | Section 4/5 | Medium | Compile and inspect with the journal manuscript class | primary agent | open |

Checks to close: clipping, overlap, font embedding, editable SVG text, 300 dpi
PNG metadata, grayscale distinction, exact numbers/directions, caption presence,
source-data traceability, float order, and table width. PNG and vector visual
inspection used the generated artifacts at original resolution.

Closed checks: 10 figure tests passed; all four PDFs report embedded/subsetted
Unicode TrueType fonts; all four SVGs contain editable text nodes; the checksum
manifest binds 20 figure deliverables.

Table checks: 11 table tests passed; the standalone two-page LaTeX harness
reported no overfull or underfull boxes; both pages were inspected at 150 dpi;
the checksum manifest binds six table deliverables.
