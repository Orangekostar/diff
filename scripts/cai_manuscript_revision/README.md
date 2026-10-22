# Frozen manuscript adapters

Only frozen text, CSV/JSON/NPZ arrays and the two explicitly bound image pairs are read. No research model is imported, loaded or executed. `run.py` requires explicit paper/artifact roots and limits numeric-library threads to four.

Run from the repository root; all commands use these exact roots:

```bash
python scripts/cai_manuscript_revision/run.py prepare-evidence --paper-root paper_cai_aei/r3_positive_review_r1 --artifact-root artifacts/cai_agent_v3/manuscript/r3_positive_review_r1
python scripts/cai_manuscript_revision/run.py render-figures --paper-root paper_cai_aei/r3_positive_review_r1 --artifact-root artifacts/cai_agent_v3/manuscript/r3_positive_review_r1
python scripts/cai_manuscript_revision/run.py build --paper-root paper_cai_aei/r3_positive_review_r1 --artifact-root artifacts/cai_agent_v3/manuscript/r3_positive_review_r1 --languages en,bilingual
python scripts/cai_manuscript_revision/visual_qa.py
python scripts/cai_manuscript_revision/run.py verify --paper-root paper_cai_aei/r3_positive_review_r1 --artifact-root artifacts/cai_agent_v3/manuscript/r3_positive_review_r1
```

Requirements: Python, NumPy, matplotlib, Pillow, PyMuPDF, pypandoc/Pandoc, XeLaTeX with xeCJK, Liberation and WenQuanYi fonts. Automatic figure audits used the installed nature-figure scripts at `/home/ww/.codex/skills/nature-figure/scripts`; these tools are not copied into the research repository. Their availability and all raw outputs are recorded. `package.prepare_package` refreshes claim/figure/requirement maps and README files after evidence preparation.

`visual_qa.py` intentionally resets manual review status to pending. After actually viewing final PDF pages and figure panels, record the reviewer and PASS status in DOCUMENT_VISUAL_REVIEW.json and FIGURE_PANEL_REVIEW.csv; verify rejects missing review, changed PDFs, page overflow, invalid DOCX structure or unresolved figure failures. Do not mark an uninspected page passed. Verify runs only the eight functions in test_revision.py, hashes the explicitly bound sources and writes a nonrecursive inventory.

Initial `bind` additionally requires `--repo . --packet <extracted-packet-path>` at evidence commit 6082b69f2792130a321c74b0efcb198f5d65cefb and the precise local originals. It hashes existing model bytes without tensor deserialization. The delivered branch already contains bound subsets, so manuscript rebuilding does not require rebinding or access to weights. Full scientific reproduction/training is outside this revision contract.

Publication is the normal two-commit protocol in docs/cai/positive_review_r1/REVIEW_AND_PUBLICATION.md. The manifest omits itself, the receipt, Python cache and temporary page contact sheets. Read the handoff and REVIEW.md before resuming.
