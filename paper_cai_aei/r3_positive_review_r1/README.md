# CAI positive review R1

Task: CAI_POSITIVE_REVIEW_R1. Scientific scope: selected development configuration only.

The `build` folder contains English and aligned bilingual manuscript and supplement, each in editable DOCX and PDF. Canonical bilingual blocks are in `source`; shared tables, equations and figures appear once in the combined version. `figures` contains six main SVG/PDF/PNG figures and six saved-diagnostic plates. `tables` and `figure_data` contain frozen source subsets and explicit source keys.

Run from the repository root (Python, matplotlib/numpy/Pillow, Pandoc and XeLaTeX required):

```bash
python scripts/cai_manuscript_revision/run.py prepare-evidence --paper-root paper_cai_aei/r3_positive_review_r1 --artifact-root artifacts/cai_agent_v3/manuscript/r3_positive_review_r1
python scripts/cai_manuscript_revision/run.py render-figures --paper-root paper_cai_aei/r3_positive_review_r1 --artifact-root artifacts/cai_agent_v3/manuscript/r3_positive_review_r1
python scripts/cai_manuscript_revision/run.py build --paper-root paper_cai_aei/r3_positive_review_r1 --artifact-root artifacts/cai_agent_v3/manuscript/r3_positive_review_r1 --languages en,bilingual
python scripts/cai_manuscript_revision/run.py verify --paper-root paper_cai_aei/r3_positive_review_r1 --artifact-root artifacts/cai_agent_v3/manuscript/r3_positive_review_r1
```

Rendering and building use already supplied arrays and image subsets, without loading research weights. Initial `bind` additionally needs `--repo . --packet <extracted-packet-path>` and the precisely resolved local image pairs. Re-preparing evidence and text requires the private local originals and extractions under the docs root; these are ignored by Git and excluded from the delivery inventory. Building the supplied canonical sources does not require those private originals. No research computation is authorized. `SOURCE_BINDINGS.json`, `MODEL_IDENTITY_LOCK.json`, `REVIEW.md` and the handoff under the artifact root document evidence and QA. The publication receipt records actual remote verification after the content commit.

The main title, six chapters, five main tables, thirteen numbered equations and fifteen references are retained. Author identity, affiliations, funding and competing interests require author confirmation before submission. Independent evaluation, multiple training initializations, component interventions and physical scanner-time validation were not performed.
