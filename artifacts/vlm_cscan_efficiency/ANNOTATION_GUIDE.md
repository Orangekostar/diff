# VLM C-scan annotation guide

## Scope

The pilot contains 60 hash-selected Hasebe specimens, ten from each of six
domains. Source data remain under `/home/ww/paper3/cmc_damage_inference`; Git
contains manifests and review templates, not a duplicate raw dataset.

Surface review and C-scan reference review are separate blinded activities.
Do not give a surface reviewer the C-scan, CAI, impact energy, domain label, or
the automatic C-scan proposal. Do not use a surface annotation as internal
damage truth.

## Generate the queues

```bash
python scripts/run_vlm_cscan.py export-annotations
```

This creates:

- `results/vlm_cscan_efficiency/surface_annotation_queue/`: 60 surface-only
  templates with the fixed clockwise ROT90 rendering identity;
- `results/vlm_cscan_efficiency/annotation_queue/`: 60 registered C-scan
  templates with algorithmic starting polygons;
- `results/vlm_cscan_efficiency/visualizations/`: 12 review examples and an
  index. These examples combine views and therefore are for C-scan review, not
  blinded surface review.

Paths relative to the external data root are listed in
`input_manifest.csv`. Re-running export never overwrites existing polygon
annotations; it only refreshes provenance on untouched automatic templates.

## Surface-only view

Inspect only the impacted-surface image in `registered_surface_rot90` frame.
Use normalized `(x, y)` coordinates in `[0,1]`. Add visible regions as
`indentation_like`, `crack_like`, or `other_suspicious`; add reflection,
contamination, and texture variation under `interference_regions`. Set
`unable_to_determine=true` when no reliable visible cue exists. A surface cue
is a hypothesis, not evidence of internal damage.

Surface labels are not used as hidden LOCATE or CHARACTERIZE references in
this pilot. They are retained for later surface-plan and appearance-stratum
analysis only.

## Registered C-scan view

Inspect the complete registered C-scan without consulting the surface plan or
method outputs. Polygon coordinates are normalized `(x, y)` in `[0,1]`.

- `regions`: certain-positive C-scan-defined indications;
- `uncertain_regions`: uncertain boundary/band excluded from IoU and recall;
- certain background is the remainder outside both sets.

The supplied polygons are only `ALGORITHM_DERIVED_NOT_REVIEWED` proposals.
Reviewers must edit or delete them; their presence is not human confirmation.
The current benchmark evaluates positive-indication tasks only. A reviewed
case with no certain indication remains formally ineligible and should record
`NO_ELIGIBLE_INDICATION` in `notes`.

For an attributable completed review, set:

```json
{
  "reference_type": "EXPERT_REVIEWED",
  "review_state": "reviewed",
  "reviewer_alias": "R1"
}
```

`AUTHOR_PROVIDED` is allowed only for a traceable author-supplied region, not
for the existing registration correspondence. Never change `specimen_key`,
`frame`, or `source_image_sha256`.

## Resume after review

```bash
python scripts/run_vlm_cscan.py prepare
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/run_vlm_cscan.py run --cohort pilot --workers 4
python scripts/run_vlm_cscan.py evaluate --cohort pilot
python scripts/run_vlm_cscan.py verify
```

Reviewed-reference bytes enter the resumable execution identity. A changed
human reference therefore triggers fresh hidden scoring while identical VLM
responses remain cached. Partial review is reported with its actual per-domain
coverage; unreviewed cases retain null formal success.
