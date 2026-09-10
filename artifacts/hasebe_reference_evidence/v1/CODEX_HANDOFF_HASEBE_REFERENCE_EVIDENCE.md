# Codex Handoff: Hasebe Reference Evidence

## Repository identity

- Branch: `research/hasebe-reference-evidence-trace`
- Base SHA: `a48c76fcead10d4f833c54ab6e892eb524432269`
- Expert reviewed evidence: `6954e37c8c6aa5fc8530d2668d689eae85604691`
- Reference version: `REVIEWED_02339eda86d486c3`
- Final SHA is reported after push; it is intentionally not embedded in this self-containing commit.

## Result

The bound evidence yields author scalar damage measurements for 24/24 TEST specimens and source C-scan screenshots for 24/24. It yields 0/24 author spatial contours and 0/24 raw quantitative ultrasound arrays.

`published_damage_measurements.csv` is a direct, exact-ID transcription of columns G/H in the CAI LVI workbook, with deterministic hash-split calibration/validation assignment added downstream. `physical_descriptors.csv` is a local RGB threshold/morphology result calibrated against author area scalars; it is not author GT.

For q8-4, no new spatial evidence resolves the expert/Reader disagreement. The author scalar constrains magnitude only. A usable current protocol can evaluate task correctness against the reviewed expert spatial reference and separately audit magnitude against author scalars; it cannot claim a unique correct scan route or physical time/path cost.

## Fixed-case checks

- `ykhs7s2dck:q8-4`: raw screenshot -> registered crop -> Reader input is `PIXEL_IDENTICAL`; axes `75 x 75 mm`; `NOT_PRESENT_IN_INSPECTED_SCREENSHOT_OR_BOUND_FILES`.
- `74t7kcdgkr:c8-24`: raw screenshot -> registered crop -> Reader input is `PIXEL_IDENTICAL`; axes `75 x 75 mm`; `NOT_PRESENT_IN_INSPECTED_SCREENSHOT_OR_BOUND_FILES`.
- `cgtnjyggtm:q24-40`: raw screenshot -> registered crop -> Reader input is `PIXEL_IDENTICAL`; axes `75 x 75 mm`; `NOT_PRESENT_IN_INSPECTED_SCREENSHOT_OR_BOUND_FILES`.

## Scientific and resource integrity

- No training, threshold tuning, VLM/Actor/STOP forward pass, BC prefix replay, formal reviewed rescore, or expert-label change was performed.
- Exactly six deterministic full-input Reader reports were recovered for the three predeclared cases; all report hashes matched the frozen reviewed table.
- The historical frozen results and the prior reviewed negative conclusion remain unchanged.

## Reproduction

```bash
python scripts/trace_hasebe_reference_evidence.py inventory --source-root /path/to/cmc_damage_inference --output-root results/hasebe_reference_evidence/v1
python scripts/trace_hasebe_reference_evidence.py extract --source-root /path/to/cmc_damage_inference --output-root results/hasebe_reference_evidence/v1
python scripts/trace_hasebe_reference_evidence.py compare --source-root /path/to/cmc_damage_inference --output-root results/hasebe_reference_evidence/v1 --artifact-root artifacts/hasebe_reference_evidence/v1
python scripts/trace_hasebe_reference_evidence.py summarize --output-root results/hasebe_reference_evidence/v1 --artifact-root artifacts/hasebe_reference_evidence/v1
```

The final verification commands and observed pass counts are appended only after they are actually run.

## Verification evidence

- `PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_hasebe_reference_evidence.py`: 8 passed in 77.36 s.
- `PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_single_case_overlay_diagnostic.py tests/test_cscan_expert_pooled_rescore.py`: 18 passed in 48.14 s.
- `python -m ruff check` on the new module, CLI and test: passed.
- `python -m ruff format --check` on the new module, CLI and test: passed.
- All three exported PNGs were decoded, visually inspected, and verified pixel-identical to their bound source JPEG decode at 891 x 891 px.
- `git diff --check` and frozen-path diff checks are run again immediately before commit.
