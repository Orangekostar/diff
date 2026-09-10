# Visual QA Ledger

| Issue | Artifact | Severity | Fix or evidence | Owner | Status |
|---|---|---:|---|---|---|
| Registered source could differ from annotation display | all panels | High | Source file, HTML packet, Reader input, and authority array are pixel-identical; hashes and dimensions match | Codex | Passed |
| Mask boundaries were vertically mirrored relative to their fills | panels 01-07, summary | High | Removed the second y-origin transform from the shared contour call, added an asymmetric-mask coordinate regression test, and regenerated all affected PNGs | Codex | Fixed |
| Dense 153-action trajectory could obscure the C-scan | panel 04, summary | Medium | Reduced path/grid opacity, separated trajectory and measured-cell colors, and labeled only action milestones | Codex | Passed |
| Three masks could be hard to distinguish | panels 06-07 | Medium | Fixed color plus solid/dashed/dotted redundancy; added boundary-only panel | Codex | Passed |
| Legend could cover the defect | all overlay panels | Medium | Legends placed below the registered image field | Codex | Passed |
| Raster export could be stretched or too small | all PNGs | High | Equal data aspect retained; panels are 2016 x 2016 and summary is 2160 x 1440 | Codex | Passed |
| Expert uncertainty could be hidden | panel 01 | Medium | Certain and uncertain encodings are distinct; this reference has no uncertain region and the legend states `uncertain: none` | Codex | Passed |
| PNG could be blank or unreadable | all PNGs | High | PIL decode and non-zero image-variance checks pass for every output | Codex | Passed |
