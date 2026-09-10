# Visual Contract: Single-Case C-scan Overlay Diagnostic

- Artifact: nine raster panels for one reviewed TEST specimen.
- Target use: AEI manuscript figure assembly and expert visual diagnosis.
- Core question: whether the BC first-STOP mask follows the expert reference or the legacy full-input Reader target.
- Evidence mode: qualitative diagnostic display with machine-readable identity and overlap measurements; not a new experiment or performance estimate.
- Specimen: `ykhs7s2dck:q8-4`, `LOCATE`, `BC_S3`, seed `3`, first STOP at step `153`.
- Registered source: `source_root:data/public/hasebe/processed/ykhs7s2dck/internal_cscan/q8-4.png`.
- Expert source: `results/bc_cscan_expert_pooled_rescore/v1/inputs/references/reference-023-4c81fe9c.json`.
- Proxy source: frozen full-input Reader report recovered by the existing reviewed pipeline.
- BC source: stored actions replayed only through the actual calibrated first STOP.
- Coordinate contract: all mask fills and contour vertices use the unchanged 675 x 674 registered C-scan pixel frame, upper-left origin, x rightward, y downward.
- Encoding: `EXPERT_GT` green solid; `PROXY_GT` blue dashed; `BC_FINAL` red-orange dotted; trajectory magenta; measured cells amber.
- Layout: independent square panels plus one 2 x 3 browsing summary; legends sit below the image field.
- Export: PNG, long edge at least 1800 px, white background, embedded title and description metadata.
- Traceability: `single_case_overlay_manifest.json` binds image, report, first-STOP, trajectory, and coordinate identities.
