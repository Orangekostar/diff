# External Public-Data Feasibility, 2026

Audit date: 2026-08-27 UTC. Public accessibility is based on official archive
records; pairing and file counts use the frozen repository manifests. No raw
external cache was locally available during this revision.

| Dataset | DOI | Publicly downloadable now? | Specimen-level C-scan? | Mechanical endpoint? | CAI-compatible endpoint? | Exact pairing? | N | Material/layup mismatch | Target mismatch | Scanner/cost metadata | Allowed role |
|---|---|---|---|---|---|---|---:|---|---|---|---|
| Hasebe/JAXA main program | `10.17632/8scdmfdcfb.3` | Yes | Yes, across the linked public releases | Yes | Yes, CAI ratio used by the paper | Yes under the frozen paper authority | 276 | Main program; not external | None for the current paper | Native raster only; no physical scanner-time model | Main six-domain evidence program |
| Imperial Interlock | `10.5281/zenodo.1476887` | Yes | Yes, post-impact C-scan CSV | Yes, compression raw data and `Pmax_kN` | Descriptively compatible, not the paper's normalized CAI ratio | Yes: 10 exact pairs | 10 | 5 baseline and 5 reinforced; different material/layup | Peak load rather than normalized CAI ratio | No compatible end-to-end acquisition-cost contract | `POSSIBLE_EXTERNAL_MICRO_PILOT` only; not run here |
| Imperial RSS | `10.17632/wg4dmwddjy.2` | Yes | Shared C-scan containers; specimen ROI unresolved | Yes | Peak compression force only | No: `UNRESOLVED_CSCAN_SPECIMEN_ROI`; `paired_cscan_cai = false` | 7 candidate CAI rows | AFP/RSS groups differ from main program | Peak force rather than normalized CAI ratio | No compatible cost contract | `EXTERNAL_METHOD_EXPERIMENT_NO_GO` |
| TU Delft | `10.4121/21621381` | Yes | Yes, two images per CAI specimen | Yes, compression traces | Peak force only | Yes | N=3 | Different material and layup | Peak force and acoustic-emission study context | No compatible acquisition-cost contract | Illustrative micro-case only |
| Cranfield WP2 | `10.5281/zenodo.4405277` | Yes | Yes: processed and raw phased-array scans | No paired mechanics outcome | No | No method-endpoint pairing | 26 processed scan pairs; 29 raw files | Multiple coupons/laminates outside the main program | Missing CAI endpoint | Acquisition-format evidence only | Raw acquisition realizability only |
| Bath hybrid carbon-glass archive | `10.15125/BATH-00103` | No; retention period ended | Historically yes | Historically yes | Historically yes | Not currently auditable | Not available | Hybrid carbon-glass laminates | Raw target cannot be re-audited | Not available | Literature evidence only |

## Gate decision

- Exact pairing fails for RSS and no compatible endpoint exists for Cranfield.
- TU Delft has only N=3.
- Interlock has 10 exact pairs but is split 5/5 across a design intervention,
  uses a different target, lacks a compatible acquisition-cost contract, and
  has no local sealed cache for a pre-specified replay in this revision.
- Bath is no longer downloadable.

An n=10 or N=3 resource cannot be described as benchmark-scale validation.
No mapping is inferred from ordering, filenames, group labels, or visual
similarity.

Decision: `EXTERNAL_MICRO_PILOT_NO_GO` and `MANUSCRIPT_ONLY_PRIMARY`.
