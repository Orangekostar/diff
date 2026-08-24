# External CAI Data Feasibility Audit

Date: 2026-08-24

## Scope and discipline

The official archives for Imperial RSS, Imperial Interlock, and TU Delft were
downloaded, unpacked, hashed, inspected, paired, and counted. No Uniform,
Reconstruction, MVD, CAI predictor, or other method was run on these data. The
data remain sealed from method-performance inspection.

| Dataset | Exact paired N | C-scan | CAI target | Groups | License | Role |
| --- | ---: | --- | --- | --- | --- | --- |
| Imperial RSS | 0 (7 potential) | Two raw multi-specimen A-scan containers; specimen ROIs unresolved | Peak force derivable for 7 CAI curves | Baseline, AP-ply, RSS amplitude variants, hybrid inner/outer | CC BY 4.0 | Sealed, pending ROI map |
| Imperial Interlock | 10 | One post-impact depth C-scan CSV per specimen | `Pmax` and full force-displacement/strain record | 5 baseline, 5 reinforced | CC BY 4.0 | Sealed small external pilot |
| TU Delft | 3 | Two C-scan JPGs per specimen | Peak absolute force derivable from MTS CSV | One CAI group | CC0 | `MICRO_CASE_VALIDATION_ONLY` |

## Imperial RSS

Source: `10.17632/wg4dmwddjy.2`. Downloaded archive SHA-256:
`957ea11e12b49fb59848b5a00ca4764eefd33c08efae26aa4e68481f21a11ced`.

The archive contains seven CAI force-displacement-strain CSVs and seven 35 J
LVI specimen files. Their names support the following potential links:
`AP-3`, `B24-4`, `HBI-1`, `HBO-1`, `SP2-3`, `SP4-3`, and `SP8-3`.

The post-impact C-scan evidence is not published as seven specimen files. It is
stored in two very large containers,
`LVIspecimens_after35Jimpact_set1_A-Scan_CSV.txt` and `set2`, with headers and
numeric scan tensors but no included table mapping a specimen ID to a container
and ROI. Consequently:

- exact machine-resolvable paired C-scan+CAI N: **0**;
- potential filename-linked specimens awaiting an ROI map: **7**;
- formal replication status: **not currently feasible**.

Counting the seven potential links as paired samples would silently assume the
missing spatial correspondence and is forbidden.

## Imperial Interlock

Source: `10.5281/zenodo.1476887`. Downloaded archive SHA-256:
`d15e7f405f7120663a151fb944b4b41fb1525f96f45cf5b00d1651400ff04292`.

All ten CAI specimens have an exact specimen-ID match between a post-impact
C-scan CSV and a compression-test CSV: `BL-007` through `BL-011` and `RE-007`
through `RE-011`. The compression header supplies `Pmax`; the remainder of each
file supplies load, displacement, corrected displacement, four strain gauges,
average strain, and time. Exact specimen dimensions are included separately.

The C-scan is a 0.2 mm by 0.2 mm pixel grid. Each value is the recorded return
depth in mm, with zero meaning no recorded return. The two groups use the same
material system and layup; the reinforced group adds interlocked thin-ply
interfaces.

The pairing is technically complete, but N=10 and group N=5 are too small for
a stand-alone, well-powered statistical external benchmark. The dataset is
appropriate as a sealed small external pilot after a future M2/M3 protocol is
frozen.

## TU Delft

Source: `10.4121/21621381`. The selected official files and full article
metadata were downloaded and hash-bound. There are exactly three CAI specimens.
Each has two C-scan JPGs and one MTS force-time-displacement CSV.

The README binds the specimens to Toray M30SC / Deltapreg DT120-200-36 UD,
`[-45,0,45,90]4s`, 150 x 100 x 5.15 mm geometry, 34 J target impact energy,
ASTM D7136 impact, and ASTM D7137 CAI testing. N=3 is explicitly classified as
`MICRO_CASE_VALIDATION_ONLY`; it cannot support statistical external claims.

## Decision

No downloaded dataset is presently sufficient by itself for formal statistical
external replication. Interlock is the only immediately usable specimen-level
paired pilot (N=10). RSS could become useful if an authoritative specimen-to-ROI
map is obtained. TU Delft remains case-level only. All manifests contain data
identity and feasibility evidence, not method results.

Machine-readable evidence is under `artifacts/external_data/`.
