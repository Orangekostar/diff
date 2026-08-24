# Cranfield Raw PA Acquisition Audit

Date: 2026-08-24

## Source and scope

Source: CompInnova WP2, `10.5281/zenodo.4405277`. Downloaded archive SHA-256:
`2e180fc3109f5510cba40ef083dea26972c37fb897e878ad30ebfac139b13565`.

The archive was downloaded, unpacked, hashed, and structurally parsed. No CAI
label is present and no prediction or acquisition method was run. This audit can
only establish raw-location realizability, not CAI validity or scanner-time
savings.

## Inventory and pairing

- 5 MHz wheel PA: 11 raw CSVs and 11 exact-stem processed PNGs.
- 10 MHz PA: 18 raw CSV components and 15 processed TIFF scans.
- Total: 29 raw files and 26 processed scan pairs.

Most pairs are one raw file to one processed image. The laminate `AB` scan binds
two raw A/B components to one TIFF; the repaired `ABC` scan binds three raw
A/B/C components to one TIFF. All 29 raw components are accounted for exactly
once in the 26 pair rows.

## Raw CSV semantics

Every file is a sequence of frame blocks:

```text
#Frame,<frame index>,of,<frame count>
#Scan,1,of,1
#Spec,<Buckets|Amplitudes>,<sample count>,FL,57
#FL1,...,#FL57
<sample rows>
```

Observed frame counts are 100, 150, or 200. Each sample row has 57 focal-law
values. The waveform/sample dimension is 504 buckets for the 10 MHz and 1.6 mm
5 MHz records, and 836 buckets for the 3.2 mm 5 MHz records. Every declared
frame count equals the number of parsed frame headers.

One recoverable physical acquisition record is therefore a
`(frame index, focal-law index)` spatial location with all 504 or 836 amplitude
samples. The raw tensor shape is
`frame_count x waveform_buckets x 57`.

## Answers to the acquisition questions

1. **Record semantics:** a spatial frame/focal-law location carries a complete
   file-declared amplitude-sample vector.
2. **Spatial grid recovery:** yes, as discrete frame and focal-law indices.
3. **Scan spacing:** not authoritatively recoverable. The 5 MHz README states a
   150 mm scan and the files contain 150 frames, but no coordinate vector or
   encoder-step field proves a 1 mm mapping. The 10 MHz files provide neither a
   coordinate vector nor spacing metadata.
4. **Processed/raw correspondence:** exact by stem for 24 scans, plus explicit
   A/B and A/B/C component grouping for the other two. Processed pixel geometry
   is presentation geometry and is not treated as a raw-index calibration.
5. **8x8 mapping:** yes at normalized-index level. Endpoint-preserving rounded
   linspace boundaries partition the frame and focal-law axes into 8x8 cells.
6. **Sparse raw mask:** yes, as a subset of unique frame/focal-law pairs while
   retaining the full waveform vector at selected pairs.
7. **Exact measurement fraction:** yes for the declared location unit,
   `selected unique pairs / (frame_count * 57)`. A different hardware trigger
   unit would require scanner documentation.

## Claim boundary

Supported:

> Normalized spatial acquisition masks can be mapped to realizable indexed raw
> PA measurement locations in the Cranfield archive.

Not supported:

- CAI prediction validation;
- calibrated millimetre coordinates for every file;
- actual scanner control or acquisition-time reduction;
- equivalence between one frame/focal-law pair and a particular hardware timing
  unit.

The full raw-file manifest, processed/raw pair manifest, grid schema, and an
example 8x8 mapping are under `artifacts/external_data/cranfield_wp2/`.
