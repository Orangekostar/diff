# MVA A4 Visual Iteration Log

## Iteration 0: preregistered structure

- Frozen three artifacts and their panel maps before formal outcomes.
- Selected a color-vision-safe semantic palette with redundant markers/styles.
- Reserved one common sequential heatmap scale for all ranking objectives.
- Prohibited outcome-based specimen selection, smoothing, and axis truncation.
- Next action: implement table-only rendering, then inspect PNG and SVG outputs.

## Iteration 1: formal render QA

- Rendered all three artifacts solely from the validated six-domain tables.
- Confirmed common heatmap limits, cell orientation, editable SVG text, and
  redundant curve markers and line styles.
- Found the CAI legend obscuring upper-budget curves and moved it into the
  unused low-MAE region without changing data, axes, or method styling.
- Next action: re-render and complete pixel, SVG, and source-data checks.

## Iteration 2: final visual verification

- Confirmed the relocated legend does not overlap any curve or reference line.
- Verified all PNG outputs are nonblank at publication resolution and all SVG
  outputs retain editable DejaVu Sans text.
- Verified `source_data.csv` contains exactly 288 plotted records, including 192
  ranking-consensus cells, with all three figure identifiers present.
- Closed all render-QA ledger items without changing the frozen data contract.
