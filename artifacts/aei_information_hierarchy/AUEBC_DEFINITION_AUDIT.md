# AUEBC Definition Audit

## Authority binding

- Manuscript equation: `paper_aei_information_hierarchy/main.tex`, Eq. `eq:auebc`.
- Closed-loop authority: `src/cmc_bbdm/mavis/closed_loop_metrics.py::_auebc`.
- Independent consistency check:
  `src/cmc_bbdm/mavis/task_specificity.py::normalized_auebc`
  (`task_specificity.normalized_auebc`).

Both implementations evaluate `np.trapezoid(y, x=x) / (x[-1] - x[0])` after
ordering strictly increasing cost coordinates. The x-axis is the
actual/effective specimen budget, not the nominal checkpoint label.

## Manuscript correction

The revised equation defines specimen-specific effective budgets
`x_{i,1}<...<x_{i,K}` and divides the trapezoidal sum by
`x_{i,K}-x_{i,1}`. AUEBC is the budget-span-normalized trapezoidal mean error
over the observed effective-budget range. Lower is better.

## Historical result status

The implementation produced every frozen historical AUEBC result using this
normalized definition. The mismatch was confined to manuscript notation;
historical numbers do not require recomputation.
