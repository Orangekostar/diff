# Evidence Authority Reconciliation

Generated deterministically from frozen machine-readable evidence.

## Registered scalar/spatial authority

Paper 1 uses `B_scalar minus B_field_selected` as its confirmatory contrast. The
reference/candidate MAEs are `0.1892044107` and
`0.1284893565`; the effect is `0.0607150541` with
simultaneous interval `[0.0066387325, 0.1536429519]`
and `5/6` held-out domains improved.

`I_field_selected` is a different independent metadata-only-prefix estimator
with equal-domain MAE `0.0896358047`. It is sensitivity evidence,
not the registered B-family endpoint. The historical `0.099568606` comparison
must not replace the matched confirmatory effect.

## Sparse-retention authority

At 25% nominal normalized-raster density, full and sparse MAEs are
`0.1284893565` and `0.1345137120` and gain
retention is `0.8989734770`. The distinct sparse-minus-full gap is
`0.0060243555` with simultaneous interval
`[0.0017300690, 0.0108331488]`; sparse is worse
in all six domains. The gap interval is not the interval for surface-to-sparse
improvement.

## Paper rule

`PAPER_CANONICAL_METRICS.csv` is the only numeric source for manuscript prose,
captions, and main tables. Historical artifacts remain unchanged. P10's
explicit `I_field_selected` recovery endpoint remains a separately labeled
diagnostic.
