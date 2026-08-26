# P0 Evidence Conflicts

Audit date: 2026-08-26

## Conflict 1: registered full-field path versus independent estimator

The P1 payload contains two semantically different estimator families.

| Method | Feature semantics | Equal-domain MAE | Paper role |
|---|---|---:|---|
| `A_surface` | 13 metadata + 21 surface statistics | 0.1881207811 | surface reference |
| `B_scalar` | `A_surface` prefix + 3 scalar internal descriptors | 0.1892044107 | matched scalar reference |
| `B_field_selected` | same metadata+surface prefix + fold-selected full-field representation | 0.1284893565 | registered confirmatory field estimator |
| `I_field_selected` | independent metadata-only prefix + fold-selected full-field representation | 0.0896358047 | internal sensitivity estimator |

`src/cmc_bbdm/cpb_v3/models.py` defines the B candidates and the independent I
candidates separately. `select_internal_candidate` is the inner-LODO selector
for the independent metadata-only candidates.

The historical science-closure estimate
`scalar_minus_full_field_equal_domain_mae = 0.09956860599738323` is
`B_scalar - I_field_selected`. It is not the registered matched B-family
contrast.

Paper 1 therefore uses this confirmatory headline:

- comparison: `B_scalar -> B_field_selected`
- MAE: `0.1892044107 -> 0.1284893565`
- reference-minus-candidate effect: `0.0607150541`
- relative improvement: `32.0897%`
- domains improved: `5/6`
- simultaneous interval: `[0.0066387325, 0.1536429519]`

The related `A_surface -> B_field_selected` effect is `0.0596314245`, relative
improvement `31.6985%`, `5/6`, simultaneous interval
`[0.00721773, 0.14842086]`.

The `I_field_selected` result may appear only as an explicitly labeled
independent-estimator sensitivity result. Historical closure files remain
unchanged.

## Conflict 2: sparse-retention interval attached to the wrong contrast

The selected 25% bilinear sparse condition has MAE `0.1345137120`; the full
field has MAE `0.1284893565`; and the surface reference has MAE
`0.1881207811`. Two different contrasts are present:

| Contrast | Estimate | Simultaneous interval | Domain result |
|---|---:|---|---|
| surface minus sparse | 0.0536070691 | `[0.0056111806, 0.1354427482]` | sparse improves 5/6 |
| sparse minus full | 0.0060243555 | `[0.0017300690, 0.0108331488]` | sparse is worse in 6/6 |

Retention `0.8989734770` is the ratio of sparse gain to full-field gain. The
frozen summary associates `5/6` with the surface-to-sparse improvement but
quotes `0.0017300690`, which is the lower bound for the sparse-minus-full gap.
Paper-specific authority must keep those statements separate.

Nominal density is unique observed native-raster locations divided by native
location count. It is not measured scanner time and cannot support a scanner-
time reduction claim.

## Conflict 3: P10 full-field reference has a different purpose

P10 explicitly uses `I_field_selected` as its full-field endpoint
(`full_field_method` in `summary.json`) when reporting partial-state utility
recovery. That diagnostic answers how much error reduction remains relative to
the historical independent dense estimator. It does not redefine the P1
confirmatory scalar-versus-spatial contrast.

Accordingly:

- P1 usefulness headline: matched `B_scalar -> B_field_selected`.
- P10 recovery fraction: retain the explicit `I_field_selected` endpoint and
  label it as a closure diagnostic.
- Never place the P10 recovery fraction beside the B-path effect without
  explaining the different endpoint.

## Conflict 4: internal decision labels versus scientific prose

`M0_GO_NOGO.md`, `M1_GO_NOGO.md`, and `GO_NOGO.md` are valid tracked project
records. Their labels are not manuscript constructs. Paper text must translate
them into bounded findings:

- retrospective one-shot acquisition headroom exists;
- static pre-inspection value was not reliably observable;
- conditional value changes, but the learned state/control evidence is mixed;
- valuation and bounded set planning are separable limitations;
- the frozen learned policy did not outperform the strongest deployable
  baseline.

## Resolution

No historical artifact will be edited. Phase P1 will create a paper-specific,
hash-bound authority that records exact method semantics, contrast direction,
statistical unit, oracle/deployable status, and allowed/forbidden wording.
