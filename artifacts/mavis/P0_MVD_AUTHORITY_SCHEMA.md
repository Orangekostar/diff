# MAVIS P0 MVD Authority Schema Audit

## Files

| File | Bytes | SHA256 |
|---|---:|---|
| `artifacts/mvd_authority/a4_candidate_bank_0p015625.npz` | 38,232,468 | `42882df095ac2935bf7cf0efec5d17c3fee257ddee4f4735fbb64d6931976b08` |
| `artifacts/mvd_authority/a4_candidate_bank_0p03125.npz` | 38,293,282 | `cfeff617df9208bdfbd8434474f63134661f8fb6f7b59676bced5054ac0d2443` |
| `artifacts/mvd_authority/observed_candidate_features_0p015625.npz` | 332,767 | `234be30caba568d80f0a5c488858d2d2f2ea1c68d85326e920a3a96ac35081b7` |
| `artifacts/mvd_authority/observed_candidate_features_0p03125.npz` | 334,554 | `7bda69e4e0e2ec0fb8c9cbd2eefb81cd4e42c5b26da5e382fbf279d77e119883` |

All four load with `numpy.load(..., allow_pickle=False)` and match the hashes
registered in `paper_v3/configs/mvd_feasibility.yaml`.

## Classification

- `candidate_features` and `candidate_costs` are deployable pre-action geometry,
  budget, and observed-state descriptors.
- `initial_embeddings` are post-scout features derived from real sparse C-scan
  values. They are not pre-ultrasound context.
- candidate `embeddings`, `reconstruction_values`, and `appearance_values` are
  privileged counterfactual fields derived using the full scan.
- no NPZ contains raw RGB observations, acquired masks, or an action-indexed
  measurement payload that can be causally revealed.
- no NPZ directly contains true CAI, although source mechanical labels generated
  downstream use source true CAI under strict OOF prediction.

The complete per-array classification is in `P0_AUTHORITY_ARRAYS.csv`.

## Causal reveal decision

Current MVD authority is insufficient by itself. Reusing candidate embeddings as
post-action observations would leak future scan content.

Regeneration is feasible from the upstream registered authority. The existing
loader re-verified 276 hash-bound read-only RGB scans across six domains:

```text
authority_state_sha256 = ae03bf35f5b32665007e7f928ee4e2ed098083a4d0982c73100307886acee394
decoded bytes           = 340206552
dtype                   = uint8
shapes                  = (338,340,3), (338,352,3), (674,675,3)
```

MAVIS must create `artifacts/mavis_authority/` from those upstream records and
expose full scans only through a typed action-bound reveal interface.
