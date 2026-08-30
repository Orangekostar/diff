# P2 Cross-Domain Damage-to-Response Baselines

Status: `MACK_EXTENSION_NO_GO`

## Authority and scope

- P0: `9d44ead975119db2181a91efbf14b74165671a9d25b7b576d90f6e104757a633` / `P0_GO`
- P1: `37da95962395a0915f586820ab03f06d8d859856e8637d975bc302b1d555ebc7` / `P1_GO`
- Cohort: 276/276 across all six registered domains.
- Estimator: source-selected Ridge only; no neural model.
- F5 is privileged sensitivity only. F0-F4 contain no true impact context.
- No CAI strength, raw CAI trace, or post-CAI image is an inference input.

## Metrics

- `extension_peak_mm` / `F0`: equal-domain MAE `0.25165073108790009`, RMSE `0.31281392448111667`, pooled R2 `0.2021402886947995`.
- `extension_peak_mm` / `F1`: equal-domain MAE `0.21877049439568927`, RMSE `0.29103254844902604`, pooled R2 `0.30938261557404223`.
- `extension_peak_mm` / `F2`: equal-domain MAE `0.2025732174465229`, RMSE `0.27405430584971152`, pooled R2 `0.38761063249736283`.
- `extension_peak_mm` / `F3`: equal-domain MAE `0.18938663004381404`, RMSE `0.24917458003082091`, pooled R2 `0.49375371208380647`.
- `extension_peak_mm` / `F4`: equal-domain MAE `0.21270364757610419`, RMSE `0.27162918950534515`, pooled R2 `0.39840078986970973`.
- `extension_peak_mm` / `F5`: equal-domain MAE `0.21474304298978839`, RMSE `0.27186327518070241`, pooled R2 `0.39736344599409101`.
- `slope_u20_u60_mpa_per_mm` / `F0`: equal-domain MAE `35.346647237323104`, RMSE `53.859639084000548`, pooled R2 `0.74455606773125094`.
- `slope_u20_u60_mpa_per_mm` / `F1`: equal-domain MAE `48.285880069886964`, RMSE `65.525956199633697`, pooled R2 `0.6219097764901963`.
- `slope_u20_u60_mpa_per_mm` / `F2`: equal-domain MAE `35.940429247103104`, RMSE `54.314204336769585`, pooled R2 `0.74022607458673217`.
- `slope_u20_u60_mpa_per_mm` / `F3`: equal-domain MAE `32.745625191712492`, RMSE `52.041943257021387`, pooled R2 `0.76150695254512213`.
- `slope_u20_u60_mpa_per_mm` / `F4`: equal-domain MAE `45.168856540692865`, RMSE `63.769316433441887`, pooled R2 `0.64190996409910717`.
- `slope_u20_u60_mpa_per_mm` / `F5`: equal-domain MAE `49.570012333922676`, RMSE `67.360822281666458`, pooled R2 `0.60043864858008045`.
- `normalized_prepeak_auc` / `F0`: equal-domain MAE `0.034273698113643565`, RMSE `0.053712523943665304`, pooled R2 `-0.13512928432333915`.
- `normalized_prepeak_auc` / `F1`: equal-domain MAE `0.036571016255757899`, RMSE `0.054614124070463899`, pooled R2 `-0.17355690333167373`.
- `normalized_prepeak_auc` / `F2`: equal-domain MAE `0.033335534237596311`, RMSE `0.052840819521024408`, pooled R2 `-0.098584067881400506`.
- `normalized_prepeak_auc` / `F3`: equal-domain MAE `0.031504773437589756`, RMSE `0.05126955289207108`, pooled R2 `-0.034220798814433406`.
- `normalized_prepeak_auc` / `F4`: equal-domain MAE `0.035684720073733987`, RMSE `0.054495612410769639`, pooled R2 `-0.16846923496775856`.
- `normalized_prepeak_auc` / `F5`: equal-domain MAE `0.036349257777404921`, RMSE `0.054875038023077478`, pooled R2 `-0.18479680840643331`.

## Paired contrasts

- `extension_peak_mm__F3_vs_F2`: relative improvement `0.065095413741898062`, domains `5/6`, familywise CI [-0.0066943245447845506, 0.035128376601972489].
- `extension_peak_mm__F4_vs_F2`: relative improvement `-0.05000873391496375`, domains `3/6`, familywise CI [-0.031222496803089254, 0.012822351105466062].
- `slope_u20_u60_mpa_per_mm__F3_vs_F2`: relative improvement `0.088891649941774759`, domains `4/6`, familywise CI [1.1184686316057244, 5.3024313736289468].
- `slope_u20_u60_mpa_per_mm__F4_vs_F2`: relative improvement `-0.2567700911455752`, domains `2/6`, familywise CI [-13.019689683974992, -5.5584686191078134].
- `normalized_prepeak_auc__F3_vs_F2`: relative improvement `0.054919197843297456`, domains `3/6`, familywise CI [-0.00025596902736786247, 0.0040453357420633749].
- `normalized_prepeak_auc__F4_vs_F2`: relative improvement `-0.07047092209154443`, domains `2/6`, familywise CI [-0.0054025784788464398, 0.00069526806214420295].
- `extension_peak_mm__F4_vs_F3`: relative improvement `-0.12311860413217041`, domains `3/6`, secondary.
- `slope_u20_u60_mpa_per_mm__F4_vs_F3`: relative improvement `-0.37938598747916219`, domains `1/6`, secondary.
- `normalized_prepeak_auc__F4_vs_F3`: relative improvement `-0.13267661309879314`, domains `1/6`, secondary.

## Gate

- `extension_peak_mm` / `F3` vs `F2`: `FAIL`; relative equal-domain MAE improvement is below 10%; familywise bootstrap lower bound is not positive.
- `extension_peak_mm` / `F4` vs `F2`: `FAIL`; relative equal-domain MAE improvement is below 10%; fewer than four held-out domains improve; familywise bootstrap lower bound is not positive.
- `slope_u20_u60_mpa_per_mm` / `F3` vs `F2`: `FAIL`; relative equal-domain MAE improvement is below 10%.
- `slope_u20_u60_mpa_per_mm` / `F4` vs `F2`: `FAIL`; relative equal-domain MAE improvement is below 10%; fewer than four held-out domains improve; familywise bootstrap lower bound is not positive.
- `normalized_prepeak_auc` / `F3` vs `F2`: `FAIL`; relative equal-domain MAE improvement is below 10%; fewer than four held-out domains improve; familywise bootstrap lower bound is not positive.
- `normalized_prepeak_auc` / `F4` vs `F2`: `FAIL`; relative equal-domain MAE improvement is below 10%; fewer than four held-out domains improve; familywise bootstrap lower bound is not positive.

- Passing primary contrasts: []
- P3-P5: `NOT_RUN_NOT_AUTHORIZED`
- Existing Paper 1 manuscript/evidence: unchanged.
