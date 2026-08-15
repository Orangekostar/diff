# V3 P2 Distillation Baselines

Response: raw damaged-to-intact CAI strength ratio (unit 1).

## Equal-domain MAE

| Method | Equal-domain MAE |
|---|---:|
| D0_equal_capacity | 0.148362319222 |
| MSPD | 0.14192667503 |
| scalar_teacher | 0.139798308383 |
| global_shuffled_teacher | 0.14119327034 |
| stratified_shuffled_teacher | 0.142724630085 |
| random_teacher | 0.141686847889 |

## Registered effects

| Effect (reference - MSPD) | Estimate | Relative improvement | Improved domains | Simultaneous interval |
|---|---:|---:|---:|---:|
| D0_vs_MSPD | 0.00643564419182 | 0.0433778888438 | 4 | [-0.010214260255, 0.0246829665328] |
| scalar_teacher_vs_MSPD | -0.00212836664677 | -0.0152245522237 | 4 | [-0.0150580664725, 0.010199586024] |
| global_shuffled_teacher_vs_MSPD | -0.000733404690331 | -0.0051943317735 | 2 | [-0.00180232212241, 0.000361459397988] |
| stratified_shuffled_teacher_vs_MSPD | 0.000797955054698 | 0.00559087141598 | 5 | [-0.000293005166069, 0.00235516752394] |
| random_teacher_vs_MSPD | -0.000239827140676 | -0.00169265633507 | 4 | [-0.00382117045876, 0.00262229393535] |
