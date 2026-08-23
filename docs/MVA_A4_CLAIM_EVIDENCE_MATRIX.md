# MVA A4 Claim-Evidence Matrix

Frozen before formal A4 result generation on 2026-08-23.

| Claim | Required evidence | Status after formal A4 |
|---|---|---|
| A3 establishes oracle headroom | Validated A3 H1-H4 package | PROVEN: `MVA_ORACLE_GO` |
| A2 rows can train an outer-safe A4 mask | Predictor-roster audit | REJECTED: current outer domain may enter an A2 source predictor |
| Source-learned global MVoM beats uniform | Paired AUEBC, positive lower bound, >=4/6 domains | REJECTED: point effect -0.000275, 95% interval [-0.001181, 0.000532] |
| Global MVoM beats global reconstruction ranking | Paired AUEBC with positive lower bound | REJECTED: point effect -0.000316, 95% interval [-0.001297, 0.000638] |
| Global MVoM beats global appearance ranking | Paired AUEBC with positive lower bound | REJECTED: point effect -0.000137, 95% interval [-0.001562, 0.001218] |
| Image fidelity is equivalent to mechanical utility | CAI MAE plus MSE/SSIM comparison | NOT SUPPORTED: reconstruction fidelity and CAI curves rank methods differently |
| A specimen-adaptive policy remains necessary | >=3% oracle-vs-global relative AUEBC gap, positive lower bound, >=4/6 domains | SUPPORTED: 39.766% relative gap, positive lower bound, 6/6 domains |
| A fixed normalized-raster mask is a physical scanner mask | Raw scanner-coordinate validation | NOT SUPPORTED |
| A5 deployable imitation policy improves acquisition | A5 held-out-domain evidence | AUTHORIZED: TO TEST |
| Laminate context improves acquisition | A6/A7 structured-transfer evidence | LOCKED |

Source full images and true source CAI may label offline rankings only through
the frozen OOF protocol. The target mask is fixed before any target image or
target CAI is read.
