# G1 Surface-Robustness Result

Status: `NO_SUPPORTED_OBSERVABLE_SURFACE_BENEFIT`

Each effect is control AUEBC minus correct-surface AUEBC, so positive values
favor the correctly registered surface hypothesis.

| Task and control | Effect | 95% CI | Improved domains |
|---|---:|---:|---:|
| FIELD NO_SURFACE | 0.00000045434448030215293 | [-0.000012668568159756931, 0.000009324295637885728] | 4/6 |
| FIELD SHUFFLED_SURFACE | -0.000002043157793511139 | [-0.000015982797216280143, 0.000010279149341085201] | 3/6 |
| CAI NO_SURFACE | -0.00076450645226755 | [-0.001571119192022546, 0.00005345423365951419] | 3/6 |
| CAI SHUFFLED_SURFACE | 0.000056822168960363406 | [-0.0006233811088856258, 0.000720148044443176] | 4/6 |

All four intervals include zero. No held-out result supports a reliable benefit
from the registered surface input under the selected policy class.

## Frozen Misleading-Surface Diagnostic

The strata were defined from frozen G0 initialization evidence before G1 target
evaluation: AGREE 1, PARTIAL 114, MISLEADING 161. Authority SHA-256 is
`519b96c0cf525cda32ff54e91792524c77fcc858e3fa33b10ee292d3be8141e6`.

Within MISLEADING, FIELD NO_SURFACE had effect -0.00001180417977104304
(95% CI [-0.00003145472859875145, 0.0000003686436919438881], 3/6 domains),
FIELD SHUFFLED_SURFACE -0.000010973357028361278
([-0.000030307127432930103, 0.000003068018171640835], 2/6), CAI NO_SURFACE
-0.0009367797589696085
([-0.0019160364715854607, 0.00003920981062707374], 3/6), and CAI
SHUFFLED_SURFACE -0.000002552665529868171
([-0.0008164456643181878, 0.0007990916485116415], 2/6). These subset analyses
are diagnostic only and do not alter any G1 gate.
