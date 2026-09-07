# G1 Observable Task-Conditioned Inspection Policy

Final status: `G1_POLICY_OBSERVABILITY_NO_GO`

## Engineering policy

- FIELD: `G1_FIELD_POLICY_NO_GO`; fixed/learned/oracle AUEBC 0.002047837067539446/0.002120618581802132/0.0019655610066778156; effect -7.27815142626863e-05, 95% CI [-8.689672989010225e-05, -6.182037360547485e-05], domains 0/6; oracle-gap closure -0.8846013469833925
- CAI: `G1_CAI_POLICY_NO_GO`; fixed/learned/oracle AUEBC 0.02859496549734356/0.029653166484753902/0.007802877380960266; effect -0.0010582009874103384, 95% CI [-0.0020696313885150072, -5.9502896739864196e-05], domains 2/6; oracle-gap closure -0.0508944066361726
- Task conditioning: `G1_TASK_CONDITIONING_NO_GO`

## Task and surface controls

- FIELD WRONG_TASK minus correct: effect 3.0030636740644057e-06, 95% CI [1.2334469453508708e-06, 5.033437921055967e-06], domains 4/6
- FIELD NO_TASK minus correct: effect 1.7949810500134293e-06, 95% CI [6.563384341036675e-07, 3.1700703012368995e-06], domains 4/6
- CAI WRONG_TASK minus correct: effect 6.731435252771757e-05, 95% CI [-7.233853622756369e-05, 0.0002084059247615635], domains 4/6
- CAI NO_TASK minus correct: effect 3.107987501816518e-05, 95% CI [-0.00011143823251693787, 0.0001704391507672262], domains 4/6
- FIELD NO_SURFACE minus correct: effect 4.5434448030215293e-07, 95% CI [-1.2668568159756931e-05, 9.324295637885728e-06], domains 4/6
- FIELD SHUFFLED_SURFACE minus correct: effect -2.043157793511139e-06, 95% CI [-1.5982797216280143e-05, 1.0279149341085201e-05], domains 3/6
- CAI NO_SURFACE minus correct: effect -0.00076450645226755, 95% CI [-0.001571119192022546, 5.345423365951419e-05], domains 3/6
- CAI SHUFFLED_SURFACE minus correct: effect 5.6822168960363406e-05, 95% CI [-0.0006233811088856258, 0.000720148044443176], domains 4/6

## Source-only decision diagnostics

- High-level decision accuracy: 0.8068218988443118
- Primitive top-1 match: 0.13508592485972484
- Top-5 utility recall: 0.303918434815895
- Expected teacher regret: 0.00014463970670956552
- Candidate-utility NDCG: 0.8425698860949694
- FOCUS/BROADEN/REFINE proportions: FOCUS=0.11447263387192201, BROADEN=0.454213789251211, REFINE=0.431313576876867
- Teacher-to-predicted transition matrix:
  - FOCUS->FOCUS: 0.03653093496669891
  - FOCUS->BROADEN: 0.07629399736015613
  - FOCUS->REFINE: 0.00043081560918872987
  - BROADEN->FOCUS: 0.06439971572733601
  - BROADEN->BROADEN: 0.3441306161425799
  - BROADEN->REFINE: 0.004722413532645092
  - REFINE->FOCUS: 0.013541983177887086
  - REFINE->BROADEN: 0.03378917574847501
  - REFINE->REFINE: 0.42616034773503325

## Frozen misleading-surface diagnostic

- G0 authority SHA-256: `519b96c0cf525cda32ff54e91792524c77fcc858e3fa33b10ee292d3be8141e6`; strata (('SURFACE_INTERNAL_AGREE', 1), ('SURFACE_INTERNAL_PARTIAL', 114), ('SURFACE_INTERNAL_MISLEADING', 161))
- field_no_surface_misleading: effect -1.180417977104304e-05, 95% CI [-3.145472859875145e-05, 3.686436919438881e-07], domains 3/6
- field_shuffled_surface_misleading: effect -1.0973357028361278e-05, 95% CI [-3.0307127432930103e-05, 3.068018171640835e-06], domains 2/6
- cai_no_surface_misleading: effect -0.0009367797589696085, 95% CI [-0.0019160364715854607, 3.920981062707374e-05], domains 3/6
- cai_shuffled_surface_misleading: effect -2.552665529868171e-06, 95% CI [-0.0008164456643181878, 0.0007990916485116415], domains 2/6
- This frozen subset is diagnostic only and does not alter the G1 gate.

## Stopping

- FIELD: `G1_FIELD_STOPPING_NO_GO`; normalized saving 0.0, 95% CI [0.0, 0.0], domains 0/6; loss ratio 1.4281895236048958; premature rate 0.0; authorized domains 0/6
- CAI: `G1_CAI_STOPPING_NO_GO`; normalized saving 0.0, 95% CI [0.0, 0.0], domains 0/6; loss ratio 2.538297473468669; premature rate 0.0; authorized domains 0/6

All target policy trajectories were frozen before hidden target truth was opened.
Inference uses 100,000 synchronized physical-specimen bootstrap replicates with equal held-out-domain weighting.
