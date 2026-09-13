# W3 seed1 VALID 结果与结论边界

固定有效 W2 的 P_all=MEAN_SC@1750；所有九种方法共用同一权重。下表 A/early A 先在试样内平均 RANDOM repeat，再域内物理试样均值、六域等权；终点 MAE 是物理试样池化，不能混为同一聚合量。J=A+0.25终点误差仅是训练目标，checkpoint 以 A 选择。

| 方法 | A (MPa) | early A | 终点 MAE | 所选 update | 实际更新 |
|---|---:|---:|---:|---:|---:|
| CENTER_FIRST | 48.475341 | 51.820645 | 48.026569 | 0 | 0 |
| GEOMETRY_SPREAD | 47.310645 | 51.879792 | 46.909917 | 0 | 0 |
| LEARNED_STATIC_TRUE | 47.976721 | 52.945606 | 46.993118 | 250 | 750 |
| NO_VLM_SPATIAL_FEEDBACK | 43.597088 | 48.770837 | 42.384681 | 1000 | 1250 |
| RANDOM | 49.594392 | 54.116690 | 46.886383 | 0 | 0 |
| SERPENTINE | 48.911416 | 52.170184 | 48.012802 | 0 | 0 |
| VLM_MEAN_FEEDBACK | 45.262566 | 50.389752 | 43.610874 | 250 | 1250 |
| VLM_SPATIAL_FEEDBACK | 45.110377 | 49.642437 | 44.285799 | 250 | 1250 |
| VLM_SPATIAL_OPEN_LOOP | 46.500981 | 51.047975 | 44.821928 | 750 | 1250 |

正数才表示主方法更好：固定收益 +2.200268 MPa（BEST_NONADAPTIVE=GEOMETRY_SPREAD），反馈收益 +1.390605 MPa，VLM 早期收益 **−0.871600 MPa**，空间相对均值结构 +0.152189 MPa。主方法整体 A 比无 VLM 方法高 1.513289 MPa；不能宣称 VLM 带来整体或早期收益。空间差异很小且并非严格参数匹配，也不是显著性结论。

主 A≤0.98×最佳非自适应 A 且低于开环，原 pilot 条件为 true。这只是后续资源建议；本轮不自动执行 GDFS、seed2/3 或 TEST。无 VLM 对照更好和 VLM 早期负方向均保留，不更换种子、模型或 C0 规则救结果。

有效范围是重复用于开发与 checkpoint 选择的 VALID：50 件、48 capture groups、六域。650 是最终 episode 数（学习5×50，三个确定固定3×50，RANDOM5×50），不是650个独立样本；23个checkpoint也不增加独立N。本轮仅一个策略种子面板，未作正式显著性检验、独立 TEST 确认或多seed鲁棒性证明。RANDOM先平均损失而不集成预测；RMSE平均平方误差后开方，R²逐repeat计算后平均。

`absolute_cai_performance.csv` 给出所有方法五个成本点的 MAE/RMSE/R²；`per_domain_metrics.csv`、`paired_effects.csv`、`per_domain_effects.csv` 给出六域与逐试样方向。成本点取最后一个不超预算的当前预测，不插值、不取历史最好预测。零/完整 P_all 参照引用 `p_all_saved_reference.json` 指向的 W2 既存结果，不将其四路线平均 A 当最佳非自适应门槛。

VLM真实接入既有冻结缓存，205可用+6终止不可用，未新增调用；最高可靠 C0 仅限制第一步。学习策略的调用号在执行时记录，每购买一格后再决策。无反馈/静态只是动作决策权限受限，其评价预测器仍使用实际已测 C-scan。真实像素 rint 图格、float64预算0.25、容限1e-12不变；不代表真实探头控制、扫描时间或设备收益，未学 STOP。

图稿取训练/新评分前已冻结的三个 VALID 案例，21张独立PNG来自保存动作；未测区域遮蔽，曲线含明确评分侧 target。案例含预测较差的 q24-48，没有按效果删除或替换。原图来源不重复上传，保留实际索引和哈希核对。

状态：implementation=EXECUTION_COMPLETE；科学证据=VALID_ONLY_PILOT_NOT_INDEPENDENT_CONFIRMATION；工程状态=ENGINEERING_ACCURACY_CRITERION_NOT_SPECIFIED。旧 new_protocol W2–W4 仍失效，本轮新目录结果不追认旧运行。完整协议/Git交付状态以 REQUIREMENTS_REVIEW.json 和 final_manifest.json 的最终记录为准。
