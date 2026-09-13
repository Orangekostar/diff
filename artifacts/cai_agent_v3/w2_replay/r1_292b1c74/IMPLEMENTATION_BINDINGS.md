# W2正式重放代码绑定

主规范：docs/cai/w2_formal_replay/CODEX_CAI_V3_W2_REPLAY_EXECUTION.md；基点292b1c74。

| ID | 实际symbol | 本轮处理与证据 |
|---|---|---|
| C01 | predictor_training._train_candidate | 复用AdamW/Huber/采样/250评分/4次早停/重载；context仅资源、归档附加、完成复用 |
| C02 | run_predictor_candidates | context隔离RUN并stage A分配；旧默认28100检查保留；新入口不重拟合Ridge |
| C03 | run_oof_reward_predictors / require_predictor_selection_evidence | 新gate、全部归档、fold均显式跟随RUN；A准备失败拒绝B |
| C04 | _cell_costs / build_validation_library / _require_exact_validation | 原生rint分区、float64与1e-12不改；prepare一次逐数组比对，实际VALID身份固定 |
| C05 | CheckpointArchive / inspect_archive | 原保存/finish逻辑复用；附加同次预测与latest optimizer/RNG；重载赢家不覆盖参选预测 |
| C06 | predictor_readiness_gate / choose_common_predictor | 零改动，独立两路线门槛与1e-8平局规则保留 |
| C07 | load_feature_bank | project_root真实工作树；旧六shard只读，TEST标签NaN、不前向 |
| C08 | actor_training._optimizer_update_upper_bound / ReplayContext | 旧Actor零改动；新作用域累计34264与A/B各6000；STARTED预留只结算一次 |
| C09 | scripts/run_cai_agent_v3_w2_replay.py / w2_replay.main | prepare-run/train-candidates/train-oof/summarize；不调用旧prepare/precheck/refresh |
| C10 | models / evaluate_predictor / _constant_metrics | 模型与指标数学不改，兼容details_callback保存同次预测；scale下界1保留 |

新增w2_replay_results.summarize只读保存预测，画VALID/OOF单图，无模型加载或前向。完成job按归档和账目复用；未完成job不自动续训，latest_training_state不是通用续训CLI。
