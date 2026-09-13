# W3实现绑定

规范：docs/cai/w3_valid_pilot/CODEX_CAI_V3_W3_PILOT_EXECUTION.md；主代理自检，未更换RL或网络。

| ID | 实际symbol | 处理 / 证据 |
|---|---|---|
| B01 | w3_pilot.load_inputs/train_pilots；actor_training._load_oof_predictors | 新薄入口显式W2来源、RUN输出；旧run_policy_pilots/precheck不调用，默认历史路径未放开 |
| B02 | require_predictor_selection_evidence(output_dir=W2) | prepare一次验证完整新W2证据，不重新前向44个上游候选；四实际消费权重hash绑定 |
| B03 | seeded_actor/_train_actor；ActorArchive | 所有seed先于初始化；每参选权重和同次动态轨迹保存，完整计划才COMPLETE，实际重载赢家且不重跑VALID |
| B04 | _training_rollout_loss/_predict_by_fold/_sample_specimens | 零改动；按域→试样、排除组的OOF、原cost-to-go、每episode等权 |
| B05 | SpatialCAIActor/MeanFeedbackActor/TrueStaticActor | 网络零改动；实际forward检查两层四头和信息权限；真实静态64共享logits |
| B06 | vlm_first_action_mask | 零改动；真实不可用等reason按原规则，C0只首步 |
| B07 | _legal_action_mask/_evaluate_one | 原float64/rint/1e-12复用；callback记录整数像素和真实前后成本，合成24/99边界通过 |
| B08 | metrics/evaluate_policy/episode_metrics | 原A/early/J/cost-to-go数学复用；RANDOM在试样内先平均损失，终点池化单列 |
| B09 | PilotContext/ActorArchive | W3独立ID/5750/方法cap/40014/21600秒；唯一预留、零增量进度、完成结算；latest optimizer和全部RNG |
| B10 | policy_pilot_gate/w3_results.summarize | 原2%及优于开环规则，只输出建议；入口不暴露扩展/GDFS/TEST |
| B11 | _evaluate_one实际调用点；w3_results.export_figures | 调用计数运行时产生，固定路线无虚构调用号；复用_draw_cells/_measured_image、rint边界显示，旧三个VALID案例先冻结 |
| B12 | tests/test_cai_agent_v3_w3_pilot.py + 九个直接已有测试 | 20项CPU检查，无真实optimizer.step；不运行旧真实优化预检或全库 |

输入证据：INPUT_BINDINGS.json、RUN/protocol_snapshot.json、pilot_authorization.json和case_manifest.csv。结果与绘图只读保存轨迹；无完整源图时只标图稿缺失。
