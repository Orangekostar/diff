# W2正式重放：源码依据、需求绑定与未验证边界

核对日期：2026-09-13。仓库固定提交：`292b1c74b27bb344af370f6738ac9c493cfd137d`。

本文件仅列本轮实际读取的源码/规范，不把计划中的接口说成已实现。分级：[R]是仓库事实，[P]是原任务定义，[L]是官方资料，[D]是本轮提出的新执行安排。

## 1. 本轮事实与判断

| 判断 | 直接依据 | 对新任务的限制 |
|---|---|---|
| 缺失的37份权重不能从6个赢家或scalar score重构 | R01恢复交接，43个时点只有6个真实快照 | 不再查全盘、不反复给旧赢家评分；从头重训最多6个模型，不是训练37个模型 |
| 已有训练循环已包含逐时点归档，不需要重造算法 | R03 `_train_candidate`：每250步评价后record，finish后实际torch.load赢家 | 复用循环；只补输出/授权/细节导出或最小容错 |
| “拿现有train-predictors直接跑”仍不合适 | R02：旧结果根硬绑定；`28100-used <12000`立即拒绝；默认重新写Ridge和VALID库 | 必须显式传本轮run root和新授权；不能只在MD里说预算已增加 |
| OOF仍默认连接旧gate和旧归档 | R02的run_oof_reward_predictors及require_predictor_selection_evidence | 新A、B和依赖核验必须用同一新结果根；feature读取仍用真实project_root |
| exact-cost已修，不应重做或放宽容差 | R04/R05：float64原生费用、固定前缀、rint边界、1e−12合法性 | float32可用于神经输入，不能用于是否可采下一格的判定 |
| 保存完整“选择历史”不代表可中断续训 | R05的payload只有state_dict及manifest；R01也明确没有optimizer/RNG续训CLI | 本轮保存一份轻量latest training state为容错；不冒称已有完整恢复功能 |
| 科学门槛不因补做而改变 | R06条件函数，P01第5.4/5.5节 | 不能见MEAN_SC历史更好就跳过其他候选，不能替换失败折或调门槛 |
| 简单基线和特征应尽量直接复用 | R07固定六shard加载；R01最小恢复清单 | 不回到原始图重复编码；部分Ridge精确库不同才一次有限CPU重算 |
| 类别预算也需要明确调整 | R01额度表与R08累计求和规则 | 本轮12,000不是“原预算还够”；明确转4,586并净增6,164，不重置历史 |
| 测试通过只证明直接检查 | R11记录28项定向检查通过，仍保持历史结果失效 | 前置review只查新增接线；训练结束给真实准备条件，不以测试数当性能 |

### 已确认的一处算术错误

R01写了“28,100＋6,164＝34,464”。实际为 **34,264**。
本包使用正确值，并在新授权中保留更正依据；旧恢复交接不覆盖。

### 一处明确的汇总口径

R04 `evaluate_predictor` 的 `valid_area_mpa` 先平均四路线、再平均域，六域等权。full/zero/center/geometry MAE/MSE则对VALID物理试样池化；R04 `_constant_metrics`的参照同样池化。本轮明确继承，不将它们混称为全部等域指标，也不默改生产数学。

R06的路线条件是CENTER和GEOMETRY各自低于zero，而不是二者取平均后可以一条好一条坏。本包明确其含义，防止口头“路线平均”被当成放宽条件。

## 2. 当前实际源码入口

所有链接固定在同一提交，数字范围为本次读取的文件内范围；最终Codex仍需对它实际运行的工作树核对。

### R01 — 恢复结论与资源

文件：`artifacts/cai_agent_v3/W2_RECOVERY_HANDOFF.md`

`https://github.com/Orangekostar/diff/blob/292b1c74b27bb344af370f6738ac9c493cfd137d/artifacts/cai_agent_v3/W2_RECOVERY_HANDOFF.md`

已读全文。包括43/6/37、真实ledger路径、6模型恢复、保守消耗22,264、剩余5,836、类别余额、GPU时间缺测、旧*.pt忽略及交付边界。

### R02 — W2编排、输出和依赖

文件：`src/cmc_bbdm/cai_agent_v3/predictor_training.py`，主要读取1–1300行。

`https://github.com/Orangekostar/diff/blob/292b1c74b27bb344af370f6738ac9c493cfd137d/src/cmc_bbdm/cai_agent_v3/predictor_training.py`

重点：`run_predictor_candidates`、`run_oof_reward_predictors`、`require_predictor_selection_evidence`、`_manifest_selection_verified`、`rescore_predictor_archive`、`refresh_cost_precision_evaluations`入口。默认RUN硬绑定旧new_protocol；OOF读取selected_p_all，不应单靠替换外层路径字符串。

### R03 — 实际训练循环

同R02文件，约560–715行的 `_train_candidate`。

已确认：构造种子固定；采样仅TRAIN/fit；AdamW和Huber；归档前核对精确VALID；每250步保存再排名；4次不改进终止；最终从真实权重载入，保存manifest/parameter_count/elapsed_seconds。没有自动恢复optimizer/RNG分支。

### R04 — VALID库和指标

同R02文件，约1–380行的 `_cell_costs`、`build_prefix_library`、`build_validation_library`、`_sample_training_masks`、`_predict_batches`、`_constant_metrics`、`evaluate_predictor`；约1120行之后的 `_require_exact_validation`。

成本矩阵float64；已修正库以1e−12比较B=.25；模型输入cost转float32；full64独立前向；A调用 `left_error_area_mpa` 而非梯形积分。输入/路线必须在整个选模过程固定。

### R05 — 候选归档与重评分

文件：`src/cmc_bbdm/cai_agent_v3/checkpoint_selection.py`，已读全文。

`https://github.com/Orangekostar/diff/blob/292b1c74b27bb344af370f6738ac9c493cfd137d/src/cmc_bbdm/cai_agent_v3/checkpoint_selection.py`

符号：`COST_DEFINITION`、`validation_identity`、`CheckpointArchive.record/finish`、`_validate_schedule`、`inspect_archive`、`rescore_archive`。已有目录拒绝覆盖；选择证据不完整/VALID改变拒绝；全时点才构成完整重排名。

### R06 — 固定准备条件

文件：`src/cmc_bbdm/cai_agent_v3/gates.py`，已读全文。

`https://github.com/Orangekostar/diff/blob/292b1c74b27bb344af370f6738ac9c493cfd137d/src/cmc_bbdm/cai_agent_v3/gates.py`

符号：`predictor_readiness_gate`、`choose_common_predictor`。有限数值检查＋常量比较＋2%完整输入改善＋两条前缀各低于zero；候选A差≤1e−8时按参数数/名称。不能从“可恢复”推导“科学支持”。

### R07 — 特征复用

文件：`src/cmc_bbdm/cai_agent_v3/feature_bank.py`，已读1–230行。

`https://github.com/Orangekostar/diff/blob/292b1c74b27bb344af370f6738ac9c493cfd137d/src/cmc_bbdm/cai_agent_v3/feature_bank.py`

`load_feature_bank(project_root=...)`从 `results/cai_agent_v3/new_protocol/feature_bank_index.csv`读取相对shard路径。`V3FeatureBank`保证TEST targets为NaN。不能把输出文件夹作为project_root绕过真实数据路径。

### R08 — 资源上界

文件：`src/cmc_bbdm/cai_agent_v3/actor_training.py`，已读1–210行。

`https://github.com/Orangekostar/diff/blob/292b1c74b27bb344af370f6738ac9c493cfd137d/src/cmc_bbdm/cai_agent_v3/actor_training.py`

只复用 `_optimizer_update_upper_bound`：历史actual求和，未完成run保留reservation，旧unknown_upper_bound仍计入。不是借此启动Actor。本轮授权边界需新增薄适配，不允许全局提高该文件的Actor预算常量。

### R09 — 实际命令

文件：`src/cmc_bbdm/cai_agent_v3/cli.py` 及 `scripts/run_cai_agent_v3.py`，已读全文。

`https://github.com/Orangekostar/diff/blob/292b1c74b27bb344af370f6738ac9c493cfd137d/src/cmc_bbdm/cai_agent_v3/cli.py`

`https://github.com/Orangekostar/diff/blob/292b1c74b27bb344af370f6738ac9c493cfd137d/scripts/run_cai_agent_v3.py`

旧命令存在 `train-predictors/train-oof/refresh-cost-evaluation`；没有新授权或run root参数。`prepare`会执行旧W0/W1。新薄入口命令在本包中明确为拟开发，不能直接抄进终端假装已实现。

### R10 — 模型固定

文件：`src/cmc_bbdm/cai_agent_v3/models.py`，已读1–220行。

`https://github.com/Orangekostar/diff/blob/292b1c74b27bb344af370f6738ac9c493cfd137d/src/cmc_bbdm/cai_agent_v3/models.py`

`MeanSCPredictor`64宽；`SpatialPredictor`128宽/两层/四头/FFN256/dropout.1，`use_surface`区分SC/C。512维独立cell输入，显式measured遮挡；不是原始波形模型。

### R11 — 已做的必要检查

文件：`artifacts/cai_agent_v3/W2_RECOVERY_VALIDATION.md`，已读全文。

`https://github.com/Orangekostar/diff/blob/292b1c74b27bb344af370f6738ac9c493cfd137d/artifacts/cai_agent_v3/W2_RECOVERY_VALIDATION.md`

报告CPU定向28项通过、Ruff/diff通过、真实训练0；包含删除非赢家权重使重评失败、VALID变化拒绝、早停完整性、中断reservation核算。这里只核对了提交的记录，没有代替用户服务器重新执行这些命令。

### R12 — 分支身份

`https://api.github.com/repos/Orangekostar/diff/git/ref/heads/research/cai-vlm-agent-v3-controlled-reuse`

本次读取返回292b1c74b27bb344af370f6738ac9c493cfd137d。它是可变引用；以上源码内容链接均已固定提交。

## 3. 规范与有限外部依据

### P01 — 原科学规范

`basis/CODEX_CAI_AGENT_V3_EXECUTION.md`第3、5、9、10节；配套原review与metric/golden文件逐字保留。本轮明确替代其资源上限和后续阶段授权，不替代W2科学定义。

旧2026-09-09的BC交接和作者ROI追溯文件用于解释研究历史，不是本轮执行依据，不从其“冻结0训练”状态否决现在的W2任务。

### L01 — PyTorch官方保存/恢复文档

`https://docs.pytorch.org/tutorials/beginner/saving_loading_models#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training`

官方指出：要恢复训练，除model state_dict外还应保存optimizer state_dict等训练状态；推理时使用eval模式。该资料支持“选择用快照”和“训练恢复状态”分开，不能证明本仓库已经实现了可恢复训练。RNG、stale计数和本轮身份字段是根据本训练循环提出的具体容错设计。

### L02 — Git官方git-add

`https://git-scm.com/docs/git-add`

默认忽略的文件不会被普通git add纳入；可对明确路径使用-f。用于防止只推了manifest没推模型，不授权force push或全仓库强制添加。

## 4. 本轮新增设计与未知边界

| 项目 | 性质 |
|---|---|
| 只正式重放W2-A和条件W2-B | 当前用户任务的实施范围；不启动W3–W5 |
| `RUN=r1_292b1c74`、新薄CLI、授权JSON | 新接口/目录设计，尚未在用户仓库实现 |
| 12,000次本轮上限，历史累计34,264 | 新额度分配建议，随用户将本任务交给Codex而生效；不是已发生训练 |
| 新增6 GPU小时 | 新的有界执行窗口；不是历史剩余时长，也不是工期保证 |
| 一份latest optimizer/RNG状态 | 轻量防返工措施；不要求大型恢复平台或容灾测试 |
| 逐checkpoint状态预测在既有前向中导出 | 新输出要求，用于减少未来重复重评分 |
| 新W2会选哪种结构、MAE能提高多少 | 未知，不能根据旧MEAN_SC诊断预设赢家 |
| 新W2通过后旧W3/GDFS是否可用 | 不自动可用；历史仍失效，下游重放需独立授权 |

本次完成的是源码核对与任务设计；没有连接用户服务器、运行其训练、生成新科学结果或推送研究仓库。
