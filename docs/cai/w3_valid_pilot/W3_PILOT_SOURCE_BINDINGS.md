# W3 seed1：源码依据、复用关系与判断边界

核对日期：2026-09-13。主仓库固定提交：`0e11452ac6590b3b2b694364bd4d1cef7c9315cf`。

## 1. 事实／推断／本轮设计分开

| 类型 | 内容 | 对本轮安排的意义 |
|---|---|---|
| 已读结果事实 | 新W2共同模型和OOF均准备通过，44个实际参选权重及同次预测保存，旧结果没有追认 | 复用新W2，不重训、不再寻找旧37个缺失快照 [R02,R03] |
| 已读代码事实 | W3默认loader/runner还指向new_protocol；W2检查函数已支持output_dir | 只补显式来源和输出参数，不复制新权重回旧目录 [R06,R07,R08] |
| 已读代码事实 | `_train_actor`先构造网络后设置seed；只在内存存best_state | 修seed顺序和候选归档，不更换算法 [R06] |
| 已读代码事实 | 空间Actor、均值Actor、真实静态Actor和C0规则均已实现 | 复用网络、从头训练W3；不再重复实现注意力框架 [R09,R10] |
| 机制推断 | seed设置在初始化后，记录的training_seed不能单独决定初始权重 | 可复现性修复；不是证明旧性能差由此造成 [R06,L01] |
| 设计选择 | 五个seed1、固定W2、5750次新更新、6 GPU小时、无GDFS/TEST | 本次小范围新授权；不是测得最优超参数，也不是已发生的训练 |
| 研究不确定性 | W2预测能力通过准备条件，不保证“新增反馈→选点收益”可学会 | W3需要实际比较固定、开环、无VLM及结构对照，不预设正向 |

历史上传的Hasebe参考追溯和旧BC交接，只能用于说明此前的目标与模块分工。其旧60件、P4克隆、零训练限制、proxy任务不是本次科学规范。这里按用户最新CAI目标及原v3W3规定执行，不混合这些历史实验的最大数字。

## 2. 精确代码映射

以下链接全部固定提交；“新增文件名”仅是本轮设计。

### [R01] 研究分支实际引用

已读取并返回`0e11452ac6590b3b2b694364bd4d1cef7c9315cf`：
https://api.github.com/repos/Orangekostar/diff/git/ref/heads/research/cai-vlm-agent-v3-controlled-reuse

只说明核对时的远端；Codex仍需检查本地是否存在用户后续提交和未提交工作，不能强制回退。

### [R02] 新W2交接及真实结果

https://github.com/Orangekostar/diff/blob/0e11452ac6590b3b2b694364bd4d1cef7c9315cf/artifacts/cai_agent_v3/w2_replay/r1_292b1c74/W2_REPLAY_HANDOFF.md

https://github.com/Orangekostar/diff/blob/0e11452ac6590b3b2b694364bd4d1cef7c9315cf/artifacts/cai_agent_v3/w2_replay/r1_292b1c74/W2_REPLAY_RESULTS_AND_BOUNDARIES.md

MEAN_SC@1750为新P_all；外部VALID50上的四固定路线平均A约48.517，完整输入MAE约41.690、R²约.711。OOF折选1250/750/1000。这些不是新W3结果，也不能将四路线平均A当最佳固定路线A。

### [R03] W2机器状态与额度

https://github.com/Orangekostar/diff/blob/0e11452ac6590b3b2b694364bd4d1cef7c9315cf/results/cai_agent_v3/w2_replay/r1_292b1c74/final_manifest.json

状态为W2_READY_FOR_FUTURE_POLICY_TASK；既有授权仍W2-only。累计使用上界33264、累计上限34264。下一任务必须显式增加W3权限，不能消费W2余额自动运行。

### [R04] W3基础采样、加载字段、模型构造

https://github.com/Orangekostar/diff/blob/0e11452ac6590b3b2b694364bd4d1cef7c9315cf/src/cmc_bbdm/cai_agent_v3/actor_training.py#L1-L250

`_PILOT_SPECS`有五方法；`_VLMFeatures`读取已有区域/置信及可用性；`_sample_specimens`按域再按物理试样；`_actor_forward`对TrueStatic仅输出共享logits和零critic。可直接复用，不改成新P4教师。

### [R05] 真实反馈训练循环

同文件 `_training_rollout_loss`，约L251-L385：
https://github.com/Orangekostar/diff/blob/0e11452ac6590b3b2b694364bd4d1cef7c9315cf/src/cmc_bbdm/cai_agent_v3/actor_training.py#L251-L385

使用float64成本计算合法动作，C0仅约束首步；采样动作后更新已测mask，再调用排除当前试样组的预测器；按episode构建整轨迹cost-to-go，输出Actor/value/entropy损失。真实目标只用于TRAIN损失。

### [R06] 实际评价、OOF加载和seed／归档问题

https://github.com/Orangekostar/diff/blob/0e11452ac6590b3b2b694364bd4d1cef7c9315cf/src/cmc_bbdm/cai_agent_v3/actor_training.py#L387-L710

- `_evaluate_one`按当前观测逐动作argmax，实际保存cells/costs/predictions。
- `evaluate_policy`对RANDOM五repeat，`_domain_equal_episode_score`按试样再域聚合。
- `_load_oof_predictors`写死DATA，未传W2 output_dir。
- `_train_actor`先`_actor`后`manual_seed`，仅内存best_state。

因此修复范围是路径、seed和存储回调。以“换个更大模型”取代这几项修复没有依据。

### [R07] 旧W3编排入口

https://github.com/Orangekostar/diff/blob/0e11452ac6590b3b2b694364bd4d1cef7c9315cf/src/cmc_bbdm/cai_agent_v3/actor_training.py#L736-L1040

旧precheck每结构真实更新一次、写旧目录；run_policy_pilots直接用旧DATA并在各job结束才存单赢家。training_seed为2026091301+固定offset。新薄入口不可仅调用旧默认函数后期待自动读取新W2。

### [R08] W2选择证据已有可复用扩展

https://github.com/Orangekostar/diff/blob/0e11452ac6590b3b2b694364bd4d1cef7c9315cf/src/cmc_bbdm/cai_agent_v3/predictor_training.py#L1266-L1316

`require_predictor_selection_evidence(root,bank,include_oof=True,output_dir=None)`明确允许独立结果目录。它检查完整候选、winner、OOF、固定VALID身份。W3传入W2，不需要再修改已有W2 gate内容，也不需要重新前向其44个候选。

### [R09] 当前已有Actor网络

https://github.com/Orangekostar/diff/blob/0e11452ac6590b3b2b694364bd4d1cef7c9315cf/src/cmc_bbdm/cai_agent_v3/models.py

`SpatialCAIActor`为2层、4头、128维、FFN256、dropout0；C-scan未测值先替换mask，再与表面/坐标/VLM等融合。`MeanFeedbackActor`是现有结构对照，`TrueStaticActor`只有共享64logits。本轮不修改这些结构，也不把W2选MEAN与Actor选MEAN混为一事。

### [R10] VLM首动作规则

https://github.com/Orangekostar/diff/blob/0e11452ac6590b3b2b694364bd4d1cef7c9315cf/src/cmc_bbdm/cai_agent_v3/policy.py

`vlm_first_action_mask`区分不可用、无可靠cue、无中高置信、无可负担候选；第一步取最高可靠置信候选，第二步及以后释放。其作用是检查优先假设，不是CAI信息增益GT。

### [R11] 统一数学与原gate

https://github.com/Orangekostar/diff/blob/0e11452ac6590b3b2b694364bd4d1cef7c9315cf/src/cmc_bbdm/cai_agent_v3/metrics.py

`left_error_area_mpa`为左端常值面积，`prediction_at_budget`不取未来状态，`torch_policy_cost_to_go`带尾段及终点.25。已修实现保留，不再用np.trapezoid。

https://github.com/Orangekostar/diff/blob/0e11452ac6590b3b2b694364bd4d1cef7c9315cf/src/cmc_bbdm/cai_agent_v3/gates.py

`policy_pilot_gate`比较主A≤.98×最佳非自适应A并低于开环A。保持原阈值，本轮只记录建议，不打开后续。

### [R12] W2成功的归档与记账模式

https://github.com/Orangekostar/diff/blob/0e11452ac6590b3b2b694364bd4d1cef7c9315cf/src/cmc_bbdm/cai_agent_v3/w2_replay.py#L250-L453

`ReplayContext`实现STARTED/reservation、PROGRESS不双计、完成结算、同次预测保存、latest optimizer/RNG和首250检查。类内有W2硬绑定，适合借用模式或提取最小工具，不适合直接原样实例化做W3。

https://github.com/Orangekostar/diff/blob/0e11452ac6590b3b2b694364bd4d1cef7c9315cf/src/cmc_bbdm/cai_agent_v3/checkpoint_selection.py

已有原子保存、完整计划/合法早停检查；W2身份绑定的是固定前缀。W3的固定身份应是评价环境，路线作为各checkpoint产物，不能强制路线一致。

### [R13] 当前VLM缓存真实覆盖

https://github.com/Orangekostar/diff/blob/0e11452ac6590b3b2b694364bd4d1cef7c9315cf/results/cai_agent_v3/new_protocol/vlm_manifest_fit.json

211条fit记录中205可用、6不可用；模型revision为`cc594898137f460bfe9f0759e9844b3ce807cfb5`。cache和actor_features有实际路径与SHA，保留失败项，不新增调用或筛掉困难试样。

### [R14] 已有图稿薄函数

https://github.com/Orangekostar/diff/blob/0e11452ac6590b3b2b694364bd4d1cef7c9315cf/src/cmc_bbdm/cai_agent_v3/diagnostics.py#L1-L175

`_draw_cells/_measured_image`可复用。`build_action_trace`默认旧DATA，部分actor_call_index由后验动作序列构造；本轮真实调用号与决定必须在执行时记录，不能把后验重建计数称为真实日志。

### [R15] 冻结数据和特征载入

https://github.com/Orangekostar/diff/blob/0e11452ac6590b3b2b694364bd4d1cef7c9315cf/src/cmc_bbdm/cai_agent_v3/feature_bank.py

本次另读取实际输入复用manifest：
https://github.com/Orangekostar/diff/blob/0e11452ac6590b3b2b694364bd4d1cef7c9315cf/results/cai_agent_v3/w2_replay/r1_292b1c74/input_reuse_manifest.json

其明确列出TRAIN161/152组、VALID50/48组、TEST65/59组及六份shard路径。W2已核对六shard与既有split复用；本轮以其输入复用manifest为权威。main root仍是真实仓库根，模型来源与数据来源分开。不重新编码，不把全图diagnostic token作为Actor输入。

### [R16] 原v3 W3科学条款

本包basis/中提供上传的原文件逐字副本：
`CODEX_CAI_AGENT_V3_EXECUTION.md`第3节、6节、8节；`CODEX_CAI_AGENT_V3_REVIEW.md`对应验收。

原第6节规定五个seed1、C0、2层空间Actor、真实静态、共享P_all、VALID gate。本包新范围收窄为W3-only，覆盖其GDFS/多seed/TEST自动继续许可。本包资源另行明确，不从历史文档推断授权。

## 3. 外部依据仅用于必要的实现说明

[L01] PyTorch官方 Reproducibility / RNG文档：
https://docs.pytorch.org/docs/stable/notes/randomness.html
https://docs.pytorch.org/docs/main/random.html

已检索：固定seed可控制同环境随机序列，但不同版本/平台不保证完全复现；`fork_rng`可保存恢复RNG。此处只支持seed顺序和审查不污染训练随机流的说明，不要求升级依赖或切换计算后端。

本轮不新增外部算法。既有GDFS与Set Transformer为以后有条件对照；无需重新安装或重新读完整外部仓库作为W3前置工作。

## 4. 仍未验证的内容

- 本次聊天未运行用户服务器上的W3训练或CI，未验证模型会优于固定路线。
- 本轮拟新增CLI、context、Actor归档与授权并未在基点中存在；Codex应实际实现并回报真实路径。
- 6h与5750是有界执行设计，不是收敛时间预测；不由W2耗时推断W3耗时。
- 预测器、VLM、反馈三者贡献必须靠同条件比较区分。代码路径有输入不等于该输入已产生有用收益。
- 历史和当前VALID都参与过开发，不能称未经观察的确认集；未启动TEST不能写成TEST成功。
