# Codex执行规范：固定有效W2的VLM引导CAI Agent——W3 seed1受控实验

日期：2026-09-13。
仓库：`Orangekostar/diff`；接续分支：`research/cai-vlm-agent-v3-controlled-reuse`。
已核对代码／证据基点：`0e11452ac6590b3b2b694364bd4d1cef7c9315cf`。
本轮ID：`W3_VALID_PILOT_R1_0e11452a`。

## 0. 这是下一项实际任务，不是又一次零训练准备

用户将本文件和本包交给Codex并要求执行时，授权下述 **W3 seed1训练、VALID比较、交接和GitHub推送**。来源事实见配套 `W3_PILOT_SOURCE_BINDINGS.md` 的[Rxx]；新增实施安排标为[D]。本包尚未在聊天端执行研究训练。

任务目标：
> 冻结本次有效W2的共同预测器与三折回报模型；VLM提出起始优先区域，学习Actor决定首个具体格子，之后每获取一格真实C-scan图像就重新决策；在同预测器、同原生像素成本下，比较VLM、反馈和空间结构对CAI预测效率的贡献。

执行顺序：任务/输入绑定 → 三处接线修正与有限预检 → 固定路线评价 → 五个seed1学习策略 → 只从已存轨迹汇总VALID比较和图稿 → 交接并实际push。

### 0.1 范围与优先级

用户本轮明确指令 > 本文件及 `W3_PILOT_AUTHORIZATION.json` 的本轮权限 > 原v3相关科学定义（本包basis）> 实际配置/测试/历史交接。

- 本轮明确覆盖W2任务中“Actor禁止、下游待授权”的限制，仅开放本文列出的W3任务。
- **不重训W2，不重做W0/W1，不重新配准/编码，不调用新VLM，不执行GDFS、STOP、seed2/3、TEST感知/预测/标签评分。**即便pilot达到原扩展条件，也只报告是否值得后续授权。
- 原v3曾预授权GDFS/条件TEST的文字，不构成本次继续执行W4/W5的许可。
- 当前任务是CAI MPa信息采集，不回到P4行为克隆、LOCATE/CHARACTERIZE proxy或专家标注路线。历史附件用于背景，不能覆盖本轮授权。
- 当前只有一个策略种子面板、反复用于开发的VALID；不得宣布独立确认、工程达标或鲁棒三种子优势。
- 已完成同身份job直接复用。不得删除不利结果、更换seed或增加另一套算法“救成功”。

### 0.2 保留用户要求的模块职责

| 模块 | 本轮作用 | 是否训练 |
|---|---|---|
| 已有冻结Qwen VLM缓存 | 表面优先区域与置信；引导首动作 | 否；新增调用0 |
| 已有冻结ResNet图格特征 | 表面与已测C-scan视觉证据 | 否；新增编码0 |
| 有效W2 `P_all` | 对所有VALID策略的累计观测预测CAI | 否；所有策略共用同一权重 |
| 有效W2三个OOF模型 | TRAIN试样按排除其capture组的折提供预测和回报 | 否 |
| 五个seed1 Actor | 根据各自允许的信息选择位置 | 是；仅本文固定面板 |

MEAN_SC是已选预测器；主Actor仍是SPATIAL。不能以预测器均值结构获胜为由删掉空间Actor，也不能强行把预测器换回Transformer。[R02,R03,R09]

## 1. 输入根、输出根与接续方式

### 1.1 三类根目录必须分开

相对于真实仓库根 `ROOT`：

```text
DATA = results/cai_agent_v3/new_protocol/
W2   = results/cai_agent_v3/w2_replay/r1_292b1c74/
RUN  = results/cai_agent_v3/w3_pilot/r1_0e11452a/
ART  = artifacts/cai_agent_v3/w3_pilot/r1_0e11452a/
LEDGER = results/cai_agent_v3/compute_ledger.jsonl
```

- DATA只读：名单、split、六份特征shard、VLM缓存、原生尺寸等。
- W2只读：新共同预测器、OOF模型/折表、完整选模证据、final_manifest及VALID结果。
- RUN只写本轮新Actor、参选轨迹、比较表和图稿；ART写本轮规范绑定、review、交接。
- 不把W2或RUN假扮成project_root，不把新模型复制回旧DATA去“自动解锁”。
- 旧W2–W4失效结果和旧37份缺失权重的结论保留。本轮不再查找这些权重。[R02,R06,R08]

### 1.2 明确绑定，不回退或重开已完成任务

检查当前工作树、分支、HEAD、一次fetch后指定远端引用。若已有后续提交/未提交工作，保留并说明与基点关系；不reset、不覆盖、不擅自切换其他实验。

把本包复制到 `docs/cai/w3_valid_pilot/` 或实际可用的同义目录；在现有 `docs/stagetask/TASK.md` 写入本文件绝对路径、来源基点、任务ID、阶段、额度和完成标准。保留STATE历史，但解除旧“未授权W3/未绑定任务”的重复阻塞。不要另造通用任务管理平台。

必读输入：
- `artifacts/cai_agent_v3/w2_replay/r1_292b1c74/{W2_REPLAY_HANDOFF.md,W2_REPLAY_RESULTS_AND_BOUNDARIES.md,RESULT_POINTER.json}`；
- W2下 `final_manifest.json`、`predictor_gate.json`、`oof_readiness.json`、`oof_fold_manifest.csv`、`input_reuse_manifest.json`；
- DATA下 `feature_bank_index.csv`、`feature_bank_manifest.json`、`split_manifest.csv`、`vlm_manifest_fit.json`、`vlm_actor_features_fit.csv`；
- 实际全局LEDGER。

已核对：276件、259组；TRAIN161/152组，VALID50/48组，TEST65/59组。W2选中MEAN_SC@1750，三个OOF选1250/750/1000。把它们作为应核对的身份，不把聊天中的舍入分数硬编码为计算输出。

VLM fit为211条记录、205可用、6条明确不可用；复用原记录及fallback，不删除这6件、不改写为no_reliable_cue、不重采样。真正文件缺失与已有的终止不可用记录不同；缺失应定位一次并据实交付。[R13]

## 2. 定向源码阅读与具体任务绑定

文件前缀默认 `src/cmc_bbdm/cai_agent_v3/`。生成至多两页 `ART/IMPLEMENTATION_BINDINGS.md`，每行包含原需求、实际函数、修改点、验收证据。

| ID | 当前真实接口 | 已核对问题／可复用部分 | 本轮动作 |
|---|---|---|---|
| B01 | `actor_training.py::_load_oof_predictors/run_policy_pilots/precheck_policy_training` | 默认读写旧DATA的gate/model | 新增显式W2来源与RUN输出；不调用旧默认一键入口 |
| B02 | `predictor_training.py::require_predictor_selection_evidence(...,output_dir=...)` | 已支持独立目录 | 明确传入W2；不要重复造一套W2选择审计 |
| B03 | `actor_training.py::_train_actor` | `_actor()`在manual_seed之前；只留best_state | 先seed再构造；增加实际参选权重、同次轨迹、latest状态保存 |
| B04 | `_training_rollout_loss/_predict_by_fold/_sample_specimens` | 已有按域/物理试样采样、OOF路由、整轨迹代价与分样本终止 | 复用，不改为P4监督，不均匀抽折导致样本权重不同 |
| B05 | `models.py::SpatialCAIActor/MeanFeedbackActor/TrueStaticActor` | 空间、均值、真实固定排序均已实现 | 复用现有结构和开关，从头初始化新Actor |
| B06 | `policy.py::vlm_first_action_mask` | 最高可靠C0仅约束首动作、之后放开 | 复用；保留reason和真实不可用分支 |
| B07 | `_legal_action_mask/_evaluate_one`及`NativeCellGrid` | 已修float64合法性、原生rint格与严格B | 复用，前向输入可float32，硬预算不得round-trip回float32 |
| B08 | `metrics.py`、`evaluate_policy/_domain_equal_episode_score` | 已有左阶梯、cost-to-go、重复内均值后六域等权 | 训练/选模/表格共用；导出同次评价轨迹 |
| B09 | `w2_replay.py::ReplayContext`及`checkpoint_selection.py` | 原子保存、reservation结算、首250核验可借鉴 | 薄Actor适配；不要直接实例化写死W2 ID/2000额度的context |
| B10 | `gates.py::policy_pilot_gate` | 已有2%且优于开环的VALID规则 | 保留，只产出后续建议，不执行扩展 |
| B11 | `diagnostics.py::_draw_cells/_measured_image/build_action_trace` | 可复用绘图；旧导出绑定DATA且部分调用号是事后重建 | 复用绘图，实际Actor调用/动作日志由执行时产生；新图只读RUN |
| B12 | `tests/test_cai_agent_v3*.py`与basis数值文件 | 已有大量直接检查 | 只跑受影响项，补本包review的少量反例，不全库回归 |

具体证据见[R04–R12,R14–R16]。

### 2.1 推荐实施方式 [D]

给原W3编排/训练增加兼容可选 `predictor_root/output_dir/run_context`，默认历史行为保持。新增薄脚本 **拟命名** `scripts/run_cai_agent_v3_w3_pilot.py`，只开放：

```text
prepare-run       # 绑定数据、W2、授权、评价格式和模型定义；0更新
run-fixed         # 用新P_all评价四种固定方法；0更新
train-pilots      # 本文五个seed1 job，完整或阶段/资源终止
summarize         # 只读取保存轨迹/权重manifest做数值汇总；0前向、0更新
export-figures    # 只读取保存轨迹及至多三件原图；0模型前向
```

这是待实现入口，不假称当前存在。不要让CLI提供或隐式调用W2训练、GDFS、TEST、扩种子；不要创建一个同训练逻辑重复的平行框架。

`require_predictor_selection_evidence(ROOT, bank, include_oof=True, output_dir=W2)`可用于一次启动检查。检查manifest、四个实际消费的权重及上游归档，不重放44个W2模型前向；训练每步不重新hash特征或上游归档。[R08]

## 3. 固定输入和训练／评价权限

### 3.1 新有效预测器绑定

从W2 gate/OOF manifest读取确切路径和hash，不凭目录里最新mtime挑模型：
- `P_all`是新W2选择的MEAN_SC@1750；
- TRAIN的每个试样用W2折表中的query fold，加载未拟合该组的相应回报模型；
- VALID所有策略用同一个P_all；
- 四模型 `eval()`、`requires_grad_(False)`，不加入Actor optimizer。预测器当前输出可detach提供反馈Actor；无反馈策略必须屏蔽它。

旧OOF固定路线状态不是动作GT。训练中Agent产生新可见状态，才按对应OOF预测器计算新预测与TRAIN任务损失。未知位置可留在环境cache，但在任何模型全局运算之前必须mask/gather；不得输入全图embedding、真实CAI、误差或oracle下一动作。

读取feature bank时允许存在无标签TEST缓存，但本轮所有前向、采样、训练、评价和图稿仅使用TRAIN/VALID。不得连接TEST标签，不以TEST可视化解释模型。

### 3.2 VLM和采集定义保持

- 64个原生图格，每动作揭示一个此前未测格的完整自有像素；预算B=.25，早期B=.0625。
- rint边界与float64硬成本通路、1e−12已有容限不改；记录整数像素累计和归一化成本。
- 初始无C-scan。VLM最高medium/high候选并集只约束第一步，由Actor在可负担候选中选具体cell；无可靠cue/不可用/不可负担分支分别记录。
- 第二步起解除C0，全部未测且可负担动作开放；不额外强制扩散或留在中心，不把C0改为软先验。
- 每步先用已获得信息预测和选动作，再揭示图格；不能先算完整路线、不免费warm-start。
- 只按预算耗尽结束；不是学会STOP，不宣称原始探头单点控制或实际设备时间收益。

## 4. 唯一学习面板与随机性

| 方法 | 学习角色 | seed_panel | training_seed | 最大更新 |
|---|---|---:|---:|---:|
| `VLM_SPATIAL_FEEDBACK` | 主方法：表面+真实VLM+新增内部反馈 | 1 | 2026091301 | 1250 |
| `NO_VLM_SPATIAL_FEEDBACK` | 同表面CNN/内部反馈，去VLM和C0 | 1 | 2026091302 | 1250 |
| `VLM_SPATIAL_OPEN_LOOP` | 同表面/VLM/C0，不读内部内容或CAI预测决定动作 | 1 | 2026091303 | 1250 |
| `LEARNED_STATIC_TRUE` | 仅64个共享位置logits，不读任何图像/预测 | 1 | 2026091304 | 750 |
| `VLM_MEAN_FEEDBACK` | 同主信息/目标的均值结构诊断 | 1 | 2026091305 | 1250 |

这些seed值来自既有W3编排，**修正为模型构造前设置**。不因为数值不好换seed，也不声称五方法使用相同初始化或严格参数量匹配。[R06,R07]

四固定方法：CENTER_FIRST、GEOMETRY_SPREAD、SERPENTINE、RANDOM。RANDOM保留已有五个repeat seed `2026091250..2026091254`，不是五个独立物理队列。每个方法按自己的实际原生成本执行，超额格子按既有规则跳过。完成训练前可先将这四方法一次性评分并保存，后续直接复用。

### 4.1 随机性和模型模式 [D：实现修复]

在`_actor()`、任何随机层初始化之前，设置Python／实际使用的NumPy RNG、Torch CPU和当前可见CUDA RNG。固定本轮运行环境并记录版本；不升级PyTorch、不承诺跨设备/跨版本位级一致。

- 每个job独立初始化，不受前一个job消费随机数影响；记录initial_state_dict_sha256。
- 读checkpoint、做首250审查或可视化不得额外消耗训练RNG。需要构造临时模型时使用保存恢复RNG或`fork_rng`，也可只加载tensor比较而不构造模型。
- Actor训练模式明确，VALID为eval且使用argmax；返回训练后恢复train。当前空间Actor dropout=0，勿误搬W2预测器的dropout=.1到Actor。
- 验收用同seed在不同构造次序下的初始权重一致性，不要求不同seed的全零TrueStatic初始权重不同。[R09,L01]

### 4.2 优化目标与超参数保持

对轨迹状态 `(c_t,p_t)` 和真实TRAIN标签y：

```text
e_t = abs(p_t - y)
A = [sum_t (c_(t+1)-c_t) e_t + (B-c_T)e_T] / B
J = A + 0.25 e_T
G_t = sum_{j=t}^{T-1} [(c_(j+1)-c_j)e_j/B] + (B-c_T)e_T/B + 0.25e_T
```

按全TRAIN的既有target_scale缩放回报。已有cost-to-go最小化：

```text
L_actor = mean_episodes sum_steps(log_prob * detach(G_scaled - value))
L_value = mean_episodes mean_valid_steps((value - detach(G_scaled))**2)
L = L_actor + 0.5 L_value - beta * mean_episodes mean_valid_steps(entropy)
```

TrueStatic保留已有zero baseline、不训练value头；其余结构保留已有value头，不借机更换RL方法。gamma=1；batch16，先均匀选域再均匀选物理试样；AdamW lr3e−4、wd1e−4、clip1；beta从.01按每job已定上限线性降到0。

只复用`_training_rollout_loss`和`metrics.py`，不再写第二套代价。不用动作后误差覆盖动作前成本、不用历史最好预测、不用固定路线的真实最优动作作标签。

某个batch成员装不下下一格时只终止该成员；其他成员继续。每个episode等权，不按动作数量增加独立样本数。静态与开环的预测器照常读实际已测内容，屏蔽的是动作决策权限。[R04,R05,R07,R11]

## 5. Actor选模和保存：不重复W2丢失候选问题

### 5.1 固定的是评价环境，不是Actor路线

每250更新评价全部VALID50，最大更新处也评价；patience4，改善阈值1e−12。选最小VALID六域等权A的checkpoint，不按早期增益或终点MAE选。平局按首次保留，不额外试checkpoint/seed。

五job的最大正常参选点数为5+5+5+3+5=23；**完整是指所有实际参选点**，不是强行凑23，也不是沿用W2的43/44点计数。

W2固定VALID前缀不能用作学习Actor的执行路线。Actor每个checkpoint会产生不同路径；共同不变的评估环境身份应包含：
- VALID样本顺序、组/域、真实标签来源与当前VALID数组身份；
- 新P_all及OOF来源、图格特征/已知几何、VLM缓存身份；
- B/成本/左阶梯公式、执行模式/合法性/C0规则、固定方法和随机种子面板；
- 实际执行代码SHA、方法开关/结构、training_seed。

每个checkpoint自己的动作、成本、预测序列另存并关联其hash；**不得把所有checkpoint路线要求为一致**，也不得让实现者的自写配置反过来覆盖本文。[R06,R12]

### 5.2 必须落盘的内容

对每个实际参选时点，原子保存：
1. Actor权重、method、seed、update、结构与input flags；
2. **同次VALID前向的完整episode状态预测与实际动作序列**，含零状态和末状态；
3. 根据这些轨迹计算的A、early A、J、终点误差及评价环境身份；
4. 追加选择清单，非赢家不删除。

建议RUN结构：

```text
models/selection_history/<method>_seed1/selection.json
models/selection_history/<method>_seed1/update_000250.pt
candidate_episodes/<method>_seed1/update_000250.csv.gz
models/selection_history/<method>_seed1/latest_training_state.pt
models/actor_<method>_seed1.pt
```

latest保存model/optimizer、NumPy Generator/Torch CPU及可见CUDA RNG、update/best/stale、训练输入身份和runID。它是容错快照，不等于已经验证自动续训。无完整状态不得偷偷从旧赢家续训；中断后优先保留并报告，任何恢复消耗仍受本文总量约束。

每个job首个真实250更新结束，立即检查权重、轨迹、状态序列、模型hash和ledger绑定，然后继续同一个job；不要为审查额外再优化一轮或再前向全部VALID。

完成/合法早停时从实际归档重载赢家，并直接取赢家对应的保存轨迹，不让“重载后预测”覆盖最后时点轨迹。只有计划完整才selection COMPLETE，异常中断保持INCOMPLETE；summary不可把它改成完成。

**复用W2原子保存/记账模式，而非把W2 `CheckpointArchive`的固定前缀identity和model_name强塞给Actor。**只做薄适配，不重构整套归档框架。

## 6. 结果计算：只回答VALID pilot问题

### 6.1 从真实保存轨迹计算

每条轨迹至少保留：specimen_key、domain/group（仅评分侧）、method、seed/repeat、checkpoint、cells、整数新增像素、float64 costs、predictions。`len(costs)=len(predictions)=len(cells)+1`。

- A用所有真实动作的左阶梯积分（含尾段），六域等权：随机repeat先在试样内平均误差/面积，再域内物理试样均值，再六域均值。
- early A截取0..0.0625；不是重新跑一条早期优化路线。
- 成本点`0,.0625,.125,.1875,.25`取最后一个不超该点的当前预测。不得线性插值，不用未来状态。
- 终点MAE为物理试样池化；随机先算各repeat损失再平均，RMSE先平均平方误差再开方；R²逐repeat算再报告均值，不先平均预测造无成本集成。
- 同时保存域等权与池化的清晰字段，不拿W2四固定路线平均A=48.517当“最佳固定A”，不把几何分散终点MAE当面积。
- `P_all`零/完整参照优先引用W2已保存数据及来源；不重跑完整W2评估。选定checkpoint下的必要局部预测属于本次真实策略运行。

最终数据应包括五个学习方法×50件，以及三个确定性固定方法×50件、RANDOM五repeat×50件；正常完成是650条最终episode，不把650当独立N。个别真实错误不能以删样本掩盖，需标运行不完整。

### 6.2 比较与继续建议

从四固定方法＋LEARNED_STATIC_TRUE按VALID A最低（平局名称排序）选BEST_NONADAPTIVE，保留所有方法表。比较：

```text
固定收益 Δ_fixed = A(best_nonadaptive) - A(main)
反馈收益 Δ_feedback = A(open_loop) - A(main)
VLM早期 Δ_vlm = early_A(no_vlm) - early_A(main)
空间结构 Δ_spatial = A(mean_actor) - A(main)
```

正数才是主方法更好；结构比较不声称参数量匹配，VLM早期比较不严格隔离首动作本身。

继承pilot资源规则：主A≤.98×最佳非自适应A，且主A<开环A，才记录`pilot_expansion_condition_met=true`。VLM早期方向独立报告，不以它为由追加seed或修改C0。此规则只给下一轮建议，**无论true/false本轮都到此结束**。

本轮不做正式三项显著性检验、不把用于选checkpoint的VALID区间当独立确认。输出逐样本配对差、六域方向和绝对误差即可。未来TEST/多seed的CI方案仍按原科学规范另行授权。

### 6.3 单张图稿与真实执行日志

优先用执行时callback记录：actor_call_index、action_index、可见状态、C0/环境legal mask及reason、动作、before/after cost、当前预测、权重ID；固定路线标FIXED_ORDER，不能生成虚假的Actor调用号。可记录为稀疏表或episode列表，不需要给每个tensor反复hash。

只在VALID取最多三件：优先复用已冻结case list；若无明确名单，在读本轮得分前按域排序、每域对`sha256("cai-v3-w3-case|"+specimen_key)`取首件，再取前三域，并保存case_manifest。不要按效果挑最好案例。

输出独立PNG：注册表面+VLM候选、真实首步、k=1/4/8/末已测C-scan+下一动作、CAI预测随成本变化。图中未测区域显式遮蔽；完整C-scan只在明确的事后参考图出现。无原图仅阻塞图稿，不阻塞模型/数值交付，不造可视化。

图稿只从已存轨迹生成，不重规划、不让VLM再判一次、不重跑全部VALID。旧diagnostics的网格绘制如有均匀浮点线与rint格边的细微差异，以真实NativeCellGrid框为准；只改显示，不改动作几何。

## 7. 实际执行安排与有限review

| 阶段 | 工作 | 产出 | 是否优化 |
|---|---|---|---|
| P0 | 明确任务/资源、读取W2和DATA、修路径/seed/归档；冻结本轮协议与初始review | INPUT_BINDINGS、AUTH、IMPLEMENTATION_BINDINGS、PREFLIGHT_REVIEW | 0 |
| P1 | 新P_all固定路线一次评分；记录固定输入基线 | fixed_episodes与fixed_metrics | 0 |
| P2 | 依表训练五个seed1，逐job首250核验、所有参选时点保存 | Actor/进度/候选轨迹与manifest | ≤5750 |
| P3 | 只读汇总，完整比较和pilot建议，有限单图 | pilot_metrics、effects、case_figures | 0 |
| P4 | 最终核对本轮结果及资源；交接、commit/push | HANDOFF/REVIEW/final_manifest | 0 |

没有独立真实优化预检：用合成数据、零optimizer.step的接线检查、真实job的首步/首250完成检查。已有precheck_policy_training会真实优化且写旧目录，**不得直接调用**。

review用配套文件的8项行为核对，优先独立审查上下文；不具备时如实标SELF_CHECK_WITH_EXTERNAL_NUMERIC_ORACLE。两模式都必须引用原条款和实际源码，不能只核对配置值。

先完成路径/seed/成本/冻结模型/信息权限/保存机制检查再长训练；首250失败立即修受影响项，不等五模型跑完。修复改变实际科学输入或损失时保留作废记录，不能仅改状态继续。

## 8. 资源授权 [D：新的W3-only配额]

采用**不挪用W2余额**的简单新配额，避免再次混淆类别：

| 项目 | 数值 |
|---|---:|
| 已核对历史已用上界（含未知750） | 33,264 |
| 原累计授权上限 | 34,264 |
| 原W2剩余 | 1,000，仍属于W2，本轮不得使用 |
| 本轮五job正式及所有重试总上限 | 5,750 |
| 新累计授权上限=34,264+5,750 | **40,014** |
| 正常用满本轮后历史已用上界 | 39,014（仍有原W2未用1,000） |
| 本轮独立GPU时间上限 | **6小时=21,600秒** |

类别只新增W3本轮：主/无VLM/开环共3,750，真实静态750，均值结构诊断1,250。原所有已用账目和W2权限不重置。模型上限不能通过失败换名来绕过；超过本轮总量或任一方法上限即停止并交付。

GPU时间窗口是新增W3授权，不宣称旧窗口还剩多少；包括训练、VALID/预检前向及本轮GPU操作，不要求跑满。默认仅一张现有GPU、CPU≤4线程，不建分布式环境。前250实测吞吐用于估算后续，不根据W2的324秒承诺Actor也同样快。

每个job使用唯一run_id，STARTED预留该job上限；COMPLETED一次结算实际更新，PROGRESS实际更新增量为0或另一种明确且不双计的已验证语义。未完整结算保持保守预留。全局LEDGER仅追加，同时保存W3视图；不得新建空账目冒充累计归零。

本轮入口每个job前检查上游、授权、本轮剩余、类别、GPU时间与review。不修改旧全局28,100/34,264常量使所有历史训练接口突然开放；本W3薄入口必须显式读新授权。

6h或更新上限用尽：保存当前状态、实际结果、未执行原因并push。不得删除VLM、减少队列、放宽成本、降低gate或启用新TEST“完成任务”。

## 9. 必须实际交付并推送GitHub

至少交付：

```text
ART/IMPLEMENTATION_BINDINGS.md
ART/INPUT_BINDINGS.json
ART/PREFLIGHT_REVIEW.md（或合并有限review）
ART/W3_PILOT_HANDOFF.md
ART/RESULTS_AND_CLAIM_BOUNDARIES.md
ART/REQUIREMENTS_REVIEW.json
ART/RESULT_POINTER.json
RUN/pilot_authorization.json
RUN/protocol_snapshot.json
RUN/fixed_episodes.csv.gz、fixed_metrics.csv
RUN/policy_training_progress.csv
RUN/policy_validation_episodes.csv.gz（全部最终650条或明确不完整）
RUN/policy_pilot_metrics.csv、pilot_effects.csv、pilot_gate.json
RUN/absolute_cai_performance.csv、per_domain_metrics.csv
RUN/initialization_summary.csv、case_manifest.csv、figures/
RUN/models/ 与 candidate_episodes/（所有实际参选权重和对应轨迹）
RUN/final_manifest.json、ledger_view.jsonl
```

名称可做不改变语义的轻微调整，须在交接中写清实际命令/路径，不只保存空模板。最终checkpoint只复制选中的实际权重，来源归档保持。旧已存W2/特征不重复上传，但新必要模型和轨迹必须实际上传；`*.pt`可能被忽略，对本RUN精确`git add -f`，不是force push。

不追加全库测试、渗透、fuzz、压力、反复全量hash、上游模型重评分或新算法调参。只验证被改调用链和本包8类核心行为。

训练前提交本轮代码/授权/协议绑定；最终提交代码、review、所有实际结果与交接，实际push同一研究分支。核对local HEAD、upstream、指定`git ls-remote`一致；不PR、不merge、不force push，不提交无关未跟踪文件，不循环把文件自身的最终commit写回文件。

摘要状态分开：
- implementation_status / protocol_conformance；
- W3 seed1完成度与pilot_expansion_condition_met；
- `scientific_evidence=VALID_ONLY_PILOT_NOT_INDEPENDENT_CONFIRMATION`；
- 工程阈值仍未指定；
- W2训练、新VLM、GDFS、seed2/3、TEST次数均0；历史失效不追认。

最终回复用一张方法结果表（A、early A、终点MAE、所选update、更新量）、三项主方向及空间诊断、资源和Git SHA。负结果必须完整交付。不要只回复“流程已读”，也不要把VALID pilot通过说成已经发表级验证。
