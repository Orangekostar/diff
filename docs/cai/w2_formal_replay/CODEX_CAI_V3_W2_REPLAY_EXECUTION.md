# Codex执行规范：CAI v3 W2正式重放（共同预测器 → 条件OOF回报模型）

日期：2026-09-13。仓库：`Orangekostar/diff`。
证据基点：`292b1c74b27bb344af370f6738ac9c493cfd137d`。
继续分支：`research/cai-vlm-agent-v3-controlled-reuse`。
本任务不是零训练恢复准备；准备已经完成。本文交给Codex执行时，授权的是以下有界W2训练，不授权后续Agent实验。事实依据见配套 `W2_REPLAY_SOURCE_BINDINGS.md` 的 [R01–R12]；本轮新增安排标作 [D]。

## 0. 本轮到底做什么，怎样才算完成

执行顺序：

> 复用已修好的exact-cost/归档实现 → 建立本轮隔离输出和额度 → 完成有限训练前review → 重训三个共同预测器并完整选模 → 仅准备条件通过后重训选中结构的三个OOF模型 → 实际评分、交接并push；到此结束。

必须交出真实的新W2模型与VALID结果，或如实交出阶段未通过/资源停止结果。不能只返回计划、又一份零训练准备报告或“等待具体任务”。

### 0.1 优先级与新授权

用户最新指令 > 本文件（本轮范围、额度、隔离和执行绑定）> 原v3主规范相关科学定义 > 当前代码/交接。原完整v3文件在 `basis/` 中提供逐字副本，仓库中已有副本可复用。

- 本文件明确替代上一份 `CAI_V3_W2_RESUME_TASK_BINDING.md` 的“本轮训练=0”和“需另行授权W2训练”限制。
- 原v3的标签、数据划分、模型、种子、损失、成本、验证库、早停、准备条件不改变。
- 只覆盖W2；即使新W2就绪，W3/GDFS/多seed/TEST仍不授权。不能以原v3全文有W3–W5为由继续。
- 已批准本轮执行后，不再要求用户重复确认同一额度或阶段。真实新冲突、资源超额或必要文件缺失仍需说明，不得伪造解除。
- 本包不会在聊天端自动执行任何训练、commit或push；用户把它作为任务发给Codex后才进入执行。

### 0.2 范围与历史边界

保持整体研究目标：未来由VLM提供起始先验、Agent每次新增C-scan后选下一位置，以更少图像采集获得更准确的CAI预测。本轮只重建这个系统赖以评价和训练的CAI预测器，不删VLM、不替换Agent设计。

本轮禁止：Actor/STOP训练或前向、GDFS、VLM新调用、CNN重编码/微调、TEST感知/标签读取/评分、W0重算、W1重建、重新找37份已确认缺失权重、专家重标、幅值恢复、模型搜索、全库测试。读取现有feature bank时允许其中仍带TEST无标签缓存，但任何拟合、选模、预测与误差导出均只取TRAIN/VALID或TRAIN内OOF；不得重新连接TEST标签。

旧W2–W4仍保留 `DIAGNOSTIC_ONLY_INVALIDATED`。新W2是新执行结果，不是追认缺失的历史checkpoint；也不自动使旧策略有效。[R01]

## 1. 固定任务绑定与目录

### 1.1 接续，不回退

核对 `git status --short`、当前分支、`git rev-parse HEAD` 和一次远端fetch。基点应是上述提交或能说明的后续延续；保留用户未提交/无关文件，不reset、不重新从v2开一条重复实验。

定向读取：
- `docs/cai/v3/CODEX_CAI_AGENT_V3_EXECUTION.md` 及配套review/metric文件；若缺，使用本包 `basis/`；不要误用旧启动README从W0开始。
- `artifacts/cai_agent_v3/W2_RECOVERY_HANDOFF.md`、`W2_RECOVERY_EVIDENCE.json`、`W2_RECOVERY_VALIDATION.md`。
- `results/cai_agent_v3/compute_ledger.jsonl`。这是实际账目路径，不是 `new_protocol/compute_ledger.jsonl`。

更新现有 `docs/stagetask/TASK.md`：本文件实际绝对路径、分支/HEAD、本轮W2-A/W2-B、执行额度和停止条件。STATE保留历史，并将“只做零训练准备”更新为“W2正式重放已绑定”。不要另建通用工作流平台。

### 1.2 本轮独立输出 [D]

固定本轮ID：`W2_EXACT_COST_REPLAY_R1_292b1c74`。
建议只创建：

```text
results/cai_agent_v3/w2_replay/r1_292b1c74/
artifacts/cai_agent_v3/w2_replay/r1_292b1c74/
```

前者简称 `RUN`，后者简称 `ART`。现有 `new_protocol/` 是只读数据/特征/验证库来源，不是本轮写入目录。

- 新权重、候选归档、score、gate、OOF折表全部进入RUN。
- 原 `new_protocol/models/`、`predictor_gate.json`、`oof_readiness.json`、`cost_precision_audit.json`、W3/GDFS与历史review不覆盖、不自动刷新为PASS。
- 全局 `results/cai_agent_v3/compute_ledger.jsonl` 只追加本轮记录；同时可输出本轮ledger视图，不用新空账目代替历史累计。
- RUN若已有本轮真实结果，先检查其身份与完成状态：正常完成且配置一致的job直接复用，不重复训练；不完整的job不得改为COMPLETE或删除后重跑。
- ART中的最终结果指针供未来任务显式引用，不将新权重偷偷复制到旧目录让下游自动解锁。

## 2. 必须读到的代码与最小改动绑定

仅围绕下表阅读，生成一份最多约两页的 `ART/IMPLEMENTATION_BINDINGS.md`；不要复述泛化研发流程。

| ID | 当前实际代码 | 已确认事实 | 本轮处理 |
|---|---|---|---|
| C01 | `predictor_training.py::_train_candidate` | 现有循环已接入归档、250-step评分、4次无改善早停、从文件重载赢家 | 复用；不重新实现优化算法 |
| C02 | `predictor_training.py::run_predictor_candidates` | 写死旧输出路径，且按 `28100-used<12000` 拒绝 | 增加本轮显式输出和额度参数；不能只改说明文件 |
| C03 | `run_oof_reward_predictors`、`require_predictor_selection_evidence` | 默认读取旧目录gate/归档；OOF依赖完整选模证据 | 让它们显式跟随同一本轮RUN，不能混用旧P_all |
| C04 | `_cell_costs/build_validation_library/_require_exact_validation` | 硬成本float64，固定四路线，模型cost输入可转float32 | 复用；验证库按实际数组身份锁定 |
| C05 | `checkpoint_selection.py::CheckpointArchive/inspect_archive` | 所有实际参选权重已归档，计划完成或合法早停才COMPLETE | 保留已修逻辑，不再只存最佳快照 |
| C06 | `gates.py::predictor_readiness_gate/choose_common_predictor` | 预测能力条件和1e−8结构平局规则已实现 | 不改门槛，不按结果挑模型 |
| C07 | `feature_bank.py::load_feature_bank` | 从旧索引加载六个shard；TEST标签为NaN | 只读复用，不重编码，不把新RUN冒充project_root |
| C08 | `actor_training.py::_optimizer_update_upper_bound` | 对历史实际更新及未结算reservation求上界 | 复用账目语义；新增W2范围授权，不全局放开Actor额度 |
| C09 | `cli.py`、`scripts/run_cai_agent_v3.py` | 旧train-predictors/train-oof无独立run或授权参数；prepare会重做W0/W1 | 新薄入口或兼容参数，只调用W2；不调用旧prepare |
| C10 | `models.py`、`evaluate_predictor/_constant_metrics` | 三候选结构与现有VALID聚合口径明确 | 本轮冻结；只增结果导出，不改模型或标签 |

文件路径均相对 `src/cmc_bbdm/cai_agent_v3/`，入口脚本除外。[R02–R10]

### 2.1 推荐实现方式 [D]

优先给现有W2编排函数增加兼容的显式 `output_dir/run_context` 与授权参数，旧默认行为不变；`project_root`始终是真实仓库根，feature bank仍从原只读路径取。训练循环、模型构造、采样、指标函数只用一份。

可以新增一个薄 `scripts/run_cai_agent_v3_w2_replay.py`，提供以下拟开发命令：`prepare-run`、`train-candidates`、`train-oof`、`summarize`。这些名字是本轮设计，不假称已经存在；实际实现后把准确命令写入交接。

- 新入口只暴露W2动作；`summarize`零优化、不隐式训练/重评分全部模型。
- `train-oof`必须读取RUN中的真实gate和完整归档；旧目录READY字段不能满足新依赖。
- 无需复制整个pipeline或建立临时假仓库来欺骗旧路径解析。
- 旧 `refresh-cost-evaluation` 会写旧库和旧gate，不作为本轮恢复命令。

## 3. 本轮明确额度 [D：新资源授权，不是历史事实]

用户将本文件作为执行任务发送时，采用“重分配未用类别额度＋净新增额度”的W2-only方案；不用上一份零训练上限阻塞本轮。

| 项目 | 数值/含义 |
|---|---|
| 历史已知更新 | 21,514 |
| 历史未知更新上界 | 750，继续保留，不凭估计释放 |
| 历史保守已用 | 22,264 |
| 旧累计上限 | 28,100 |
| 旧剩余下界 | 5,836，其中W2类1,250、其他类4,586 |
| 本轮共同预测器新训练 | 最多6,000（3×2,000） |
| 本轮条件OOF新训练 | 最多6,000（3×2,000） |
| 本轮新增实际优化总上限 | 12,000，含失败/中断/重试，不另开优化预检配额 |
| 明确转给W2的其他旧类别余额 | 4,586 |
| 净新增累计额度 | 6,164 |
| 新累计总上限 | **34,264 = 22,264 + 12,000 = 28,100 + 6,164** |
| W2共同/OOF历史＋本轮类别上限 | 11,000 / 11,750 |
| 本轮新增GPU活动时间 | **最多6 GPU小时**，一张现有GPU，CPU≤4线程；含训练和必要验证 |

旧交接的“34,464”是200次的算术错误。保留历史文件，在本轮authorization中写清更正，不照抄。[R01]

新的6 GPU小时是本任务单独提出的增量时间限制，不是声称旧12小时还剩6小时。旧GPU时长只有记录下界，保留缺测说明；本轮从开始完整记时。上限不是必须跑满，也不是对完成工期的预测。

### 3.1 授权必须落实到运行入口

随包 `W2_REPLAY_AUTHORIZATION.json` 是待绑定到仓库的授权配置模板。执行时记录用户任务来源、代码基点、实际ledger起始位置、run ID和输出目录；不要伪造用户签名或已执行状态。

- 起始核对历史账目；若实际新增消耗已超过本基点22,264上界，先计算真实可用额度。不能把34,264再自动加大。
- W2-A开始前，检查可以覆盖本轮已承诺的6,000＋条件6,000以及新时间窗口；作stage allocation，不把整批12,000再写成重复STARTED消耗。
- 每个job只保留一次最多2,000的reservation，同run完成后仅结算一次实际更新。不要在已有reservation之外重复加一次同样额度。
- W2-B开始时检查的是该阶段6,000余额，以及每个剩余job，不得再次要求还剩整个12,000，导致A完成后B永远被拒绝。
- 进度记录不重复计入 `actual_optimizer_updates`；沿用旧“STARTED预留＋同run完成实际结算”含义，未知中断按保守上界占用。
- 资源上限只在新W2作用域生效。不要全局把Actor/GDFS/TEST阈值抬高，不把W2节省量当下游授权。
- 达到时间/更新限制时保存真实进度并交付 `RESOURCE_LIMITED`；人工暂停不是科学早停，归档不能被强行finish。

## 4. P0：一次输入与训练前检查，不重复恢复准备

### 4.1 复用输入

从原 `new_protocol/`读取：现有cohort/split、feature_bank_index及六shard、精确VALID库、必要constants/Ridge表。历史基线为276试样、259组，TRAIN161/152组、VALID50/48组、TEST65/59组；使用实际文件核对，不重新分组。成员或有效标签变化则报告差异，不能在一次W2恢复中静默换数据。

固定：作者CAI MPa、原生尺寸、已有图像预处理、现有标签精度、所有模型种子、OOF分组、原VALID路线。旧权重不作warm start，预训练图像特征可以复用。

现有feature bank加载TEST无标签行不等于TEST评分；禁止任何对这些行的模型预测、目标连接与错误分析。VLM fit缓存包括已记录失败项也只保留，不补调，测试感知仍关闭。

### 4.2 精确VALID库

使用 `build_validation_library(bank, _cell_costs(bank))` 的结果，并与已修复保存库的数组逐项核对一次。路线为CENTER_FIRST、GEOMETRY_SPREAD、SERPENTINE及按specimen key固定hash的RANDOM；每条从0开始至不超过B=.25的最后合法状态，完整64格另作全输入评价。

保留 `_require_exact_validation` 和 `validation_identity`；复制小型库到RUN或保存同身份引用。新模型/所有update/OOF均用相同库。不要用旧float32库，也不要重新采样VALID。

成本来自原生像素分区；继承rint(linspace)边界与1e−12合法性容差。不能更换为floor边界、给预算加epsilon以多采一格，或把float32模型输入反用来判定硬动作。

### 4.3 有限review与第一检查点

执行配套本轮review，不先开长训练。复用已完成28项检查的结果，只重查改动影响的输出隔离、额度作用域、归档接线、gate及固定数值。

使用零优化合成参数和已有metric_reference。不要运行旧 `precheck-predictors`，它会执行3次真实optimizer更新且写旧目录。需要接线检查可禁用optimizer.step，或将第一候选真实训练的首个250-step检查点作为运行中验收，计入正式2,000，不能丢弃另起一轮。

第一真实候选到250步后，立即检查：权重确实在RUN、selection仍INCOMPLETE、评分/VALID身份完整、ledger run ID正确。没通过就停止受影响job；不要先跑完六个模型再检查文件。

## 5. W2-A：三个共同预测器的正式重放

复用 `_train_candidate`；使用新输出的真实 `archive_dir`，不能默认None。

| 模型 | 构造 | 固定seed | 上限 |
|---|---|---:|---:|
| MEAN_SC | MeanSCPredictor，64宽、表面＋已测C-scan | 2026091201 | 2,000 |
| SPATIAL_SC | SpatialPredictor(use_surface=True)，128宽2层4头、FFN256、dropout .1 | 2026091202 | 2,000 |
| SPATIAL_C | 同空间结构，回归不使用表面 | 2026091203 | 2,000 |

采样与优化全部继承：均衡选域→选物理试样；batch32；mask混合10%零、60%1..16格四路线、20%17..48随机、10%完整；AdamW lr3e−4/wd1e−4、clip1；按各自fit TRAIN尺度标准化的Huber delta1。Mask不能根据隐藏图像或CAI选择。当前代码的尺度下界1.0保留且记录，不借恢复改变目标单位。[R03,R10]

每250更新评价固定库；改善需严格低于旧最好减1e−12，4次连续无改善则科学早停；否则最多2,000。实际forward结构、参数数、使用surface开关必须记录。早停可少于8个checkpoint，不强制复制旧43个时点。

### 5.1 每个参选点必须保留

1. 用同一次VALID前向取得真实predictions/metrics。
2. 原子保存 `update_XXXXXX.pt`、update、seed、fit组、标准化常量、VALID identity和成本定义，再更新排名。
3. 建议从这次评价直接导出轻量逐状态预测（NPZ或压缩CSV）；不要为多保存一个文件再重复前向一遍。可给 `evaluate_predictor` 增加兼容的details返回/回调，数学保持不变。
4. 每个正常参选点都保存，不只保留赢家。一次完整六模型重放最多48个参选权重，但合法早停可能更少。
5. 正常结束调用finish，实际重载文件中的赢家，再核对身份与同口径指标。不是只保留内存里的best_state。

不要在最终验收再对所有候选进行多轮全量前向。训练时保存的真实完整评分＋一次选中模型重载核对足够；确需检查具体不一致时，仅针对该项调用 `rescore_predictor_archive`，保持同VALID、同设备/批大小/精度。不得将CPU/GPU末位差异误称新观测或为了过关放宽科学选择阈值。

### 5.2 保存选模快照不等于可继续训练 [D：最小容错]

现有archive只存权重，不含optimizer/RNG。优先在参选检查点另原子保存一份 `latest_training_state.pt`：model、optimizer、numpy Generator状态、Torch CPU/CUDA RNG、update、best/stale计数、run ID与VALID身份。保留当前checkpoint，不做通用调度/续训平台。[L01]

正常运行无需调用续训。发生中断时先保留并结算真实消耗；没有完整状态就不得从旧赢家续训并声称原训练恢复。新建自动恢复路径不是本轮硬性完成项，无法可信续跑就交付明确状态，不偷偷增加试跑/重启次数。

### 5.3 指标口径与选模

主选模指标仍是 `evaluate_predictor()` 的 `valid_area_mpa`：每条真实轨迹左端阶梯面积，每试样平均四路线，再各域平均、六域等权。

```text
A(B) = [Σ Δc_t * |p_t−y| + (B−c_T)*|p_T−y|] / B
B = .25
```

W2预测训练是Huber；checkpoint选择为A；不把Actor的 `.25*terminal` 辅助项加入W2预测损失或选模。当前门槛中的full/zero/center/geometry指标为VALID物理试样的池化均值，常量参照同口径；这与A的六域等权是不同汇总，分别记录，不在本轮偷偷统一或修改。[R04,R06]

每个候选先按固定A选其update，再判断该选中update是否通过全部条件，不能为了通过门槛改选另一个checkpoint：
- full MAE < fit TRAIN中位数常量的VALID MAE；
- full MSE < fit TRAIN均值常量的VALID MSE；
- full MAE ≤ .98×zero MAE；
- CENTER_FIRST和GEOMETRY_SPREAD在B=.25的MAE **各自** < zero MAE；
- 数值、标签、可见性及归档均有效。

这里明确采用当前 `gates.py` 的两个独立路线条件，不能只合并成平均值来放宽。再从通过的模型中选A最小者；1e−8内平局选参数更少者，再模型名。没有通过者就 `PREDICTOR_NOT_READY`，跳过W2-B、整理并push。全部模型即使不通过也必须报告。[R06]

## 6. W2-B：仅对新选中结构恢复三个OOF回报预测器

W2-A已有新 `PREDICTOR_READY`，并经**指向RUN**的 `require_predictor_selection_evidence(..., include_oof=False)`验证三候选及赢家真实归档后，才可进入本阶段。

- 不预设仍选MEAN_SC，按新A阶段结果构造。
- 复用 `_oof_assignments` 的TRAIN capture-group hash和逐域均衡分配；输出并核对每折fit/query名单、组数、物理N和域数。不得重新分折或使用query折挑checkpoint。
- 每折各从头初始化，seed依次2026091211、2026091212、2026091213；每份≤2,000，同W2-A训练/验证/归档规则。
- 每折的均值、尺度和常量参照只从该折fit计算；checkpoint仍用外部VALID的同一固定库选择。
- 记录新选中结构、折ID、fit/query身份；使用fold对应常量做独立核对，不能误用P_all的常量。
- 已进入本阶段且资源/正确性允许时完成固定三折，不根据某折成绩换seed或改结构。若某折不通过准备条件，其余既定折可按计划完成诊断，但后续Agent仍不能启动。
- 三份选中模型分别按同样准备条件评价；全通过且所有归档有效才 `REWARD_MODELS_READY`，否则 `REWARD_MODELS_NOT_READY`。
- 导出query折零/四路线部分/完整输入预测用于OOF误差诊断，不用于选择这些模型的权重。未知新TEST保持关闭。

新W2-A和W2-B是否就绪分别报告；准备条件是后续资源筛选，不是工程达标、更不是VLM或闭环策略有效的证据。

## 7. 复用Ridge、输出结果，不开展新算法实验

已有TRAIN常量和完整Ridge在输入一致时直接复用。部分Ridge若旧前缀成本身份不同，允许按正确库一次有限CPU重算；仍alpha10/PCA既定规则，不网格搜索。不要因新编排默认调用整个Ridge流程而无条件重做所有拟合。

必交结果：

```text
RUN/
  replay_authorization.json
  input_reuse_manifest.json
  validation_prefix_library.npz                 # 或可解析的同身份只读引用
  predictor_training_progress.csv
  predictor_comparison.csv
  predictor_gate.json
  candidate_state_predictions/                 # 同次VALID前向保存，不额外训练
  models/predictor_<name>.pt
  models/selection_history/predictor_<name>/selection.json + 全部真实参选权重
  oof_fold_manifest.csv                        # 若B未执行，明确状态，不伪造空成绩
  oof_predictor_training_progress.csv
  oof_state_predictions.csv
  oof_readiness.json
  models/reward_predictor_<name>_fold<k>.pt
  models/selection_history/reward_<name>_fold<k>/...
  final_manifest.json
ART/
  IMPLEMENTATION_BINDINGS.md
  W2_REPLAY_REQUIREMENTS_REVIEW.md
  W2_REPLAY_RESULTS_AND_BOUNDARIES.md
  W2_REPLAY_HANDOFF.md
```

实际后缀/目录可作非科学调整，但交接给出准确路径。所有manifest中的路径相对真实project_root；原始特征和数据索引只引用，不复制六shard或原图。

结果表至少列：候选/折、fit/query N、selected update、实际updates、VALID A、zero/full/center/geometry MAE、full RMSE/R²、各准备条件通过/失败、归档完整性。各域误差为诊断，不增加新的主检验。OOF query效果不得混作VALID或TEST。

最多2张单独PNG：共同候选的VALID CAI误差—成本曲线、各OOF模型的预测/误差对照。优先从保存预测绘图，不重新前向。所有图标VALID/OOF，不画或重跑Agent轨迹，不把固定前缀说成自主规划成绩。

旧失效结果可在另表作同队列诊断，但不得将新W2 VALID与旧v2 TEST的MAE直接相减称性能提升，也不宣称已达到之前的41 MPa或其它历史数字。

## 8. Review与状态：少而关键，训练前必须有

用 `CODEX_CAI_V3_W2_REPLAY_REVIEW.md` 完成8项有限验收。真正检查“本轮规范→实际代码→独立预期/真实文件”，不是把已写的配置当授权。

阶段：P0前置；W2-A第一次250-step快速归档确认；A完成后；B完成/停止后最终。每次只核对对应项目，不重复整套旧28/52项，也不每个checkpoint重复哈希所有数据。

独立上下文可用时用独立审查；否则标 `SELF_CHECK_WITH_EXTERNAL_NUMERIC_ORACLE`。不编造独立审查者或“review通过”过程。

结果至少区分：
- implementation_status：薄适配与执行是否完成；
- checkpoint_selection_status：哪些新run证据完整；
- predictor_readiness / reward_readiness：实际数值条件；
- scientific_scope：仅本次W2 TRAIN/VALID、历史W2–W4仍失效；
- downstream_authorization：`NOT_AUTHORIZED_W2_ONLY`；
- engineering_status：`ENGINEERING_ACCURACY_CRITERION_NOT_SPECIFIED`。

## 9. 提交与GitHub交付

训练前先commit本轮任务绑定、授权配置、代码/前置review，使科学执行有可追溯基点。每个阶段完成可提交阶段文件；最后必须实际push同一分支。

必须上传：本轮改动代码、配置、CSV/JSON、所有实际参选checkpoint、最终选择权重、轻量训练恢复文件（如生成）、图稿和交接MD。`.pt`/`models/`可能被忽略，先检查本run路径，必要时只对该路径 `git add -f`；这个动作不是force push。[R01,L02]

不要只上传manifest却遗漏非赢家权重。不要为了上传重新打包全部原始图像/feature bank。单文件接近既有存储限制时沿用仓库可取得的LFS/分片方式，记录真实获取方式，不分享不可访问的临时路径。

完成后核对本地HEAD、upstream和 `git ls-remote` 指定分支SHA一致；不PR、不merge、不force push、不覆盖其他工作。最终交付SHA写在最终回复，不循环修改文件来包含自身提交号。推送失败真实报告，不能称上传成功。

最终回复只需阶段表＋关键数值：A是否完成/哪个P_all；B是否执行/三折准备；本轮实际更新和GPU时间、累计上界；文件路径；三方SHA。没有通过门槛也要交付，不扩范围去追求好数字。
