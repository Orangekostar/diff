# Codex执行规范：CAI Agent冻结结果的论文证据整理

日期：2026-09-14。仓库：`Orangekostar/diff`。
接续分支：`research/cai-vlm-agent-v3-controlled-reuse`。
已核对证据基点：`e2a1115468da6e8695321204a13fa9a5322ea809`。
任务ID：`CAI_V3_PAPER_EVIDENCE_R1_e2a11154`。

## 0. 本次到底做什么

用户已决定停止新增实验，转入论文证据制作。本任务是**实现并运行只读结果分析、生成论文表图与写作证据、实际提交推送**，不是新训练，也不是只交研究计划。

> 以已冻结W3的九种方法、650条VALID轨迹和W2同一预测器的完整输入预测为依据，形成“为什么好、好在哪里”的证据链，回答“相同采集成本上限谁预测更准”和“达到相同经验预测质量谁需要更少采集”。

### 0.1 最新范围覆盖历史继续授权

用户最新指令 > 本文与 `PAPER_EVIDENCE_SCOPE.json` > 已冻结W3的指标/数据/方法定义 > 当前脚本默认行为。此前pilot通过不构成本次继续W4/W5的许可。

| 允许 | 本轮禁止 |
|---|---|
| 读取既有CSV/JSON/gzip轨迹及已保存预测NPZ；统计、分组重采样、表格与图稿整理 | 所有新优化更新、模型拟合、Actor/预测器/VLM/Reader/STOP/CNN前向 |
| 引用并整理完整64格输入＋同一个有效CAI预测器 | 新稀疏采样、经典插值、企业启发式等基线，或更换回归器 |
| 复用既有21张图；必要时仅用同三个已冻结案例原图重排/提升输出分辨率 | 新seed、TEST感知/预测/标签接入、新试样/新配准/新专家标注/幅值恢复 |
| 完整保留正负消融、现有全部方法、逐域差异 | 重新选checkpoint、换主方法、筛除失败样本、选择有利阈值救结果 |
| 常规记录本次脚本处理时间与资源以便交接 | 硬件性能/设备时间/推理延迟/运动路径成本评测或其额外实验 |

**预测、动作和评价样本完全冻结；可以新增派生分析，但它们仍是已有VALID结果的事后分析。**本轮不再要求补种子或新TEST才能完成交付；相关限制在论文适用范围里如实披露。

主方法仍叫 `VLM_SPATIAL_FEEDBACK`。`NO_VLM_SPATIAL_FEEDBACK`是必须保留的强对照，不能因更好而在写作中删除。当前策略按CAI任务训练，不改称历史P4行为克隆BC。

### 0.2 研究对象与表述

- 主任务：利用逐步取得的C-scan**图像**预测实验测得CAI强度，单位MPa。
- 成本：取得的唯一原生图像像素数／完整图像像素数；不是设备时间、货币成本或扫描速度。
- 范围：VALID 50个物理试样、48个capture group、六个已有数据域，一个学习策略种子面板。
- 650为episode数，不是独立试样数；RANDOM五repeat不是五个数据集或部署集成。
- 当前没有空间损伤GT任务、部署STOP或工程允许误差线。本轮不生成这些任务的成功率。
- 方法机制可用源码和轨迹说明；动作变化不自动等于语义推理，采后误差下降不自动等于真实损伤的因果贡献。
- 不把本比较称为已覆盖整个工业检测流程。完整扫描是同一预测器的全信息参照，不是经过另行验证的工业最强模型。

## 1. 固定来源、路径与任务绑定

相对于真实仓库根ROOT：

```text
DATA = results/cai_agent_v3/new_protocol/
W2   = results/cai_agent_v3/w2_replay/r1_292b1c74/
W3   = results/cai_agent_v3/w3_pilot/r1_0e11452a/
A3   = artifacts/cai_agent_v3/w3_pilot/r1_0e11452a/
RUN  = results/cai_agent_v3/paper_evidence/r1_e2a11154/
ART  = artifacts/cai_agent_v3/paper_evidence/r1_e2a11154/
LEDGER = results/cai_agent_v3/compute_ledger.jsonl
```

DATA/W2/W3/A3及历史失效结果只读。本轮表格、图、参考提取和manifest写入RUN；解释、来源、review、交接写入ART。不要调用旧 `w3_results.summarize(root)` 或旧 `export_figures(root)`：它们会写原W3结果，前者还会重置原final_manifest的review状态。[R03,R04]

复用现有工作树；先检查分支、HEAD和一次远端引用。若已有后续提交，保留并说明与证据基点关系，不reset、不覆盖用户文件。将本包放在 `docs/cai/paper_evidence/` 或一个明确记录的目录，更新现有TASK/STATE，写明本文件实际路径、任务ID、零训练权限和完成标准。历史“待授权训练”不阻塞本次纯计算任务。

### 1.1 必读资料，不做全仓库搜索

1. A3中的 `W3_PILOT_HANDOFF.md`、`RESULTS_AND_CLAIM_BOUNDARIES.md`、`REQUIREMENTS_REVIEW.json`。
2. W3中的 `final_manifest.json`、`actor_manifests.json`、`policy_validation_episodes.csv.gz`、`policy_pilot_metrics.csv`、`absolute_cai_performance.csv`、`pilot_effects.csv`、`per_domain_metrics.csv`、`per_domain_effects.csv`、`paired_effects.csv`。
3. W3中的 `p_all_saved_reference.json`、`case_manifest.csv`、`figure_manifest.json`；A3 `INPUT_BINDINGS.json`。
4. `p_all_saved_reference.json`明确指向的W2 `candidate_state_predictions/A_MEAN_SC/update_001750.npz`；DATA `feature_bank_index.csv`及已绑定split元数据。只需要索引，不需要加载六份图像特征bank。
5. 解释VLM时只读已有 `vlm_actor_features_fit.csv`、`vlm_manifest_fit.json`及必要的缓存记录。

若确有必要文件缺失，定向查看本RUN/manifest引用的文件一次。已有完整输入汇总不能被扩写为50个虚构预测；缺原始预测时将对应配对项标为 `MISSING_SAVED_FULL_PREDICTIONS`，保留可完成部分，不补模型前向。

## 2. 源码—任务对应关系

以下路径默认前缀 `src/cmc_bbdm/cai_agent_v3/`；符号已经从固定提交核对。[R03–R09]

| 编号 | 实际代码/文件 | 已有能力 | 本次使用方式 |
|---|---|---|---|
| C01 | `w3_results.py::values/pooled_errors/method_metrics` | 按预算取当前预测、repeat内算误差、池化MAE/RMSE/R² | 复用纯计算逻辑；新写入目标，不调用原summarize |
| C02 | `actor_selection.py::read_episodes/episode_metrics` | gzip轨迹读取、A/early A/J重算 | 只读选中轨迹，禁止重新验选23个候选或加载.pt |
| C03 | `metrics.py::prediction_at_budget/left_error_area_mpa` | 不看未来的阶梯预测及面积 | 原数学不变；新增“可查询范围”封装，禁止越过.25外推 |
| C04 | `predictor_training.py::evaluate_predictor`的details_callback | 已保存 `full_specimen_indices/full_predictions_mpa/full_targets_mpa` | **只读其既有NPZ**，按冻结索引连接50个VALID key；不执行该函数 |
| C05 | `w2_replay_results.py` | 依据索引读取既存预测的范式 | 借用读取方式；不运行其写回W2的summarize，不混入OOF query |
| C06 | `actor_training.py::_evaluate_one/evaluate_policy` | 执行时记录可见集合、C0、调用号、像素和前后预测 | 仅查字段语义；用已保存 `execution_trace`，**不执行评价函数** |
| C07 | `w3_results.py::export_figures` | 既有三案例绘图和真实图像路径 | 优先复用21图；如重绘，仅提取纯绘图函数并写新目录 |
| C08 | `models.py::SpatialCAIActor/MeanFeedbackActor/TrueStaticActor`、`policy.py::vlm_first_action_mask` | 方法结构、输入权限和C0先验规则 | 只用于核对Methods/机制表述；不改代码、不实例化模型 |
| C09 | `w3_results.py::summarize`及W3报告 | 原对照选择、聚合和写入副作用 | 保留原BEST_NONADAPTIVE=GEOMETRY_SPREAD；不重新挑checkpoint或改历史状态 |

推荐新增一个小型只读分析模块，例如 `paper_evidence.py`，及薄脚本 `scripts/build_cai_agent_paper_evidence.py`，最多再拆出独立数值函数文件。可以借用现有纯函数，不复制训练系统、不建立数据库/网页/新工作流平台。

建议命令名 `prepare`、`analyze`、`export`、`finalize`或等价单入口。这些是**本次拟实现接口**，实现后写入真实可运行命令；不得把尚不存在的命令当已执行。`analyze/export`不导入训练入口执行逻辑、不加载checkpoint、不探测GPU；导入纯数学模块间接导入Torch不等于前向，但不要因此使用CUDA。

## 3. P0：建立唯一的冻结证据视图

### 3.1 轨迹与身份

读取最终650条轨迹一次，键至少是 `(specimen_key, method, seed_panel或training_seed, run)`。真实字段以文件为准，已由manifest绑定的缺省种子字段可引用，但不能虚构其余seed。

验证九方法、同50个key、48组、六域；五个学习方法各50条、三个确定固定各50条、RANDOM250条。`specimen_key`保留域前缀，不能只按裸试样编号连接。核对各key的target与分组一致，`len(predictions)=len(costs)=len(cells)+1`，cost从0单调增加，动作唯一。

从 `execution_trace`读取实际 `actor_call_index/action_index/cell/visible_cells_before/environment_legal/proposal_legal/c0_reason/before_cost/after_cost/new_pixels/cumulative_pixels/before_prediction_mpa/after_prediction_mpa`。不存在的字段不得通过叙述补成真实记录。可以从已存动作生成mask或box，但列为派生字段。

形成 `frozen_episode_index.csv`、`acquisition_events.csv`；new observation的误差变化为 `abs(p_before-y)-abs(p_after-y)`，明确y仅在分析侧使用。不要把它称为在线已知奖励或独立物理重要性标签。

### 3.2 完整扫描参考：不需要再运行全图模型

从W3 `p_all_saved_reference.json`核对模型路径、hash、selected_update=1750和已有NPZ路径。读取NPZ的三个 `full_*`数组；用 `feature_bank_index.csv`的**原行顺序**将索引转换为key，不能排序索引CSV后再映射。确认所取恰为同一50个VALID key，并与W3逐key标签相符。

输出 `full_scan_predictions.csv`：key/domain/group/target/full_prediction/abs_error/squared_error/model_hash/source_npz/source_row。`FULL_SCAN_P_ALL`成本只有1.0，完整参考只有一个独立结果点，**没有0到1的采集轨迹或A**。

保留模型仍接受表面输入的事实：它是完整C-scan＋现有表面输入的同一MEAN_SC，不改叫“只用C-scan的工业模型”。复算MAE/RMSE/R²应与41.690010/53.944040/.710536的来源一致；这些数字仅用于小数容差核对，不作为伪造原始预测的目标。

W2批处理与W3逐episode的零输入小数差可以作为数值实现差异披露，不为凑齐更改预测。完整参考不能来自三份OOF模型平均或旧v2 TEST。

## 4. P1：相同采集投入下谁预测更准

### 4.1 预算、当前报告与实际采集量

固定主表预算 `B_grid=[0,.0625,.125,.1875,.25]`，所有九方法均报告。对每条冻结轨迹：

`j(b)=max{j: c_j<=b}`，`p(b)=p_j`。末状态后至.25若无更多合法动作，保持最后预测；这不表示新增采集。使用既有bisect规则，不把动作后的值提前到动作前。

整格不可分割，因此主表标题为“相同采集成本上限”，同时输出实际成本均值/最小/最大、累计像素和unused budget。不得声称每条轨迹恰好采集b；不得按步数冒充相同像素成本。

### 4.2 保持两种聚合口径，不能混为一个量

- MAE：先每试样在RANDOM的五repeat上平均绝对误差，再在50试样池化平均。
- MSE：同样平均平方误差；RMSE是总体MSE开根号，不平均每repeat的RMSE。
- R²：每repeat在同50试样上计算后平均，保持原定义；没有“单个试样的R²”。
- A和early A：先在试样内平均repeat，再域内平均，最后六域等权；用原左端阶梯积分，范围分别[0,.25]与[0,.0625]。
- J=A+.25×终点误差仅作方法说明/补充；不是新的主评价，也不重新选checkpoint。

输出 `same_cost_metrics.csv`、`same_cost_metrics_by_domain.csv`、`same_cost_paired_losses.csv`、`same_cost_paired_summary.csv`。配对增益为“对照损失−主方法损失”；MAE差按上述池化，面积差按六域等权。RMSE/R²差用方法级统计，不能杜撰逐试样RMSE/R²差。保留全部八个主方法对照，不只选最好看的几个。

完整输入只另列1.0成本行和 `full_scan_gap.csv`；例如 `MAE_main(b)-MAE_full`，这不是同成本比较。不得为FULL_SCAN填一个虚假的A，也不得在每个b给它免费全图预测。

### 4.3 不确定性：有限、探索性，不新增确认实验

在当前冻结selected checkpoints上，对MAE配对差和原四项机制面积差生成一次5000次分组bootstrap（seed=2026091401）。六域固定，域内有放回抽capture groups，抽中一个组带入所有成员、重复次数作权重；所有方法和预算共用这些抽样。RANDOM先平均损失，repeat不增加N。池化与域等权分别按原估计量计算。

输出95%**逐点探索性**区间，字段 `inference_scope=POSTHOC_SELECTED_VALID_CONDITIONAL`、`simultaneous=false`。这些区间不校正此前的VALID选模和本轮看过结果后的分析选择，不能用于“确认显著”“因果证明”或新的继续关口。不是之前计划但未执行的正式TEST区间。原无区间结果保留，不更改其注册身份。不新增p值筛选、多个置信水平或跨seed推断。

## 5. P2：相同经验预测质量谁需要更少采集

### 5.1 只分析已观测范围，建立两套明确网格

**主表**使用既有五个预算点。**细粒度补充**使用所有冻结episode在[0,.25]内成本断点的并集，加0和.25；在每个共用b上按P1计算群体MAE。所有方法都用同一断点并集，不只使用主方法断点。

这是已有阶梯轨迹的精确重表达，不是新前向；底层MAE曲线不做单调回归、插值、滚动最优或平滑。保留误差反弹。`prediction_at_budget`本身会在任意后续b返回末值，因此新封装必须**明确拒绝b>.25**；仅FULL_SCAN允许独立点b=1。

### 5.2 等质量定义是群体水平，不能逐试样偷看真值选择停止

设 `M_m(b)`为P1池化MAE。在某个网格G上：

`c_m(q;G)=min{b in G: M_m(b)<=q}`。

无满足点输出null和 `NOT_REACHED_WITHIN_OBSERVED_RANGE`，不能填.25或外推。q是MAE目标，不是每个试样的真实损伤阈值。计算顺序是**先在共同b汇总所有试样、再求最小b**，不是给每个试样按y找最好停止点再平均。

`c_m(q)`是事后经验取得质量所需的成本上限，不是可部署STOP；同时记录该b下actual_cost_mean/min/max、M_m(b)及后来是否反弹超q。可列 `later_recrosses_target`作为描述，不据未来信息变更已观测曲线。

### 5.3 不挑有利阈值，不临时宣称非劣

本轮目标网格是**新分析约定**，不声称它在原训练前已注册。

1. 生成统一q网格：以1 MPa间距覆盖九方法所有已观测MAE及FULL_SCAN MAE的联合最小到最大范围（下取整/上取整）。完整CSV保存全部，不用任意截取的q范围得出普遍优越。
2. 生成完整对照锚点表：五个非自适应方法（CENTER、GEOMETRY、SERPENTINE、RANDOM、LEARNED_STATIC）各四个非零原预算的实际MAE，全部作为目标q；另列FULL_SCAN MAE。重复数值可共享计算，但保留所有来源记录。
3. 对每个q，比较双方的**最早达标成本**。对照在.25的MAE被拿来作q，不代表对照必须花.25；仍应查它是否在.0625或.125已经达到。
4. 有定义时，输出绝对成本差 `c_control-c_main`与相对差 `1-c_main/c_control`。c_control=0或任一未达到时比率为null，保留原因；负差也保留。
5. FULL_SCAN的可观察成本集合只有{1}。它在q满足时的c=1描述预定义完整输入流程，不证明传统方法只有扫完整图才能达到q。
6. 不设置事后5%/5MPa等“非劣”宽限来使主方法达标。输出全输入质量差 `M_m(b)-M_full`及 `max(0,M_m(b)-M_full)`的描述量即可；没有预先合理确定的非劣界值，就不实施或宣称确认性非劣。

输出 `quality_targets.csv`、`equal_quality_grid.csv`、`equal_quality_anchors.csv`、`cost_error_event_curves.csv`和 `full_scan_gap.csv`。将fixed-grid主结果与event-grid补充结果分栏，不取其中更有利的一套代替另一套。等质量比率暂不做bootstrap显著性检验，避免给离散、可能未达的逆问题制造虚假精度。

**独立检查示例**：q=46.9099169921875，原显示预算下GEOMETRY最早.125、主方法最早.0625、学习静态也最早.0625。相对GEOMETRY的预算上限减少50%，不是75%；相对该静态策略此阈值并无更早达标优势。真实细粒度网格结果另由保存轨迹计算，不硬编码这三个成本为全轨迹最小值。

## 6. P3：“为什么好”的证据，而不是后配推理故事

### 6.1 分层主张—证据矩阵

| ID | 主张/问题 | 必须使用的真实证据 | 不能越界 |
|---|---|---|---|
| E1 | 有限投入下的信息利用更有效 | 同预测器、同预算主表与固定对照A/终点配对差 | 不称超过所有工业方法 |
| E2 | 内部反馈对后续选择有价值 | 主方法对同VLM开环；逐域结果与已保存路径 | 不将一个seed差异写成严格因果贡献 |
| E3 | 不是单一通用排序 | 主方法对LEARNED_STATIC_TRUE；冻结多试样真实顺序 | 路线不同本身不构成效果证据 |
| E4 | 怎样随观测推进 | 真正execution_trace、C0释放、测量后预测变化与下一位置 | 不编造Agent语言思维链、注意力热图、未记录logits |
| E5 | VLM额外作用 | 主对无VLM早期及全程差；首步C0描述 | 现有负方向必须保留，不能换名为已证实优势 |
| E6 | 空间结构作用 | 主对均值A、early A、终点MAE/RMSE全部并列 | 不仅报告有利A、不称严格容量匹配或必胜 |
| E7 | 相对完整信息的权衡 | 同key的FULL_SCAN预测及质量差/等质量可达性 | 不将全图当免费基线，不虚构25%到100%轨迹 |

输出 `CLAIM_EVIDENCE_MATRIX.md` 和机器可读CSV；每条包含原文拟用句、估计量、方向、来源文件/行键、scope、支持层级、限制、表图编号。负方向也必须有行。

### 6.2 有限的过程分析

从既有事件统计：首步C0是否实际限制；何时解除；每次新增观测后的误差变化；主/开环/无VLM的首个分歧位置、同成本误差差。参考主方法真实首步proposal判断无VLM首动作是否位于其中；这是描述关联，不能据此判定“C0导致失败”。

比较路径时同k的图只说明过程，对效率的数值比较仍按同b。离开C0不自动说明纠错，预测变化也不自动说明观测的是损伤边界。只报告已存可验证行为，不新增反事实执行/强制路径/输入扰动前向。

预选案例严格复用：[R10]
- `74t7kcdgkr:c8-16`
- `cgtnjyggtm:q24-48`
- `w68dtmpfyf:q16-29`

复用原21张图及图像来源。需要比较图时仅从这些案例的已存九方法轨迹取同成本状态；不用效果最好的seed，不新挑案例，不删除预测差的q24-48。必要重绘最多仍这3件（至多6张原始表面/C-scan文件），原图缺失就复用既有PNG并说明。

输出 `mechanism_effects.csv`、`mechanism_event_summary.csv`、`case_narratives.md`与 `FIGURE_CAPTIONS.md`。每个叙述至少包含一次真实输入/动作/预测变化和不利或无法解释的部分（按实际存在记录），不强行找反转故事。

## 7. P4：论文可直接使用的交付

### 7.1 表图与文本

至少生成以下独立材料：

| 产物 | 内容 |
|---|---|
| 主性能表（CSV/Markdown/LaTeX片段） | 九方法五预算；正文可用25%摘要，但完整表必须提供；完整扫描100%单列 |
| 配对效果表 | 固定、开环、无VLM、静态、均值全部结果及探索性区间范围 |
| 等质量表 | 全部预定义锚点、经验最小成本及未达到状态；不能只留成功目标 |
| 六域补充表 | 保留原物理试样权重、域差异和负向域 |
| 图稿 | 0–.25的MAE、RMSE、R²阶梯曲线分别成图；等质量图、机制效应图、完整扫描差距图。最多6类新汇总图，PNG＋SVG；原案例图复用 |
| 论文文字材料 | `PAPER_RESULTS_DRAFT.md`、`METHODS_FACT_SHEET.md`、`LIMITATIONS_AND_SCOPE.md`、`CLAIM_EVIDENCE_MATRIX.md`、图表说明 |

曲线坐标写“native-raster acquisition fraction”或准确中文；所有图明确VALID/单seed/成本范围。全输入100%可独立点或独立表格/参照线，禁止画出没有数据的.25→1实线。图的q阈值只是经验MAE目标，不能标为合格线。

正文沿“任务驱动信息获取→同成本质量→反馈机制→等质量采集→完整信息参照”组织，不按历史开发阶段写研发日志。措辞以观察结果为依据：正向证据可突出，VLM/空间有限或负向结果不删除。Methods中既有训练事实只从已执行代码与manifest摘取，不回到历史P4 BC描述。

不需要额外文献检索/投稿模板/完整论文编译；本轮只整理可追溯的结果和方法材料。相关工作与新颖性评价不靠这些结果自动得出。

### 7.2 科学状态保留

`implementation_status`可为 `PAPER_EVIDENCE_ASSEMBLY_COMPLETE`；
`evidence_scope=POSTHOC_FROZEN_VALID_SINGLE_SEED`；
`independent_confirmation=NOT_PERFORMED`；
`noninferiority_status=NO_PRESPECIFIED_MARGIN`；
`new_training_updates=0`、`new_model_forwards=0`、`new_test_access=0`。

本轮任务完成不是论文已获录用/工程达标，也不把原pilot改成正式TEST。无需因这些边界反复提出已经被用户排除的新实验。

## 8. P5：有限验收、资源与GitHub交付

### 8.1 只做八类直接检查

执行配套review。检查：来源与50key连接；repeat/域聚合；左阶梯与边界；完整参考只在1.0；等质量正确最小值/未达到；真实事件与非因果说明；旧输入只读与零前向；最终表图/数值/推送完整。

先运行少量合成数值检查，再构建真实分析一次，最后核对已算结果。失败只重查受影响项。不要重新跑W2/W3审计全套、加载23或44个候选模型、全库pytest、渗透/fuzz/压力测试、反复hash整库。可复用本包独立数值参照；不能改正确预期去迎合生产代码。

CPU最多4线程，GPU和所有新模型前向=0，新增优化/新VLM/新TEST=0。只记录普通分析耗时及新生成文件，不制作硬件性能表；不新增训练额度。原ledger累计使用39,014、上限40,014作为背景记录，不消费其中1,000余额。若ledger在实际HEAD有变化照实记录，不重置。

原inputs启动时计算一次必要文件hash并在结束核对，不重复全量模型或bank hash。输入metadata完整，但程序可不用.pt；保留模型hash引用不等于重新加载模型。原W3/W2的manifest、review、PNG均不得改写。

### 8.2 必须实际交付和推送

生成 `ART/CODEX_HANDOFF_CAI_V3_PAPER_EVIDENCE.md`：本轮实际执行、源/代码提交、新分析定义、真实结果、配对与等质量边界、未完成项、复现命令、资源和写作建议。生成 `RESULT_POINTER.json`指向RUN `final_manifest.json`。

将新模块/脚本、有限测试、本包规范、全部实际派生表/图/文本及零更新账目追加提交到同一v3分支并push。只提交相关文件，不PR、不merge、不force push，不reset用户工作，不重新上传原始数据/六shard/已有模型。核对local/upstream/`git ls-remote`指定引用一致；原分支需保护旧产物。

无新训练任务卡或“待预算批准”作为本轮收尾。真实缺数据则逐项标明，不输出虚构结果，也不把缺一个辅助图变成全部分析阻塞。最后回复：一张交付表＋主要已支持/未支持内容＋具体文件路径＋三方SHA；不要只列测试数。
