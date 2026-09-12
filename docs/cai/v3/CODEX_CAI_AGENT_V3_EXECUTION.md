# Codex执行任务：复用既有证据的VLM引导CAI主动采集 v3

日期：2026-09-11。仓库：`Orangekostar/diff`。
证据基点：`4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d`。
建议分支：`research/cai-vlm-agent-v3-controlled-reuse`。

## 0. 执行范围、优先级和真实目标

这是新的、完整的执行任务，不是再往旧v1/v2附加一个补丁。就本轮开发而言，本文件是唯一执行规范；`CODEX_CAI_AGENT_V3_REVIEW.md`和`metric_reference.py`是配套验收依据，不能被实现方的配置或测试反向覆盖。旧提示词用于解释历史偏差，不继续叠加过期冻结限制。

用户授权的是有条件的新开发与新训练，不是立即重跑所有实验。必须依次完成：

> 修正v2的评价 → 使用完整候选队列验证CAI预测器 → 固定共同预测器，检验空间反馈Agent → 有条件完成一个GDFS适配pilot → 通过VALID条件才扩种子并进行一次内部TEST评价 → 无论正负都交接并push。

必须保留的用户目标：
1. 冻结VLM从真实表面照片提出优先检查区域，参与首动作；不能用CNN替代这条通路。
2. 学习型Agent每获得一次新的C-scan图像观测，就重新决定下一个位置；不能预先看完整C-scan或执行一张预排路线冒充闭环。
3. 主目标是较小检测投入下更准确预测真实实验CAI强度，单位MPa；不是区域mask自洽、P4动作模仿、CAI比值或“1−MAE准确率”。
4. 本轮仍是8×8整格、原生图像栅格成本的离线采集研究，不是原始超声测点控制或真实扫描时间优化。

本任务不保证效果正向。接口接入、需求验收、科学支持、工程达标必须分开报告。

所有新增设计用[D]标记；源码事实和公开依据分别见来源文件[Rxx]/[Lxx]。新增文件/CLI/状态均待实现，不得在开始前声称已经存在。

### 0.1 不得自动开展

不恢复幅值；不重新标专家ROI；不修改旧标签/划分/结果；不微调VLM或CNN；不建立新大模型平台；不做STOP、三层细化、硬件接口；不重跑旧G1教师银行、P1–P7整套实验、六折LODO或全仓库测试。

### 0.2 分支与文件隔离

- `git fetch`后核对上述基点。若远端已前进，记录差异，仍按用户指定证据基点建新工作树，不静默跟随main，不覆盖用户其他工作。
- 新代码建议 `src/cmc_bbdm/cai_agent_v3/`，入口建议 `scripts/run_cai_agent_v3.py`。
- 新输出 `results/cai_agent_v3/`、`artifacts/cai_agent_v3/`，其中 `legacy_v2_rescore/` 与 `new_protocol/` 分开。
- 旧代码优先import纯函数；硬绑定旧数据/Task/0..1损失的类只借用必要数学或结构，用薄适配实现。不要放松旧类型断言来强塞新任务。
- 不复制大型capability、逐对象hash、安全反篡改框架；保留必要的来源和checkpoint身份即可。

## 1. W0开始前的定向读取与需求绑定

必须实际读取下面的文件/函数，并生成最多约两页的 `IMPLEMENTATION_BINDINGS.md`。每行写：需求ID、现有函数、适用部分、实际新增/修正位置、验收方式。不是只复述本任务。

| 需求ID | 要读的实际文件/函数 | 绑定到本轮的工作 |
|---|---|---|
| C01 | `cai_active_image/statistics.py::normalized_error_area_mpa`；`learned_cscan/metrics.py::exact_step_integral` | 左阶梯积分与尾段。旧StepSnapshot约束损失≤1，不能直接装MPa |
| C02 | `cai_active_image/pipeline.py::_aggregate_prediction_rows/_effect_rows` | 修正先平均预测的集成口径；保留逐seed损失 |
| C03 | `cai_active_image/training.py::train_actor/train_predictor` | 纠正动作前/后误差、终点权重、折采样、不同长度episode同步停止 |
| C04 | `cai_active_image/features.py::build_feature_bank`、`protocol.py` | 绕开旧60件硬绑定；复用兼容的独立图格编码，不把旧名单当全队列 |
| C05 | P0R `surface_manifest.csv`、`learned_cscan/hasebe_reference_evidence.py` | 276候选、作者MPa、来源capture分组 |
| C06 | `learned_cscan/perception.py`、`cai_active_image/perception.py`、实际表面渲染调用链 | 真实VLM、缓存、编号映射、最高可靠C0只作用首步 |
| C07 | `learned_cscan/policies.py::LearnedCellActor` | 复用两层四头128维空间交互结构，不加载旧BC任务权重 |
| C08 | `mva/cai_evaluator.py::fit_pca_projection/fit_cai_predictor`、`mva/encoder_session.py` | 低成本Ridge诊断与已绑定冻结视觉底座 |
| C09 | `V3_EXTENDED_GATE_STATUS.md`、旧G1交接 | 空间破坏、稀疏保持、表面融合及学习策略负结果，不只挑P1正结果 |
| C10 | `cai_active_image/environment.py/episodes.py` | 真正逐次揭示、准确原生像素计费与日志 |
| C11 | `tests/test_cai_active_image_v2.py` | 不继承错误的6.0积分预期、无效动作历史测试或“当前配置自己通过” |
| C12 | 本包review与独立数值文件 | 训练前验收、阶段关口、资源累计与GitHub交付 |

外部项目固定：`iancovert/dynamic-selection@e2b6f7403fdac4d217ac2ec5dea96acd60240b60`。读`models.py::MaskingPretrainer`、`utils.py`的mask层与ConcreteSelector、`greedy.py::GreedyDynamicSelection.fit/forward`及MIT LICENSE。

Set Transformer只作为注意力汇总的设计参考；本轮优先复用已有LearnedCellActor的Transformer，不同时增加第二套注意力骨干。[R07,L01,L02]

## 2. W0：固定v2模型和轨迹，先修正评价，不训练

输入来自 `results/cai_active_image/v2/` 的真实 `trajectories.csv`、`test_episodes.csv`、`per_specimen_predictions.csv`、`config.yaml`与相关manifest。先检查列名，不根据文件名假定seed级信息齐全。

### 2.1 轨迹规范化

将每个 `(specimen_key, method, seed)` 还原成状态序列 `(c_t,p_t)`，必须有真实零C-scan状态和末状态。旧逐动作行包含before/after和prediction_before/prediction_after时，按真实顺序去重连接；检查相邻边界一致，不重复积分。

标签使用v2当时绑定的真实MPa列，以保持“只改统计”的意义。发现与作者来源不一致，先单列差异，不能在同一次统计修正里偷偷换标签。

不得为补表重新前向Actor。原文件缺失必要状态则报告`LEGACY_RESCORE_INCOMPLETE`，保留能完成的部分；不能从集成预测反推出单seed预测。

### 2.2 独立运行统计

先对每个seed、试样计算误差/面积，再按试样平均seed损失。MAE是平均绝对误差；聚合RMSE是独立运行平方误差均值再开方，同时单列各seed RMSE。R²逐seed计算并报告均值，不能先平均预测。

旧集成表保留，标记`ENSEMBLED_PREDICTIONS_UNPRICED`，不得当一次0.25成本运行。该表只作历史解释，不据它论证工程能力。

### 2.3 历史方法与统计边界

- v2 `LEARNED_STATIC`仍消费表面信息的事实写进更正说明，展示别名可用`V2_SURFACE_DEPENDENT_LEARNED_CONTROL`；不改原文件，也不假称无需重训就修成静态。
- v2训练目标仍是其实际实现；新统计不意味着旧模型已经按正确目标训练。
- 用P0R来源表连接v2 TEST的capture group；如果缺组关系，不编造“逐试样独立”，可先输出点估计和明确的CI缺口。
- 输出旧口径/更正口径并列表与结论变化，不预设会从负变正。

产出：`legacy_v2_rescore/{state_predictions.csv,per_run_metrics.csv,paired_effects.csv,old_vs_corrected.csv,scope_deviations.md}`。

## 3. 统一的数学契约：训练、选择checkpoint与正式评价不得各写一套

### 3.1 误差—成本面积

真实误差只由训练损失或评价器计算：`e_t=abs(p_t−y)`，原生累计成本`c_0=0`，终点`c_T≤B`。

```text
A(B) = [Σ(t=0..T−1) (c_(t+1)−c_t) e_t + (B−c_T)e_T] / B
J(B) = A(B) + 0.25 e_T
```

主指标A；0.25终点项仅用于训练，必须在配置和日志中明确。该系数为继承的工程设计，不是物理常数。

早期B=.0625，完整B=.25。报告成本点固定`0,.0625,.125,.1875,.25`，另列完整64格。任意b取最后一个`c_t≤b`的已获得预测，不插值预测，不用未来状态或历史最好值。若只到c_T<B，尾段保持末预测。

实现一个纯数学函数，再给Torch做薄适配；两者对本包固定案例必须一致。不能`np.trapezoid`，也不能重新按稀疏展示点积分。[R01]

### 3.2 策略梯度

使用cost-to-go最小化写法：

```text
d_t = (c_(t+1)−c_t) * e_t / B
terminal = (B−c_T)*e_T/B + .25*e_T
G_t = sum(j=t..T−1,d_j) + terminal
L_actor = mean_episodes[sum_t logπ(a_t|o_t) * detach(G_t−V(o_t))]
L_value = mean_episodes[mean_valid_steps (V(o_t)−detach(G_t))²]
L = L_actor + .5 L_value − β mean_episodes[mean_valid_steps entropy]
```

用TRAIN标签标准差缩放所有代价，不改变MPa报告；`gamma=1`，β从.01线性降到0。损失按episode均衡，不把长轨迹当更多独立试样。

同一批中某件无可容纳动作时只终止该件，不允许`if not all(any_legal): break`提前终止其他试样。终止件的动作loss mask掉，尾段/终点只加入一次。

部署Actor/critic不输入y、真实误差、奖励、未来回报。预测器参数冻结，但其当前输出可作为已测信息的派生量输入反馈Actor。

### 3.3 固定测试参照

必须调用本包`metric_reference.py`或用其独立预期核对实现，不能先看production结果再改预期。例如：
- 成本[0,.125,.25]，预测[190,194,198]，y=200：A=8，J=8.5；不是6。
- 成本[0,.10,.20]，预测[190,196,198]，y=200，B=.25：A=6，J=6.5。
- y=200、两seed预测190/210：独立MAE=10，集成MAE=0。主表必须是前者。

## 4. W1：完整候选队列、标签和缓存

### 4.1 队列与标签

以P0R授权276件为候选，不是保证最终全部有效。读取：
- `results/agentic_task_driven_nde/p0r_author_registration/surface_manifest.csv`；
- `results/hasebe_reference_evidence/v1/author_measurements.csv`；
- 作者试验有效性/纳入排除依据和已保存crosswalk；
- v2 MPa来源 `results/multiview/e1_audit/oof_predictions.csv`只作交叉核对，不能把OOF预测列当真实标签。

取`measurement_semantics=CAI_STRENGTH`、`unit_normalized=MPa`的作者记录，保留source cell/sheet/version。不能拿面积、凹痕、下降率、健康组均值归一化比值代替。精确键为`(dataset_id,specimen_id)`；若CAI源域ID和图像域ID不同，使用已有作者crosswalk，不按数值相近或随意去掉t/astm后缀连接。

主输入只允许真实表面图和冲击后/CAI前C-scan图像；材料域、试样ID、冲击能量、力学结果、完整表面轮廓标量都不进主模型。必要的来源字段留在评价侧。

排除仅遵从作者有效性与真实缺失/重复冲突。不得按模型误差、VLM没线索、图不好看排除。无法取得足够完整队列时交付具体阻塞，不自动退回旧60件冒称完成全队列实验。

### 4.2 新划分[D]

复用队列的内部新协议，名称`INTERNAL_REUSED_COHORT_V3`，不能称未触碰确认集。

- 同物理试样、同原始C-scan截图（含多面板）、相同表面来源图像、重复图像身份构成union-find capture group。
- 仅来源关系建组；从已存在SHA/manifest开始，不重复全盘hash。
- 每域按`sha256("cai-agent-v3|"+group_key)`排序；前floor(.6G) TRAIN，接着floor(.2G) VALID，其余TEST。保证各域三split有组；不满足时标阻塞，不换seed挑结果。
- 跨域组先有限核对来源，不能为了分层拆组；确实跨域且不能满足本次固定分层协议时记录具体缺口并停止新训练，不静默丢弃。
- 全276候选的去留、组数、各split物理N必须输出；不预写精确166/55/55。
- 保留v2成员历史标记。新TEST可能曾在旧研究出现，所有结论均按复用队列范围解释。

源表和cohort建立可检查标签是否存在，模型选择不得使用新TEST的分布/得分。保存TEST标签于评分侧，学习feature bank中为NaN或不包含该列。

### 4.3 图像与VLM

复用P0R已确认表面顺时针90°和显示编号映射；不得重复旋转或将display ID重复反解。优先从已绑定裁图读取C-scan；对复用v2缓存的对象核对解码数组身份和编码器/预处理相同。

- 表面和C-scan分别8×8独立裁图，再冻结ResNet18编码，每格512维。先整图CNN再切特征图是不允许的。
- 复用兼容60件cell特征；仅编码缺失项。标签变化不强制重算无标签特征，图像/预处理变化则必须重算受影响项。
- 为完整输入Ridge诊断可额外编码整张表面及完整C-scan；这套`diagnostic_full_tokens`绝不能成为部署Actor输入。
- VLM必须真实。复用Qwen2.5-VL-7B-Instruct固定revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`及已冻结prompt/schema/render，补齐实际缺失缓存，不因旧配置cap=60限制全队列。
- 优先等预测器准备通过后，再批量补齐新TRAIN/VALID VLM；新TEST感知仅在正式评价前补齐，prompt不改、标签不看。
- 每件一条有效响应；最多一次格式修复，不重采样挑更有利内容。调用失败与no_reliable_cue区分。不可访问时交付预测部分并明确VLM阻塞，不降级主方法。

## 5. W2：先验证CAI预测能力，再训练Actor

本轮有意把“预测器好不好”与“策略好不好”拆开。旧P1、P3/P5正信号是动机，不是新MPa模型已经有效的证明；旧P7表面融合负信号要求检查回归侧是否受表面输入影响。[R10]

### 5.1 必做的便宜参照

在同一新split和MPa标签下，固定拟合：
1. TRAIN均值、TRAIN中位数常量；
2. `RIDGE_SURFACE_FULL`：完整表面冻结embedding；
3. `RIDGE_CSCAN_FULL`：完整C-scan冻结embedding；
4. `RIDGE_SURFACE_CSCAN_FULL`：上述两者拼接；
5. `RIDGE_PARTIAL`：可见图格的均值embedding、已测mask、成本及完整表面embedding。未知C-scan不参与均值；零观测有明确零向量/标记。

PCA/标准化只在各自TRAIN拟合；PCA维数固定min(16,独立TRAIN试样数−1,实际秩)，Ridge alpha=10，无网格搜索。部分Ridge用每件固定32个、覆盖零/部分/完整状态的训练样本，按试样等权。借用`mva/cai_evaluator.py`中的数学与接口思想；外部依赖不全时可用薄numpy/sklearn实现，不复制旧庞大的验证权限系统。

前四项只有完整可见条件下可用，不能被Agent免费调用。旧P1有其他输入/比值端点，以上是新图像/MPa基线，不是假称P1精确复现。

### 5.2 三个有限的神经预测器候选[D]

| 名称 | 输入与结构 | 用途 |
|---|---|---|
| `MEAN_SC` | v2型逐cell MLP/均值汇总，64宽；表面+C-scan | 重训的结构对照，不用旧split权重 |
| `SPATIAL_SC` | 128宽、2层、4头、FFN256的Transformer；表面+C-scan | 原拟议空间表示 |
| `SPATIAL_C` | 与SPATIAL_SC相同，回归器无表面内容 | 检验回归侧表面融合风险；不删除规划器的表面VLM |

SPATIAL模型必须在实际forward使用Transformer，不能仅定义unused模块。每个cell由可见图块、位置、measured标记融合成token，加全局query后回归。未知C-scan在任何全局运算前换成相同mask embedding；SC可保留全表面，C不接收表面内容。权重和PCA均从新TRAIN拟合。

模型间差异是预先指定的表示/输入比较，不宣称纯参数量匹配或更大模型必胜。所有方法保存参数数、实际forward结构、输入开关和选择checkpoint。

### 5.3 缺失模式与拟合

每批先均匀选域再均匀选物理试样，图块/状态不增加独立N。
固定训练混合：10%零；60% K∈1..16（随机、中心、蛇形、几何分散前缀各1/4）；20% K∈17..48随机；10%完整64格。所有mask生成与y和隐藏图像内容无关。

每个模型最多2000 updates，batch32，AdamW lr3e−4、wd1e−4、clip1、Huber delta1（TRAIN标准化CAI），每250步VALID检查；4次无改进可早停。SPATIAL dropout=.1。不因为结果不好自动改变损失或扩模型。

VALID固定前缀库在训练前一次产生，跨checkpoint/模型保持不变，含四种路线、零、部分、完整状态。RANDOM用固定seed，不在每个epoch重新随机验证mask。选择checkpoint使用固定前缀平均A(B)，并单列完整输入MAE/RMSE；不用训练loss选checkpoint。[L01]

### 5.4 共同预测器选择与硬关口

对神经候选分别检查以下预设资源规则[D]，不是工程达标或显著性：
- 完整输入VALID MAE < TRAIN中位数常量MAE，且完整输入MSE < TRAIN均值常量MSE；
- 完整输入MAE相对本模型零C-scan至少改善2%；
- 预定CENTER_FIRST与GEOMETRY_SPREAD两条路线在B=.25的平均MAE至少比本模型零C-scan低；
- 数值、可见性和标签连接验收通过。

从通过者中选固定VALID前缀A最小的一个为`P_all`；1e−8内平局选参数更少者，再按模型名排序。不得在TEST换预测器。若没有通过者：输出`PREDICTOR_NOT_READY`并结束新Actor训练，交接所有结果，不因为流程要完成就开12个Actor。

若便宜Ridge更好而神经模型未准备好，明确报告，但不把不可微Ridge偷偷塞入后续既定梯度接口。需要另一方法设计时由用户再授权。

### 5.5 三折回报预测器

只对选中结构再训练三份TRAIN内capture-group交叉拟合模型。分折使用固定hash后按各域组的试样量确定性均衡分配；组不拆，不能机械逐域rank%3造成一折样本明显集中而不披露。记录每折fit/query的组数、物理N、域数。

只以外部VALID固定库选checkpoint，不以query折选权重。Actor采样按域/物理试样均衡，再根据该试样所属query折路由至排除它的预测器；不能均匀轮流折而使小折试样被过采样。

输出OOF各状态预测与真实误差，检查零/部分/完整的稳定性。三份回报模型也必须具备上面的完整/部分信息方向性条件；不通过标`REWARD_MODELS_NOT_READY`，不扩大策略。训练与部署预测器不同是交叉拟合设计，记录差异，不把它解释成泛化保证。

## 6. W3：共同预测器固定后的VLM＋空间反馈Agent

### 6.1 保持不变的采集协议

复用`NATIVE_8X8_FULL_CELL_V1`：格子互不重叠，一步购买未测格的原生像素；B=.25；不能免费warm start，不超过B，不允许重复采集。费用用整数原生计数计算再归一化，不能由resize后的224图计费。

每步真实调用Actor，执行后更新观察和CAI预测。日志含cell、box、before/after cost、visible mask、VLM cache key、C0、actor call index、预测、模型hash。y/误差只在评分侧join。

### 6.2 VLM起始与后续自由选择

主方法使用v2已修正规则：最高medium/high等级候选的并集C0只限制首步；Actor在C0中选择具体位置。只有low/unknown或真实无可靠线索则全合法集合，仍传入弱先验。预算放不下C0时记录fallback。

从第二步起全部未测且可负担格子开放。VLM只需每件一次感知；Agent必须每步决策。保留cue/alternative解释文本，但不声称全文语义进入Actor。

为了隔离空间反馈，这轮不同时把C0改成软先验；若VLM无收益，按负结果报告，后续是否改起始机制另议。[R06]

### 6.3 主Actor和结构对照

- `VLM_SPATIAL_FEEDBACK`：复用旧LearnedCellActor的2层/4头/128宽contextualizer及每cell评分结构。新token输入为表面512、仅可见C-scan512或mask token、二维坐标、measured、已执行次序/64、VLM indicator/confidence。全局token含成本、剩余成本、VLM可用性与当前预测（detach/TRAIN尺度）。
- 修改的是输入适配，不把512维塞进旧17维typed ObservationPacket。构造新Actor，不加载旧P4/LOCATE/CHARACTERIZE权重。
- `VLM_MEAN_FEEDBACK`：按v2均值反馈结构重训一个seed，输入权限、C0、共同预测器、成本和目标与主Actor一致。明确参数数不同，因此是结构比较，不作严格容量匹配因果宣称。
- 历史/坐标应进入可执行动作的上下文，不接受“只有已测、已非法格子分数变化”的伪验证。是否在真实数据利用反馈，仍以任务效果和诊断为准，不强制每次扰动argmax都变。

### 6.4 必做对照矩阵

| 方法 | 决策可见信息 | 训练/评价 |
|---|---|---|
| `VLM_SPATIAL_FEEDBACK` | 表面+VLM+新增C-scan | 主方法，先seed1 |
| `NO_VLM_SPATIAL_FEEDBACK` | 保留表面CNN+C-scan，无VLM/C0 | 独立从头CAI任务训练 |
| `VLM_SPATIAL_OPEN_LOOP` | 表面+VLM+几何，不读C-scan/其latent/当前预测 | 同C0，从头CAI任务训练 |
| `LEARNED_STATIC_TRUE` | 仅一组全试样共享64个可训练位置logits；不读任何图像/VLM/预测 | 采样无放回训练，推理排序固定 |
| `CENTER_FIRST/GEOMETRY_SPREAD/SERPENTINE/RANDOM` | 既定几何/固定随机排序 | 无学习；RANDOM固定5组，不增物理N |
| `VLM_MEAN_FEEDBACK` | 同主方法，均值汇总 | 仅一seed结构诊断，不作为额外主检验 |

每种策略均用同一`P_all`评价。即便选择`SPATIAL_C`为共同预测器，Actor仍保留表面VLM与表面图，只有回归分支不直接使用表面；这是一项明确允许的分工，不是取消VLM。

`BEST_NONADAPTIVE`只由VALID在四固定方法与`LEARNED_STATIC_TRUE`中选，不在TEST选最弱对手。无反馈Actor不得经预测器侧路读入C-scan，但其结果预测器使用实际采得图像。

### 6.5 pilot训练和继续条件

主/无VLM/开环各seed1最多1250 updates；静态seed1最多750；均值诊断seed1最多1250。batch16，其他优化参数按第3节。每250步评固定VALID，patience4，不因到达上限而声称收敛。

所有pilot完成后，只基于VALID决定扩展：主方法A相对BEST_NONADAPTIVE至少降低2%，且低于VLM_OPEN_LOOP；早期主−无VLM比较照实报告但不作为追加seed搜索条件。结构诊断效果不正向时不能偷偷换主网络为更有利版本。

不通过标`POLICY_PILOT_NOT_SUPPORTED`：不扩seed、不打开新TEST。仍可按W4做唯一预授权的GDFS学习方式诊断，然后交付；GDFS变好也不能自动取代主方法或打开TEST。

## 7. W4：只做一个GDFS适配pilot，不复制整套外部平台

### 7.1 已授权的复用范围

W2可借鉴`MaskingPretrainer`的缺失观测思想，但本规范的固定mask库与采样规则优先。当前预测器/三折准备通过且W3 pilot完成后，默认允许一个`GDFS_ADAPTED_FROZEN_PREDICTOR_S1`，最多1250 updates。若预测器未准备好，GDFS实现/有限检查可交付，但不得训练。

从固定commit提取必要mask/Concrete代码并保留MIT声明；不要完整复制datasets、notebooks、Lightning训练器。Set Transformer本轮不另建训练分支。

### 7.2 适配必须说清楚

原GDFS `fit()`会同时优化selector和predictor，还使用软mask辅助梯度。这里为了同预测器比较：
- 选择器采用主Actor空间结构，继续接入真实VLM/C0；冻结当前query对应的OOF预测器参数。
- 一格512维为同一组，mask形状[B,64,1]；禁止让网络单独购买某维embedding。
- 选择器只看当前hard-visible状态。训练代理分支可用TRAIN完整隐藏token计算软mask后的预测损失；这属于明确记录的训练松弛，不是可部署观测。
- 候选logits先屏蔽已测/超预算/C0外位置，Concrete温度预定从1到.1几何退火；不执行原项目多温度×多epoch的大矩阵。
- predictor需单独支持训练专用soft-mask混合入口，参数仍冻结但允许对输入求导；不能在`no_grad`里将选择器梯度切断。
- 下一步hard状态只揭示实际选中的一个合法格子。VALID/TEST一律hard acquisition，禁止低透明度全图、软mask或完整图像latent进入部署。
- 原GDFS以逐步预测损失训练；本pilot记录为“冻结预测器、VLM和成本适配的贪心松弛对照”，不声称精确复现原论文，不声称和REINFORCE只有优化器不同，也不继承CMI/Bayes最优保证。

选择器训练损失使用标准化CAI的逐步Huber平均；评价仍用第3节统一的真实A/终点指标。该训练目标差别必须写入method表，不能暗称整轨迹REINFORCE等价。

只报告同协议VALID学习方式比较及软/硬差距。该pilot不加入三项正式主比较，不自动扩成三seed。梯度入口无法正确实现时标`GDFS_ADAPTER_INCOMPLETE`，不要用重复RL冒名替代。

## 8. W5：条件扩种子、一次正式内部评价

只有W3既定pilot门槛通过、所有核心review无阻塞且资源够，才补主/无VLM/开环seed2/3各1250与静态seed2/3各750。固定seed面板1/2/3；不挑好seed，不让GDFS pilot或新TEST改变主方法。

扩展后再次读取VALID只用于锁定各checkpoint和BEST_NONADAPTIVE；不能按三seed均值好坏重选协议。新TEST模型全部锁定后评价一次。若资源不足保留已完成模型，状态`RESOURCE_LIMITED`，不提前用部分seed作完整正式宣称。

### 8.1 三项主比较

正向均为对照误差−主方法误差：
1. 完整A：BEST_NONADAPTIVE减主方法；
2. 完整A：VLM_OPEN_LOOP减主方法；
3. 早期A：NO_VLM_FEEDBACK减主方法。

三项各98.333333%双侧区间，5000次同一组预生成的重采样，控制总体.05。固定六域分层，不重采样域ID；域内重采capture group并带入全组试样，组重复次数作为样本权重，然后按物理试样平均、六域等权。避免“大组和小组各算一个样本”。无法确定的来源组不伪造。

固定区间只说明本复用队列下精度，不证明外部泛化；状态/seed/mask不增加N。

### 8.2 绝对预测、过程与图稿

输出零、预设成本点、终点、完整64格的逐seed MAE/RMSE/R²、聚合独立运行误差、各域MAE；另列TRAIN常量、完整Ridge参照。没有容许MPa误差，保持`ENGINEERING_ACCURACY_CRITERION_NOT_SPECIFIED`。

报告实际累计像素、动作数、unused_budget，及VLM/CNN/Actor/预测器计算延迟；不将cache hit或像素比例解释成真实设备时间。

最多三个预选案例：在新TEST每域按固定hash取候选，再取前3域；如停在VALID，则同规则使用VALID。不得按效果选。导出单张PNG：表面+VLM、起始、固定k=1/4/8/末的已测C-scan与下一动作、CAI预测—成本图。完整C-scan仅作明确的事后诊断图；不能生成虚假测量或重新摆放真实动作。

## 9. W6：review与需求验收，必须在长训练前有一次

执行配套`CODEX_CAI_AGENT_V3_REVIEW.md`。核心要求是“原需求→代码→独立数值/行为检查”，不是Ruff+测试数+自写配置一致就宣布通过。

三个检查时点：
- A：W0/W1完成、长训练前，核对指标、数据范围、可见性、C0、真实静态及阶段控制；
- B：W2结束，核对真实预测能力条件与OOF，再允许Actor；
- C：正式结果汇总前，核对seed、cost、统计、累计资源和来源。

若运行环境支持新上下文审查Agent，使用配套审查提示词，审查者不能改需求、golden预期或production以制造PASS。若无此能力，仍执行独立数值参照与逐条源码核对，如实标`SELF_CHECK_WITH_EXTERNAL_NUMERIC_ORACLE`，不伪称独立人工/Agent审查。

任何核心阻塞都不得开始其后的长训练。修复后仅重查受影响项。必须以实际命令输出和代码位置关闭问题，不能仅写“已修复”。不要求生成巨型审计册。

### 9.1 状态必须拆开

`implementation_status`、`protocol_conformance`、`scientific_evidence`、`engineering_status`分别保存。允许“开发完成，pilot不支持”；不允许“46 tests pass，所以任务完整通过”。

### 9.2 变更管理

上述指标公式、队列范围/分组、输入权限、主方法结构、真实静态、三项主比较、阶段规则是硬契约。路径/批处理/缓存等非科学优化可做并记录。改变硬契约须在训练前列明偏差并得到用户确认，不能先改配置再让测试跟着通过。

## 10. 资源：累计计数，失败/作废也算

[D] 本轮新计算最多一张现有GPU、CPU≤4进程，累计12 GPU小时；上限不是必须用满。该轮新增优化更新最高28,100次：

| 阶段 | 更新上限 |
|---|---:|
| 三个神经预测器候选 | 3×2000=6000 |
| 选中结构的三折回报预测器 | 3×2000=6000 |
| 主/无VLM/开环三seed | 3×3×1250=11250 |
| 真实静态三seed | 3×750=2250 |
| 均值Actor一个seed | 1250 |
| GDFS适配一个seed | 1250 |
| 小型预检优化步骤预留 | 100 |
| 合计 | 28100 |

相比旧21,500，上限明确新增预测器/结构/学习方式的有界对照，并非偷偷延长相同实验。多数情况下会因早停/阶段关口少于上限；不得并行启动全表再判断可行性。

Ridge为有限CPU拟合，不记为神经optimizer updates，但记录拟合次数和时间。没有足够时间时，保留未完成，不用删VLM、缩成60或降低验收要求救进度。

一份简单JSONL追加记录job、开始/结束、设备、实际updates、缓存复用、失败/作废/续跑。所有新训练含失败均计数；旧已完成v2消耗只在历史披露，不重复算入本轮也不重复执行。挂起续跑保留optimizer/RNG；修复数学或科学输入导致旧checkpoint无效，必须据实注销并计入资源，剩余额度不够则停。

不做全库pytest、渗透、fuzz、压力、多浏览器、多框架矩阵；不重复九工作簿审计、全盘hash、模型库搜索。只保留约12类核心检查，参数化后的测试数不是目标。旧图/旧模型只读。

## 11. W7：必须交付并实际推送GitHub

至少交付：
- 可运行代码/CLI、冻结配置、来源与版本记录、第三方许可证；
- W0更正表；新候选/排除/分组清单、特征bank索引；
- 预测器比较、OOF准备结果；实际训练到的Actor/对照及checkpoint manifest；
- 已到阶段的CSV/JSON、单张图稿；未执行阶段的原因，不造占位成绩；
- `CODEX_HANDOFF_CAI_AGENT_V3.md`、`RESULTS_AND_CLAIM_BOUNDARIES.md`、`REQUIREMENTS_REVIEW.md`、`compute_ledger.jsonl`。

可用性与完成度检查：新增CLI `--help`真实可运行；复现命令明确哪些只汇总、哪些会训练；`summarize`不得隐式重训；原目录diff为空或只列明确批准变化。

实际执行`git add/commit/push -u`到新分支；不要只打印待用户执行的命令。核对本地HEAD、tracking分支与`git ls-remote`指定远端引用一致；不PR、不merge、不force push、不reset用户工作。失败明确报告，不谎称上传。

代码、文本、表格、小图和必要轻量模型必须上传。大feature bank拆成每文件<50MiB的NPZ shards；使用仓库既有LFS/存储方式时核对可取得性，不能只上传不可用的绝对路径。原始数据不重复上传，遵守已有许可；索引记录实际位置、来源和重建方式。

交接正文写已有的协议/训练/结果来源提交；最终交付commit SHA在最后回复报告，避免“为了把文件自身SHA写进它本身”循环提交。

## 12. 最终回复格式

只需一张阶段表和关键结果：

| 阶段 | 是否实际执行 | 真实结果/阻塞 | 文件 |
|---|---|---|---|
| v2评价修正 | | | |
| 新队列与分组 | | | |
| 预测器及回报模型 | | | |
| VLM＋空间Agent pilot | | | |
| GDFS适配pilot | | | |
| 条件三seed/内部TEST | | | |
| review/累计资源/GitHub | | | |

明确：VLM是否真实接入、每步是否实际Agent决策、CAI绝对能力、早期VLM贡献、反馈贡献、相对固定排序贡献。支持与不支持逐项写；别把代码交付完成解释成研究已成功。
