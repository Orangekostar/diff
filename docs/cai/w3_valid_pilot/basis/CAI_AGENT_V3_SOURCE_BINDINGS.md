# CAI Agent v3：实际源码依据、复用判断与未核实项

整理日期：2026-09-11。

## 0. 证据边界与版本

- [R] 固定仓库代码/产物的事实。
- [F] 用户提供或此前交付并重新读取的文件。
- [L] 本次核查的公开原始论文、作者仓库。
- [D] 本轮提出的新实验设计；不冒充旧代码或已证明的结果。
- [U] 需Codex在服务器实际核实的路径、数据完整性、性能与运行兼容性。

本次GitHub查询确认`research/cai-vlm-guided-agent-v2`仍指向：
`4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d`。
分支查询：https://api.github.com/repos/Orangekostar/diff/git/ref/heads/research/cai-vlm-guided-agent-v2

本包只生成执行任务与独立数值参照；没有训练、没有改仓库、没有读取服务器GPU，也没有运行新实验。读取源码能够证明实现方式，不能证明运行性能或单一因果根因。

## 1. 当前v2：哪些可以保留，哪些要修

### [R01] 积分与独立运行统计

源码：
- https://github.com/Orangekostar/diff/blob/4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d/src/cmc_bbdm/cai_active_image/statistics.py
- https://github.com/Orangekostar/diff/blob/4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d/src/cmc_bbdm/cai_active_image/pipeline.py
- https://github.com/Orangekostar/diff/blob/4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d/src/cmc_bbdm/learned_cscan/metrics.py

实际代码：`normalized_error_area_mpa`使用`np.trapezoid`；`_aggregate_prediction_rows`先平均seed预测再算误差；`_effect_rows`则先算每episode面积再平均，不应混为同一错误。旧`exact_step_integral`有左阶梯思想，但`StepSnapshot.task_loss`限定0..1，不能直接承载MPa。

[D] 抽取新纯数学函数，配独立手算，不修改旧Task枚举/值域。固定模型重算只修评价，不等于修复训练。

### [R02] v2实际训练

- https://github.com/Orangekostar/diff/blob/4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d/src/cmc_bbdm/cai_active_image/training.py
- https://github.com/Orangekostar/diff/blob/4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d/results/cai_active_image/v2/predictor_training_manifest.json

实际Actor训练用动作后误差，另加1倍终点；批内任意样本无合法动作会终止全批；各折轮换采样。共同预测器实际24件，三折分别拟合12/18/18件；这些是记录数，不是几万状态的独立N。

[D] 恢复明确的左阶梯/终点.25目标，并采用每episode终止；按域/试样采样再选其排除折预测器。小样本和模型能力可能相关，但不是已证明的唯一负结果原因。

### [R03] 当前数据入口不是全276候选入口

- https://github.com/Orangekostar/diff/blob/4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d/src/cmc_bbdm/cai_active_image/features.py
- https://github.com/Orangekostar/diff/blob/4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d/src/cmc_bbdm/learned_cscan/runtime.py

`build_feature_bank()`调用旧`load_study_context()`并遍历旧assignments；它确实先裁64格再编码，可复用这个顺序，但名单不能原样复用为新全队列。它用`MVAEncoderSession`封装冻结ResNet，核对外部执行权重后编码，不能只看repo本地路径就认定模型实际从那里加载。

[D] 新名单薄加载；复用相同独立图格的无标签特征；不重跑旧background prior和MAVIS教师模型。

### [R04] 276候选与真正的来源组入口

- https://github.com/Orangekostar/diff/blob/4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d/results/agentic_task_driven_nde/p0r_author_registration/surface_manifest.csv

实际列包括：`dataset_id`、`specimen_id`、`cscan_source_path/sha256`、`cscan_panel_index`、`registered_cscan_crop_path/sha256/height_px/width_px`、`impacted_surface_path`、`surface_sha256`、`cai_identity`、`p0r_roster_status`。

已读例子有`c8-7and11.jpg`这类多面板原截图，因此按裁图文件名随机分割不够。完整276候选身份有旧P0R交接支持，但本包没有执行全部新MPa有效性连接。

[D] 用来源关系建capture group，再分split。新split不是未触碰确认；历史数据角色变化如实标注。

### [R05] 作者MPa的实际字段来源

- https://github.com/Orangekostar/diff/blob/4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d/src/cmc_bbdm/learned_cscan/hasebe_reference_evidence.py
- https://github.com/Orangekostar/diff/blob/4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d/results/hasebe_reference_evidence/v1/author_measurements.csv
- https://github.com/Orangekostar/diff/blob/4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d/src/cmc_bbdm/cai_active_image/data.py

提取器将CAI工作簿C列映射`CAI_STRENGTH`、`MPa`，保留source_cell/sheet；D列是另一种下降量。实际作者CSV字段含`measurement_semantics/value_numeric/unit_normalized/source_kind/source_sheet/source_cell/source_dataset_version`。v2 loader从OOF表只读取`cai_strength_mpa`标签列，不读取其预测列；需交叉核对来源，不能仅见文件名OOF就认为标签必假。

本次`fetch_file`对大CSV片段曾返回空content，改用contents读取成功；这是读取工具的表现，不是文件为空的证据。

### [R06] VLM和采集循环确实已经存在

- https://github.com/Orangekostar/diff/blob/4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d/src/cmc_bbdm/learned_cscan/perception.py
- https://github.com/Orangekostar/diff/blob/4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d/src/cmc_bbdm/cai_active_image/perception.py
- https://github.com/Orangekostar/diff/blob/4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d/src/cmc_bbdm/cai_active_image/episodes.py
- https://github.com/Orangekostar/diff/blob/4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d/paper_v3/configs/learned_cscan_same_perception.yaml

已有Qwen2.5-VL冻结表面感知、缓存、显示cell映射、最高可靠C0及第二步解除机制。60件缓存不等于276件齐全。VLM只提供表面优先假设，不是CAI信息增益标签。

[D] 本轮保留C0规则以隔离其他因素；不同时改成软约束。缺失感知按相同prompt补齐；不重复采样挑答案。

### [R07] 已有空间Actor是真实可复用代码

- https://github.com/Orangekostar/diff/blob/4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d/src/cmc_bbdm/learned_cscan/policies.py

`LearnedCellActor`含2层TransformerEncoder、4头、128宽、FFN256，接global+64cell，输出合法动作logits。但输入是17维cell、16×10子块、9维global等旧观察，任务为LOCATE/CHARACTERIZE规则克隆。

[D] 复用contextualizer/评分结构并新建图像token适配；不加载其旧BC权重、不伪称输入不变。v3使用该结构仍需实际验证，不保证优于均值模型。

### [R08] v2模型与static反例

- https://github.com/Orangekostar/diff/blob/4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d/src/cmc_bbdm/cai_active_image/models.py

v2是cell MLP＋均值。Actor未测位置局部C-scan是共同未知输入，已测内容主要通过均值和当前预测影响候选；不是完全没有空间信息。static关闭VLM/feedback，但还读表面特征。因此新真正静态必须单独构造成64共享logits。

### [R09] 简单CAI回归可以借用接口，不复制大型依赖层

- https://github.com/Orangekostar/diff/blob/4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d/src/cmc_bbdm/mva/cai_evaluator.py
- https://github.com/Orangekostar/diff/blob/4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d/src/cmc_bbdm/cpb_v3/models.py
- https://github.com/Orangekostar/diff/blob/4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d/src/cmc_bbdm/mva/encoder_session.py

`fit_pca_projection`、`fit_cai_predictor`明确将fit_indices限制在调用者提供的训练集。原CPB候选还可含13 metadata＋21表面统计；旧性能不能当成当前纯图像MPa的成绩。旧层有许多外部typed依赖，不需要为一个Ridge基线把整套验证框架搬来。

## 2. 本次新增查到的历史证据，如何改变开发安排

### [R10] 不只看旧P1，要读后续extended状态

- https://github.com/Orangekostar/diff/blob/4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d/results/cross_stage/V3_FINAL_GATE_STATUS.md
- https://github.com/Orangekostar/diff/blob/4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d/results/cross_stage/V3_EXTENDED_GATE_STATUS.md

固定记录：
- P1：276件、CAI比值、完整信息有增量；不是新MPa协议。
- P3：破坏8×8块空间组织，MAE从0.128489到0.183940；支持该旧预测器对空间组织敏感，不是定位真值。
- P5：旧25%稀疏采样/双线性重建保留约89.90%完整输入增益；其采样几何不等于v3的25%整格，不得直接拼表或免费插值全部未测内容。
- P4：更复杂dense表示未超过旧冻结ResNet基线，不能默认换大模型就有效。
- P6：扩散重建没有机械增益，本轮不重复扩散路线。
- P7：旧surface+sparse融合差于sparse-only，是特定表示/估计器下的结果，不证明表面VLM对规划无价值。

[D] 因此v3增加一个回归侧SPATIAL_C对照，但保留策略侧VLM；先查表示和预测能力，不只盲目更换Actor。这里只查到摘要状态，部分详细P1路径曾返回404，不能声称已读取或复现全部checkpoint。

### [R11] 旧G1不支持简单重建大教师银行

- https://github.com/Orangekostar/diff/blob/4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d/artifacts/inspection_agent_g1/CODEX_HANDOFF_INSPECTION_AGENT_G1.md
- https://github.com/Orangekostar/diff/blob/4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d/src/cmc_bbdm/inspection_agent/oracle.py

G1已经在276件上做软效用蒸馏/DAgger，CAI AUEBC固定0.028595、学习0.029653、特权oracle0.007803。旧单位、测量几何、上下文不同，不能与新MPa数字相加。特权机会不等于可观察策略必然可学。

[D] 不把旧oracle标签当新GT，不重建百万候选；只做限定的新学习方式对照。

### [R12] 旧review的问题确实有源码反例

- https://github.com/Orangekostar/diff/blob/4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d/tests/test_cai_active_image_v2.py
- https://github.com/Orangekostar/diff/blob/4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d/artifacts/cai_active_image/v2/CAI_VLM_GUIDED_AGENT_V2_COMPLETION_AUDIT.md

测试对[190,194,198]/y200/成本[0,.125,.25]期待6；原左阶梯应为8。history测试比较全部logits，不能区分变化是否只发生在非法格子。审计承认作废8000 predictor＋13500 Actor，不能只报告保留run的21500为累计消耗。

[D] 独立golden预期、合法候选通路检查、累计资源日志是必要正确性工作，不是过度安全测试。

## 3. 公开方法：精确复用范围与限制

### [L01] GDFS／dynamic-selection

论文：Covert et al., Learning to Maximize Mutual Information for Dynamic Feature Selection, ICML 2023。
出版页：https://proceedings.mlr.press/v202/covert23a.html
作者仓库：https://github.com/iancovert/dynamic-selection
本次查询main commit：`e2b6f7403fdac4d217ac2ec5dea96acd60240b60`。

固定文件：
- https://github.com/iancovert/dynamic-selection/blob/e2b6f7403fdac4d217ac2ec5dea96acd60240b60/dynamic_selection/models.py
- https://github.com/iancovert/dynamic-selection/blob/e2b6f7403fdac4d217ac2ec5dea96acd60240b60/dynamic_selection/utils.py
- https://github.com/iancovert/dynamic-selection/blob/e2b6f7403fdac4d217ac2ec5dea96acd60240b60/dynamic_selection/greedy.py
- https://github.com/iancovert/dynamic-selection/blob/e2b6f7403fdac4d217ac2ec5dea96acd60240b60/LICENSE

实际读取：MaskingPretrainer随机mask训练；其VALID随机mask有“应预先计算共享”的TODO。GreedyDynamicSelection联合优化selector/predictor，训练soft mask、推进hard mask，部署hard选择。utils有分组/二维mask及ConcreteSelector。

[D] 借用缺失观测训练和选择循环。冻结预测器、VLM、原生cost和一格多维联合采集都是本轮适配，不是原论文已实现的组合。训练soft分支不可冒充真实可见信息；所有评价必须hard。同预测器适配不等于原论文精确复现。MIT文本必须保留。本轮不复制多温度完整epoch循环。

论文的greedy最优/互信息分析不能自动推广为本轮Huber、整轨迹代价、有限数据下的保证。仅作为一个有限训练对照。

### [L02] Set Transformer

论文出版页：https://proceedings.mlr.press/v97/lee19d.html
作者仓库：https://github.com/juho-lee/set_transformer
已读`modules.py`的MAB/SAB/ISAB/PMA及`models.py`组装。源码原接口无padding mask；不能把未测token当有效观测。输入集合无序并不妨碍将位置作为特征，但空间信息要显式提供。

[D] 本轮仅作为原理参考，优先使用[R07]已有Transformer；不额外vendor或训练Set Transformer。因不纳入执行依赖，不为它强行读取所有环境/下载权重。

### [L03] 图像预测CAI的外部依据

Mack et al., Deep learning for predicting impact energy and compression after impact strength of composite materials using C-scan images, AEI 72 (2026) 104518。
DOI：https://doi.org/10.1016/j.aei.2026.104518
出版页：https://www.sciencedirect.com/science/article/pii/S1474034626002107

本次读到摘要及公开节选：ResNet18、C-scan、CAI回归。仅支持图像监督路径有先例；不能据摘要恢复全部划分/训练细节，不拿其R²作为本轮承诺或直接可比baseline。

## 4. [F] 用户材料与范围延续

- 旧`AEI_CSCAN_AGENT_NEXT_SESSION_HANDOFF`明确用户要VLM表面感知＋内部观测＋学习决策；其2102阶段的零训练冻结已被当前用户明确的新开发授权取代，不能拿过期状态阻止本轮。
- Hasebe执行汇报说明作者面积/凹痕与截图可对应24件，未找到作者空间mask/定量数组；当前CAI任务因此保留图像回放边界，不要求再次人工标注。
- 原v1执行稿和v2补丁本次重新读取了数据、公式、网络、对照和验收条款。v3将其明确合并为单个完整规范，避免多个补丁含混覆盖。

## 5. 新设计与必须诚实保留的不确定性

[D] 新split hash、三预测器候选、2%资源筛选、.25终点项、28100累计上限、12小时、有限GDFS pilot，都是本轮资源/方法设计，不是旧结果推导的必然最优选择。

[U] 需Codex实际确认：
1. 276候选的有效MPa配对数、capture group、各split规模与跨域组。
2. 外部根、权重/图像路径、全队列作者CSV/crosswalk是否可读。
3. 未命中VLM的真实数量、单次耗时，能否在资源内补齐。
4. 原生裁图与v2缓存的解码/预处理兼容性，不依据名称推断。
5. 新预测器是否真的在VALID准备好；没有独立证据时不能只因完整图有信息就称强预测。
6. Transformer、GDFS或更多样本是否提升——尚未实验验证。
7. 实际review是否独立上下文、实际推送是否成功，只能按真实日志报告。
