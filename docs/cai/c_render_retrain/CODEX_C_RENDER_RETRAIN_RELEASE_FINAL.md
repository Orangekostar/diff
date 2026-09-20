# Codex最终执行指令：C配置全量接入、受影响Actor重训、选模、论文证据更新与GitHub发布

**任务ID：`CAI_V3_C_RENDER_RETRAIN_RELEASE_R1_331f5295`**  
**依据版本：`331f52952b93bd9442ad2e44243cc71c7d966b4d`**  
**仓库：`Orangekostar/diff`**  
**分支：`research/cai-vlm-agent-v3-controlled-reuse`**

这是一份自包含的实际开发、运行和交付授权，不是仅供讨论的研究建议。不需要拼接历史提示词。完成目标是：**固定C=P0原提示词+R1清晰编号图，对211件TRAIN/VALID建立同一C先验，重训三个VLM相关Actor，完成规定选模，合并可复用对照，生成C版本论文证据、更新稿件并实际推送GitHub。** 不得完成缓存后就结束，也不得只返回计划或PASS清单。

## 0. 决策、范围与完成含义

### 0.1 已确定的决策

用户已目视认可C，并明确要求它替换原论文路线后重做受影响实验。将此记录为**用户选择C作为下一开发版本**，不是替用户编造六件、所有区域或全队列的人评准确率。无需等待额外人工标注才能继续本任务。

本轮采用C，不在A/B/C/D之间重新选赢家。C的定义保持为**P0+R1及已执行pilot的解析/一次格式修复合同**；不加入P1文字，不改置信数值，不取消或软化C0，不改训练目标，不增加新网络。定位图不是内部attention。

**“发布”指向原GitHub分支提交代码、模型、数据产物和作者审阅稿，并给出可复现入口。不是向AEI自动投稿，不创建GitHub Release/tag，不改变仓库可见性，不代表统计显著或投稿就绪。**

### 0.2 一次执行完成的阶段

| 阶段 | 必做工作 | 不得误当完成的替代物 |
|---|---|---|
| W0 | 绑定版本、冻结C定义、211件名单与资源；实现新入口 | 只写TASK.md |
| W1 | 建立211件C先验、终态和可直接加载的CSV | 只更新6个看过的案例 |
| W2 | 三Actor从原定seed初始化训练，完整归档与选模 | 旧Actor换新先验直接测一次 |
| W3 | 150条新最终轨迹+500条可复用对照，形成650条主矩阵 | 把旧VLM轨迹改名为C |
| W4 | 全部同成本、配对区间、等质量、时机分析和案例重出 | 仅更新主表或复制旧区间 |
| W5 | 新C稿件、MD/TeX/HTML、主稿PDF/SI PDF与发布索引 | 方法已改，摘要/图注仍用旧结果 |
| W6 | 有限验收、交接、实际commit/push、远端文件核对 | 只写“建议上传” |

只根据**工程与数据完整性**决定能否进入下一阶段，不以效果必须优于无VLM、C0必须覆盖某些格子、置信区间必须为正作为开发通过条件。效果不佳时仍完成三个策略、证据、真实稿件和GitHub交付，禁止追加调参直到变正。

### 0.3 明确不做

- 不重训W2共同预测器/三折回报模型，不重新编码CNN特征，不微调或更换Qwen。
- 不重训无VLM、学习固定排序或固定路线；通过第6节复用条件后复用其结果。
- 不扩seed2/3，不做GDFS、STOP、旧BC、LOCATE/CHARACTERIZE、专家mask或新采样算法。
- **本轮不运行65件TEST的VLM/Actor/预测器，不接入其标签，不把重做VALID称为独立测试。** 旧feature bank加载器可能整体读入已存在且TEST标签为NaN的分片；允许只读载入这种既有容器和划分元数据，但不得索引TEST参与拟合、选模、前向、图稿或评价。
- 不提取新attention，不做提示词/字体/分辨率/温度搜索，不手工纠正个别格号。
- 不重建文献库、不做硬件性能benchmark、不全库pytest/安全扫描/压力测试/反复全量哈希。
- 不覆写旧C-pilot、旧A结果和原稿，不将旧模型仅修改manifest当作新训练。

## 1. 已核实依据与需要开发的内容

以下事实来自固定提交的实际文件，文末S编号给出永久路径。**表中“本轮动作”是新开发要求，不声称已经实现。**

| 依据 | 已核实事实 | 本轮动作 |
|---|---|---|
| S1–S3 | C已有`P0/R1`生成记录；R1由`render_readable(clean)`产生。原clean先旋转后缩放，R1只加标记 | 精确复用R1实现及字体身份，不重新设计“更好看的”R1 |
| S2 | `parse_contract`还检查跨region重复格和empty/no-cue一致性；历史production parser没有完全执行相同检查 | 将该合同作为C管线的一部分冻结，保留首次及repair结果；不要将所有变化都叫作纯字号效应 |
| S4 | `SurfacePerceptCache.resolve()`内部写死P0；production VLM入口写旧`new_protocol/`并依赖旧gate | 不调用这些有写入副作用的旧入口；新runner显式传入实际P0文本和C图像 |
| S5–S7 | `_VLMFeatures`接受路径但旧W3调用硬编码旧CSV；旧W3绑定205 available/6 unavailable及40014/5750额度 | 明确数据根、W2根、C先验根、输出根；计数改为真实C结果，额度在实际执行入口生效 |
| S5、S6 | Actor输入VLM特征，训练目标是CAI轨迹代价；MEAN_SC不接收VLM；无VLM路径将VLM通道置零 | 重训三个VLM相关策略，复用W2及六个不依赖VLM的对照 |
| S6–S8 | `seeded_actor`已在初始化前设seed；ActorArchive每250步保存权重/同次轨迹；latest状态有optimizer/RNG，但旧入口拒绝不完整任务重开 | 不再修一遍seed；增加新任务的有界恢复加载能力，不把“保存过状态”当成已有resume |
| S9–S11 | 统计和时机脚本多处硬编码旧W3/旧evidence目录；原始统计区分域等权A与试样池化MAE | 只做路径/版本薄适配，复用数学，重算依赖新轨迹的量 |
| S12 | 全输入参考是同一MEAN_SC@1750及50件VALID已存NPZ | 复用且核对，不新增全扫描前向 |
| S13 | 稿件由abstract、六份sections和declarations生成；只改manuscript.md会被覆盖 | 从真正源文件修改，再重建所有输出 |
| S14 | 最近W3记录累计使用39014/40014，后续两轮VLM诊断训练为0 | 运行时核对一次历史账目，本轮使用独立增量授权，不借用W2余款 |

附带的9月9日BC交接属于旧方法；其中24件TEST、192动作、P4示范、STOP和PROXY端点均**不适用于本轮CAI v3**。本轮不存在“必须先拿到作者损伤mask”的前置条件。

## 2. 路径与版本绑定

由当前工作树的`git rev-parse --show-toplevel`确定ROOT，不硬编码`/home/ww/diff`为执行根。以下均相对ROOT：

```text
DATA        = results/cai_agent_v3/new_protocol
W2          = results/cai_agent_v3/w2_replay/r1_292b1c74
OLD_W3      = results/cai_agent_v3/w3_pilot/r1_0e11452a
OLD_W3_ART  = artifacts/cai_agent_v3/w3_pilot/r1_0e11452a
OLD_EVID    = results/cai_agent_v3/paper_evidence/r1_e2a11154
GROUNDING   = results/cai_agent_v3/vlm_grounding_pilot/r1_e8d9ef01
OLD_PAPER   = paper_cai_aei/r1_84bea60e

CODE        = scripts/cai_c_retrain
OUT         = results/cai_agent_v3/c_render_retrain/r1_331f5295
ART         = artifacts/cai_agent_v3/c_render_retrain/r1_331f5295
C_PRIOR     = OUT/vlm
C_W3        = OUT/w3
C_EVID      = OUT/evidence
PAPER       = paper_cai_aei/r2_c_331f5295
AUTH        = docs/cai/c_render_retrain/C_RETRAIN_RELEASE_SCOPE.json
```

OUT/ART/CODE/PAPER/AUTH为**本任务要创建的路径**。保留OLD_*字节与Git历史。允许创建本轮任务卡；不清除用户其他未提交改动。若HEAD已在基点之后，读实际增量并沿用兼容成果，不reset、不切回旧SHA、不追随main。发生同目录其他新任务冲突时停止该写入，不另起一个“最新”目录静默绕过。

### 2.1 最短读取顺序

1. S1交接、`GROUNDING/experiment_lock.json`、六个`C_P0_R1/state.json`及原输入身份；确认C不是D。
2. `scripts/vlm_grounding_pilot/{common,prepare,run}.py`，尤其渲染、真实prompt传入、签名、一次repair。
3. `W2/{predictor_gate.json,oof_readiness.json,oof_fold_manifest.csv,input_reuse_manifest.json}`；S12全输入指针。
4. S5–S8相关函数、`OLD_W3/protocol_snapshot.json`、`actor_manifests.json`、`initialization_summary.csv`；旧W3授权只作科学参数参照。
5. S9–S11、原论文`EVIDENCE_MAP.csv`及S13构建源。
6. 当前`compute_ledger.jsonl`与执行环境。旧额度和旧task_id不是本任务权限；本文件覆盖本轮范围内的旧“零训练”“不允许新VLM”限制。

旧环境曾存在另一份editable `cmc_bbdm`。在任何真实计算前记录实际模块`__file__`并确认属于当前ROOT。不要为了导入几段函数升级整个环境。Qwen使用上次成功的环境；Actor优先使用原W3训练环境，记录两者解释器、库版本与差异，禁止将差异隐去。W0创建`runtime_lock.json`，明确`vlm_python`、`actor_python`和`report_python`的本机绝对可执行路径、实际库版本以及源码位置；相同环境可指向同一路径。`all`须用这些解释器启动独立子进程，不能假定一个Python同时兼容所有阶段，更不能为此升级现有环境。

W0锁定科学参数后先完成新代码的有限接线检查；正式训练前提交可追溯的实现版本，或记录训练所用文件的内容hash及基点。不同阶段的代码身份分别保存。写作/图稿文件后来变更不会使已完成训练失效；改变训练所用数学、模型、采样或输入签名才触发不兼容处理。

## 3. 冻结C：精确输入与解析合同

### 3.1 不可改变的定义

| 项目 | 冻结值 |
|---|---|
| 逻辑版本 | `C_P0_R1_GLOBAL_V1` |
| 模型 | `Qwen/Qwen2.5-VL-7B-Instruct` |
| revision | `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| prompt | pilot实际`prompts/P0.txt`，逐字节；与生产`SURFACE_PERCEPT_PROMPT`核对 |
| schema/repair | 原schema、原repair文本、pilot `parse_contract`；最多一次格式修复 |
| 图像顺序 | clean在前、R1编号图在后；同方向同范围 |
| clean | 原surface RGB→`ROTATE_270`一次→最大边1024 LANCZOS；不做第二次旋转 |
| R1 | 精确复用pilot `prepare.py::grid_only/render_readable`；不得手工移标签 |
| 模型状态 | eval、requires_grad=False、BF16、SDPA |
| processor | use_fast=False、min_pixels=200704、max_pixels=1003520 |
| generate | batch=1、do_sample=False、temperature=None、use_cache=True、max_new_tokens=500 |
| C0 | 复用`vlm_first_action_mask`，最高medium/high且合法可负担；只限制首步 |

不要新增system指令、`add_vision_id`、新schema字段或bbox，不将当前“23/31”“27/28”、试样ID、旧候选、CAI标签、attention或人工圈图输入模型。

R1的已执行实现使用DejaVu Sans，字号由`max(12, round(24*min_cell_side/128))`起始，inset为`max(3, round(10*min_cell_side/128))`；用textbbox校正字框，底色(0,0,0,170)，白字(255,255,255,230)，必要时按几何统一减小字号。**这些公式是帮助核对，不替代读取原函数和锁文件。** 字体身份不同或重渲染六件的R1像素hash不同，先恢复相同渲染依赖，不能仍称exact C。只记录字体名/hash，不提交字体文件。

### 3.2 新代码与复用方式

实现独立薄入口，建议模块职责：

```text
scripts/cai_c_retrain/
  cli.py             # 单一任务入口，阶段依赖和状态
  prepare.py         # C配置、211件输入、依赖/资源锁
  vlm.py             # 全量C先验与签名缓存；非旧production写入口
  train.py           # 三策略训练、选模、有限resume
  context.py         # 新路径、额度、checkpoint/ledger接口
  assemble.py        # 150新+500复用；另存A/C比较输入
  evidence.py        # 路径适配后的原统计与事件输出
  paper.py           # 新稿件与结果引用更新
  validate.py        # 有限定向检查
```

可合并相邻模块，但必须实现后文统一CLI。优先导入原纯函数；如果pilot函数因`common`或ROOT全局副作用不能直接导入，可复制**已核实的纯渲染/解析/生成封装**到新目录并记录来源、逐字节输入输出等价性，不修改旧文件，不全局monkeypatch旧目录常量。

训练核心复用`seeded_actor`、`_sample_specimens`、`_training_rollout_loss`、`evaluate_policy`、`_load_oof_predictors`和原metrics。不要调用旧`w3_pilot.main`后篡改常量。新训练壳可从`_train_actor`定向移植以接入恢复与新路径，但损失、采样、优化、验证顺序保持一致；用小范围diff复核，不复制一整套研究框架。

## 4. W0–W1：211件C先验

### 4.1 队列和数据权限

严格沿用161 TRAIN+50 VALID，按`feature_bank_index.csv`原索引顺序生成。每个`specimen_key=dataset_id:specimen_id`唯一。先在元数据中筛选split，VLM阶段只取表面路径/hash、key、split、组身份、已登记尺寸；不向VLM传文件名、材料域、真实CAI或C-scan。

本轮C配置不改变CNN的clean或内部图，因此不得将R1编号图送入ResNet或重建feature bank。原W2模型和三折划分完全保持。

外部数据根从`feature_bank_manifest.json::encoder_execution_root`定位；遇到软链接/挂载差异可解析现有等价位置并核对源hash，禁止凭相似文件名替换缺失试样。

### 4.2 复用6件C，其余生成

逐件比较pilot C的模型revision、P0文本、repair合同、clean/R1像素hash、font/render配置、processor/chat/generate参数及实际环境。逻辑缓存键可以加入新task_id，但语义复用检查必须比较真实签名，不靠文件夹名字。

- 完整相符的6件：导入其最终状态和全部原始attempt来源，**包括已有修复的C结果**；标`reused_from_pilot=true`、本轮新生成0。不再次调用以“复核质量”。
- 某件签名不相符：不得混合；记录不复用理由，在本轮统一锁定配置下最多一次主生成+一次格式修复。不得因候选不好而拒绝复用。
- 正常路径：6件复用、205件新主生成；最坏无复用：211件新主生成。新生成总尝试上限422。
- 原R0六件不可用状态不迁移为C的失败状态；它们有新输入身份，应按本轮合同正常尝试。所有试样统一C，不允许失败时暗退R0。

保存`C_PRIOR/input_manifest.csv`、`config_lock.json`、`runs/<key>/state.json`、`raw_1.txt`、必要`raw_2.txt`、真实prompt/chat、输入和生成token IDs、尝试计数与时间。每个调用签名包括两张实际输入图hash。新主任务计数指唯一case的首次生成机会；中断后的同输入重试另记`INTERRUPTED_RETRY`，占该case仅剩的第二次总机会，不能另获第三次repair。生成尝试、主任务数与重试数分栏。

输入存档分为运行侧和Git交付侧：运行侧保存或按相同源文件/渲染器可靠重建211件的实际clean/R1，保存生成前图像hash和配置；Git交付必须含211件原始回答、签名、64格特征、终态和轻量缩略图，以及六个pilot匹配案例的原尺寸输入入口。已在仓库中的输入直接引用；不重复上传211份原始3357像素照片或两套clean。其他原尺寸图可通过明确的零模型`export-inputs --specimen-key`命令复现；HTML不能链接到未交付的本地绝对路径。

### 4.3 不把失败、无线索和中断混成一类

| 终态 | available/no_reliable | 后续处理 |
|---|---|---|
| 合同有效且有regions | True/False | 按原_features编码 |
| 合同有效、空regions且no_reliable=true | True/True | 零indicator/confidence，不修复 |
| 两次后仍格式/唯一性/一致性不合法 | False/False | 零特征+明确`SCHEMA_INVALID_AFTER_ONE_REPAIR`，保留试样，C0按原规则回退 |
| 资源耗尽、源图缺失、模型运行异常、状态STARTED但无raw | 非正常完成，不冒充无可靠线索 | 标INCOMPLETE，停止依赖未完成cache的训练；保留其他已完成结果 |

跨region重复格要按pilot合同触发一次repair；**同时保存`parser_valid`与`contract_valid`，报告修复依赖**。不得手动去重/删语义不喜欢的格，或重新生成直到落在作者认可的位置。不要把更高置信、更少格子或更大C0说成定位更好。

模型调用前写STARTED，调用后先原子保存raw及token，再解析。已有终态绝不重发。raw已存而解析/汇总中断，只离线继续；有原始格式不合法回答且第二次机会尚未使用，继续同一case的一次原repair即可。状态恢复不得删除既有STARTED/attempt记录。调用未留raw的中断保守计一次尝试；只有仍有每件最多两次总机会且尚未用过恢复机会时，允许一次同输入重试，不新增第三次。跨次请求不累计对话历史；每次repair只带本件本配置的原回答。

### 4.4 明确的消费接口

输出`C_PRIOR/vlm_actor_features_fit.csv`，必须恰好211行且无重复key，列至少为：

```text
specimen_key,dataset_id,split,cache_key,vlm_available,no_reliable_cue,
region_count,region_indicator,confidence,failure_reason,prior_version,
clean_image_sha256,gridded_image_sha256,reused_from_pilot
```

`region_indicator`和`confidence`各为64个分号分隔float32值；unknown/low/medium/high仍为0、1/3、2/3、1。CSV布尔字符串须为原loader接受的`True`/`False`，不得输出小写导致静默不可用。解码语义用原`_features`。

输出`C_PRIOR/vlm_manifest_fit.json`，记录211个终态、实际available/no-cue/schema失败数量、字段/文件hash、生成/repair/forward计数、pilot复用数量以及P0/R1身份。禁止固定期待205/6。

所有211行有正常终态（包括明确schema失败）才锁定`C_PRIOR_COMPLETE`。无人工参考不阻塞。全体有终态但可用率或可靠候选很低，仍如实报告，不能调prompt“修到通过”；若模型根本无法执行造成INCOMPLETE，不启动假装具备C先验的训练。

## 5. W2：三个Actor重训、选模与恢复

### 5.1 四条独立依赖链

必须显式传递`data_root`、`predictor_root`、`vlm_manifest_path`和`output_root`。不得为满足硬编码而把C文件复制覆盖旧DATA目录。每次`_VLMFeatures`都使用新C CSV。

W2仍绑定`w2_replay/r1_292b1c74`。共同模型为MEAN_SC@1750，当前权重SHA256为：

```text
f67e912bd5faa26ae5cff8a9a0241439797fccef8cec825f43ebe5dc9f9d57ff
```

以当前W2 gate/manifest为权威再次核对这一个权重与三个OOF权重、fold表和数据身份，不重新评分44个W2候选。训练回报使用排除当前试样来源组的OOF模型；VALID用共同P_all。四个预测器eval、requires_grad=False；真实CAI不进入Actor状态。

### 5.2 固定方法和训练参数

| method键（保持原loader可识别） | training_seed | 逻辑update上限 | 实际消耗上限含中断回放 |
|---|---:|---:|---:|
| VLM_SPATIAL_FEEDBACK | 2026091301 | 1250 | 1500 |
| VLM_SPATIAL_OPEN_LOOP | 2026091303 | 1250 | 1500 |
| VLM_MEAN_FEEDBACK | 2026091305 | 1250 | 1500 |

`method`保留原枚举，另加`prior_version=C_P0_R1_GLOBAL_V1`、`experiment_id`和`protocol_identity`。不得仅把method命名成一个原`_uses_vlm()`不认识的新字符串，导致C0静默关闭。A/C比较用独立`method_instance_id`，不要覆盖原method。

固定：seed_panel=1；从原seed初始化，不用旧Actor暖启动；域均匀再域内试样均匀；batch16；AdamW(lr=3e-4,weight_decay=1e-4)；clip1；gamma1；critic权重0.5；entropy从0.01按原公式线性至0；budget0.25；terminal_error_weight0.25；validation_interval250；patience4；改善容差1e-12。

损失和采样逐行复用S6：`J=A(B)+0.25e_T`的cost-to-go策略梯度、值基线与熵项。不得改成即时reward、BC/PPO、teacher选点、硬mask后的固定贪心训练或增加loss项。训练动作采样、评价动作argmax保持。

C0仍仅第一步生效，第二步开始放开硬候选但保留VLM特征；空间/均值反馈及开环各自的信息权限与原实现相同。未测内部特征在模型内屏蔽；预算采用原生格像素float64、1e-12容差，网络cost通道float32。

首次初始化保存state_dict身份，与对应原W3初始hash比较；同设置却不一致时在首步前定位模块/版本/seed问题，不另选seed。加载与检查放在初始化前，或保存/恢复RNG；不能因审查改变训练随机序列。

### 5.3 精确选模规则

**每个方法独立选模，不按A/C谁好选择论文版本。**正式候选仅为update=250、500、750、1000、1250；每候选用同一50件VALID、同一P_all、同一C先验、同一成本规则生成自己的完整轨迹。

评价分数：先对同试样repeat平均A，再在域内对试样平均，再六域等权平均。选择顺序为：按update升序遍历，只有`score < best_score - 1e-12`才替换赢家；平局或容差内保留更早update。不得按最低终点MAE或无VLM差值改选。

patience=4的合法早停不得改写；以本轮1250上限与250验证间隔，完整运行自然最多五个参选点，**资源/异常中断不是早停**。缺少后续计划点时，只能标PROVISIONAL，不发布完整赢家。

每个实际候选须同时保存：权重、50件同次VALID轨迹（带execution_trace）、真实评分、状态身份、输入C manifest hash。不能只保留内存best_state。固定的是评价环境，不是强求不同checkpoint走同一路线。

正常完整结果为**15个候选、750条候选轨迹、3个已选Actor、150条已选轨迹**；150条是750中的子集，不重复计成新增独立样本。对非科学中断残留另存来源，不混入正式候选。

### 5.4 受控恢复，而非反复重训

这是本轮需要补的实际开发项。旧代码已有`latest_training_state.pt`，但`ActorArchive`默认目录必须不存在，旧context拒绝不完整job，旧训练循环总从1开始；**不得只保留这些旧拒绝路径然后宣称支持resume**。

新训练入口须实现：

1. update0在模型/optimizer/RNG初始化后保存独立的`initial_training_state.pt`并保留；每个250步候选完成时保存一个原子提交的恢复快照，含模型、optimizer、Python/NumPy全局/Generator/Torch CPU/CUDA RNG、update、best_score/update、stale、progress、C/W2/队列/训练签名。候选和快照用同一commit-marker闭合。
2. 一次性故障允许**每个方法最多一次**同签名恢复，从最近完整快照的下一逻辑update继续，原entropy按逻辑update计算。若首250前中断，从update0同初始化恢复，不换seed。
3. 已提交候选权重/轨迹不重评、不覆盖；快照之后未闭合的candidate文件保留到orphan目录并记录，本轮回放重新产生的有效候选不得与其混用。
4. 记录真实执行步与逻辑进度两个计数。丢失的计算不免费；无法精确确定时，对自上次提交至下一250边界按最多250步保守计费。每方法1500实际上限=1250计划+250恢复储备，三个合计4500。储备不能用来把逻辑训练延长到1500或新增seed。
5. 新C manifest、fold、网络/损失/采样语义或训练环境改变时不得恢复同一科学run。纯路径/原子保存等工程修复可记录source变更及语义检查后恢复；影响计算的改动须保留部分结果，不在本任务内重设计。
6. 第二次故障、坏快照或不足以完成余下逻辑步的额度时标`PARTIAL_EXECUTION`并交付；不得用部分赢家生成完整新论文成绩。不要开启容错平台，只实现这条250步快照的有限恢复路径。

### 5.5 首250与最终加载检查

每方法首次真实250步即检查权重、50条轨迹、C缓存身份、真实首步合法性和RNG未被检查污染。不另开真实优化“预检训练”。

选中后从归档磁盘**实际重载赢家**，构造发布用checkpoint与manifest。只需每方法在同一固定案例`cgtnjyggtm:q24-48`复跑一次发布smoke（共3个episode），核对动作/成本和已存预测；它不是新的评价样本，也不替换同次VALID选模轨迹。固定环境下动作应一致、cost误差≤1e-12、预测差≤1e-4 MPa；不符则检查加载/配置，不能悄悄用新轨迹覆盖原记录。

## 6. W3：复用六个对照，建立版本清楚的主矩阵

六个对照为`CENTER_FIRST, GEOMETRY_SPREAD, SERPENTINE, RANDOM, LEARNED_STATIC_TRUE, NO_VLM_SPATIAL_FEEDBACK`。

### 6.1 复用通过条件

核对同50 VALID key/组/目标、同feature-bank身份、同P_all权重、同成本和合法性、同原method实现、同Random五run种子2026091250–2026091254。无VLM的VLM通道与C0实际被禁用；固定/学习静态排序不使用VLM。代码或推理环境不同的地方记录，不能伪称这些对照本轮重新运行。

条件通过：从原`policy_validation_episodes.csv.gz`按method筛选并复制原始行，**保留原seed、checkpoint、execution_code_sha、轨迹内容**；在独立`row_provenance.csv`记录REUSED，不把旧数据的身份字段改成新训练。

条件不通过：先查明是否只是无关wrapper/路径变化。若存在真正方法、数据、预测器或成本变化，本任务不自动扩大训练，标明不可比并保留局部结果；禁止为凑650而混表。

### 6.2 行数与版本

- 四固定规则：50+50+50+250=400条。
- 学习固定排序和无VLM：50+50=100条。
- 三个新C策略：3×50=150条。
- 合计650条、50物理试样、48来源组、6域。

主矩阵只包含**三个C版本VLM策略+六个不受VLM改变影响的对照**。旧三个A版本VLM策略另存`historical_A_comparison/`，不作为重复主行混入9方法聚合。

主发布算法键保持不变，但manifest与图表元数据明确C。A/C专项表使用`prior_version`或`method_instance_id`区分；标题不能把版本字符C解释成第四种网络。不得按新结果将无VLM改为预设主方法。

## 7. W4：从新轨迹重新生成全部论文证据

优先薄适配原纯数学S9–S11。旧`w3_results.summarize/export_figures`、`paper_evidence.Evidence`和论文`timing_analysis.py`含旧目录与常量；**不直接执行旧main，也不为满足它们复制C覆盖旧目录**。参数化新实例，或复制必要输入/输出壳，数学函数保持原样。

### 7.1 指标、区间与范围

输出九方法五预设cap（0、.0625、.125、.1875、.25）的MAE/RMSE/R²、真实取得像素比例及范围；域等权A/early A与池化试样终点指标分开。小cap均查询固定总预算.25轨迹前缀，不重训或重新执行不同总预算策略。

Random先在同试样内平均绝对/平方误差，再汇总试样；不对预测平均后冒充误差，不将五run当250个物理样本。R²沿用原repeat聚合。

沿用原5000次、seed2026091401的域内capture-group bootstrap。优先读取OLD_EVID的`bootstrap_group_weights.npz`，核对key/组/域顺序，做显式索引重排后复用**相同抽样权重**；权重不存在或无法匹配才按原算法生成一次并记录原因。必须用新误差向量算新区间，不能复用旧CI端点。已完成分析再次执行只读缓存，不能重抽到区间变正。

必须报告：

| 对比 | 主报告指标 |
|---|---|
| C主方法 vs 最佳非自适应（原五类） | A、终点MAE、五cap配对差 |
| C主方法 vs C开环 | A、终点MAE |
| C主方法 vs 无VLM反馈 | early A，同时列全程A与终点MAE |
| C空间 vs C均值反馈 | A、终点MAE |
| C主方法 vs 历史A主方法 | A、early A、终点MAE与逐域差 |
| C开环/C均值 vs 各自历史A | 相同量，放版本对照表 |

差值统一`control_error - C_error`，正值有利C。最佳非自适应仍按A选，平局沿用原顺序/名字规则并记录；六个复用对照不变时应复现旧选择。区间是固定已选VALID上的探索性分析，不因新一轮重训变成独立验证，不声明因果字号贡献。

### 7.2 全输入参照

只读：

```text
W2/candidate_state_predictions/A_MEAN_SC/update_001750.npz
```

读取`full_specimen_indices/full_predictions_mpa/full_targets_mpa`，按原feature索引映射同50件VALID。使用S12指针核对，同P_all下完整输入指标应仍为原值（MAE约41.690010 MPa），这是一项应保持不变的参照，不是新改善。成本单列1.0，无虚构A，不将25%—100%连成实测曲线。

### 7.3 等质量与时机解释

保留原定义`c_m(q)=min{c: pooled_MAE_m(c)<=q}`，分别输出五cap主网格与**根据新九方法轨迹重建的**共同事件网格。网格断点数量不再固定为599。

质量目标沿用原`docs/cai/paper_evidence/PAPER_EVIDENCE_SCOPE.json`的规则：对新九方法在共同事件网格的全部MAE及完整输入MAE，取全局最小值的floor至全局最大值的ceil，间隔1 MPa生成完整整数网格；另加五个非自适应方法的全部四个非零cap锚点及完整输入目标（保留锚点来源，数值重复可共用计算）。不要为了C好看删阈值或临时改非劣界限。双方都查最早达标cost；未达标、零分母、负节省、首次达标后反弹保持。旧“50%/0%”“66.4%”不得硬拷贝，只有新计算恰好一致才能继续用。

时机贡献复用：`g_t=(1-c_t/.25)*(e_{t-1}-e_t)`；按(0,.0625]、(.0625,.125]、(.125,.1875]、(.1875,.25]完成时刻汇总，保持负项。先episode求和、试样内repeat平均、域内试样平均、六域等权。验证`A=e0-sum(g)`；对照初始误差不同须保留e0差项。**旧第二阶段+1.296 MPa和最大阶段结论均重新计算，不预设还成立。**

### 7.4 图、事件和可读报告

重新输出当前九方法曲线、配对表/森林图、等质量图、九方法四阶段贡献图、六域表。继续使用原三件q24-48、c8-16、q16-29；必须用**新C主方法的真实选模轨迹和C候选**出表面候选、首动作、1/4/8/终点已测图和CAI过程图，不能把旧21张轨迹图当新图。原始底图可引用，历史图另标A作比较。已测集合单调增加，grid严格8×8，预测可反弹。

输出本地HTML总览，包含211先验状态清单及缩略图入口、九方法主表、A/C版本对照、三案例与结果下载。小图可CPU批量产生；模型不参与图形生成。不要重建专家平台。

## 8. W5：更新为C版本论文，不只交一张实验表

从OLD_PAPER建立独立PAPER，保留六章源和已核实文献，不连同旧build结果不加检查地当新稿发布。按S13编辑`abstract.md`、`sections/01...06`、`supplementary.md`、`declarations.md`、图注和表格源，再执行新目录的build脚本生成整稿。

至少逐项更新：

1. Methods明确VLM仍P0，编号采用精确R1；模型/Actor/预测器分工、X_obs、C0、budget、预测器训练说明保持正确。补充C的合同检查与repair实际数量，不暗改为P1置信定义。
2. 实验设置给出本轮三策略seed/逻辑步/所选update、211先验可用性；对六个复用对照注明共同不变输入条件与来源，不称其本轮新训练。
3. 摘要、结果、结论、主表、SI、图注、例子、时机解释全部来自C evidence。旧44.286/45.110/第二阶段1.296等只允许出现在明确的历史A比较中。
4. 同成本正向观察、反馈对照、等质量节省、全输入差距按新结果组织，保留无VLM和负向结果。不要因C未赢而回写A为主线，也不要把“用户认为C正确”当定量定位验证。
5. 图1沿用已接受结构，不大改科学内容；必要时仅更新编号呈现说明。不得伪造新的卡通attention。C案例图全部来自真实新轨迹。
6. 更新`EVIDENCE_MAP.csv`、`RESULT_ALLOCATION.csv`、作者导读、模型/输入/图表索引；原引用台账能复用，不拓展文献任务。
7. 作者信息、基金、利益冲突、许可和实际AI型号等未确认项保持作者清单，不编造、不因此阻塞科学初稿。新稿是作者审阅稿。

使用已经安装的nature-skills中相关写作/润色和主文分配片段，不全局更新skill。**核心比较和实际发现用正向、清楚的工程语言叙述，验证范围集中交代；这不允许删除不利数据或写“显著”替代探索性点估计。**

构建沿用已成功Pandoc/LuaLaTeX，不换Office模板。对新PAPER内复制的分析/绘图/构建脚本逐一检查旧目录常量，将其显式指向C_EVID及PAPER；不只修改源稿中的链接。从PAPER源重建MD/HTML/TeX，先创建build目录；编译主稿和SI。用现有PDF渲染工具检查页面概览以及摘要、算法、主表、变化最大的图、SI表，不能用“编译退出码0”替代视觉检查。只修失败页；不要求精确页数或重复三轮全稿检查。

## 9. 资源授权与账目（本轮新增）

### 9.1 明确上限

| 项目 | 授权上限 |
|---|---:|
| C队列 | 211件TRAIN/VALID |
| 新主生成任务 | 最多211个唯一case；正常精确复用6件后205 |
| 新生成总尝试，含repair/异常尝试 | 最多422；每个非复用case最多2次，不执行A/B/D |
| 输出token预算 | 每次500；理论合计最多211000（不等于必须用完） |
| VLM累计GPU会话时间 | 5400秒（90分钟），含加载/修复/失败；单调用120秒 |
| 正式Actor逻辑更新 | 3×1250=3750 |
| 工程中断回放储备 | 每方法最多250，合计750；无故障为0 |
| 新Actor实际更新/保守上界 | 最多4500；每方法1500 |
| Actor训练、VALID、发布smoke累计GPU时间 | 10800秒（3小时） |
| 合计新GPU会话上限 | 16200秒（4.5小时），不是预期耗时 |
| CPU | 最多4线程；不并行多卡 |
| 额外seed/W2训练/CNN/TEST/attention/GDFS/STOP | 0 |

原使用39014，原总上限40014；**追加本轮4500**，新累计授权44514。正常无故障新增3750后使用42764；即使用满恢复储备也不超过43514，原W2未用1000仍不转移。本轮3750计划与750故障储备不能调换为更多科学训练。

这些是本任务的明确新执行窗口，不从旧GPU时间余额推算。执行前核对历史账本，若与39014不一致，列出真实差项；额度不自动随差项增大，优先判断本任务是否已部分执行，不能把重复调用当新任务重置额度。

VLM与Actor分阶段、不同进程运行，确保先释放Qwen后再训练Actor。沿用单个空闲GPU；Qwen至少核对34GiB可用显存（旧实测峰值约29.1GiB）。不停止/迁移他人进程，不在正在训练的GPU上强行挤入；无合适GPU时保存PREPARED_WAITING_RESOURCE并给出已完成CPU结果及继续命令，不循环忙查。

### 9.2 记账和恢复规则

新task_id、session_id、case/variant、method/run_id必须可区分；新本地账目append-only。原全局ledger前缀不改。**logical_update、actual_optimizer_updates、lost/replayed_upper_bound分别记录，不能把多次保存的累计步数逐项相加。**

先按旧`update_usage`对本任务以外的历史前缀计算一次；本轮以独立段级账目计算增量。每个训练段最多250步，在段前持久化预留、段后记录完成或中断实际/上界；恢复时未闭合段按上界收费，重放另收费。这样崩溃最多损失一段，可在每方法250储备内恢复一次。汇总只将每个已闭合段的增量写全局ledger一次，并保存唯一event_id；禁止在方法完成行再次重复写其全量步数。旧函数不能用来把本轮未完成整方法预留与已完成段双重相加。

例如：某方法已提交update250，第二段中断且执行量无法精确恢复，则第一段计250、失败段上界计250；从250恢复到500再计250，继续500→1250计750，合计保守实际1500而逻辑终点仍1250。没有故障则只计1250。对应全局行中每个段只写一次非零`actual_optimizer_updates`或`actual_optimizer_updates_upper_bound`，方法/任务摘要行写0并用其他字段记录累计值。不要在全局账本再添加未释放的整方法预留；本轮运行中的预留只由本地段账本持有。

生成记录区分`generation_attempts`、主调用、repair、失败、输出token和实际Qwen顶层forward；使用轻量hook计数，不存attention。**一次generate不是一次forward。** 原pilot六个C条目的历史生成及repair不能重复记成本轮生成；只保留来源计数，实际数从状态读取，不硬编码。

所有阶段入口需检查本轮剩余额度，而非每进入第二/第三模型又要求完整4500剩余。不把旧28100/40014、205/6或旧task结束状态作为新任务门槛。

## 10. 不超过八类的定向验收

一次开发前静态/合成检查，三模型首250真实检查，交付一次合并验收；失败只修对应项。以下为需求类别，不要求每格再建一套测试框架。

| ID | 验收对象 | 最低充分证据 |
|---|---|---|
| Q1 输入与C定义 | P0/R1实际传入、精确复用、211key、无TEST运行、旋转一次 | fake backend捕获+六已知C输入hash+实际首个新case input日志 |
| Q2 缓存与异常 | 签名隔离、最多一次repair、schema失败/无cue/中断分开 | 合成回复和已有状态，不执行模型做单元测试 |
| Q3 模型和可见性 | 四W2冻结、正确OOF、C0/未知屏蔽、seed | 静态源码对应+首个实际batch/首250元数据，无额外优化预检 |
| Q4 归档/选模/恢复 | 五候选计划、同次轨迹、平局、一次resume与双计费防止 | 小标量模型/RNG的中断恢复例+15正式候选与3次赢家smoke |
| Q5 九方法复用 | 500复用+150新；method/版本/行身份清楚 | row_provenance、key目标一致、只读原轨迹hash；不重跑全部对照 |
| Q6 数值与图 | 原指标、配对权重、full key、等质量与时机式、图状态 | 旧6对照数值复算不变+一个手算例+新真实表图，不重跑全库数学测试 |
| Q7 论文源同步 | 新C源MD/表图/PDF一致，无旧统计冒充C | 结果字段映射与实际PDF视觉检查，非仅搜索几个旧数字 |
| Q8 发布 | 权重/代码/结果/稿件可追踪，local/upstream/remote一致 | Git受影响文件范围、发布manifest和远端选取文件身份 |

不进行渗透、模糊、并发压力、全库pytest、旧44权重全前向、全数据反复哈希或多轮模拟同行评审。不能用一个fake backend测试通过代替真正211先验生成，也不能用“图片不如预期”当理由做语义重试。

## 11. 统一CLI与发布契约

### 11.1 要开发的入口

以下命令是**本任务需要实现的接口，不是声称仓库已经存在**：

```bash
python scripts/cai_c_retrain/cli.py prepare --config docs/cai/c_render_retrain/C_RETRAIN_RELEASE_SCOPE.json
python scripts/cai_c_retrain/cli.py vlm --config docs/cai/c_render_retrain/C_RETRAIN_RELEASE_SCOPE.json
python scripts/cai_c_retrain/cli.py train --config docs/cai/c_render_retrain/C_RETRAIN_RELEASE_SCOPE.json
python scripts/cai_c_retrain/cli.py assemble --config docs/cai/c_render_retrain/C_RETRAIN_RELEASE_SCOPE.json
python scripts/cai_c_retrain/cli.py analyze --config docs/cai/c_render_retrain/C_RETRAIN_RELEASE_SCOPE.json
python scripts/cai_c_retrain/cli.py paper --config docs/cai/c_render_retrain/C_RETRAIN_RELEASE_SCOPE.json
python scripts/cai_c_retrain/cli.py verify --config docs/cai/c_render_retrain/C_RETRAIN_RELEASE_SCOPE.json
python scripts/cai_c_retrain/cli.py publish --config docs/cai/c_render_retrain/C_RETRAIN_RELEASE_SCOPE.json
```

另实现`export-inputs --specimen-key <实际key>`，只读已有表面来源并按冻结C配置重建输入，不调用模型，不覆盖旧目录。支持`all`顺序调用上述阶段，并由新入口将不同阶段分派到已锁定解释器；阶段已完整则按签名复用；改变输入必须显式报错，不能默默沿用旧标志。训练支持`--resume`及可选`--method`原枚举，只执行第5.4节的一次同签名恢复，不重置限额。`all`遇到可恢复的一次性工程失败时调用这条有界恢复路径一次，而不是重新执行从头训练；已有完整任务只执行尚未完成的阶段。`publish`本身不启动模型、训练或重抽bootstrap。

Codex完成必要代码后实际运行全部阶段；不要把这组尚未实现的命令原样交给用户，要求用户继续写实现。若个别路径/依赖真正不可用，保存阶段状态、具体证据和一条准确继续命令，执行仍可完成的不依赖工作。

### 11.2 必须交付的文件

```text
ART/
  SOURCE_AND_REQUIREMENT_BINDINGS.md
  AUTHOR_C_DECISION.md
  EXECUTION_REVIEW.md
  CODEX_HANDOFF_CAI_C_RETRAIN_RELEASE.md
  GIT_DELIVERY.json
OUT/
  authorization_snapshot.json
  protocol_lock.json
  task_state.json
  resource_usage.json
  vlm/                         # 211行先验、完整调用/复用来源、真实输入身份
  w3/
    actor_manifests.json
    models/                    # 15参选权重+3selected+各方法initial/latest状态
    candidate_episodes/        # 各实际参选点50轨迹
    policy_validation_episodes.csv.gz
    row_provenance.csv
    historical_A_comparison/
  evidence/                    # 新主表、配对、等质量、事件、时机、案例及HTML
  release_manifest.json
  index.html
PAPER/
  abstract.md
  sections/01_...md ... 06_...md
  declarations.md
  manuscript.md
  manuscript.html
  main.tex
  supplementary.md
  supplementary.tex
  references.bib
  figures/
  tables/
  build/main.pdf
  build/supplementary.pdf
  EVIDENCE_MAP.csv
  README.md
```

manifest记录三新赢家及共享W2/复用对照的准确路径/hash、C输入与解析版本、科学设置、结果和稿件身份。共享权重已经在仓库时只引用，不重复复制44个W2候选；新增15参选权重及3最终权重实际追踪上传，不仅上传空manifest。检查gitignore，必要时仅对本任务具体模型/轨迹/PDF文件使用定向add。不要`git add .`收走无关文件。

原分支正常commit/push，不PR、不merge、不force push、不修改远端设置。不把自身最终commit写入同一commit造成无限循环：先结果提交，再最多一次交接提交，最终回复给最终SHA。网络失败保留本地提交并重试push，不重跑实验。远端有未预料增量导致非fast-forward时不重写历史，报告分歧。

逐项核对选中模型/新C manifest/650轨迹/两个PDF/交接均被追踪且远端同SHA；上传权重超过当前文件限制时不得悄悄丢弃，可用已有仓库大文件方案或可校验分片，不能迁移历史。无需对数百文件反复逐blob验证。

### 11.3 发布状态不与科学胜负绑定

| 状态 | 条件 |
|---|---|
| C_RETRAIN_RESULTS_COMPLETE | 211正常终态、3模型完整选模、650主轨迹和证据闭合 |
| C_RETRAIN_RELEASE_COMPLETE | 上项+新C稿件/主稿与SI构建视觉检查+GitHub实推成功 |
| PARTIAL_EXECUTION | 明确资源/输入/恢复/构建等缺项；仍上传真实已完成内容，禁止用旧结果填C空缺 |
| EFFECT_REPORT | 固定报告各比较点估计/探索性区间/未达标，不作为发布通过门槛 |

发布完成后将新C manifest作为**本项目C版本的默认入口**（新目录README/配置明确指向），旧A仍保留并可复现。不得要求先“C显著更好”才交付，也不得因为C变差就悄悄把旧A改回默认。

## 12. 机器可读范围

以下JSON与本MD一致，是本轮唯一参数来源。Codex将其原样写入AUTH；允许另添实际执行身份、预算消耗和文件hash，不更改冻结科学值。若同时收到包内同名JSON，以两者完全一致为前提加载；不依赖其他旧授权文件决定本轮权限。

<!-- SCOPE_JSON_BEGIN -->
```json
{
  "schema_version": 1,
  "task_id": "CAI_V3_C_RENDER_RETRAIN_RELEASE_R1_331f5295",
  "repository": "Orangekostar/diff",
  "branch": "research/cai-vlm-agent-v3-controlled-reuse",
  "source_commit": "331f52952b93bd9442ad2e44243cc71c7d966b4d",
  "authorization": "User-assigned final instruction authorizes only the stages and finite allowances below; this file is not evidence of execution.",
  "scientific_scope": "C_P0_R1_TRAIN_VALID_SEED1_RETRAIN_WITH_PAPER_AND_GIT_DELIVERY",
  "prior_version": "C_P0_R1_GLOBAL_V1",
  "roots": {
    "data": "results/cai_agent_v3/new_protocol",
    "predictors": "results/cai_agent_v3/w2_replay/r1_292b1c74",
    "old_w3": "results/cai_agent_v3/w3_pilot/r1_0e11452a",
    "old_w3_artifacts": "artifacts/cai_agent_v3/w3_pilot/r1_0e11452a",
    "old_evidence": "results/cai_agent_v3/paper_evidence/r1_e2a11154",
    "grounding_pilot": "results/cai_agent_v3/vlm_grounding_pilot/r1_e8d9ef01",
    "old_paper": "paper_cai_aei/r1_84bea60e",
    "code": "scripts/cai_c_retrain",
    "output": "results/cai_agent_v3/c_render_retrain/r1_331f5295",
    "artifacts": "artifacts/cai_agent_v3/c_render_retrain/r1_331f5295",
    "vlm": "results/cai_agent_v3/c_render_retrain/r1_331f5295/vlm",
    "w3": "results/cai_agent_v3/c_render_retrain/r1_331f5295/w3",
    "evidence": "results/cai_agent_v3/c_render_retrain/r1_331f5295/evidence",
    "paper": "paper_cai_aei/r2_c_331f5295",
    "authorization": "docs/cai/c_render_retrain/C_RETRAIN_RELEASE_SCOPE.json",
    "global_ledger": "results/cai_agent_v3/compute_ledger.jsonl"
  },
  "stages": [
    "prepare",
    "vlm",
    "train",
    "assemble",
    "analyze",
    "paper",
    "verify",
    "publish"
  ],
  "cohort": {
    "total_physical": 276,
    "total_groups": 259,
    "train_physical": 161,
    "train_groups": 152,
    "valid_physical": 50,
    "valid_groups": 48,
    "reserved_test_physical": 65,
    "reserved_test_groups": 59,
    "vlm_authorized_splits": [
      "TRAIN",
      "VALID"
    ],
    "c_prior_rows": 211,
    "test_scoring_allowed": false,
    "test_stored_redacted_arrays_may_be_loaded_as_container": true,
    "test_models_labels_or_image_access": false
  },
  "vlm": {
    "model_repository": "Qwen/Qwen2.5-VL-7B-Instruct",
    "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    "prompt_id": "P0",
    "prompt_source": "results/cai_agent_v3/vlm_grounding_pilot/r1_e8d9ef01/prompts/P0.txt",
    "prompt_equals_original_surface_percept": true,
    "render_id": "R1",
    "render_source": "scripts/vlm_grounding_pilot/prepare.py::render_readable",
    "clean_render": "RGB_ROTATE_270_ONCE_MAX_EDGE_1024_LANCZOS",
    "font_identity": "EXACT_PILOT_FONT_AND_RENDER_METADATA_NO_FONT_FILE_UPLOAD",
    "image_order": [
      "clean",
      "R1_numbered"
    ],
    "new_system_message": false,
    "add_vision_id": false,
    "dtype": "bfloat16",
    "attention": "sdpa",
    "eval": true,
    "requires_grad": false,
    "processor": {
      "use_fast": false,
      "min_pixels": 200704,
      "max_pixels": 1003520
    },
    "generation": {
      "batch_size": 1,
      "max_new_tokens": 500,
      "do_sample": false,
      "temperature": null,
      "use_cache": true
    },
    "parsing_contract": "scripts/vlm_grounding_pilot/common.py::parse_contract",
    "repair_policy": "ONE_ORIGINAL_FORMAT_REPAIR_PRESERVE_FIRST_RAW_AND_GENERATED_IDS",
    "pilot_reuse": "EXACT_SIGNATURE_ONLY_PRESERVE_REPAIR_HISTORY",
    "expected_exact_pilot_reuse": 6,
    "expected_new_unique_primary_jobs_if_reuse_exact": 205,
    "maximum_new_unique_primary_jobs": 211,
    "maximum_generation_attempts_per_case": 2,
    "maximum_generation_attempts_including_repair_failure_retry": 422,
    "maximum_total_output_tokens": 211000,
    "interrupted_retry": "ONE_RETRY_ONLY_WITHIN_THE_SAME_TWO_ATTEMPT_CASE_LIMIT_NO_THIRD_REPAIR",
    "final_schema_failure_policy": "AVAILABLE_FALSE_NO_RELIABLE_FALSE_ZERO_FEATURES_KEEP_SPECIMEN",
    "valid_no_cue_policy": "AVAILABLE_TRUE_NO_RELIABLE_TRUE_ZERO_FEATURES_NO_RETRY",
    "incomplete_policy": "DO_NOT_TRAIN_UNTIL_ALL_211_HAVE_TERMINAL_RECORDS",
    "confidence_values": {
      "unknown": 0.0,
      "low": 0.3333333333333333,
      "medium": 0.6666666666666666,
      "high": 1.0
    },
    "feature_boolean_strings": [
      "True",
      "False"
    ],
    "generation_counts_distinguish_unique_jobs_and_attempts": true
  },
  "predictors": {
    "reuse_w2": true,
    "retrain": false,
    "common_model": "MEAN_SC",
    "common_selected_update": 1750,
    "common_checkpoint": "results/cai_agent_v3/w2_replay/r1_292b1c74/models/predictor_mean_sc.pt",
    "common_checkpoint_sha256": "f67e912bd5faa26ae5cff8a9a0241439797fccef8cec825f43ebe5dc9f9d57ff",
    "oof_folds": 3,
    "oof_routing": "ORIGINAL_TRAIN_CAPTURE_GROUP_FOLDS",
    "shared_encoder_reencoding": false
  },
  "models": [
    {
      "method": "VLM_SPATIAL_FEEDBACK",
      "training_seed": 2026091301,
      "seed_panel": 1,
      "max_logical_updates": 1250,
      "max_actual_updates_including_replay": 1500,
      "max_resumes": 1
    },
    {
      "method": "VLM_SPATIAL_OPEN_LOOP",
      "training_seed": 2026091303,
      "seed_panel": 1,
      "max_logical_updates": 1250,
      "max_actual_updates_including_replay": 1500,
      "max_resumes": 1
    },
    {
      "method": "VLM_MEAN_FEEDBACK",
      "training_seed": 2026091305,
      "seed_panel": 1,
      "max_logical_updates": 1250,
      "max_actual_updates_including_replay": 1500,
      "max_resumes": 1
    }
  ],
  "training": {
    "optimizer": "AdamW",
    "batch_size": 16,
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "gradient_clip_norm": 1.0,
    "discount_gamma": 1.0,
    "critic_weight": 0.5,
    "terminal_error_weight": 0.25,
    "entropy_start": 0.01,
    "entropy_end": 0.0,
    "entropy_progress": "LOGICAL_UPDATE_NOT_ACTUAL_REPLAY_STEPS",
    "sample_order": "UNIFORM_DOMAIN_THEN_UNIFORM_TRAIN_PHYSICAL_SPECIMEN",
    "seed_before_initialization": true,
    "warm_start_old_actor": false,
    "validation_interval": 250,
    "patience": 4,
    "improvement_tolerance": 1e-12,
    "selection_metric": "DOMAIN_EQUAL_LEFT_ERROR_AREA_MPA",
    "tie_break": "KEEP_EARLIEST_UNLESS_STRICT_IMPROVEMENT_EXCEEDS_1E_12",
    "candidate_updates": [
      250,
      500,
      750,
      1000,
      1250
    ],
    "save_all_candidate_weights_and_same_evaluation_trajectories": true,
    "expected_candidate_count": 15,
    "expected_candidate_episode_rows": 750,
    "expected_selected_models": 3,
    "expected_selected_episode_rows": 150,
    "snapshot_at_zero_and_each_validation": true,
    "snapshot_fields": [
      "model",
      "optimizer",
      "python_rng",
      "numpy_global_rng",
      "numpy_generator_rng",
      "torch_cpu_rng",
      "torch_cuda_rng",
      "logical_update",
      "best_score",
      "best_update",
      "stale",
      "progress",
      "environment_identity",
      "run_id"
    ],
    "resume_lost_segment_max_updates": 250,
    "maximum_method_resume_count": 1,
    "partial_checkpoint_not_final_selection": true
  },
  "evaluation": {
    "budget": 0.25,
    "early_budget": 0.0625,
    "costs_dtype": "float64",
    "neural_cost_dtype": "float32",
    "legality_tolerance": 1e-12,
    "cost_unit": "UNIQUE_NATIVE_RASTER_FRACTION",
    "cost_points": [
      0.0,
      0.0625,
      0.125,
      0.1875,
      0.25
    ],
    "smaller_budgets": "PREFIXES_OF_FIXED_TOTAL_BUDGET_025_TRAJECTORIES",
    "first_step": "EXISTING_HIGHEST_MEDIUM_HIGH_C0_THEN_RELEASE",
    "reused_method_episode_counts": {
      "CENTER_FIRST": 50,
      "GEOMETRY_SPREAD": 50,
      "SERPENTINE": 50,
      "RANDOM": 250,
      "LEARNED_STATIC_TRUE": 50,
      "NO_VLM_SPATIAL_FEEDBACK": 50
    },
    "expected_reused_episode_rows": 500,
    "expected_combined_primary_episodes": 650,
    "random_repeat_seeds": [
      2026091250,
      2026091251,
      2026091252,
      2026091253,
      2026091254
    ],
    "new_release_smoke_episodes": 3,
    "release_smoke_case": "cgtnjyggtm:q24-48",
    "release_smoke_prediction_tolerance_mpa": 0.0001,
    "release_smoke_cost_tolerance": 1e-12,
    "release_smoke_must_match_actions": true,
    "old_A_three_vlm_methods": "SEPARATE_HISTORICAL_COMPARISON_NOT_NINE_METHOD_PRIMARY_ROWS"
  },
  "evidence": {
    "main_method": "VLM_SPATIAL_FEEDBACK",
    "methods": [
      "CENTER_FIRST",
      "GEOMETRY_SPREAD",
      "SERPENTINE",
      "RANDOM",
      "LEARNED_STATIC_TRUE",
      "NO_VLM_SPATIAL_FEEDBACK",
      "VLM_MEAN_FEEDBACK",
      "VLM_SPATIAL_FEEDBACK",
      "VLM_SPATIAL_OPEN_LOOP"
    ],
    "endpoint_aggregation": "REPEAT_LOSS_MEAN_THEN_PHYSICAL_SPECIMEN_MEAN",
    "area_aggregation": "REPEAT_MEAN_THEN_WITHIN_DOMAIN_SPECIMEN_MEAN_THEN_SIX_DOMAIN_EQUAL",
    "bootstrap": {
      "replicates": 5000,
      "seed": 2026091401,
      "confidence": 0.95,
      "sampling": "PAIRED_CAPTURE_GROUP_WITHIN_DOMAIN",
      "reuse_old_draw_weights_after_key_check": true,
      "recompute_new_error_intervals": true,
      "scope": "POSTHOC_SELECTED_VALID_CONDITIONAL_POINTWISE"
    },
    "difference_sign": "CONTROL_ERROR_MINUS_C_ERROR",
    "full_reference_pointer": "results/cai_agent_v3/w3_pilot/r1_0e11452a/p_all_saved_reference.json",
    "full_reference_forward_calls": 0,
    "full_reference_cost_support": [
      1.0
    ],
    "full_reference_area": null,
    "secondary_cost_grid": "UNION_OF_NEW_PRIMARY_NINE_METHOD_EPISODE_BREAKPOINTS_AND_0_025",
    "do_not_hardcode_secondary_grid_length": true,
    "equal_quality": {
      "metric": "POOLED_PHYSICAL_SPECIMEN_MAE_MPA",
      "q_step": 1.0,
      "q_range": "FLOOR_MIN_TO_CEIL_MAX_OF_ALL_NEW_OBSERVED_METHOD_AND_FULL_REFERENCE_MAES",
      "anchors": "ALL_FIVE_NONADAPTIVE_METHODS_ALL_FOUR_NONZERO_CAPS_PLUS_FULL_MAE",
      "preserve_unreached_negative_savings_recrossing": true,
      "noninferiority_margin": null,
      "not_a_deployable_stop": true
    },
    "timing": {
      "stage_edges": [
        0.0,
        0.0625,
        0.125,
        0.1875,
        0.25
      ],
      "bins": "RIGHT_CLOSED_COMPLETION_COST",
      "formula": "g_t=(1-c_t/B)*(e_previous-e_current)",
      "identity": "A=e0-sum(g_t)",
      "retain_negative_terms": true,
      "recompute_all_stage_claims": true
    },
    "cases": [
      "74t7kcdgkr:c8-16",
      "cgtnjyggtm:q24-48",
      "w68dtmpfyf:q16-29"
    ],
    "case_source": "SELECTED_NEW_C_MAIN_EPISODES_AND_C_PRIORS",
    "case_acquisition_steps": [
      1,
      4,
      8,
      "endpoint"
    ],
    "all_new_case_pngs": true
  },
  "budget": {
    "historical_used_upper_bound_expected": 39014,
    "previous_global_cap": 40014,
    "unused_W2_allowance": 1000,
    "transfer_W2_allowance": false,
    "new_planned_optimizer_updates": 3750,
    "new_interruption_replay_reserve": 750,
    "new_actual_optimizer_updates_upper_bound": 4500,
    "additional_global_authorization": 4500,
    "new_global_cap": 44514,
    "normal_cumulative_usage": 42764,
    "max_cumulative_usage_this_task": 43514,
    "vlm_gpu_session_seconds": 5400,
    "vlm_attempt_seconds": 120,
    "actor_gpu_session_seconds": 10800,
    "total_new_gpu_session_seconds": 16200,
    "max_visible_gpus": 1,
    "max_cpu_threads": 4,
    "minimum_free_qwen_gpu_gib": 34,
    "update_accounting": "UNIQUE_SEGMENT_DELTA_INCLUDING_LOST_AND_REPLAYED_UPPER_BOUNDS_NO_DOUBLE_COUNT",
    "accounting_quantum_updates": 250,
    "refuse_silent_budget_reset": true
  },
  "manuscript": {
    "update_all_affected_results_and_claims": true,
    "sections": 6,
    "formats": [
      "md",
      "html",
      "tex",
      "pdf"
    ],
    "main_pdf": "build/main.pdf",
    "supplementary_pdf": "build/supplementary.pdf",
    "actual_visual_review_required": true,
    "preserve_original_paper": true,
    "status": "AUTHOR_REVIEW_DRAFT",
    "select_C_independent_of_performance_direction": true,
    "no_new_literature_project": true,
    "do_not_claim_AI_or_author_approval_not_recorded": true
  },
  "permissions": {
    "train_only_three_registered_C_actors": true,
    "new_C_Qwen_generations_within_caps": true,
    "new_registered_actor_and_frozen_predictor_forwards_for_training_valid_smoke": true,
    "cpu_recompute_evidence": true,
    "update_C_manuscript_in_new_directory": true,
    "git_commit_push": true,
    "TEST_evaluation_or_label_join": false,
    "extra_seed_panels": false,
    "W2_training": false,
    "CNN_encoding": false,
    "Qwen_finetuning": false,
    "new_attention_diagnostics": false,
    "GDFS": false,
    "STOP": false,
    "new_baselines_or_hyperparameter_search": false,
    "hardware_benchmark": false,
    "full_repository_test_sweep": false,
    "overwrite_frozen_production_sources_results_or_old_paper": false,
    "paper_journal_submission": false,
    "create_PR_merge_force_push_or_release_tag": false
  },
  "delivery": {
    "handoff": "CODEX_HANDOFF_CAI_C_RETRAIN_RELEASE.md",
    "main_instruction": "CODEX_C_RENDER_RETRAIN_RELEASE_FINAL.md",
    "actual_commit_and_push": true,
    "same_branch": true,
    "check_local_upstream_remote_sha": true,
    "keep_all_actual_candidate_weights": true,
    "keep_raw_generation_and_repair_results": true,
    "keep_211_case_status_and_feature_rows": true,
    "full_resolution_input_delivery": "SIX_PILOT_MATCHED_CASES_OR_EXISTING_REPO_REFERENCES_PLUS_ALL_CASE_HASHES_THUMBNAILS_AND_REBUILD_CLI",
    "never_upload_font_or_base_Qwen_weights": true,
    "no_model_result_cherry_picking": true,
    "full_success": "C_RETRAIN_RELEASE_COMPLETE",
    "partial": "PARTIAL_EXECUTION",
    "scientific_effect_is_not_completion_gate": true,
    "result_and_handoff_commits_max_after_implementation": 2
  },
  "runtime": {
    "lock_file": "runtime_lock.json",
    "interpreter_keys": [
      "vlm_python",
      "actor_python",
      "report_python"
    ],
    "phase_subprocesses": true,
    "no_automatic_library_upgrade": true,
    "scientific_file_hashes_fixed_before_training": true
  }
}
```
<!-- SCOPE_JSON_END -->


## 13. 源码与证据索引（固定提交，非运行结论）

以下均已实际读取相应内容。本任务的新额度、恢复设计与新目录是本文设计，不是下面旧文件已授权/已实现的事实。执行时用工作树实际文件核对，不需要重新联网逐项下载。

- **S1 已完成C小样本对照与范围**：`artifacts/cai_agent_v3/vlm_grounding_pilot/r1_e8d9ef01/CODEX_HANDOFF_VLM_GROUNDING_PILOT.md`
  https://github.com/Orangekostar/diff/blob/331f52952b93bd9442ad2e44243cc71c7d966b4d/artifacts/cai_agent_v3/vlm_grounding_pilot/r1_e8d9ef01/CODEX_HANDOFF_VLM_GROUNDING_PILOT.md
- **S2 C解析合同、repair与原模块绑定**：`scripts/vlm_grounding_pilot/common.py`
  https://github.com/Orangekostar/diff/blob/331f52952b93bd9442ad2e44243cc71c7d966b4d/scripts/vlm_grounding_pilot/common.py
- **S3 精确R1绘图实现、font、signature生成**：`scripts/vlm_grounding_pilot/prepare.py`
  https://github.com/Orangekostar/diff/blob/331f52952b93bd9442ad2e44243cc71c7d966b4d/scripts/vlm_grounding_pilot/prepare.py
- **S3b 原图旋转/缩放/编号流程**：`src/cmc_bbdm/vlm_cscan/runtime.py`
  https://github.com/Orangekostar/diff/blob/331f52952b93bd9442ad2e44243cc71c7d966b4d/src/cmc_bbdm/vlm_cscan/runtime.py
- **S4 原schema/parser/cache resolver**：`src/cmc_bbdm/learned_cscan/perception.py`
  https://github.com/Orangekostar/diff/blob/331f52952b93bd9442ad2e44243cc71c7d966b4d/src/cmc_bbdm/learned_cscan/perception.py
- **S4b 原VLM调用、_features及旧写入目录**：`src/cmc_bbdm/cai_agent_v3/vlm_perception.py`
  https://github.com/Orangekostar/diff/blob/331f52952b93bd9442ad2e44243cc71c7d966b4d/src/cmc_bbdm/cai_agent_v3/vlm_perception.py
- **S4c 实际Qwen processor/generate参数**：`src/cmc_bbdm/vlm_cscan/vlm.py`
  https://github.com/Orangekostar/diff/blob/331f52952b93bd9442ad2e44243cc71c7d966b4d/src/cmc_bbdm/vlm_cscan/vlm.py
- **S5 MEAN_SC、空间/均值Actor输入与参数**：`src/cmc_bbdm/cai_agent_v3/models.py`
  https://github.com/Orangekostar/diff/blob/331f52952b93bd9442ad2e44243cc71c7d966b4d/src/cmc_bbdm/cai_agent_v3/models.py
- **S6 Actor特征加载、seed、训练和OOF/VALID执行**：`src/cmc_bbdm/cai_agent_v3/actor_training.py`
  https://github.com/Orangekostar/diff/blob/331f52952b93bd9442ad2e44243cc71c7d966b4d/src/cmc_bbdm/cai_agent_v3/actor_training.py
- **S7 旧W3硬编码、context与已存optimizer/RNG快照**：`src/cmc_bbdm/cai_agent_v3/w3_pilot.py`
  https://github.com/Orangekostar/diff/blob/331f52952b93bd9442ad2e44243cc71c7d966b4d/src/cmc_bbdm/cai_agent_v3/w3_pilot.py
- **S8 完整候选归档、严格更新/平局和早停检查**：`src/cmc_bbdm/cai_agent_v3/actor_selection.py`
  https://github.com/Orangekostar/diff/blob/331f52952b93bd9442ad2e44243cc71c7d966b4d/src/cmc_bbdm/cai_agent_v3/actor_selection.py
- **S9 原论文证据输入、full连接、输出与统计调用**：`src/cmc_bbdm/cai_agent_v3/paper_evidence.py`
  https://github.com/Orangekostar/diff/blob/331f52952b93bd9442ad2e44243cc71c7d966b4d/src/cmc_bbdm/cai_agent_v3/paper_evidence.py
- **S10 原聚合、held、bootstrap、等质量、full映射**：`src/cmc_bbdm/cai_agent_v3/paper_evidence_math.py`
  https://github.com/Orangekostar/diff/blob/331f52952b93bd9442ad2e44243cc71c7d966b4d/src/cmc_bbdm/cai_agent_v3/paper_evidence_math.py
- **S10b 原误差面积与任务cost-to-go**：`src/cmc_bbdm/cai_agent_v3/metrics.py`
  https://github.com/Orangekostar/diff/blob/331f52952b93bd9442ad2e44243cc71c7d966b4d/src/cmc_bbdm/cai_agent_v3/metrics.py
- **S11 已验证的时机分解及聚合入口**：`paper_cai_aei/r1_84bea60e/analysis/timing_analysis.py`
  https://github.com/Orangekostar/diff/blob/331f52952b93bd9442ad2e44243cc71c7d966b4d/paper_cai_aei/r1_84bea60e/analysis/timing_analysis.py
- **S11b 原质量网格、锚点和探索性区间配置**：`docs/cai/paper_evidence/PAPER_EVIDENCE_SCOPE.json`
  https://github.com/Orangekostar/diff/blob/331f52952b93bd9442ad2e44243cc71c7d966b4d/docs/cai/paper_evidence/PAPER_EVIDENCE_SCOPE.json
- **S12 同一P_all权重、1750选中点和完整输入NPZ**：`results/cai_agent_v3/w3_pilot/r1_0e11452a/p_all_saved_reference.json`
  https://github.com/Orangekostar/diff/blob/331f52952b93bd9442ad2e44243cc71c7d966b4d/results/cai_agent_v3/w3_pilot/r1_0e11452a/p_all_saved_reference.json
- **S13 MD源文件到整稿/TeX/HTML的真实构建链**：`paper_cai_aei/r1_84bea60e/build_manuscript.py`
  https://github.com/Orangekostar/diff/blob/331f52952b93bd9442ad2e44243cc71c7d966b4d/paper_cai_aei/r1_84bea60e/build_manuscript.py
- **S14 旧W3阶段总量与历史使用上界**：`results/cai_agent_v3/w3_pilot/r1_0e11452a/final_manifest.json`
  https://github.com/Orangekostar/diff/blob/331f52952b93bd9442ad2e44243cc71c7d966b4d/results/cai_agent_v3/w3_pilot/r1_0e11452a/final_manifest.json
- **S14b 既有训练与两轮VLM诊断账目**：`results/cai_agent_v3/compute_ledger.jsonl`
  https://github.com/Orangekostar/diff/blob/331f52952b93bd9442ad2e44243cc71c7d966b4d/results/cai_agent_v3/compute_ledger.jsonl
- **S14c 旧保守使用上界计算，仅作为历史段读取参照**：`src/cmc_bbdm/cai_agent_v3/w2_replay.py`
  https://github.com/Orangekostar/diff/blob/331f52952b93bd9442ad2e44243cc71c7d966b4d/src/cmc_bbdm/cai_agent_v3/w2_replay.py
- **S14d 原seed1科学训练与评价设置**：`docs/cai/w3_valid_pilot/W3_PILOT_AUTHORIZATION.json`
  https://github.com/Orangekostar/diff/blob/331f52952b93bd9442ad2e44243cc71c7d966b4d/docs/cai/w3_valid_pilot/W3_PILOT_AUTHORIZATION.json


## 14. 最终给用户的汇报格式

结束后按W0—W6报告实际完成状态，必须同时提供：

- C先验211件的可用/无线索/schema失败分布、pilot复用与新生成/repair/前向计数。
- 三个新Actor的完整候选点、所选update、真实/逻辑更新数、恢复储备使用，以及9方法650条主矩阵来源。
- C主方法对原A、无VLM、C开环、C均值和最佳非自适应的结果（含既定区间）；完整输入差距、等质量及最大贡献阶段如实报告。
- 新MD/两份PDF/HTML/发布manifest/交接的实际路径和最终Git SHA；真实未决项单独说明。

不要只报测试数量或一句“C更好”。完整完成时结束本轮，不自动开启TEST、扩种子或再次改提示词。遇到部分阻塞则交付已完成的真实内容，不把本任务作为再次全库审计的起点。
