# Codex执行提示词：AEI初稿的定向科学写作修订 R2

## 0. 本轮授权与完成目标

- 仓库：`Orangekostar/diff`。
- 接续分支：`research/cai-vlm-agent-v3-controlled-reuse`。
- 已核对稿件基点：`f4758829c621fa02a83595f1549e41dabe9e1c64`。
- 任务ID：`CAI_AEI_TARGETED_REVISION_R2_f4758829`。
- 任务性质：**修改已经完成的论文初稿，不重新做研究，不重新写一套大纲。**

用户认可现有六章结构与研究主线，现授权你依据下列问题实际修订正文、补充材料、相关图表、引用和生成脚本，重新生成可编辑整稿与PDF，完成有限复核及GitHub提交推送。

完成目标：

> 将当前较偏证据交接的初稿，修订为以“面向CAI评估的状态依赖采集”为中心的研究论文；补全真实方法细节，突出观测组合和取得时机的解释，压缩重复的审计式说明，同时保持原始方法、全部结果、统计含义与证据身份不变。

不等待另一轮大纲批准；不只返回修改建议。当前规范本身已经绑定具体任务。

### 0.1 不得扩大范围

| 允许 | 不允许 |
|---|---|
| 修改英文段落、公式、伪代码、表头、图注、文献元数据和对应作者说明 | 新训练、参数拟合、checkpoint重选、策略重复运行、新模型前向、VLM调用、特征编码 |
| 从冻结源码补充“已经执行的方法” | 修改研究代码来让实现符合想写的故事 |
| 复用既有数表；读取已有时机贡献表做必要算术核对 | 重算bootstrap、修改估计量、重新筛选有利阈值或案例、删除失败试样 |
| 将既有案例图组成一个简短过程图，保持源图像内容不变 | 生成新的观测图、损伤GT、注意力、语言推理、超声幅值或反事实轨迹 |
| 使用现有Pandoc/LuaLaTeX构建与PDF检查工具 | 全仓库回归、训练重放、渗透/fuzz/压力测试、全模型哈希、反复模拟审稿 |
| 为修正引用定向查询正式出版方记录 | 扩展新基线、重开文献综述项目、因期刊模板暂不可得而阻塞正文 |

新增研究更新、研究模型前向、新VLM、新TEST接入、新bootstrap、研究GPU任务均为 **0**。写作与编译使用CPU，线程上限4。Codex写作本身不计作研究VLM调用。

不得借本次修订启动已排除的稀疏插值基线、硬件时延、设备移动成本、专家mask或STOP实验。不存在新的训练额度申请任务。

## 1. 路径与最短阅读顺序

相对于实际工作树根目录ROOT：

```text
PAPER = paper_cai_aei/r1_84bea60e/
OLD_HANDOFF = artifacts/cai_agent_v3/manuscript/r1_84bea60e/
EVID = results/cai_agent_v3/paper_evidence/r1_e2a11154/
EVID_ART = artifacts/cai_agent_v3/paper_evidence/r1_e2a11154/
W2 = results/cai_agent_v3/w2_replay/r1_292b1c74/
W3 = results/cai_agent_v3/w3_pilot/r1_0e11452a/
NEW_ART = artifacts/cai_agent_v3/manuscript_revision/r2_f4758829/
```

`NEW_ART`是本任务建议新建的修订交接目录，不声称它已经存在。正文在现有PAPER内作定向修改，旧稿由Git基点保留；不要复制一整套带有失效硬编码路径的新论文工程。

先执行非破坏性状态核对：当前分支、HEAD、upstream、未提交文件。若HEAD是基点之后的相关提交，查看受影响稿件差异并保留已完成修订；不reset、不覆盖用户改动、不自动切换到main。只有实质冲突时才询问，其他工作继续。

### 必读输入

1. `OLD_HANDOFF/CODEX_HANDOFF_CAI_AEI_MANUSCRIPT.md`、`WRITING_REVIEW.md`。
2. `PAPER/abstract.md`及六份`PAPER/sections/0[1-6]_*.md`。
3. `PAPER/supplementary.md`、`PAPER/build_manuscript.py`。
4. `EVID_ART/METHODS_FACT_SHEET.md`及`PAPER/analysis/timing_contributions.csv`。
5. 当前`PAPER/references.bib`、`PAPER/references/reference_ledger.csv`、`PAPER/SKILL_USE.md`。

按问题定向读取源码，而非重新审计全部仓库：

| 修订内容 | 源码/结果依据 |
|---|---|
| Actor真实状态、逐步执行、固定总预算 | `src/cmc_bbdm/cai_agent_v3/actor_training.py::_actor_forward/_evaluate_one` |
| VLM首步候选与后续输入 | 同目录`policy.py::vlm_first_action_mask`、`models.py::SpatialCAIActor.forward` |
| 预测器训练 | 同目录`predictor_training.py::_sample_training_masks/_train_candidate/evaluate_predictor/run_oof_reward_predictors`及新W2实际manifest |
| 共同预测器输入与输出 | 同目录`models.py::MeanSCPredictor` |
| 指标定义 | 同目录`metrics.py`、`paper_evidence_math.py` |
| 时机解释 | `PAPER/analysis/timing_contributions.csv`、`timing_identity_checks.json` |
| 真实图与案例 | `PAPER/FIGURE_INDEX.csv`、`EVID/case_figure_reuse.csv`及原图指针 |

若单个文件名字在当前版本有所变化，先从其父目录、现有交接或manifest定向定位一次；不得根据文件名猜科学内容，也不得回到历史BC主张。对新发现而非本清单已确认的问题，先记录证据，再决定是否属于本轮范围。

## 2. 先修生成链，防止改稿在编译时被覆盖（R0）

已核对`build_manuscript.py`的真实行为：

```text
abstract.md + sections/01...06_*.md
    → manuscript.md
    → 各section.tex + main.tex
    → manuscript.html
supplementary.md → supplementary.tex
reference_ledger.csv → Markdown文末引用列表
references.bib → TeX/HTML引用
```

此外，当前脚本把`declarations`直接写在Python多行字符串中；仅修改`manuscript.md`里的声明，重建时会被覆盖。

因此：

- **正文修订以`abstract.md`和六份分节MD为编辑源。不要只修改生成的`manuscript.md`、`main.tex`或`sections/*.tex`。**
- SI修订以`supplementary.md`为源。
- 建议将硬编码声明迁移到新增的`PAPER/declarations.md`，让构建脚本读取它，形成单一声明来源。只作这项薄修改，不开发新的论文构建框架。
- 摘要解析目前假定title/abstract/keywords三块。保持兼容；确需调整时同步修改解析并作一次小范围检查。
- `reference_ledger.csv`与`references.bib`必须同步，避免MD显示2017而PDF显示2019。
- 使用已安装工具链；`/tmp`中的上轮Pandoc路径可能失效，先检查`PANDOC`和`command -v pandoc`，不机械复制旧临时路径。

验收：源MD的关键修订出现在生成MD、TeX和PDF中；再次从源构建不会恢复旧措辞。来源链在新交接中简短记录。

## 3. 科学方法的四项定向修订

### R1：补全Algorithm 1的内部观测输入和更新

当前伪代码列出`S, V, M, H, p and budget`，遗漏已经取得的内部C-scan内容；`masked_X`也没有明确初始化。这是伪代码表达缺项，不是研究代码需要重训或重写。

把算法写成与真实执行等价的以下过程，统一采用0—63编号。使用`X_obs`表示逻辑上的可见内部缓冲；说明实际回放实现可保存完整环境缓存，但模型混合内容之前执行mask，与该逻辑接口等价。

```text
Input: full surface descriptors S; cached VLM features V;
       frozen actor pi_theta; frozen CAI predictor P_phi;
       replay environment E; native cell costs d; total budget B = 0.25
Initialize M = 0, H = 0, X_obs = 0, c = 0, t = 0
p = P_phi(S, X_obs, M, c)
while True:
    L = {i in {0,...,63}: M[i] = 0 and c + d[i] <= B}
    if L is empty: break
    C = L
    if t == 0 and the VLM has affordable highest-reliable candidates:
        C = L intersect C0
    z = (S, X_obs, M, H, V, p, c, B-c)
    scores, _ = pi_theta(z)
    a = argmax of scores over C
    x_a = E.acquire(a)
    X_obs[a] = x_a
    M[a] = 1
    t = t + 1
    H[a] = t / 64
    c = sum of native cell costs over measured cells
    p = P_phi(S, X_obs, M, c)
    record the actual action, observation state, cost, and prediction
Return acquisition sequence and predictions
Compute errors with y only on the training/scoring side
```

在最终论文中保留可执行规则的必要精度说明（浮点容差等可放SI），不要让伪代码增加真实实现没有的停止、实时学习或语言规划功能。

必须与正文和Fig1一致：VLM提供先验，不逐步生成动作；Actor每步决策；预测器每次观测后更新输出但不更新权重；C0硬限制只用于首步，后续VLM数值特征仍保留。`value_head`不是CAI预测头或STOP头。

### R2：补全CAI预测器的训练，不再只说“冻结了一个模型”

在§3.3或§4.3补一个简洁段落（建议150—220英文词，非硬限）；在SI S1补一张**独立于Actor设置**的预测器训练表。

已核对源码支持以下内容：

| 项目 | 真实设置/应写内容 |
|---|---|
| 目标 | 真实CAI强度，MPa；不是BC动作或分层mask |
| 训练mask | 概率10%零内部观测；60%为1—16格；20%为17—48格；10%完整64格 |
| 小观测集合 | 从RANDOM、CENTER_FIRST、GEOMETRY_SPREAD、SERPENTINE中选择路线前缀 |
| 大部分观测集合 | 随机排列选17—48格 |
| 采样语义 | 上述是随机mask抽样概率，不是每个实际batch必然满足的固定件数 |
| 监督损失 | `Huber((prediction-y)/s, 0)`，delta=1，`s=max(train_target_std,1 MPa)`；按源码核实归一化的具体使用范围 |
| **预测器batch** | **32**（`predictor_training.py::_BATCH_SIZE`），不能套用Actor的16 |
| 预测器优化 | 按`_train_candidate`核实AdamW、学习率、weight decay、梯度裁剪、最多更新数及验证间隔 |
| 选择过程 | 三个共同预测器候选，按真实固定VALID前缀及原选择规则得到MEAN_SC@1750，不写成人工预先指定该网络获胜 |
| OOF回报 | 选中结构的三个交叉拟合模型按TRAIN来源组划分；当前试样由未拟合其来源组的模型提供训练反馈；不是三个模型部署集成 |

补充说明：用于训练预测器的17—48格和完整观测状态，是其输入覆盖设计；不意味着Actor执行预算被扩大到25%以上。

必须区分：

- 预测器训练监督损失（Huber）与策略训练代价（`A+0.25 e_T`）；
- 预测器batch=32与Actor batch=16；
- 共同预测器的VALID评分前缀与Actor自身的轨迹；
- OOF排除当前试样组拟合，不等于新的外部测试或没有任何选模依赖。

对未在源码与已执行manifest中核实的值，标注准确缺项，不从技能模板补造。不得为补写方法而运行训练入口、`precheck_predictor_training()`或加载模型验证。

### R3：准确说明小预算点是总预算0.25策略的前缀

在§4.4明确：

> All lower-cap estimates are queried from prefixes of trajectories generated with a total acquisition budget of 0.25; the policy is neither retrained nor rerun with a different total budget for each reported cap.

允许按上下文润色，但保留科学含义。执行时Actor得到的剩余预算是`0.25-c_t`，不是每个展示阈值`b-c_t`。

本修改不改变既有表格或查询。使用“performance at a common acquisition cap along the recorded trajectory”等准确措辞，不写成已验证任意预算条件的最优策略。实际采集比例可能小于cap，表格中相应范围继续保留。

### R4：统一统计层级、编号和符号

- 等质量分析中的`group-level MAE curve`改为`cohort-level MAE curve`或`pooled specimen-level MAE curve`，全稿统一。这里的MAE不是按capture group均值等权计算。
- 保留正确顺序：Random每试样先平均重复损失→池化物理试样；RMSE由池化MSE开方；A/early A为试样内repeat→域内均值→六域等权；capture group用于bootstrap依赖结构。
- 0—63与代码、VLM编号和原案例图一致。不得改原图数字迎合1—64公式。
- 分别定义原始图像与512维图格描述符，避免同一个`X`在“像素、特征、掩膜后状态”之间无说明切换。
- 将训练时`P_{-k}`/对应OOF预测器与评价时`P_all`说清，不让Algorithm 1（评价流程）暗示使用真实标签或OOF模型集成。
- 区分原生像素采集比例、cap、动作数、目标MPa和轨迹面积MPa；不把57.049（域等权初始误差）与58.551（池化初始MAE）强行改成同值。

## 4. 论文叙事的定向修订

### R5：重写摘要与引言，压缩审计语言，但不删改结果

保留标题的任务驱动采集方向和六章结构。重点修改摘要、引言收尾、相关工作冗余段、§5解释段与结论；已准确的其他段落优先保留，只做必要衔接。

#### 中心主张

> We organize C-scan acquisition around the CAI assessment objective. A state-dependent policy uses surface priors and acquired internal evidence to allocate subsequent observations, so both the acquired subset and the timing of its availability contribute to prediction quality under a fixed acquisition budget.

这是可根据稿件调整的论述框架，不是需要原样粘贴的口号。正文用准确的状态、损失和比较具体解释。

#### 三项贡献的中心

1. 将全过程CAI评估质量写入顺序采集任务，而不只处理采集完成后的图像回归。
2. 连接表面先验、可见内部证据、当前预测与下一动作，形成状态依赖采集方法。
3. 在共同预测器下，分别通过同cap性能、时机贡献和等质量采集需求说明方法作用。

不要以“建立日志、完成哈希、通过审查”充当科学贡献；也不把标准Transformer或策略梯度宣称为原创算法。

#### 摘要建议组织

工程问题 → 方法与关键设计 → 真实评价范围 → 一到两组决定性数值 → 具体的过程发现/工程含义。建议180—230英文词，属编辑目标，不冒充AEI限额。

可使用已确认的主要数值与“相对开环的最大正向阶段差异出现在第二完成阶段”这一过程观察。不能保证统计显著、因果模块贡献或全扫描非劣。

摘要中无需逐条复述四个区间、所有负消融和全部部署限制；**但必须准确注明离线回放与validation评价，不能把选模集改称独立test或笼统宣称所有方法中主方法最好**。决定结论的无VLM强对照与不确定性在主结果表和正文消融段可见。

#### 限制信息的正确位置

| 内容 | 放置方式 |
|---|---|
| 50件VALID参与checkpoint选择、单策略seed面板 | 实验协议集中写清；摘要如实称validation，不以test替代；结论必要时简短界定 |
| 关键配对区间及无VLM更优 | 主结果/消融展示保留，正文一句直接报告方向与读数，不每段重复四区间跨零 |
| 完整输入质量差距 | 主性能表与完整信息讨论保留 |
| 群体等质量并非单件STOP、cap不是时间 | 在定义及相关图注清楚限定；不要另起硬件开销研究议题 |
| 读取了摘要而未读全文、工具检查状态、哈希和浮点残差 | 引用台账、SI或交接，不作为每节的修辞中心 |
| 作者待确认事项 | AUTHOR_INPUTS及一处封面/声明占位，不散入科学段落 |

禁止将这项任务解释为删掉真实限制。也禁止用反复自我否定替代方法分析。每次出现限制，应回答该处的具体解释问题，而不是附上一段通用免责文字。

### R6：将§5.3已有时机分析提升为解释核心

不重算`timing_analysis.py`，直接引用已保存的`analysis/timing_contributions.csv`及已有图。

当前主方法相对开环的四阶段时机加权贡献差，四舍五入约为：

| 完成阶段 | 主方法减开环的贡献差（MPa） |
|---|---:|
| (0,0.0625] | +0.415 |
| (0.0625,0.125] | +1.296 |
| (0.125,0.1875] | −0.223 |
| (0.1875,0.25] | −0.097 |

这些数值须以表中未舍入数据核对。第二阶段的正差最大，四阶段净和对应A改善约1.391 MPa。

围绕这个结果写“对哪里、在何时形成预测价值”的分析：

- 同一个预测器固定了评估映射，观测组合和取得时机是比较对象。
- 相对开环的最大正向阶段差异位于获得初始内部信息后的第二采集阶段；不能仅用“起点选得好”概括全程差异。
- 阶段以观测完成成本归类，其权重影响之后到B的整个区间，**不是第二阶段局部积分下降了1.296**。
- 保留后两阶段负贡献，不写“每步持续纠错”或“93%收益由反馈造成”。
- 恒等式与signed contributions保留；逐episode残差`10^-14`等算术自检移到SI，正文不把浮点闭合作为新的经验性能证据。
- 未记录语言推理、注意力或物理损伤真值，不补造此类解释。

建议段落链为：观察 → 与开环/固定排序对照 → 时机与预测变化的解释 → 一个短而具体的适用说明。不要先连续列五句“不是……”。

## 5. 图表、参考文献与声明

### R7：减少重复展示，增强真实过程图

1. **原Table 4保留在主文**，完整展示四项差值与区间；原Fig3（同样的四项forest plot）移到SI。不要为凑“六幅主图”继续主文双重展示，也不要因移图删除区间。
2. **原Fig5升级为同一既定案例的短过程图**：优先选c8-16既存的1/4/8步状态图中可读的2—3幅；读取原manifest确认文件，保持时间顺序、原图比例、已测与未测颜色含义和真实下一动作。
3. 不再选新试样，不删除q24-48不利案例；其他两个固定案例继续保留在SI。
4. 新图只作确定性排版/组合，源PNG内容不修改。允许文字面板标签、步数、真实成本/预测注释，但必须来自该案例既有表；同k图用于说明过程，同成本性能仍用真实b对照。
5. 不用生成式图像服务，不新造mask，不从完整图补“未测”的真实观测，不运行模型恢复缺图。既有面板缺失时保留可用面板并写明，不阻塞其他修订。
6. 全部Figure/Table编号与交叉引用重新同步，更新FIGURE_INDEX和图注。主文图数可由6变5，表数也以实际组织为准，不设僵硬图表数量验收。
7. 只对新增/重排的图做单图字号、碰撞和视觉检查；整稿编译后确认其他图没有因缩放变得不可读，不重复重绘未改图。

图1的训练/执行信息流还需与R1一致：新增内部证据必须进入Actor，真实CAI只进入监督/评分侧。若图本已表达正确，不为审查而重画。

### R8：修正已知引用版本，清除检索过程语言

- 相关工作正文移除“their institutional abstract”“only abstract accessed”等获取过程表达。以实际读到内容支持的技术描述介绍文献，不因此增加未经核实的机制或缺失能力断言。
- 引用台账仍如实保留摘要/部分章节/全文的读取范围；不能因正文删除检索说明就将台账升格为FULLTEXT_READ。
- 已核实Janisch等论文存在正式AAAI 2019记录，可保留`janisch2019`键并同步更新引用元数据：

```text
Title: Classification with Costly Features Using Deep Reinforcement Learning
Authors: Jaromír Janisch; Tomáš Pevný; Viliam Lisý
Year: 2019
Venue: Proceedings of the AAAI Conference on Artificial Intelligence
Volume / issue: 33 / 01
Pages: 3959–3966
DOI: 10.1609/aaai.v33i01.33013959
Official record: https://ojs.aaai.org/index.php/AAAI/article/view/4287
```

- 若该段实际使用的是预印本特有内容，应保留相应版本依据，不把“正式发表版元数据”冒充“已读全文”。对ResNet、Transformer、AdamW仅在当前记录与正式版本明显不一致时做同类小范围核实，不为凑数量新加文献。
- 更新`references.bib`、`references/reference_ledger.csv`、相关工作表与生成MD文末列表；引用键改变时检查所有引用位置，优先不改已有键。
- 不要求19条必须变成更多，文献数量不是本轮完成标准。

### R9：集中管理声明与作者待确认项

- 从普通结果段、图1说明和讨论中移除“作者尚待审核”等工作提醒；在AUTHOR_INPUTS/新交接中保留。
- 保留真实AI使用披露，不降格为“仅拼写检查”，不删除研究中的Qwen说明。写作辅助与研究VLM分开。
- 工具精确型号仅在运行记录支持时写；不要仅从聊天工具自述或猜测补“GPT-6”等版本号。已确定使用OpenAI Codex可写产品名称，具体型号待作者核实。
- 不代作者确认署名、资助、利益冲突、公开许可或“所有作者已审核”。未知字段留一处中性占位或作者清单，不写成无冲突、无资助、已批准。
- 建议一处声明来源`declarations.md`；从build脚本移除硬编码长段，不让重建恢复已删除的内部提示。
- AEI官方格式、作者元数据未完成时维持`AUTHOR_REVIEW_PENDING`/`JOURNAL_FORMAT_VERIFICATION_PENDING`等交付身份，不声称投稿就绪；这些不阻塞正文和PDF修订。

## 6. Skill使用方式

先读`PAPER/SKILL_USE.md`记录的实际本地安装，不全局更新nature-skills，不要求为本轮重装工具。

- 主体是**定向论证修改＋润色**：对需要重构的摘要、引言和解释段使用`nature-writing`；已正确段落用`nature-polishing`局部修订。
- 保持`paper_type=methods`、`language=zh-to-en`、`journal=generic`并服从AEI已有规范；目标不是Nature，不引入Nature字数/版式硬限。
- 沿用主文分配规则：正文说明问题、方法、决定性比较与作用方式；SI保存细节。影响主结论的负向结果和区间不得被隐藏。
- 若nature-reviewer可用，仅做**一次AEI定向复核**；同上下文自查就如实记录，不伪称互盲同行评审，不为低分再开训练。
- 图形后端沿用已有Python代码。发生布局修改才调用相关绘图检查，不重新跑所有技能自检。

## 7. 推荐执行顺序与八项有限验收

| 阶段 | 实际工作 | 停止条件/产物 |
|---|---|---|
| P0 | 核对分支、读取输入、确定源文件与生成链 | 绑定当前修改源；不再询问“具体是哪项任务” |
| P1 | 完成R0—R4方法及口径补全 | 修订源MD与薄构建修改，源码依据明确 |
| P2 | 完成R5—R6叙事修改 | 更清楚的摘要、引言、结果解释和结论；保留必要证据 |
| P3 | 完成R7—R9图表/引用/声明 | 过程图、交叉引用、声明源与文献同步 |
| P4 | 从源重建MD/HTML/TeX，编译主稿和SI | 两个PDF真实生成；原数据不变 |
| P5 | 一次下表复核，只修失败项，然后commit/push | 闭合修订表和新交接 |

八项检查不是每项都需要一套测试程序；已有可用工具与少量人工/静态核对即可。

| 验收 | 必须检查什么 |
|---|---|
| Q1 源文件与构建一致 | 六章源MD、摘要、声明、生成整稿和PDF匹配；不再恢复旧文本 |
| Q2 方法完整 | 伪代码显式含X_obs；预测器训练mask/Huber/32-batch与Actor16-batch分开；VLM/Actor/预测器职责准确 |
| Q3 评价口径准确 | 固定B=.25的前缀说明；池化MAE与域等权A不混；编号0—63；训练/执行信息权限不变 |
| Q4 数据与主张未漂移 | 主表全部方法与冻结表一致；无VLM、四项区间、完整输入差距、50%/0%与反弹仍保留；阶段贡献数值未改 |
| Q5 论证集中 | 新摘要与引言有明确工程任务和机制；重复审计说明被压缩而非事实删除；第二阶段观察为解释重点而非因果百分比 |
| Q6 图表与引用 | 主文不重复Table4/Fig3；过程图来自原案例；编号/引用无断链；BibTeX与ledger版本同步 |
| Q7 构建与视觉 | 无缺字/未定义引用/内容截断；检查全部页面概览及改动段、算法、表格、过程图和SI转移图；不能用“编译成功”代替视觉检查 |
| Q8 交付与冻结范围 | 研究源码/结果/模型没有修改；实际提交推送；交接如实记录未完成项和本轮零研究计算 |

不跑整个仓库pytest、旧W2/W3模型校验或旧bootstrap。没有改动的统计模块不重新验证全部推导。正常LaTeX为解析交叉引用而多次运行属于构建，不等于多轮科学测试。

若发现新的数值冲突，不修改冻结果“修绿灯”。记录具体冲突及来源，暂停依赖该数值的句子，继续不依赖的工作。

## 8. 最终构建、交接与GitHub

沿用已安装的Pandoc和LuaLaTeX构建。以下为仓库根目录的参考方式，先核实可执行文件，不能假称运行过：

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
# 当command -v pandoc无结果但已有其他本地安装时，设置PANDOC为真实路径。
python paper_cai_aei/r1_84bea60e/build_manuscript.py
latexmk -lualatex -interaction=nonstopmode -halt-on-error -cd \
  -outdir=build paper_cai_aei/r1_84bea60e/main.tex
latexmk -lualatex -interaction=nonstopmode -halt-on-error -cd \
  -outdir=build paper_cai_aei/r1_84bea60e/supplementary.tex
```

本轮实际交付：

- 已修改的摘要、六章MD、SI和统一声明源；生成的整稿MD/HTML、TeX/BibTeX与两份PDF。
- 新/重排图及其生成脚本，准确图注、FIGURE_INDEX；不复制模型、原始数据或字体。
- 更新受影响的EVIDENCE_MAP、RESULT_ALLOCATION、引用台账、作者导读与待确认项；不要把旧审查改写成“当时已经通过新要求”。
- `NEW_ART/REVISION_RESPONSE.md`：R0—R9逐项，原位置→真实改动→新位置→依据→剩余事项；附Q1—Q8简短结果及各节词数变化。
- `NEW_ART/CODEX_HANDOFF_CAI_AEI_REVISION_R2.md`：实际入口/提交、修改范围、构建命令、PDF路径、零研究计算、未决作者事项和Git状态。

Git纪律：

1. 只stage本任务文件，不使用无范围`git add .`收走用户无关改动。
2. 研究`src/`、原`results/`和已冻结分析数表保持不变；PAPER内仅修改呈现和相应构建文件。必要普通文件hash最多开始/结束各一次，不重复整库哈希。
3. 正常commit并push到原v3分支；不PR、不merge、不force push、不reset。Git历史中的f4758829保留原稿。
4. 核对local HEAD、upstream和对应远端ref一致，检查最终PDF和MD确实已被追踪/推送，不只上传文本说明。
5. 文档记录实际结果提交，最终回复提供包含交接的最终SHA，不为把自身SHA写入文件而反复提交。

完成状态可以写`TARGETED_MANUSCRIPT_REVISION_COMPLETE`。这代表修订交付，不代表新增验证、作者已批准或期刊录用。

最终回复只需：R0—R9完成表、最重要的三项改进、实际整稿/PDF/交接路径、仍需作者决定的事项、三方SHA。不要只列检查数量，不再自动提出追加训练计划。

## 9. 本任务的已核对依据

以下为固定来源链接。执行时优先读取当前工作树对应文件，保留基点便于对照。本文中的修改安排是新任务要求；不要将其误写为旧代码已经具备的状态。

- **S1 初稿/章节**：https://github.com/Orangekostar/diff/blob/f4758829c621fa02a83595f1549e41dabe9e1c64/paper_cai_aei/r1_84bea60e/manuscript.md
- **S2 SI**：https://github.com/Orangekostar/diff/blob/f4758829c621fa02a83595f1549e41dabe9e1c64/paper_cai_aei/r1_84bea60e/supplementary.md
- **S3 真实生成链**：https://github.com/Orangekostar/diff/blob/f4758829c621fa02a83595f1549e41dabe9e1c64/paper_cai_aei/r1_84bea60e/build_manuscript.py
- **S4 预测器训练与mask**：https://github.com/Orangekostar/diff/blob/f4758829c621fa02a83595f1549e41dabe9e1c64/src/cmc_bbdm/cai_agent_v3/predictor_training.py
- **S5 Actor状态与执行**：https://github.com/Orangekostar/diff/blob/f4758829c621fa02a83595f1549e41dabe9e1c64/src/cmc_bbdm/cai_agent_v3/actor_training.py
- **S6 模型结构**：https://github.com/Orangekostar/diff/blob/f4758829c621fa02a83595f1549e41dabe9e1c64/src/cmc_bbdm/cai_agent_v3/models.py
- **S7 原指标**：https://github.com/Orangekostar/diff/blob/f4758829c621fa02a83595f1549e41dabe9e1c64/src/cmc_bbdm/cai_agent_v3/metrics.py
- **S8 聚合/等质量口径**：https://github.com/Orangekostar/diff/blob/f4758829c621fa02a83595f1549e41dabe9e1c64/src/cmc_bbdm/cai_agent_v3/paper_evidence_math.py
- **S9 真实时机分解数据**：https://github.com/Orangekostar/diff/blob/f4758829c621fa02a83595f1549e41dabe9e1c64/paper_cai_aei/r1_84bea60e/analysis/timing_contributions.csv
- **S10 原初稿交接**：https://github.com/Orangekostar/diff/blob/f4758829c621fa02a83595f1549e41dabe9e1c64/artifacts/cai_agent_v3/manuscript/r1_84bea60e/CODEX_HANDOFF_CAI_AEI_MANUSCRIPT.md
- **S11 Janisch正式出版记录**：https://ojs.aaai.org/index.php/AAAI/article/view/4287 （2019，33(01)，3959–3966，DOI 10.1609/aaai.v33i01.33013959）

此次提示词依据上一轮完整内容审查，并重新读取了分支引用、章节目录、构建脚本、预测器训练源码和时机贡献表；不是一次新模型或PDF执行检查。构建脚本硬编码声明、生成源文件关系及预测器batch=32，属于本次定向复核确认的具体事项。
