# Codex主执行规范：CAI Agent的AEI六章论文写作

日期：2026-09-14。任务：`CAI_AEI_MANUSCRIPT_R1_84bea60e`。
仓库：`Orangekostar/diff`。接续分支：`research/cai-vlm-agent-v3-controlled-reuse`。
已核对研究SHA：`84bea60e3fd2016b0b18379cd2a2a473200c2b4d`。
参考skill仓库：`Yuan1z0825/nature-skills`；已读版本`9ea7330a17813a15421fe843778a776c258b9001`。
下文[Rxx]、[Sxx]、[Jxx]、[Lxx]见AEI_WRITING_SOURCE_BINDINGS.md；[D]表示本轮提出的实施安排，不是已完成研究。

## 0. 最终目标与已确定的论证

用户已经同意停止新实验，采用六章结构撰写AEI工程方法论文。本轮要**实际写出完整初稿**，不是再次输出计划，也不是只将交接翻译为英文。

工作标题：
**Learning what to inspect: Task-driven multimodal C-scan acquisition for compression-after-impact assessment**

中心论证：
> 将C-scan图像采集组织为面向CAI评估的顺序决策，用表面先验和当前已测证据更新采集位置；在共同预测器下分析观测子集、获取时机与预测质量的关系。

贡献分别落在任务表达、可见证据驱动的采集方法、质量—采集量比较与过程解释。VLM负责表面候选和首步先验，学习型Actor负责每一步决策，独立CAI预测器负责强度估计。不是VLM每步语言规划，不是旧P4示范BC，不是损伤mask或STOP研究。[R02,R05–R08]

### 0.1 权限与禁止项

| 允许 | 禁止 |
|---|---|
| 核实相关工作、数据来源和AEI规则；读取现成CSV/JSON/NPZ/图稿/源码 | 新训练、拟合、研究模型前向、重新执行轨迹、参数/阈值/seed搜索 |
| 写六章英文初稿、补充材料、参考文献、作者审阅说明，编译和有限图文QA | 新TEST/外部样本接入、GDFS、插值/稀疏采集等新基线、VLM新调用 |
| 一次基于已有acquisition_events的收益时机恒等分解 | 重新生成bootstrap、重选checkpoint、反事实选点执行、伪造注意力/GT |
| 复用旧图；新增方法流程示意图和至多一幅时机分解图 | 生成或修饰成“真实数据”的试件/超声图、训练日志或人类标注 |
| 生成投稿辅助草稿和待确认表；提交推送本分支 | 替作者批准署名/基金/利益冲突/原创性声明；实际投稿、PR、合并、force push |

“研究模型新前向=0”不禁止Codex自身的写作工具调用。所有基于冻结结果的派生分析仍是事后VALID分析，不增加试样数或独立验证。[R01,R03]

### 0.2 写作范围与篇幅[D]

英文正文固定六个编号主章节：1 Introduction；2 Related work；3 Task-driven multimodal inspection framework；4 Experimental design；5 Results and discussion；6 Conclusions。
摘要、关键词、声明、参考文献和SI不计作额外主章节。默认英文稿；任务记录、中文论证导读与作者问题清单用中文。正文规划约6,000–7,500词，摘要草稿约200–230词：**这些是本轮写作目标，不是已经核实的AEI限额**。不为凑字数填充内容。

Nature写作skill采用task=manuscript、paper_type=methods、language=zh-to-en、journal=generic。methods是skill的论证类型，不等于AEI正式文章类别。按AEI官方要求核对最终文章类型。[S01–S07,J01]

## 1. 来源与输出隔离

以下相对于真实仓库ROOT：

```text
DATA = results/cai_agent_v3/new_protocol/
W2   = results/cai_agent_v3/w2_replay/r1_292b1c74/
W3   = results/cai_agent_v3/w3_pilot/r1_0e11452a/
EVID = results/cai_agent_v3/paper_evidence/r1_e2a11154/
EART = artifacts/cai_agent_v3/paper_evidence/r1_e2a11154/
PAPER = paper_cai_aei/r1_84bea60e/                 # 本轮拟用新根
WART  = artifacts/cai_agent_v3/manuscript/r1_84bea60e/  # 本轮拟用新交接根
```

DATA/W2/W3/EVID/EART及历史研究代码均只读。现成paper_aei_information_hierarchy、paper_v3等可能属于其他工作，不能据名字沿用其科学叙事或覆盖它们。

先检查当前工作树、HEAD、分支和一次fetch。存在后续提交或未提交内容时保留并注明关系；不reset、不自动切到main，不把另一个论文目录当本轮输出。若已有同任务正文，从已完成阶段续写，不重新整体生成。

本包放入`docs/cai/aei_manuscript/`，在已有TASK/STATE写真实主规范路径、阶段与终点。解除旧“任务未绑定/无训练授权”的非相关阻塞。不要另建任务管理系统。

### 1.1 必读事实包（按需定向扩展）

- EART：CODEX_HANDOFF_CAI_V3_PAPER_EVIDENCE.md、METHODS_FACT_SHEET.md、PAPER_RESULTS_DRAFT.md、CLAIM_EVIDENCE_MATRIX.md、LIMITATIONS_AND_SCOPE.md、FIGURE_CAPTIONS.md。
- EVID：final_manifest或analysis_manifest实际入口；method_summary.csv、same_cost_metrics.csv、same_cost_paired_summary.csv、mechanism_area_intervals.csv、same_cost_metrics_by_domain.csv、full_scan_reference.json/full_scan_gap.csv、equal_quality_grid.csv/equal_quality_anchors.csv、quality_targets.csv、cost_error_event_curves.csv。
- 过程：acquisition_events.csv、frozen_episode_index.csv、mechanism_event_summary.csv、path_divergence.csv、case_same_cost_states.csv、case_figure_reuse.csv；不重复读取全部候选模型。
- W3：最终actor_manifests/protocol_snapshot/INPUT_BINDINGS实际引用，仅核对事实与身份，不加载权重。
- DATA：索引和split、来源/配准元信息。六域铺层/厚度等须查实际记录，不能按ID或c8/q24后缀猜测。

旧上传的9月9日AEI_CSCAN_AGENT_NEXT_SESSION_HANDOFF是历史BC路径，只能说明背景。24件Hasebe追溯中的“0/24原始阵列”不能扩写为当前276件全部没有原始数据。当前方法使用截图图格是代码事实，不依赖这种扩大结论。[R12]

### 1.2 源码到文章绑定

| 文中对象 | 需阅读的实际文件/符号 | 写作任务 |
|---|---|---|
| 表面先验 | learned_cscan/perception.py的SURFACE_PERCEPT_PROMPT；cai_agent_v3/vlm_perception.py::_features | 说明VLM看表面、输出区域和序数置信；Actor不使用全文语言推理 |
| 首步初始化 | cai_agent_v3/policy.py::vlm_first_action_mask | 最高可靠medium/high等级限定首动作；后续解除硬限制但保留VLM特征 |
| 图像表示 | cai_agent_v3/feature_bank.py::_cell_crops/build_feature_bank | 先独立裁格再编码；区分环境完整缓存与模型可见输入 |
| 共同预测器 | models.py::MeanSCPredictor | 同一冻结预测器、已测集合/坐标/成本；完整输入仍有表面信息 |
| 学习决策 | models.py::SpatialCAIActor/MeanFeedbackActor/TrueStaticActor | 状态变量、两层4头128维、合法动作、critic与CAI头分工 |
| 真实训练/执行 | actor_training.py::_training_rollout_loss/_predict_by_fold/_evaluate_one | 交叉拟合回报；TRAIN采样、VALID argmax；每步再决策，非在线更新 |
| 指标 | metrics.py::left_error_area_mpa/trajectory_objective_mpa | 左阶梯A、.25终点项、误差变化时机恒等分解 |
| 当前表格定义 | paper_evidence_math.py与paper_evidence.py | repeat先算损失；A六域等权、MAE池化；等质量先群体后逆查询 |
| 数据来源 | cohort.py/cohort_export.py及其真实引用的清单 | 数据流图、CAI标签/单位/试样对应；只读，不重跑构建 |

完整文件路径和当前已知事实见SOURCE_BINDINGS。逐段内部使用`% evidence: path | row_key`或独立证据表，**不把仓库路径和审计ID大量写进论文正文**。

## 2. P0：用skill建立写作契约，而不是重新设计研究

按NATURE_SKILLS_AEI_ADAPTER.md读取本地相关skill；记录实际路径与版本。已有安装优先，无需全局更新或全量克隆依赖。公共参考仅用于需要的片段，禁止外发未发表稿至额外审稿网站/API。

本阶段最多三份紧凑工作表，可合并：
1. `AUTHOR_ARGUMENT_MAP.md`：一句中心论证，六章问题链、工程问题→设计→证据映射。
2. `TERMINOLOGY_AND_NOTATION.md`：统一CAI、VLM、Actor、capture group、acquisition fraction、A/early A、同成本上限/等质量等；明确误差MPa不是准确率。
3. `RESULT_ALLOCATION.csv`：结果功能、正文/SI/图注/数据位置、支持方向与关键限制。

**不要停在这些表。**结构已经同意，默认继续起草。仅真正无法从来源解决的事实进入AUTHOR_INPUTS，不向用户重复询问是否写六章。

## 3. P1：补齐论文所需文献和来源说明

这一步是写作检索，不授权新实验。优先4条证据线：C-scan/损伤图像与CAI；自主/自适应超声采集；动态特征/任务驱动获取；本方法实际使用的算法与模型来源。

- 先复用当前repo已核实references，再沿REFERENCE_SEARCH_SEEDS定向补充。目标约20–35条直接相关参考仅作规模建议，不能凑引用数，也不设“必须读今年20篇”的资源门槛。
- 对最接近的3–5项工作，读取能支持差异判断的方法/评价正文；其余只为特定事实读取相应段。只有摘要时，只能写摘要支持的结论，不能标FULLTEXT_READ。
- 对最接近AEI实例优先核实图像→CAI论文，研究问题、输入是否先全量可见、目标损失、评价对象；避免“所有前人只扫描不评估/从未主动选点”。
- 生成`references/reference_ledger.csv`：citekey、题目、作者/年份/venue、DOI、来源、access/read_extent、支持句、所在段/页、限制。DOI与标题作者不匹配必须修正，不能由AI补齐。
- `references/closest_work_matrix.md`比较任务目标、决策前可见信息、反馈、评价与本文差异。没有全文依据的格子标未核实，不用推断补满。
- 导出真实`references.bib`。待核实条目存单独候选清单，不伪造BibTeX；无法访问关键文献则局部保留CITATION_NEEDED及准确缺口，继续其他段落。
- 数据集原论文、版本、许可和表面/内部配准出处须对应实际使用文件。公开数据不等于本地衍生图都能无条件重发。代码可访问不等于已公开，不能未经核实写“all code publicly available”。

AEI guide本次聊天抓取403；Codex从官方页面或用户本地保存文件定向核对一次即可。记录文章类型、摘要、reference style、highlights/graphical abstract、匿名要求、模板和声明的verified/unknown状态。失败不阻塞科学初稿；不使用仿冒Elsevier域名或泛用AI生成作者指南填空。[J01]

## 4. P2：唯一新增计算——收益发生时机的恒等分解[D]

目的：把第5章“观测组合与取得时机”变成可核验的解释，不预设收益必来自早期，也不将数学恒等式称为新机制验证。

对每条保存轨迹，令e_t=abs(p_t-y)、c_0=0、B=.25，定义：

    delta_t = e_(t-1) - e_t
    g_t = (1 - c_t/B) * delta_t
    A(B) = e_0 - sum_t g_t
    J = A(B) + .25*e_T

独立推导可放Methods简短说明、完整代数放SI。终点c_T<B的保持区间已被恒等式包含；c_t=B的g_t=0，但该次误差改变仍可能作用于终点项。负g必须保留。

### 4.1 实施

只用EVID的acquisition_events/frozen_episode_index及已有A表；必要时从冻结最终轨迹取e0。按每episode动作序号读取一次，不运行旧analyze/export，不加载模型。

- 四段按**观测完成成本**分箱：(0,.0625]、(.0625,.125]、(.125,.1875]、(.1875,.25]。
- 每episode先分箱求和，RANDOM在试样内均值，域内物理试样均值，再六域等权，与A保持同估计量。
- 单项g对整个剩余过程的贡献，不等于“只发生在该箱内部的积分改善”。图注准确称timing-weighted contribution grouped by acquisition completion stage。
- 全九方法输出，保留正负；主与任意对照总贡献差加上e0差，须还原原A差。不可将各模块的算术差写成因果百分比分摊。
- 只新增`PAPER/analysis/timing_contributions.csv`、`timing_identity_checks.json`、轻量脚本和至多一幅图。保留现成探索区间，不另抽样/做新显著检验。
- 最大约10,227事件一次CPU处理。`reference/timing_identity_reference.py`为独立玩具例子，不是把已有分数硬编码成研究结果。
- 数值不闭合时仅查该映射/单位；不修原W3数据来迎合恒等式。若确有保存字段缺失，文中保留数学解释、标记派生图未完成，继续正文；不新增前向。

## 5. P3：写Methods与Experimental design

按蓝图先完成第三、四章。必须出现一段流程算法伪代码、一个信息权限表和目标函数。每个方法模块用“工程决策需求→具体输入/处理→在闭环中的作用”说明，避免只列网络层数。

硬事实：
- 独立图格64×512；VLM不是内部超声Reader，不直接预测CAI或输出每一步工具指令。
- 主Actor是两层四头128维；CAI预测器是已选MEAN_SC@1750，不因文章想突出空间结构而改写预测器。
- 已测mask先屏蔽未知内部内容；全表面输入可见。研究训练有数据缓存不等于部署可见全C-scan。
- 同一个P_all评价九方法；OOF用于TRAIN回报，不能称三模型集成的预测收益。
- 原预算B=.25，early=.0625，唯一原生像素计费；按合法可负担动作结束，不存在已验证STOP/实际设备控制。
- 总276/259来源组；TRAIN161/152，VALID50/48，TEST65/59尚未评价。论文结果用50件VALID，不能换称test set或暗示276件全用于性能评价。
- 单策略seed面板，各方法初始化种子不同；重复/事件/断点不是独立N。
- MAE等与A的聚合分别说明；选择checkpoint的VALID用于当前报告，bootstrap是事后条件探索，非独立确认。[R02–R09]

数据域缺少某项铺层/厚度时用真实domain ID及已证字段，不猜配置。作者/许可等缺失用AUTHOR_INPUTS链接，主体数据/方法已知部分照常完成。

## 6. P4：写Results and discussion，并安排图表

以已有PAPER_RESULTS_DRAFT为事实底稿，不逐句保留其审计口吻；按五个问题写第5章：
1. 同成本上限下谁预测更准？
2. 共享固定→试样相关开环→反馈组织，性能如何变化？
3. 时间加权误差变化揭示哪些实际阶段贡献？
4. 同一经验质量目标下，双方最早需要多少采集？
5. 完整输入差距和不同域/个体上的适用性是什么？

正文句法：先报告研究动作/观察，再给决定性数值和比较，最后给与证据相称的解释。不要以“证据不足”开头每个段落，也不要写“strongly demonstrates”代替缺失的统计支持。

### 6.1 必须在主文保持可见的事实

- 主法25%上限MAE44.286 vs geometry46.910；实际fraction并非每件恰.25。关键配对区间[-.488,5.925]在主表或相邻文字出现一次即可。
- 主法A45.110、static47.977、open46.501；四个既存机制区间均含0，应在同一主消融表/图明确可读，非仅藏于SI。
- 无VLM对照A43.597、终点42.385更低；正文用一两句数据说明，不用整段反复“无法归因”，更不能隐藏或换主模型。
- 空间相对均值仅.152A差；空间终点MAE更高而RMSE稍低，不能只留有利A。
- 全输入41.690MAE/.711R²为同模型成本1.0参照；部分采集全范围未达到其MAE。不称无损替代完整检测。
- 主网格q46.9099时对geometry50%而对static0%；event-grid不同且有反弹，不能改用更漂亮的一组普遍宣称节省。
- 区间与适用范围集中放实验协议、主结果图表和一段讨论；不要在每个caption堆满重复免责声明。

### 6.2 图表[D]

优先保留已有图，不使用图像生成API。主图/表结构见蓝图。允许新增**一张确定性工作流示意图**（可将状态更新并入，不含新数据图像）和**一张收益时机图**；已有21张案例只取原三件，无需全部塞入正文。

来源CSV→图→正文数值建立映射。已有图重编号/矢量排版不改变数据；不得平滑、截断负效果、填补.25至1.0曲线。复用图若文字过密，仅在新目录从原CSV重排，输出相同数据和标明出处。

统计显示用真实n/units，避免自制error bars。框架图不得把未测数据连到Actor，不能画GT反馈到部署侧、假STOP或语言思维链。

## 7. P5：Related work、Introduction、Conclusions与前置材料

方法与结果稳定后完成第二、一、六章，最后写题目、摘要、关键词。文献差异从read_extent支持的内容出发，不以没有读到等于对方没有该能力。

- 引言从“有限测量支撑剩余强度评估”起，不从大模型浪潮/无出处工业低效开始。
- 贡献用formulate/develop/characterize等可被本文完成的动作，不能把现代标准RL/Transformer包装成首次理论创新。
- 摘要清楚标明离线C-scan图像回放及50件验证评价；选择1–2个代表结果，不只摘最有利等质量阈值。若摘要称比较优势，用observed及确切范围，不称普遍显著；结论不能比结果更强。
- Related work保留自主超声选点和直接C-scan→CAI的最近邻，不能稻草人化工业方案。本轮不重开其baseline实验。
- Discussion含工程启示、与文献差异、质量—采集权衡、局限与未来验证一条链；不每段自我否定，也不把后续实验建议变成本轮任务。

## 8. P6：完整可编辑稿与作者审阅材料

建议目录[D]：

```text
PAPER/
  manuscript.md
  main.tex
  sections/01_introduction.tex ... 06_conclusions.tex
  references.bib
  references/reference_ledger.csv
  references/closest_work_matrix.md
  supplementary.md
  supplementary.tex
  figures/                         # 复制/引用的小型最终图，保留图源
  tables/                          # 正文精简表+完整SI/source tables
  analysis/                        # 唯一时机派生分析
  build/                           # PDF/log；不提交大量中间文件
  submission_drafts/               # 草稿身份明确
  AUTHOR_INPUTS.md
  AI_USE_RECORD.md
  AUTHOR_READING_GUIDE_ZH.md
```

1. 主MD与LaTeX来自同一段落/数字基准，避免维护两份互相漂移的正文。可以MD为审阅稿、分节TeX为排版稿，但最终需核对语义/数字相同。
2. 用现有可用LaTeX工具链；优先适合Elsevier的已安装模板，AEI具体要求未核实前明确是初稿模板。不让Nature模板将Methods放到文末打乱六章。
3. 在已有工具链下实际编译主稿和SI并渲染看页。若依赖缺失，限定为一次常规解决；没有权限不大规模安装或下载系统包。交付完整源码和实际BUILD_BLOCKER，不把缺PDF冒充已经编译。
4. 正文表可紧凑，45/840/270行的完整资料不必全堆正文；SI提供必要完整比较和索引，机器可读CSV作为source data。决定主结论的不利事实仍在主文。
5. 投稿辅助稿：题目/摘要/关键词、4条highlights草稿、科学内容cover-letter草稿、数据/代码可用性和AI使用声明草稿。它们不冒称AEI均强制；准确字数/匿名/声明要求写入JOURNAL_PROFILE待核实项。
6. 作者姓名、顺序、通讯地址、基金号、CRediT、利益冲突、所有作者同意投稿等不能推断。未知写显式待填，不把“无利益冲突”“全部同意”预填为事实。内部稿可含占位，标AUTHOR_REVIEW_REQUIRED。
7. Qwen在研究中的使用记Methods；Codex/其他实际写作辅助另记AI_USE_RECORD，名字/版本/用途/作者审核状态据实填写，不虚构模型版本或完成的人工审阅。保留当期出版社要求的披露待作者确认。[J02]
8. 流程图仅说明已实现流程，定量图必须由数据可复现绘制；不使用通用生成式图像制作投稿图形摘要，不生成试件/损伤影像。图形摘要如官方另需，本轮先交基于现成主图的文字brief和作者制作项，不新增外部图像服务。[J02]

中文AUTHOR_READING_GUIDE约2页，按“每章主张→证据→作者需确认点”解释，不全文重复翻译英文稿。

## 9. P7：一次有限内部审查、局部修订与交付

执行本包REVIEW。使用nature-reviewer时明确覆盖其默认3报告为**一次AEI定向内部审查**；默认不声称互盲/独立。若真正用独立子上下文，记录实际模式，不编造身份或录用概率。

顺序：完整初稿→一次审查→一次针对性修订→只复核修改处。审查指出科学边界应如实记录；本轮以准确措辞和显示修正，不改数据/启动新实验，不把Major concern自动降级为小问题，也不以模拟评分不够为由无休止重写。

只做8类：六章与论证完整、方法代码一致、事实/数字映射、同成本/等质量、区间与主张、引用/数据来源、排版可读和写作范围、Git真实交付。新增时机脚本最多6个玩具数值检查+真实恒等闭合；不运行全repo pytest、W2/W3全审计、fuzz或压力测试。

对实际新/改图使用已安装nature-figure的有限图形QA并视觉查看；未改图不重跑整套。正文编译检查引用/公式/表图溢出；发现问题只重编受影响稿件，不为降AI检测分数改写事实。

## 10. 最终验收与GitHub同步

WART生成：
- `CODEX_HANDOFF_CAI_AEI_MANUSCRIPT.md`：六章完成情况、来源SHA/实际代码SHA、正文文件、图表映射、参考核实状态、作者待确认、编译/审查结果、真实命令、写作零研究计算、Git状态。
- `WRITING_REVIEW.md`和紧凑`DELIVERY_MANIFEST.json`；可合并多个小记录，禁止庞大审计册。

交付状态分开：
`MANUSCRIPT_DRAFT_COMPLETE` / `AUTHOR_REVIEW_PENDING` / `JOURNAL_FORMAT_VERIFICATION_PENDING` / `BUILD_BLOCKED`按实情；不把“稿件写完”称PAPER_ACCEPTED或研究外部验证完成。已知性能验证边界不阻塞所有章节完稿。

只提交本包、论文正文、必要图源/数据链接、唯一派生脚本与结果、有限测试及交接；不重传模型/六feature shards/全量raw数据/字体文件，不改论文证据根。`.bib`必须含所有实际引用，不含虚构条目。

用户沿用之前的Git交付要求：在同一研究分支实际commit并push，核对HEAD/upstream/`git ls-remote`目标引用一致。push受权限/网络拒绝则保留本地提交并报告真实失败，不能声称同步成功；不force push，不PR、不merge、不reset。记录结果提交，再用最终回复报告包含记录的最终SHA，避免自引用提交循环。

最终对用户回复：六章/表图/引用/编译/待作者事项的简表＋完整稿件入口＋交接路径＋三方SHA。不能只回“规范已读、计划已生成”或只列测试数。
