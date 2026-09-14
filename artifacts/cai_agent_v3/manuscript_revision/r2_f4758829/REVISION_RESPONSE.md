# AEI targeted revision R2 — response and finite review

状态：TARGETED_MANUSCRIPT_REVISION_COMPLETE。基点：f4758829c621fa02a83595f1549e41dabe9e1c64。P指paper_cai_aei/r1_84bea60e；A指本目录。

本报告记录一次nature-reviewer方式的同上下文AEI定向自查，不是互盲同行评审，不给录用概率。配置methods / zh-to-en / generic；nature-writing用于论证重写，nature-polishing用于局部润色，nature-figure仅组合既有图像。科学证据身份仍是参与选模的50件VALID、48组、六域、单策略seed面板。

## R0—R9逐项回应

| 项目 | 原位置与问题 | 实际改动及新位置 | 依据 | 剩余事项 |
|---|---|---|---|---|
| R0 | build_manuscript.py硬编码声明；仅改生成稿会丢失 | 新增declarations.md，脚本只薄改为读取文件；编辑摘要/六章/SI源MD；生成MD/HTML/TeX/PDF同步，重复源构建10文件逐字节一致 | P/build_manuscript.py；A/SOURCE_REBUILD_CHECK.json | Pandoc为本机已安装临时路径，迁移时配置PANDOC |
| R1 | §3.4 Algorithm1缺可见内部缓冲和初始化 | §3.1定义X_obs；§3.4完整算法含零初始化、全部actor输入、0—63、H[a]=t/64、原生成本和首步C0；评价用P_all；PDF p8完整单页 | actor_training.py::_actor_forward/_evaluate_one；models.py；policy.py | 无；逻辑缓冲与真实mask实现等价已说明 |
| R2 | §3.3缺预测器训练、易与actor混淆 | 新训练段及SI S1独立训练表，Huber/归一化/32-batch、10/60/20/10%mask、优化与选择；actor仍16；共同模型与OOF区分 | predictor_training.py::_sample_training_masks/_train_candidate/evaluate_predictor/run_oof_reward_predictors；W2 predictor_gate/oof_readiness及实际manifests | OOF仍共用VALID选checkpoint的依赖如实保留 |
| R3 | §4.4小预算点易误读为逐预算策略执行 | 明确所有小cap来自总B=.25轨迹前缀；actor剩余预算.25−c，不按查询cap重跑；SI S3同步 | actor_training.py::_actor_forward/_evaluate_one；现存查询协议 | 不宣称任意预算条件最优 |
| R4 | 等质量group-level措辞、编号与图像/特征符号混淆 | §3.1原图I^S/I^X、S/X为64×512特征；§4.4 pooled MAE与domain-equal A分别定义；0—63；P_-k/P_all分开；57.049与58.551保留 | models.py；metrics.py；paper_evidence_math.py；既有表 | 无估计量变更 |
| R5 | 摘要/引言/结论偏交接审计叙事 | abstract.md、§1、§2局部、§5解释、§6聚焦CAI状态依赖采集与全过程质量；主结果保留无VLM强对照、全部区间与完整输入差距 | 冻结method_summary与机制/质量表；原六章 | selected VALID与单seed限制仍在协议、结果及结论 |
| R6 | §5.3未突出第二阶段时机差 | §5.3以+0.415/+1.296/−0.223/−0.097切入；解释剩余预算加权、非阶段局部积分；后两负差及381/793上升保留；浮点闭合留SI S2 | analysis/timing_contributions.csv及timing_identity_checks.json；A/TIMING_ARITHMETIC.json | 描述性分解，不作因果百分比 |
| R7 | 主文Table4与森林图重复；案例仅单状态 | Table4留主文，旧Fig3_contrasts移SI FigS4；新Fig4由c8-16第1/8步原PNG组合；主文现5图，SI13图；不利q24-48保留；图索引/图注/分配同步 | EVID/case_figure_reuse.csv；原PNG；figures/compose_case_r2.py；A/CASE_* | 源像素未改；图1无须重绘；不是硬件时延图 |
| R8 | Janisch版本元数据不一致；正文出现检索过程 | bib、ledger、最近邻矩阵、生成引用统一AAAI2019，33(01):3959–3966，DOI正确；正文移除institutional abstract措辞 | AAAI官方记录https://ojs.aaai.org/index.php/AAAI/article/view/4287；A/REFERENCE_VERIFICATION.md | 原author-PDF pp1–3读取范围保留，未升格FULLTEXT_READ |
| R9 | 作者提醒分散、工具型号缺证据 | 声明单源，AUTHOR_INPUTS集中待确认；删除SI末尾重复作者段；实际OpenAI Codex辅助用途保留，精确型号待核实；研究Qwen独立说明 | declarations.md；AI_USE_RECORD.md；AUTHOR_INPUTS.md | 署名/贡献/基金/利益冲突/许可/作者审核/正式格式仍待确认 |

## 一次定向复核：Q1—Q8

论文现在以明确状态、训练目标和固定预测器比较支持状态依赖采集的中心论点。最关键的限制仍可见：无VLM反馈的全程A和端点MAE优于主法，四项机制区间均含零，完整输入MAE不可达；时机分解说明观察到的差异而不确定因果模块份额。没有增加经验性主张或新实验。

| 检查 | 最终结果与证据 |
|---|---|
| Q1 | PASS：源MD→生成MD/HTML/TeX；再次从最终源构建10文件完全一致；PDF_CONTENT_CHECK检查实际PDF关键文字与公式/算法 |
| Q2 | PASS：完整X_obs算法、mask/Huber/32与16区别；预测器/VLM/actor/critic职责；SI pp1–3实际训练表可读 |
| Q3 | PASS：§4.4固定总预算前缀、池化MAE/域等权A、capture-group bootstrap依赖结构；0—63；训练/评价标签权限未变 |
| Q4 | PASS：九方法主表逐项与冻结method_summary三位小数一致；原科学表行全部保留（Janisch年份纠正除外）；全输入、四区间、无VLM、50%/0%、反弹和有符号贡献保留 |
| Q5 | PASS：摘要203词；引言任务→缺口→状态连接→三贡献；§5.3第二阶段观察居首，同时保留负差与非因果解释 |
| Q6 | PASS：5主图/13SI图文件和索引一致；旧森林图只在SI；原图字节未变，PDF嵌入图按PDF存储坐标方向还原后RGB逐像素相同；19引用键bib/ledger一致，正式Janisch同步 |
| Q7 | PASS：主稿19页、SI23页；两份编译exit0；无缺字/undefined/Overfull/LaTeX错误。已查看全部页面概览，以及算法p8、表/时机解释、过程图、SI预测器表及森林图；各图未裁切。过程图新标签10pt、对齐/碰撞通过 |
| Q8 | PASS（内容与范围）；Git闭合见GIT_DELIVERY.json和最终回复。18个普通冻结文件完成唯一结束hash比对；src/results/原analysis与tables/旧交接无差异。全部研究计算为0 |

首次版面检查发现算法跨页和SI末尾重复作者提醒，仅修这两项。算法使用有界raw-LaTeX fence后完整置于单页；直接裸minipage曾导致Pandoc把代码围栏当原始LaTeX，已根据编译日志定位并修正，最终编译成功。自动检查脚本的森林图文件名和PDF内部上下方向假设也已校正；它们不是研究数据或图像的变更。不另开一轮模拟审稿。

新增图源QA的三个提示是PNG无TIFF、300dpi而非技能通用600dpi、166mm而非通用89/183mm；本任务交付PDF/SVG及300dpi PNG、论文实际166mm宽度，属于明确采用的输出契约，非缺失数据。原嵌入文字由最终PDF人工检查，10pt自动检查只覆盖新增标签。SI较长表格正常跨页并重复表头；无内容截断。

## 词数变化

同一近似计数：排除表格、图注、公式和引用；R2另排除仅用于算法分页的raw-LaTeX围栏。字数是编辑比较，不冒充AEI正式限制。

| 部分 | R1 | R2科学文本 | 变化 |
|---|---:|---:|---:|
| Abstract | 207 | 203 | −4 |
| 1 Introduction | 806 | 560 | −246 |
| 2 Related work | 720 | 679 | −41 |
| 3 Framework | 1923 | 2187 | +264 |
| 4 Experimental design | 1291 | 1344 | +53 |
| 5 Results/discussion | 1825 | 1807 | −18 |
| 6 Conclusions | 214 | 188 | −26 |
| 六章合计 | 6779 | 6765 | −14 |

原build脚本朴素计数为6772（Framework2194），多出的7个token来自两处排版宏/围栏；未为字数改造构建器。新增真实方法细节与压缩叙事基本抵消，总体篇幅稳定。

作者待决定项集中于P/AUTHOR_INPUTS.md。TARGETED_MANUSCRIPT_REVISION_COMPLETE不等于投稿就绪、作者已批准或获得独立测试验证。

Q8 Git闭合：结果提交`664ce6dd08fa29579857a37c21392ab307d3adb9`已推送；三方SHA相同，整稿及两个实际PDF均在该提交，工作树当时干净。最终文档记录提交由最终回复提供。
