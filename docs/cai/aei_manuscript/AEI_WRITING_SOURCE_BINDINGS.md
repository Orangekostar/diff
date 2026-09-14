# 写作依据登记：实际来源、用途与核实范围

核对日期2026-09-14。本表使用固定提交链接；可变分支仅用于确认起点。来源事实、由代码推导的解释、本轮建议不能混写。

## 1. 当前来源与历史的分层

| 层级 | 可用于什么 | 不可用于什么 |
|---|---|---|
| 当前84bea60e中的论文证据 | 本稿方法、结果、图表、已有范围 | 不能升级为独立TEST/多seed证据 |
| 当前cai_agent_v3代码 | 方法和指标的实际实现、信息权限 | 仅有代码功能不等于已有性能证明 |
| 旧BC/专家/Hasebe trace | 数据出处和历史问题背景，需标明原样本/协议 | 不能用旧AUSC/三seed/LOCATE/STOP填充当前CAI结果 |
| nature-skills | 写作和审查工作方法 | 不是AEI官方政策，不提供本方法的实验结果 |
| 外部论文/官方政策 | 引言、相关工作、数据引文和投稿规则 | 不能代替本项目测得的性能，未读全文不虚称已读 |
| 本包新安排[D] | 六章段落安排、输出目录、单次时机分析、有限review | 不是此前已经执行或期刊强制要求 |

## 2. 当前研究固定来源

以下[R01–R11]当前提交均为`84bea60e3fd2016b0b18379cd2a2a473200c2b4d`。

### [R00] 分支引用
https://api.github.com/repos/Orangekostar/diff/git/ref/heads/research/cai-vlm-agent-v3-controlled-reuse

本次返回上述84bea60e。Codex实际执行时保留用户后来工作，不能reset到聊天SHA。

### [R01] 已完成的论文证据交接（本次完整读取）
https://github.com/Orangekostar/diff/blob/84bea60e3fd2016b0b18379cd2a2a473200c2b4d/artifacts/cai_agent_v3/paper_evidence/r1_e2a11154/CODEX_HANDOFF_CAI_V3_PAPER_EVIDENCE.md

确认650最终轨迹、50物理/48组、10,227事件、599公共断点、两网格与完整参考、表图/草稿已存在，旧实验只读，全部新研究计算0。说明本轮应写完整稿而非重复证据制作。

### [R02] Methods事实表（本次完整读取）
https://github.com/Orangekostar/diff/blob/84bea60e3fd2016b0b18379cd2a2a473200c2b4d/artifacts/cai_agent_v3/paper_evidence/r1_e2a11154/METHODS_FACT_SHEET.md

包含276/259，TRAIN161/152、VALID50/48、TEST65/59；共同模型、OOF、VLM、Actor结构/参数/种子/更新和全部原统计定义。来源用于第3/4章，不能用旧交接替代。

### [R03] 结果与结论范围（前序定向读取，Codex必须再读取其本地固定版）
https://github.com/Orangekostar/diff/blob/84bea60e3fd2016b0b18379cd2a2a473200c2b4d/artifacts/cai_agent_v3/paper_evidence/r1_e2a11154/PAPER_RESULTS_DRAFT.md
https://github.com/Orangekostar/diff/blob/84bea60e3fd2016b0b18379cd2a2a473200c2b4d/artifacts/cai_agent_v3/paper_evidence/r1_e2a11154/CLAIM_EVIDENCE_MATRIX.md
https://github.com/Orangekostar/diff/blob/84bea60e3fd2016b0b18379cd2a2a473200c2b4d/artifacts/cai_agent_v3/paper_evidence/r1_e2a11154/LIMITATIONS_AND_SCOPE.md

原草稿数值准确的部分保留、结构和句式按论文重写。结论范围不能作为“负面文字全部删掉”的对象；它定义真实证据身份。

### [R04] 现有图稿图注（本次完整读取）
https://github.com/Orangekostar/diff/blob/84bea60e3fd2016b0b18379cd2a2a473200c2b4d/artifacts/cai_agent_v3/paper_evidence/r1_e2a11154/FIGURE_CAPTIONS.md

六图F1/F2/F3为MAE/RMSE/R²、F4等质量、F5机制区间、F6full差距。实际文件名从EVID的manifest/list读取，不按F1自行猜文件名。旧21图已固定三例，图注不是新原图生成许可。

### [R05] 模型输入与结构（本次定向重读SpatialCAIActor；预测器在前序已读）
https://github.com/Orangekostar/diff/blob/84bea60e3fd2016b0b18379cd2a2a473200c2b4d/src/cmc_bbdm/cai_agent_v3/models.py

Actor二维位置/可见标志/历史/VLM和内部图格特征进入空间交互。MeanSCPredictor不接策略名和动作历史，输入状态相同应输出相同，这是定位“选择观测”的代码性质，而非最优性结论。

### [R06] 已执行顺序与训练目标（前序已读，Methods写作须对照）
https://github.com/Orangekostar/diff/blob/84bea60e3fd2016b0b18379cd2a2a473200c2b4d/src/cmc_bbdm/cai_agent_v3/actor_training.py

`_training_rollout_loss`、`_predict_by_fold`、`_evaluate_one`、`evaluate_policy`：Actor采样/argmax、OOF、合法动作、event callback。只读，不调用旧入口。

### [R07] 表面VLM与首步规则（前序已读）
https://github.com/Orangekostar/diff/blob/84bea60e3fd2016b0b18379cd2a2a473200c2b4d/src/cmc_bbdm/learned_cscan/perception.py
https://github.com/Orangekostar/diff/blob/84bea60e3fd2016b0b18379cd2a2a473200c2b4d/src/cmc_bbdm/cai_agent_v3/vlm_perception.py
https://github.com/Orangekostar/diff/blob/84bea60e3fd2016b0b18379cd2a2a473200c2b4d/src/cmc_bbdm/cai_agent_v3/policy.py

VLM仅观察表面并描述线索，序数置信不是物理概率；最高可靠C0只硬限首步。下游仍有VLM特征，不写成“VLM首步以后完全退出”。

### [R08] 特征与数据连接（前序已读；需读取实际元信息补域描述）
https://github.com/Orangekostar/diff/blob/84bea60e3fd2016b0b18379cd2a2a473200c2b4d/src/cmc_bbdm/cai_agent_v3/feature_bank.py
https://github.com/Orangekostar/diff/blob/84bea60e3fd2016b0b18379cd2a2a473200c2b4d/src/cmc_bbdm/cai_agent_v3/cohort_export.py

先独立裁格后编码，保留native_shape、原索引。cohort_export是下一阶段需定向读取的来源路由，不声称本次已核对其中每个数据字段。

### [R09] 原评价数学（本次完整读取）
https://github.com/Orangekostar/diff/blob/84bea60e3fd2016b0b18379cd2a2a473200c2b4d/src/cmc_bbdm/cai_agent_v3/metrics.py

`left_error_area_mpa`左端常值且有保持尾段；`trajectory_objective_mpa`为A+.25终点误差。由此推导的时机恒等式是代数解释，本包尚未计算实际分阶段贡献。

### [R10] 论文派生数值（本次完整读取math，前序已读主要Evidence函数）
https://github.com/Orangekostar/diff/blob/84bea60e3fd2016b0b18379cd2a2a473200c2b4d/src/cmc_bbdm/cai_agent_v3/paper_evidence_math.py
https://github.com/Orangekostar/diff/blob/84bea60e3fd2016b0b18379cd2a2a473200c2b4d/src/cmc_bbdm/cai_agent_v3/paper_evidence.py

`held/aggregate/bootstrap_weights/first_quality/saving/map_full`及Evidence负责已存表。不能重新调用analyze把冻结分析重写。时机分解应是新小脚本而非修改这两个文件。

### [R11] 真值、分数、图表原始键
https://github.com/Orangekostar/diff/tree/84bea60e3fd2016b0b18379cd2a2a473200c2b4d/results/cai_agent_v3/paper_evidence/r1_e2a11154

关键文件：method_summary、same_cost_metrics、same_cost_paired_summary、mechanism_area_intervals、equal_quality_grid/anchors、full_scan_predictions/full_scan_gap、acquisition_events、frozen_episode_index、case_figure_reuse。
这里仅给已读交接确认的文件清单；Codex核对实际存在的后缀和row key，不从汇总值生成假原始数据。

### [R12] 旧附件，仅历史依据
9月9日BC交接与Hasebe参考追溯已提供。旧BC模仿P4、三层动作、LOCATE/CHARACTERIZE、旧24件评测与当前任务不同。作者ROI/raw阵列0/24仅限已查24件，不能作为276全体论断。不要把这两份旧文件的“下一步授权”带入写作。

## 3. Skills固定来源

路径前缀：
https://github.com/Yuan1z0825/nature-skills/blob/9ea7330a17813a15421fe843778a776c258b9001/

| ID | 相对路径（已定向读取） | 实际支持的适配 |
|---|---|---|
| S01 | skills/nature-writing/SKILL.md | 路由manuscript/类型/章节/语言/generic，非自动outline审批 |
| S02 | skills/nature-writing/manifest.yaml | 所有轴与always_load准确路径，version1.5.0 |
| S03 | skills/nature-writing/static/core/stance.md | 作者证据优先、缺口局部标注 |
| S04 | skills/nature-writing/static/core/workflow.md | 先论证/术语/段落，再起草和针对性修订 |
| S05 | skills/nature-shared/core/main-text-discipline.md | 主文和SI按功能分配，不隐藏反向证据 |
| S06 | skills/nature-writing/static/fragments/paper_type/methods.md | 方法论文的问题→技术→比较→复现链 |
| S07 | skills/nature-writing/static/fragments/journal/generic.md | 非专门目标使用generic，不套Nature严格限额 |
| S08 | skills/nature-polishing/SKILL.md | 不改变事实的语言和版式修订 |
| S09 | skills/nature-reviewer/SKILL.md | 默认3报告可被明确改变；真实独立性和证据定位，不能假互盲 |
| S10 | skills/nature-figure/SKILL.md | 已有Python复用、依数据绘图、对新改图进行有限QA |

本次未逐项运行上游脚本，也未审查整库19个skill。Codex按本地manifest加载其实际依赖，不根据此表声称已经运行了其审稿系统。

## 4. AEI规则与外部文献核实状态

### [J01] AEI官方页面：本次未取到正文
https://www.sciencedirect.com/journal/advanced-engineering-informatics/publish/guide-for-authors
https://www.sciencedirect.com/journal/advanced-engineering-informatics/about/aims-and-scope

本次作者指南抓取403，不能从通用Elsevier规则猜AEI的精确摘要字数、highlights、图形摘要、匿名要求或引用样式。六章和词数是用户决定/本包安排。Codex可定向再核一次官方页面或作者本地PDF；仍失败标UNKNOWN，完成科学初稿，不阻塞所有章节。

### [J02] Elsevier生成式AI政策：官方页面已读
https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals

2026-09-14读取到页面标记2026年6月更新。其要求作者审核并披露实质稿件辅助；研究中AI写Methods。数据图必须真实可复现；主研究影像不得凭空生成，通用生成式图形摘要不在允许路线。最终披露/作者责任由真实作者确认，本包不虚构确认完成。

### [L01–L05] 文献种子
详见REFERENCE_SEARCH_SEEDS.md。当前只核实到摘要或metadata的条目，不能当作已读全文的机制依据。科研比较只依据原论文/作者机构库/正式会议原文，不用二次解读代替原始来源。
