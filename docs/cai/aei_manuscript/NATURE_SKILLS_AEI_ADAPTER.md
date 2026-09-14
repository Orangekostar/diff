# nature-skills到AEI写作任务的适配

参考仓库已核对版本：`Yuan1z0825/nature-skills@9ea7330a17813a15421fe843778a776c258b9001`。
这是一份本轮工作适配[D]，不修改上游skill，不冒称AEI官方规则。

## 1. 为什么这样选skill

已读nature-writing路由、manifest、核心立场/流程、methods和generic片段；也定向读了main-text-discipline、nature-polishing、nature-reviewer、nature-figure入口。其余always_load由Codex按本地manifest读，不声称聊天端已逐一读取全部库。

| 工作 | 使用skill | 本轮轴值/方式 |
|---|---|---|
| 完整论文初稿 | nature-writing | task=manuscript；paper_type=methods；language=zh-to-en；journal=generic；按当前章节加载对应fragment |
| 前置/投稿辅助草稿 | nature-writing | 同一工程定位，task=submission-package只在处理辅助材料时；不是修回回复 |
| 语言收紧 | nature-polishing | 仅在完整有证据段落后用；journal=generic；不得加强原结论 |
| 新/改图 | nature-figure | 已有Python路线，复用matplotlib；无需再问Python还是R |
| 稿件内部审查 | nature-reviewer适配 | 本包明确只要求一次AEI定向内部审查，非默认3互盲报告；准确记实际独立性 |
| 结果与篇幅管理 | nature-shared | 术语表、main-text-discipline、discussion-argument-language、必要一致性检查 |

不使用nature-response：尚未收到真实审稿意见。不使用自动idea/experiment skill重新开研究。Skill不是模型证据来源或期刊录用评分器。

## 2. 定向加载路径

相对于已定位的本地nature-skills根NS（不要假设固定安装目录）：

```text
skills/nature-writing/SKILL.md
skills/nature-writing/manifest.yaml
```

按manifest的always_load读：

```text
skills/nature-shared/core/reader-workflow.md
skills/nature-shared/core/paper-type-taxonomy.md
skills/nature-shared/core/ethics.md
skills/nature-shared/core/terminology-ledger.md
skills/nature-writing/static/core/stance.md
skills/nature-writing/static/core/workflow.md
skills/nature-writing/static/core/output-format.md
```

本任务匹配：

```text
skills/nature-writing/static/fragments/task/manuscript.md
skills/nature-writing/static/fragments/paper_type/methods.md
skills/nature-writing/static/fragments/language/zh-to-en.md
skills/nature-writing/static/fragments/journal/generic.md
skills/nature-writing/static/fragments/section/{method,experiments,discussion}.md
# 写到其他章节才读：related-work.md, intro.md, conclusion.md, abstract.md, title.md
skills/nature-shared/core/main-text-discipline.md
skills/nature-shared/core/discussion-argument-language.md
```

具体路径以本地manifest为准。当前本地若是更早/更新版本，记录差异；保留本轮任务和AEI适配，不因上游更新把计划换成Nature稿。不能无授权全量更新~/.codex/skills，不能执行上游安装/CI整套脚本。

若已安装单独skill不含`skills/`父目录，按本地真实路径映射，同样记录。找不到可定向获取所需公共文件，不自动安装全部运行环境。

## 3. 核心流程怎么用

| skill原意 | 本轮具体实现 |
|---|---|
| 先写一句论证再造句 | AUTHOR_ARGUMENT_MAP一页确定CAI任务→状态依赖获取→同预测器比较 |
| 建Terminology Ledger | 明确CAI是MPa、Agent不是VLM、当前不是BC、A越低越好、成本不是时间 |
| 一个段落一个工作 | 每段标context/gap/design/result/comparison/interpretation/qualification，标记只留工作稿 |
| Evidence outward | 写Methods/Experiment/Results先于Abstract；每个结果句绑定已有CSV rowkey |
| 主文和证据完备不同 | SHA/审计/完整840行在repo/source data/SI；关键区间和无VLM对照仍主文可见 |
| 缺字段不让全稿停摆 | 作者/基金放待确认，缺引用只阻塞相关句；其余章节继续 |
| 校准动词 | 设计职责用provides/encodes；数值用achieved/was lower on this set；不把跨零区间写significantly superior |
| 局部修订 | reviewer指出哪段就改哪段及依赖处，不多轮全稿重写 |

## 4. “positive但不审计化”的明确写作规则

论文主文不反复出现中文“当前证据尚不足以将收益明确归因…”的逐字翻译。替代方式不是隐藏证据，而是给出具体结果：

- Methods：VLM生成首步候选；Actor学习逐步选择；CAI预测器更新强度估计。
- Results：同预测器下各策略的误差和区间直接并列，解释观测集合/时间的作用。
- 消融：直接写无VLM43.597与主法45.110的A，说明比较对象；不额外写一长段“我们未能证明…”。
- Discussion：集中说明本研究中的采集分配、在本队列观察到的差异及适用条件。单seed与VALID选模的信息不能移除或改名。

允许积极论证已实现的工程解决方案，不允许把已有探索结果变成新的确认事实。禁用“AI检测器分数”“去AI率”作目标，禁止为去AI痕迹隐藏实际工具披露。

## 5. 必须覆盖的Nature默认

| 默认/风险 | 本轮明确覆盖 |
|---|---|
| Nature/NMI文章类型、字数、display数量 | AEI官方规则另查；本文字数/图表预算只是设计目标 |
| 三份互盲审稿+synthesis | **一次**有路径定位的内部review；真实独立子上下文才可标独立，不造三份假报告 |
| 跨学科“重大突破”门槛 | 面向AEI工程问题/可复现方法/公平评价，不强行宣称重大发现或首创 |
| 生成式图像API | 本任务不调用；确定性图解和真实数据制图，原研究影像不生成/修改 |
| 每步skills格式选项询问 | 六章/英文/Python/AEI均已明确，状态说明不是审批关口 |
| 核心证据不足→不断新增实验 | 记录局限和准确结论，按冻结任务完成初稿；不改证据或触发训练 |

## 6. 审查和图稿的有限执行

读取nature-reviewer/SKILL.md时同时传入本包CODEX_AEI_WRITING_REVIEW。只核验输入稿与原始证据、相关文献、AEI已核实要求，不运行上游库的CI/大量通用测试。

对新/修改图按nature-figure实际支持工具检查（先看--help，不猜参数）：`validate_figure.py`、`audit_pdf_text.py`、`audit_figure_collisions.py`；有可比多panel时才做`audit_panel_alignment.py`。每张最终改图检查一次，修正后只重查影响图。原封不动复用图以原QA和版面视觉为依据，不对21张旧PNG反复审计。

全稿一次跨段术语/数字/主张检查，一次针对性纠正，余留作者问题逐条列出。不存在“必须得到Accept分数才结束”的循环。

## 7. AEI与AI使用状态

AEI Guide for Authors本次在线抓取403，不能套用其他刊物精确要求。记录真实核实状态，不把这变成正文写作阻塞。[J01]

Elsevier官方现行政策要求对实质写作辅助披露，并由作者审核负责；研究用Qwen与稿件准备用Codex分别记录。图稿按真实数据和可复现代码制作；通用生成式图形摘要不纳入本包。具体最终声明和人工审核状态须作者确认。[J02]
