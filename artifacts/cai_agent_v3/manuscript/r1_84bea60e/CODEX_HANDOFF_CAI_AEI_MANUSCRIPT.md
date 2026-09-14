# CAI—AEI完整论文初稿交接

任务：CAI_AEI_MANUSCRIPT_R1_84bea60e。主规范：docs/cai/aei_manuscript/CODEX_AEI_MANUSCRIPT_EXECUTION.md。

## 完成状态

MANUSCRIPT_DRAFT_COMPLETE / AUTHOR_REVIEW_PENDING / JOURNAL_FORMAT_VERIFICATION_PENDING。无BUILD_BLOCKED。论文初稿完成不代表独立验证、作者已批准或投稿就绪。

入口研究SHA与实际读取代码SHA均为 `84bea60e3fd2016b0b18379cd2a2a473200c2b4d`。分支始终是`research/cai-vlm-agent-v3-controlled-reuse`，工作树`/home/ww/diff/.worktrees/cai-vlm-agent-v3-controlled-reuse`。未修改研究源码；无reset/PR/merge/force push。

推荐并实际采用配置：nature-writing task=manuscript / paper_type=methods / language=zh-to-en / journal=generic，服从本包AEI六章结构。方法类别是写作类型，不冒充已核实的正式文章类型。实际skill安装与版本见PAPER/SKILL_USE.md。

## 正文与交付材料

PAPER=`paper_cai_aei/r1_84bea60e/`。

| 项目 | 实际成果 |
|---|---|
| 六章 | Introduction、Related work、Framework、Experimental design、Results and discussion、Conclusions全部完成 |
| 篇幅 | 正文近似6779词，摘要207词；写作目标不是AEI官方限额 |
| 完整稿 | manuscript.md、manuscript.html、main.tex、六份MD/TeX章节、references.bib |
| PDF | build/main.pdf 20页；build/supplementary.pdf 21页 |
| 图表 | 主图6幅（仅流程/时机2幅新绘），主表4张加信息权限表；SI完整同cap比较、域结果、质量CSV索引、3固定案例；13份源表 |
| 来源 | 89段落/展示块EVIDENCE_MAP.csv；FIGURE_INDEX.csv；39复用图文件SHA与冻结原件一致 |
| 引用 | 19引用与19BibTeX键全部匹配；closest matrix和reference ledger保留读取范围 |
| 作者材料 | AUTHOR_READING_GUIDE_ZH、AUTHOR_INPUTS、AI_USE_RECORD、JOURNAL_PROFILE |
| 投稿草稿 | 题目摘要关键词、4highlights、cover letter、数据代码/AI声明、仅文字graphical-abstract brief |

唯一新增计算读取650保存episode、10227事件，生成9方法×4阶段36行贡献。最大episode恒等残差2.8422e-14 MPa，方法面积残差≤7.1054e-15。六类规定玩具检查通过，未重算bootstrap。详见analysis/timing_identity_checks.json。原事件、模型、feature shards、raw和字体未复制进新交付。

## 审查与验证

按授权完成一次同上下文AEI内部审查、一次针对性修订，随后只复核修改处；不声称互盲同行评审。WRITING_REVIEW.md保留Major科学证据边界和局部修订。主要修订为相关工作获取过程移到台账、预处理复现信息补SI、公式与伪代码缺字修复、主表列宽/整表排版及图注与真实画面对齐。

实际构建为pandoc 3.9 → LuaLaTeX / latexmk，使用已安装article；elsarticle不可用，未冒称Elsevier模板合规。Pandoc仅作一次常规临时依赖安装，路径/tmp/cai_pandoc。最终两PDF编译成功，日志没有缺字、未定义引用或Overfull。已查看全稿contact sheets与关键表图页；主表不再跨页拆行。SI长表按长表排版，图像保持原始比例。

新增两图字体/碰撞检查通过，0fail/0warn；源图最小8pt。静态图脚本检查不能识别扩展名循环的两个导出FAIL保留原报告，并以实际六导出文件和PDF渲染作人工处置；300dpi预览符合本轮图稿契约，主稿采用矢量图。详见build/FIGURE_QA_REVIEW.md，不把原静态报告改成绿灯。

validation.json记录：主表九方法全部数字匹配、19引用解析、13来源表/39图副本未变、两PDF最终hash。原证据与源码diff为空。检查范围限本任务，没有全仓库pytest/旧研究审计/压力测试。

实际关键命令：

```bash
PANDOC=/tmp/cai_pandoc/pypandoc/files/pandoc python paper_cai_aei/r1_84bea60e/build_manuscript.py
latexmk -lualatex -interaction=nonstopmode -halt-on-error -cd -outdir=build paper_cai_aei/r1_84bea60e/main.tex
# SI在PAPER目录执行同一latexmk命令，目标supplementary.tex。
python paper_cai_aei/r1_84bea60e/analysis/test_timing.py
git -c core.whitespace=cr-at-eol diff --cached --check -- . \
  ":(exclude)paper_cai_aei/r1_84bea60e/build/*.log" \
  ":(exclude)paper_cai_aei/r1_84bea60e/figures/*.svg"
```

## 保留的证据限制与作者事项

全部结果是50件selected VALID/48组/6域、单策略seed面板；650运行不增独立N。无VLM强对照、空间/均值混合方向、四组件区间跨0、全部partial未达full及质量主网格50%/0%均在主文。新恒等分解不是干预，图像fraction不是物理设备时间。

Mack/Fuentes仅依据实际可访问机构摘要比较；三项算法最近邻已读相关方法正文。未获全文细节留未核实，不写缺失能力断言。七数据record的版本/配置/CC BY 4.0已核实，本地衍生图最终发布仍需作者审核。AEI官方Guide 403；具体类型/匿名/限额/格式待核实，不阻塞科学初稿。

姓名顺序、机构、CRediT、资助、利益冲突、稿件批准、原创性与并行投稿声明、公开代码URL/许可及最终AI披露均待作者确认，不代填“无”或“全部同意”。

## 资源与Git交付

本轮研究训练更新=0，研究模型前向=0，新增VLM=0，编码=0，TEST评价=0，新增bootstrap=0，设备实验=0。允许的保存事件代数分解仅一次；后续图/文重建不构成研究推断。CPU限制≤4，无GPU研究活动。

本交接先随实际稿件提交，然后执行普通push同分支；成功身份记录在GIT_DELIVERY.json，最终包含同步记录的SHA由最终回复给出，避免自引用提交循环。若push失败必须据实报告，不以本段预言成功。

空白检查将CSV的CRLF作为合法行尾，并排除原始工具日志和生成SVG；这些格式输出保持生成内容，未为检查更改科学数据。
