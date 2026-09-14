# CAI AEI targeted revision R2 handoff

状态：TARGETED_MANUSCRIPT_REVISION_COMPLETE；AUTHOR_REVIEW_PENDING；JOURNAL_FORMAT_VERIFICATION_PENDING。

实际工作树：/home/ww/diff/.worktrees/cai-vlm-agent-v3-controlled-reuse。
分支：research/cai-vlm-agent-v3-controlled-reuse。稿件基点：f4758829c621fa02a83595f1549e41dabe9e1c64。入口规范：paper_cai_aei/CODEX_AEI_TARGETED_REVISION_R2.md。实际结果提交及远端核对记录于同目录GIT_DELIVERY.json；含交接的最终SHA由最终回复报告，不循环写入自身SHA。

## 入口与范围

- paper_cai_aei/r1_84bea60e/manuscript.md：完整可编辑整稿；manuscript.html：带公式的阅读版。
- paper_cai_aei/r1_84bea60e/build/main.pdf：实际主稿19页。
- paper_cai_aei/r1_84bea60e/build/supplementary.pdf：实际SI23页。
- 同PAPER下abstract.md、sections/01…06_*.md、supplementary.md、declarations.md为编辑源；main.tex/sections/*.tex/supplementary.tex由构建器生成；bib和ledger同步。
- figures/Fig4_case_process.{pdf,svg,png}和compose_case_r2.py为新增组合图，原c8-16第1/8步像素不变；主图5幅、SI13幅，索引在figures/FIGURE_INDEX.csv。
- REVISION_RESPONSE.md记录R0—R9、Q1—Q8及词数变化。STATIC_VALIDATION.json、SOURCE_REBUILD_CHECK.json、PDF_CONTENT_CHECK.json、PDF_STRUCTURE.json及两份contact PNG保存有限验收证据。

本轮只改写作源、呈现、引用元数据、声明、相应薄构建脚本、作者说明和本任务交接/状态。原src、results、冻结analysis/tables、模型与原交接未改。未扩展文献综述；仅正式出版方定向核对Janisch。

## 实际构建

工具：本机Pandoc 3.9，latexmk 4.88 / LuaHBTeX 1.24.0（TeX Live 2026）。构建日志见main_compile.log及supplementary_compile.log；最终两命令退出码均0。再次从最终源构建生成文本完全一致。CPU线程环境上限4。

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PANDOC=/tmp/cai_pandoc/pypandoc/files/pandoc python paper_cai_aei/r1_84bea60e/build_manuscript.py
latexmk -lualatex -interaction=nonstopmode -halt-on-error -cd -outdir=build paper_cai_aei/r1_84bea60e/main.tex
latexmk -lualatex -interaction=nonstopmode -halt-on-error -cd -outdir=build paper_cai_aei/r1_84bea60e/supplementary.tex
```

PANDOC是实际可执行路径，不是永久系统安装承诺；其他环境设置为其可用路径。仅新增过程图做源级字号、碰撞、对齐和视觉QA；整稿与SI均看实际页面概览及受影响重点页。算法分页问题在本次自查中修复，完整位于主稿p8。

## 冻结与零研究计算

新增训练更新0；研究模型前向0；研究VLM调用0；特征编码0；TEST接入0；bootstrap0；研究GPU任务0；timing_analysis.py重跑0。仅读取保存的四阶段贡献作差和求和，未重读episode事件。18个登记普通文件只进行开始/结束两次hash核对，无全模型hash。写作辅助不计研究VLM。

四阶段主法减开环贡献约+0.415/+1.296/−0.223/−0.097 MPa，和为1.391 MPa；第二阶段差异对应剩余预算加权贡献，不是局部积分或因果占比。无VLM更优、区间跨零、全输入差距与回升均保留。

## 待作者完成

AUTHOR_INPUTS.md集中保留署名/机构/贡献、基金/利益冲突、数据图像归属和公开许可、代码版本/归档、作者科学审核、真实投稿声明、AEI官方格式。OpenAI Codex实际辅助用途已披露，精确底层型号待运行记录核实；研究Qwen2.5-VL-7B-Instruct单独说明。未代作者确认任何批准，未实际投稿。

Git按原分支普通commit/push，不PR/merge/force push/reset。结果提交后核对其HEAD、upstream与远端ref；记录提交另作一次文档闭合，最终三方SHA由最终回复给出。

实际结果提交：`664ce6dd08fa29579857a37c21392ab307d3adb9`，已正常推送，结果HEAD/upstream/远端ref一致；上述整稿、两PDF及本交接的已提交blob与本地字节一致。此行及GIT_DELIVERY.json由随后唯一文档记录提交闭合。
