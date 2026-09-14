# CAI—AEI定向修订 R2

状态：TARGETED_MANUSCRIPT_REVISION_COMPLETE；AUTHOR_REVIEW_PENDING；JOURNAL_FORMAT_VERIFICATION_PENDING。

- `build/main.pdf`：英文主稿，19页；`build/supplementary.pdf`：SI，23页。
- `manuscript.md` / `manuscript.html`：整稿；HTML公式使用MathJax。
- `abstract.md`、六份`sections/*.md`、`supplementary.md`、`declarations.md`：编辑源。TeX/HTML/整稿MD由脚本生成。
- `AUTHOR_READING_GUIDE_ZH.md`、`AUTHOR_INPUTS.md`：论证导读与待确认项。
- `EVIDENCE_MAP.csv`、`RESULT_ALLOCATION.csv`、`figures/FIGURE_INDEX.csv`：证据及呈现分配；主文5图、SI13图。
- `references.bib`与`references/reference_ledger.csv`须同步编辑；现有19条引用，读取范围如实保留。

配置：methods / zh-to-en / generic，服从AEI定向修订要求。摘要203词，六章科学文本约6765词；原构建脚本计数6772，额外7个token来自算法排版宏，均非期刊正式字数。当前article版式供作者审阅。

从仓库根构建（已验证Pandoc 3.9、LuaLaTeX；本机Pandoc路径如下，迁移机器时替换PANDOC）：

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PANDOC=/tmp/cai_pandoc/pypandoc/files/pandoc python paper_cai_aei/r1_84bea60e/build_manuscript.py
latexmk -lualatex -interaction=nonstopmode -halt-on-error -cd -outdir=build paper_cai_aei/r1_84bea60e/main.tex
latexmk -lualatex -interaction=nonstopmode -halt-on-error -cd -outdir=build paper_cai_aei/r1_84bea60e/supplementary.tex
```

`figures/compose_case_r2.py`仅组合既定c8-16第1/8步原图。依赖本机nature-figure面板检查脚本，其他机器可用PANEL_ALIGNMENT_TOOL指定；无需重绘未改图。R2未重跑timing_analysis.py、数学测试或任何研究计算。

R2交接（相对仓库根）：`artifacts/cai_agent_v3/manuscript_revision/r2_f4758829/CODEX_HANDOFF_CAI_AEI_REVISION_R2.md`。原稿与原审查保留于Git基点f4758829及原交接目录，不回写历史结论。
