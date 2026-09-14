# CAI—AEI完整作者审阅稿

状态：MANUSCRIPT_DRAFT_COMPLETE；AUTHOR_REVIEW_PENDING；JOURNAL_FORMAT_VERIFICATION_PENDING。

入口：
- `build/main.pdf`：完整英文主稿，20页。
- `build/supplementary.pdf`：补充材料，21页。
- `manuscript.md` / `manuscript.html`：整稿可读版本；HTML公式使用MathJax。
- `sections/*.md`、`abstract.md`：唯一正文编辑基准。
- `main.tex`、`sections/*.tex`、`supplementary.tex`、`references.bib`：可编辑LaTeX。
- `AUTHOR_READING_GUIDE_ZH.md`：两页结构的中文论证导读；`AUTHOR_INPUTS.md`：作者确认项。
- `EVIDENCE_MAP.csv`、`figures/FIGURE_INDEX.csv`、`figures/reuse_manifest.json`：段落、图、源数值映射。
- `submission_drafts/`：highlights、cover letter、可用性/AI声明和图形摘要brief，均未提交。

配置：manuscript / methods / zh-to-en / generic，按AEI六章与作者规定范围适配。正文约6779词（排除表格、公式和图注的近似计数），摘要207词。19条真实引用；来源读取范围见台账。当前article模板只用于作者审阅。

从仓库根重建正文（需要Pandoc；本次使用临时安装的3.9二进制，具体运行版本见交接）：

```bash
PANDOC=/path/to/pandoc python paper_cai_aei/r1_84bea60e/build_manuscript.py
cd paper_cai_aei/r1_84bea60e
latexmk -lualatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
latexmk -lualatex -interaction=nonstopmode -halt-on-error -outdir=build supplementary.tex
```

只验证六类玩具数学检查（从仓库根）：

```bash
python paper_cai_aei/r1_84bea60e/analysis/test_timing.py
```

唯一真实事件分解已完成并保存；不要删除检查结果后重复运行timing_analysis.py。图稿重建脚本只读取已保存36行贡献，不读模型或新事件。冻结研究根保持只读。

完整交接位于 `artifacts/cai_agent_v3/manuscript/r1_84bea60e/CODEX_HANDOFF_CAI_AEI_MANUSCRIPT.md`（相对仓库根）。Git最终同步身份由同目录GIT_DELIVERY.json及最终回复记录。
