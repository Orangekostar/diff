# 开始执行：CAI v3 → AEI完整论文初稿

这是一项**实际写作任务**，不是再次论证研究方向、只交大纲或重开实验。用户已经同意六章结构与任务驱动采集主线。

## 交给Codex的启动文字

```text
继续 Orangekostar/diff 的 research/cai-vlm-agent-v3-controlled-reuse。
本轮有效任务是 CAI_AEI_MANUSCRIPT_R1_84bea60e：
基于已冻结CAI v3证据，完成面向 Advanced Engineering Informatics 的六章英文论文初稿、补充材料、可编辑来源、有限写作审查及GitHub交接。

请完整阅读本包：
1. CODEX_AEI_MANUSCRIPT_EXECUTION.md（主规范）
2. AEI_SIX_SECTION_BLUEPRINT.md（六章与段落/证据绑定）
3. NATURE_SKILLS_AEI_ADAPTER.md（实际skill使用和覆盖规则）
4. AEI_WRITING_SOURCE_BINDINGS.md（固定来源和文献核实状态）
5. CODEX_AEI_WRITING_REVIEW.md（有限验收）
6. WRITING_SCOPE.json（范围）

已核对研究来源SHA为84bea60e3fd2016b0b18379cd2a2a473200c2b4d。
nature-skills参考版本为9ea7330a17813a15421fe843778a776c258b9001。
将本包定位到实际工作树docs/cai/aei_manuscript/或记录等价路径，
在已有TASK/STATE绑定真实主规范绝对路径、阶段和输出目录，直接继续执行。
不要再次询问“本轮做什么”，不要把旧BC或零训练恢复提示当当前科学任务。

本轮允许检索核实写作引用、读取既有数据/代码/图表、写正文、LaTeX编译、
从既存事件完成一次收益时机恒等分解、形成表图及写作审查。
研究模型训练/拟合/新前向、VLM新调用、TEST接入、扩seed、GDFS、
新采集基线、专家标注和硬件性能评测全部禁止。

先完成Methods/Experimental design和Results，再写Related work/Introduction，
最后完成Conclusion、Abstract、题目、补充材料和投稿辅助草稿。
六个主章节固定；不新增大纲审批关口。未知作者/基金/声明只在相应位置标记，
不得把它们作为停止全部正文写作的理由。

沿用本地已安装nature-skills，按manifest按需读取。
本轮不是Nature投稿：journal=generic，AEI官方要求另查另记；
不能照搬Nature字数/版式/重大性门槛。
完成一次有证据定位的内部审查及一次针对性修订；
不做三份假互盲review、反复打分、全仓库审计或新实验。

最终必须有完整六章英文正文，而不只是任务卡、表格索引或review。
输出主MD、可编辑LaTeX/BibTeX、现成工具链可编译时的PDF、补充材料、
中文论证导读、作者待确认项及CODEX_HANDOFF_CAI_AEI_MANUSCRIPT.md。
实际commit/push相关写作文件到当前分支，核对local/upstream/remote SHA。
不PR、不merge、不force push、不覆盖旧证据、不实际向期刊投稿。
```

## 文件优先级

用户最新明确指令 → 本包主规范/WRITING_SCOPE → 已冻结科学事实 → AEI当期官方投稿要求（只管期刊规则）→ 本地skill写作流程 → 历史交接。期刊规范不授权改变真实结果，skill也不授权增加实验。

`reference/timing_identity_reference.py`仅是独立数学例子，可用`python reference/timing_identity_reference.py --self-test`检查；不是研究结果生成器，不包含真实CAI数据。

本包由聊天端制作，没有修改研究仓库，也没有替Codex运行正文写作、派生研究分析或推送。本轮目标是**可供作者审阅的完整初稿**，并非自动投稿或录用承诺。
