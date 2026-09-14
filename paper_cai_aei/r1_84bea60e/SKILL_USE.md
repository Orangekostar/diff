# 本轮实际skill配置

- nature-writing：`/home/ww/.codex/skills/nature-writing/SKILL.md`，manifest 1.5.0；task=manuscript，paper_type=methods，language=zh-to-en，journal=generic。依包中AEI适配，不再次审批已同意大纲。
- nature-figure：`/home/ww/.codex/skills/nature-figure/SKILL.md`，2.8.0，Python确定性图；只新增流程/时机两图，读取冻结数据。
- nature-reviewer：`/home/ww/.codex/skills/nature-reviewer/SKILL.md`；用户包覆盖默认三互盲报告，执行一次同上下文AEI内部审查。
- nature-polishing：`/home/ww/.codex/skills/nature-polishing/SKILL.md`，6.6.0；methods/en/generic，局部相关工作措辞及公式/排版修订；不扩大研究范围。
- superpowers:test-driven-development：仅用于获准时机脚本的六类数值检查，先失败后实现；不运行全仓库测试。
- superpowers:verification-before-completion：按实际编译、数值/文件检查、原证据diff与Git三方SHA决定交付状态。
- superpowers:using-superpowers：会话技能路由。以上为实际安装版本，不冒充参考仓库原commit版本。

## R2定向修订

按CODEX_AEI_TARGETED_REVISION_R2.md延续methods/zh-to-en/generic；nature-writing用于摘要、引言和5.3论证，nature-polishing用于局部措辞/排版，nature-figure沿用Python仅组既定案例。一次AEI定向同上下文复核，不宣称互盲。没有重跑R1时机分析/玩具检查或任何统计/研究模型。底层写作模型号未经独立运行记录核实，当前声明仅写OpenAI Codex。

R2最终版面自查使用systematic-debugging定位Pandoc raw-LaTeX围栏解析问题；通过有界围栏修复算法分页，最终PDF复核通过。没有调用研究测试或扩大代码修改。
