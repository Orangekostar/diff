# 本轮定向review：冻结结果到论文证据

输入必须同时包括本轮主规范、实际新增分析代码、冻结W3/W2来源和真实输出。不能只检查“当前配置与自己写的测试一致”。不再审查训练架构是否最好，不重新判断是否应做TEST。

本review不授权训练或模型前向。准备期以合成数组与文件fixture检查，真实构建结束以已保存预测核对；不重复整个W2/W3审计。

| ID | 实际检查 | 独立预期与反例 | 通过证据 |
|---|---|---|---|
| A1 | 来源、key与阶段 | 九方法/650条/50key/48组；随机repeat不增N；full_*按冻结索引原序连接同50 key；故意反转索引不能悄悄通过 | INPUT_BINDINGS、scope、full_scan_predictions连接结果 |
| A2 | 同成本和聚合 | y200、两repeat预测190/210，MAE10而非0；MSE100/RMSE10；域均值与池化分栏；不造逐试样R² | reference及真实五预算汇总与旧表差异 |
| A3 | 时间顺序和数据覆盖 | costs[0,.125,.25]、预测[190,194,198]、y200：A8，b=.0625预测190，b=.125预测194；合法末状态到.25保持；查询.5拒绝 | 纯函数测试，无未来插值，无预算平滑 |
| A4 | 完整扫描身份 | full只有b1，A为null；W2 full50个预测复算原摘要；不能混用OOF/四路线前缀或free full at25% | full_scan_reference.json及差距表 |
| A5 | 等质量最小值 | q46.9099时fixed-grid主.0625、geometry.125、static.0625；对照基准不是固定填.25；未达null，分母0比率null；非单调曲线不改写；先群体后求最小 | quality表、全q输出、primary/secondary分栏 |
| A6 | 机制及文字忠实性 | trace实际调用/动作前后预测吻合；不新增反事实/注意力；VLM负方向、均值更好的终点、负向域不得删除 | CLAIM_EVIDENCE_MATRIX、case_narratives、逐域表 |
| A7 | 零前向、只读与范围 | 禁止旧summarize/train/evaluate入口；不加载.pt；不接TEST；不修改源目录；bootstrap标posthoc；不添加工程时间基线 | 新薄入口代码、日志、旧路径diff/一次输入hash、零更新ledger记录 |
| A8 | 真交付 | CSV/MD/LaTeX表一致、单图单位/范围清楚、没有.25→1虚构曲线；未达到也输出；GitHub实际推送 | RESULT_POINTER、final_manifest、最终三方SHA |

## 执行限制

1. 首次新增入口运行前，核对A1–A5的合成数值与输出隔离；之后完成一次真实构建。
2. 结束只复核真实分析涉及的聚合、全参考连接、等质量表、解释与图稿，不重跑候选模型或大范围旧测试。
3. 允许复用现有纯函数；发现原函数没有范围限制应在新任务封装处理，不改历史科学数字。
4. 独立参照文件 `paper_evidence_reference.py --self-test`可用；它不是正式研究结果，也不替代真实输入读取。
5. 最多八类检查，参数化数量不作为成功指标。少量单元检查＋修改文件Ruff＋git diff --check足够，不增加性能/安全测试矩阵。
6. 如环境实际支持独立上下文review，可将本文件单独交给它；否则标 `SELF_CHECK_WITH_EXTERNAL_NUMERIC_ORACLE`，不伪造独立审查者。
7. 区分IMPLEMENTATION通过与SCIENTIFIC范围：置信区间正向也不能写成独立TEST或因果证据；证据负向不导致程序验收失败。
8. 缺少保存的辅助文件，只局部标缺口并保留其余真实交付；不擅自补前向，不反复要求新增实验。

最终review JSON建议仅包含task_id、code_sha、source_sha、A1–A8状态/证据/缺口、new_updates/new_forwards/new_TEST_access、old_outputs_unchanged、mode、delivery。不要生成巨型重复审计册。
