# 冻结论文证据制作交接

## 已完成的实际任务

任务 `CAI_V3_PAPER_EVIDENCE_R1_e2a11154`，实际工作树 `/home/ww/diff/.worktrees/cai-vlm-agent-v3-controlled-reuse`，分支 `research/cai-vlm-agent-v3-controlled-reuse`。入口与fetch后远端均为 `e2a1115468da6e8695321204a13fa9a5322ea809`，无reset或切换。主规范绝对路径 `/home/ww/diff/.worktrees/cai-vlm-agent-v3-controlled-reuse/docs/cai/paper_evidence/CODEX_CAI_V3_PAPER_EVIDENCE_EXECUTION.md`；配套8份文件来自用户ZIP `/home/ww/diff/docs/paper Prepare/CAI_V3_PAPER_EVIDENCE_CODEX_PACKAGE.zip`，原样保存。

数据来源为固定W3最终赢家650条VALID轨迹、同有效P_all的W2既存full_*预测，以及指定元数据/既有图。没有新训练、模型拟合、模型前向、GPU、VLM或TEST接入；没有重新选checkpoint、添加基线、改主方法或删除失败样本。所有原DATA/W2/W3/A3路径保持只读。本轮写入：

- RUN=`results/cai_agent_v3/paper_evidence/r1_e2a11154/`
- ART=`artifacts/cai_agent_v3/paper_evidence/r1_e2a11154/`
- 现有TASK/STATE追加记录；全局ledger仅追加一条CPU分析零更新记录。

数值分析在代码提交 `83cfa90` 上实际运行一次，普通分析处理时间1.052246秒（非硬件benchmark）。随后仅澄清两个文字/字段：共享日志的C0释放reason不能等同真实先前限制、空间结构终点MAE与RMSE方向不同；以及修正三列图例重叠为两列。没有重新读取全部轨迹计算/重采bootstrap，也未改任何科学数值。最终渲染、文字及review代码随结果提交，实际提交身份见GIT_DELIVERY.json及最终回复。

## 实际结果与新增分析定义

来源650 episode=九方法在50物理试样/48 capture groups/六域上的结果；10,227次真实采集事件不是独立样本。所有对照使用同MEAN_SC@1750，完整输入50件按原feature index行序连接；MAE41.6900100708、RMSE53.9440398105、R²0.7105359831与保存摘要一致。完整输入只在成本1.0有一个点，A为空。

| 交付 | 实际规模/位置 |
|---|---|
| 冻结视图与过程 | frozen_episode_index.csv 650行；acquisition_events.csv 10227行；full_scan_predictions.csv 50行 |
| 同成本主表 | same_cost_metrics.csv/.md/.tex：45个部分方法预算点＋full独立行；实际成本/像素/unused budget并列 |
| 配对与域表 | 2000逐试样同成本差、40个MAE差及区间；270域方法预算行，四套主表均有MD/TeX |
| 探索性区间 | 一次5000×50共享组权重，seed2026091401；四机制面积差及40同成本MAE差；95%逐点非同时 |
| 等质量 | 599公共事件断点；q全范围41..61 MPa共21；20非自适应预算锚点＋full锚点共21；每类420行、总840行，两网格分栏 |
| 机制与案例 | mechanism_effects/event_summary、path_divergence、400逐样本面积对照、3例真实case叙述及同成本状态；引用原21PNG |
| 新图 | 六类各PNG300dpi/SVG可编辑/PDF：MAE、RMSE、R²阶梯、主网格等质量、四机制区间、full质量差 |
| 论文材料 | PAPER_RESULTS_DRAFT、METHODS_FACT_SHEET、LIMITATIONS_AND_SCOPE、CLAIM_EVIDENCE_MATRIX、FIGURE_CAPTIONS |

所有派生区间均为 `POSTHOC_SELECTED_VALID_CONDITIONAL`，固定六域，域内重采capture group后带入全体物理成员；Random先每件平均损失，不集成预测。池化MAE和六域等权A分别计算，区间不校正先前VALID选模或本次事后选择，不作独立确认、正式显著或新gate。

同成本上限.25时：主方法MAE44.285799，几何46.909917，开环44.821928，无VLM42.384681，均值43.610874 MPa。主方法实际成本均值.247436538、范围.234491702–.249998901，不能写每件恰好25%。主方法相对几何的MAE差2.624118，探索区间[−.488466,5.925252]。四面积机制差及95%区间：

| 方向（对照−主） | 估计 MPa | 探索区间 |
|---|---:|---|
| 固定几何A | +2.200268 | [−0.786736,5.033064] |
| 开环/反馈A | +1.390605 | [−0.683122,3.405635] |
| VLM早期A | −0.871600 | [−4.870269,3.255410] |
| 均值/空间A | +0.152189 | [−2.186441,2.425374] |

四区间均跨零。无VLM整体A/终点MAE更低；均值Actor终点MAE更低但RMSE略高，完整保留。相对几何面积六域3正3负；反馈5正1负，不能报告普遍优势。主方法45/50首步proposal真收窄，793观测中381次误差增加；只是轨迹描述，不是C0或损伤因果证据。

等质量q=46.9099169921875的主网格中，主/几何/静态最早成本分别.0625/.125/.0625，因此对几何减少50%，对静态0%。补充事件网格为.031388065/.093425651/.062406858，对几何66.403%、对静态49.704%；主首次达标后反弹超q。主表与补充不可挑优替代。全840行保留147个未达到、43个负节省和54个零分母比率行（按完整重复来源行计数，不是独立目标数）。

所有九部分方法在两个网格均未达到完整参考MAE41.690010。主.25时MAE差2.595789 MPa；不放宽为事后非劣界，不外推.25→1，不声称传统流程只能全扫。等质量是群体经验阈值逆查询，不是逐件oracle STOP。

## 有限验收与图稿QA

初始数值模块不存在时测试按预期失败；实现后8项定向测试通过，包内独立9例通过。最终修改四源码/测试文件Ruff通过，git diff --check通过。测试覆盖左阶梯/尾段/覆盖范围、重复损失、组成员权重、双方最早点、非单调/未达/零分母/负值、原序full索引、真实调用号、MD/TeX数值一致。

真实构建先后各核对一次必要CSV/JSON/gzip/NPZ/21原PNG哈希，未hash模型/六bank。SAVED_EVIDENCE_REVIEW.json独立复算共享权重区间及全部840等质量行、主表CSV/MD/TeX；最大区间差1.78e-15；原W3五预算结果一致，源目录git diff为空。未重复模型审计或训练全套测试。

六个最终PDF文字/碰撞审查均退出0，最小字体8pt，6×PNG/SVG/PDF可用，逐图视觉检查无图例遮挡。静态源码heuristic未识别DejaVu Sans及循环扩展名导出，且对点估计图给出无errorbar提示；真实嵌入字体/18个输出及F5区间逐项核对，解释见FIGURE_QA.json，不伪称静态工具无报错。六图均为单轴，多panel对齐不适用。没有新增案例、原图重绘或硬件图。

review为主代理 `SELF_CHECK_WITH_EXTERNAL_NUMERIC_ORACLE`；未委派科学审查或捏造独立reviewer。A1–A8实际证据见REQUIREMENTS_REVIEW.json；Git完成后更新最后交付状态。

## 可运行命令与资源

在实际工作树执行，所有命令CPU≤4：

```bash
# 本轮已执行一次的真实分析；已存在analysis_manifest时拒绝覆盖/重新抽样
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src python scripts/build_cai_agent_paper_evidence.py analyze
# 仅从新派生表重画六汇总图；不写原W3，不读取checkpoint或重新抽样
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src python scripts/build_cai_agent_paper_evidence.py export
# 只检验保存的新表和共享权重，不生成新的bootstrap
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python artifacts/cai_agent_v3/paper_evidence/r1_e2a11154/verify_saved_evidence.py
# 合成数值例子
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_cai_agent_paper_evidence.py
```

analyze.log、export.log、verify.log为实际成功输出；analyze不提供训练选项。第一次重建需在没有本RUN输出的隔离副本中执行，不能为重建删除用户已交付证据。当前CSV/NPZ和代码已经完整交付，可直接只读检查或export。

实际全局ledger从174追加到175行，历史字节前缀保留；已知38264＋旧未知750＝39014/40014，W2余额1000未动。本轮训练/模型前向/拟合/新基线/VLM/TEST/GPU均0；普通分析处理耗时只作交接，不作为设备性能结果。没有新的训练任务或待配额项目。

## 写作使用与交付状态

正文按“任务驱动信息获取→同成本质量→反馈→等质量→全输入权衡”使用PAPER_RESULTS_DRAFT；25%摘要可放正文，T1完整五预算、T3所有目标/两网格和六域表放补充。主文同时给出无VLM强对照、四区间跨零及全输入未达；不能只选一个有利阈值或将VLM机制写成已证实。Methods只引用本事实表和固定来源，相关工作/新颖性不由本分析自动推得。

结果状态为PAPER_EVIDENCE_ASSEMBLY_COMPLETE，范围POSTHOC_FROZEN_VALID_SINGLE_SEED，独立确认NOT_PERFORMED，非劣NO_PRESPECIFIED_MARGIN。无真实缺失文件或未完成分析项目。实际源/代码/派生表图/文本及零更新账目提交到同一v3分支；不PR、merge、force push或重传模型/原始数据。GIT_DELIVERY.json记录实际结果提交核对；包含记录的最终SHA在回复中给出，避免自引用提交循环。
