# 开始执行：冻结CAI Agent结果，整理论文证据

这不是W4/GDFS、扩种子、TEST或硬件评测任务。用户已经决定停止新实验；本包只授权**已有预测/轨迹的计算分析与论文表图交付**。

## 文件用途

| 文件 | 用途 |
|---|---|
| CODEX_CAI_V3_PAPER_EVIDENCE_EXECUTION.md | 主任务；完整的范围、来源、数学、工作安排和Git交付 |
| W3_PAPER_EVIDENCE_SOURCE_BINDINGS.md | 固定提交与实际读到的源码/结果依据 |
| CODEX_CAI_V3_PAPER_EVIDENCE_REVIEW.md | 八类有限需求核对 |
| PAPER_EVIDENCE_SCOPE.json | 机器可读范围、固定预算、分析规则、禁止项 |
| paper_evidence_reference.py | 9个独立数值例子；无仓库导入、无模型或GPU |
| HAND_COMPUTED_CASES.md | 人可直接核对的成本/质量/聚合示例 |

无需先复读所有历史交接。原W3/W2规范只用于核对已经执行的方法；其训练/扩展授权不延续到本次任务。

## 直接发给Codex的启动文字

```text
继续Orangekostar/diff的research/cai-vlm-agent-v3-controlled-reuse。
本轮任务已明确：以e2a1115468da6e8695321204a13fa9a5322ea809冻结的W3结果，
以及W2保存的同一预测器完整输入预测，制作论文证据。

请完整阅读本次附上的CODEX_CAI_V3_PAPER_EVIDENCE_EXECUTION.md，
配合SOURCE_BINDINGS、REVIEW、SCOPE和独立数值文件执行。
先将实际规范路径、分支/HEAD、本轮ID和阶段写入现有TASK/STATE。

我要的实际交付是：
1. 同采集成本上限的MAE/RMSE/R²、配对误差和逐域结果；
2. 完整扫描＋同一CAI模型的既存100%输入参照；
3. 统一质量目标下的经验最小采集成本，保留未达到和不利结果；
4. “为什么好、好在哪里”的主张—证据矩阵、消融与真实过程图；
5. 论文可用CSV/Markdown/LaTeX片段、单张图、写作材料和交接MD。

本轮训练、所有新模型前向、新VLM、TEST接入、GPU任务均为0。
不做新稀疏采集/插值/工业启发式基线，不做硬件开销评测，
不补种子或GDFS，不重新选checkpoint，不删除无VLM更好的结果。
从保存的预测NPZ和650条轨迹计算；不是重新执行任何检测策略。

新结果写入results/cai_agent_v3/paper_evidence/r1_e2a11154/，
交接写入artifacts/cai_agent_v3/paper_evidence/r1_e2a11154/。
W2/W3原结果与review保持只读，不调用会写回原目录的旧summarize。

同质量比较必须给对照也找最早达标点；不能将6.25%对25%的对角线
比较直接包装成75%节省。只记录到25%的路径不得外推到100%；
完整扫描只作为100%的独立已观测点。新增分析是冻结VALID结果的
事后证据整理，不变成独立TEST确认或工程非劣检验。

完成后实际提交分析代码、相关结果和CODEX_HANDOFF_CAI_V3_PAPER_EVIDENCE.md，
push到同一分支，核对local/upstream/remote SHA一致。
不PR、不merge、不force push；只做本任务直接相关的有限检查。
不要只返回工作计划，也不要因历史任务仍要求预算授权而停止本轮纯计算。
```

## 交付说明

本包本身没有修改GitHub仓库或运行研究分析。将它作为任务交给Codex后，由Codex读取服务器上的真实数据并完成计算和推送。

完整扫描的NPZ字段和写出代码已经核对，但聊天侧未解压研究NPZ或650条gzip轨迹；实际连接和新等质量数字以执行结果为准。没有真实保存数据的项目应明确标缺失，不能补前向或编造。
