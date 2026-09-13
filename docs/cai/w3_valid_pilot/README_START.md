# 启动：有效W2之后的W3 VLM引导选点实验

本包是**下一轮实际训练任务**，不是又一份恢复准备，也不是授权重跑整个项目。

## 交给Codex的文件

| 文件 | 作用 |
|---|---|
| `CODEX_CAI_V3_W3_PILOT_EXECUTION.md` | 本轮完整任务：新W2接线、五个seed1、固定对照、VALID结果、真实交付 |
| `W3_PILOT_SOURCE_BINDINGS.md` | 当前固定提交下实际源码和依据 |
| `CODEX_CAI_V3_W3_PILOT_REVIEW.md` | 八类有限需求验收，不做全库或过度安全测试 |
| `W3_PILOT_AUTHORIZATION.json` | W3-only阶段权限与独立5750次更新／6 GPU小时额度 |
| `W3_HAND_COMPUTED_CASES.json` | 独立数值、聚合和额度预期 |
| `basis/` | 原v3科学规范、review与原有独立metric参考的逐字副本；不是重新启动W0的指令 |

## 直接发给Codex的启动文字

```text
请执行附件 CODEX_CAI_V3_W3_PILOT_EXECUTION.md，并使用本包源码依据和review文件。
本轮正式绑定W3 seed1受控实验，接续：
research/cai-vlm-agent-v3-controlled-reuse
基点：0e11452ac6590b3b2b694364bd4d1cef7c9315cf。

明确授权本包W3-only安排：五个seed1总计最多5750次新增更新，含失败/重试；
独立6 GPU小时、单GPU、CPU最多4线程。保留历史33264使用上界；
累计上限由34264增加至40014，旧W2未用1000仍不挪给W3。
不重训W2，不调用新VLM，不重编码，不做GDFS/seed2/3/TEST。

新有效预测器来源：results/cai_agent_v3/w2_replay/r1_292b1c74/。
数据/特征/VLM仍从results/cai_agent_v3/new_protocol/只读复用。
新输出进入results/cai_agent_v3/w3_pilot/r1_0e11452a/及对应artifacts。
不要将新模型复制回旧目录，也不要让下游继续读失效gate。

实际读取源码，先修W3来源路径、seed初始化顺序和逐时点归档。
复用已有空间/均值/真实静态Actor、C0、cost-to-go与float64成本；不换算法。
先做有限预检，再实际训练五方法并与固定路线比较。
每个真实250检查点保存权重和同次VALID轨迹，首250即检查，不跑完才发现缺证据。
每次新增C-scan后由Actor重新选点，VLM只引导起始，不用规则整条路线冒充Agent。

即使VALID达到2%且优于开环的条件，也只报告后续建议，本轮不扩展或打开TEST。
结果正负都生成W3_PILOT_HANDOFF.md、代码/CSV/模型/单张图并实际commit/push。
核对local/upstream/remote SHA，不PR、不merge、不force push。
不全库审计，不重新复核所有W2候选，不只回复计划或“工作方式已读”。
```

把实际规范绝对路径和上述阶段写入已有TASK/STATE，保留历史但不重复旧未授权阻塞。文件路径/内容真实缺失仍应说明，不能伪造准备通过。

## 本包不是性能保证

W2的VALID A=48.517是四固定路线平均，不是最佳固定对照；完整输入41.690 MPa也不是Agent结果。W3会实际检验VLM、内部反馈和空间结构是否有增量。仅一个seed、同一开发VALID，不作正式独立显著性结论。

**范围、5750次和6小时是这份任务的明确设计。聊天端没有执行用户仓库中的训练或推送。**
