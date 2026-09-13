# 直接启动：CAI v3 W2正式重放

这份包替代上一轮“零训练恢复准备”的执行边界，继续同一v3分支。当前要做的是三个共同预测器及条件三个OOF模型的正式训练，不是再查历史缺失权重。

## 文件用途与优先级

| 文件 | 用途 |
|---|---|
| `CODEX_CAI_V3_W2_REPLAY_EXECUTION.md` | 本轮完整执行范围、来源绑定、科学契约、真实W2训练与交付 |
| `W2_REPLAY_AUTHORIZATION.json` | 明确W2-only额度、34,264累计上限、新增6 GPU小时；发送执行时生效 |
| `W2_REPLAY_SOURCE_BINDINGS.md` | 本次实际读取的源码、固定提交链接与事实/新增设计/未知边界 |
| `CODEX_CAI_V3_W2_REPLAY_REVIEW.md` | 8项有限验收；训练前和阶段后按需核对 |
| `basis/` | 原v3主规范、依据、review、独立metric与golden的逐字副本；不得照旧README从W0再启动 |

本轮主规范 > basis中的原规范。替代的是阶段授权、额度、输出位置，不改变W2的模型、标签、split、种子、评价与准备门槛。

## 发给Codex的启动指令

将整个ZIP作为附件，同时粘贴：

```text
请执行附件中的 CODEX_CAI_V3_W2_REPLAY_EXECUTION.md。

本轮具体任务是CAI v3 W2正式重放，不再是零训练准备。
继续 research/cai-vlm-agent-v3-controlled-reuse，证据基点为
292b1c74b27bb344af370f6738ac9c493cfd137d。

我授权本轮W2-only训练：共同预测器最多6000次更新；准备条件通过后，
选中结构的三个OOF回报模型最多6000次；本轮合计最多12000次。
将原未用5836额度（其中其他类别4586）明确重分配给W2，净增6164，
历史累计上限由28100调整为34264，不是34464。
旧已用22264和未知750上界保留；失败、中断和重试仍计入。
本轮另设最多6 GPU小时的增量时间窗口，单GPU，CPU最多4线程。
这不是认定旧时间预算还剩6小时，历史缺测保持披露。

只执行W2-A、准备通过后的W2-B和交付。新VLM、CNN重编码、Actor/STOP、
GDFS、多seed扩展及TEST感知/标签评价全部不在本轮授权中。
不要重复W0/W1、全盘寻找历史权重、全仓库或过度安全测试。

读取恢复提交中已存在的训练、归档、gate、ledger逻辑，做必要薄适配：
新输出与旧new_protocol分开，真实数据/特征复用；让新额度在入口生效，
不要仅改MD，也不要全局解除Actor预算。先review，再真实训练。

所有实际参选checkpoint都保存；每250步固定exact-cost VALID评分，
保持原seed、模型、损失、早停和准备条件。A没有可用预测器就跳过B，
保存实际结果；即使A/B通过也不能自动启动Agent。

完成后输出W2_REPLAY_HANDOFF.md、实际代码/权重/CSV/JSON及必要图稿，
实际commit/push，核对local/upstream/remote SHA一致；不PR、不merge、
不force push，不覆盖历史失效结果。不要只返回计划或再次询问任务阶段。
```

授权配置只是本轮任务附件，不代表聊天端已经启动了训练或改写了仓库。若工作树存在真正的新冲突或超额，报告具体差异；不要重置账目或伪造完成。

## 预期结束状态

- A未通过：交付新候选比较与 `PREDICTOR_NOT_READY`，B未执行。
- A通过、B未全部通过：交付有效P_all和三折真实结果，`REWARD_MODELS_NOT_READY`。
- A/B均通过：交付有效新模型、完整选模证据与 `W2_READY_FOR_FUTURE_POLICY_TASK`；下游仍未授权。
- 资源或真实输入问题：交付已完成部分、真实ledger及具体剩余项。

以上都不等同于VLM/Agent性能已得到支持，也不承诺MAE下降。
