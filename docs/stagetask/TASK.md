# W2 exact-cost checkpoint恢复任务卡

- 工作树：`/home/ww/diff/.worktrees/cai-vlm-agent-v3-controlled-reuse`。
- 分支：`research/cai-vlm-agent-v3-controlled-reuse`；入口本地/远端 HEAD：`00ac8fb8800e3a3b908532d077af37f09aee90da`，基于 v2 `4f2b26b0f5e1593c8883b6ab78a6f7fdf700043d`；不重建、不reset。
- 主规范实际路径：`/home/ww/diff/.worktrees/cai-vlm-agent-v3-controlled-reuse/docs/cai/v3/CODEX_CAI_AGENT_V3_EXECUTION.md`。
- 配套实际路径：上述目录中的 `CAI_AGENT_V3_SOURCE_BINDINGS.md`、`CODEX_CAI_AGENT_V3_REVIEW.md`、`metric_reference.py`、`golden_cases.json`。
- 来源：`/home/ww/diff/docs/CODEX_HANDOFF_CAI_AGENT_V3/` 五个同名文件，逐字复制；不声称重新找到ZIP。科学契约以主规范为准，本轮权限以用户最新指令为准。
- 阶段：W2零训练恢复准备。正式训练更新=0，新VLM=0，新TEST感知/评分=0；不启动W3/GDFS/扩seed，不改变成本、预算、标签、split、候选或验收预期。
- 修改范围：W2保存/选择和必要依赖关口代码、直接测试、本任务卡/STATE、规范副本、恢复交接及证据。保留所有原失效报告和主工作树未跟踪文件。
- 已读取：用户指定六份现有交接/审查/成本/扩展证据；实际ledger为 `results/cai_agent_v3/compute_ledger.jsonl`（指定的new_protocol下路径不存在）。

| ID | 实际文件/函数 | 修改/核对 | 验收证据 |
|---|---|---|---|
| R1 | docs/cai/v3；Git | 绑定规范和现有分支 | 本卡、规范副本及交付SHA |
| R2 | models/、两份training_progress.csv、ledger | 定向寻找全部实际参选时点权重；不伪造 | W2_RECOVERY_EVIDENCE.json |
| R3 | predictor_training.py::_cell_costs/build_prefix_library/_train_candidate/evaluate_predictor | 保留原生成本通路，验证训练/前缀/评分一致 | 小型非整除图格独立整数像素预期 |
| R4 | predictor_training.py::_train_candidate；checkpoint_selection.py | 逐参选时点保存、绑定固定VALID身份、完整选择和重载 | 临时合成checkpoint，包含非赢家及缺失/输入变化案例 |
| R5 | actor_training.py::_load_oof_predictors；实际ledger | 验证失效上游阻断，重算累计资源与恢复缺口 | 定向阶段测试与ledger逐项汇总 |
| R6 | artifacts/cai_agent_v3/W2_RECOVERY_HANDOFF.md | 交接复用/重放/资源请求；实际commit/push同一分支 | local/upstream/remote一致 |

执行顺序：定向核对→独立失败测试→最小保存/选择修复→五类直接验证→交接与Git交付。CPU≤4；不运行正式训练预检。完成条件允许历史checkpoint未找回，但必须明确恢复缺口，代码准备与科学有效性分开报告。本轮已有授权，不重复询问。


## 当前任务：W2正式重放（替代上方零训练执行限制）

用户明确将 `/home/ww/diff/docs/W2正式重放/CAI_V3_W2_FORMAL_REPLAY_CODEX_PACKAGE.zip` 作为执行任务。
主规范实际路径：`/home/ww/diff/.worktrees/cai-vlm-agent-v3-controlled-reuse/docs/cai/w2_formal_replay/CODEX_CAI_V3_W2_REPLAY_EXECUTION.md`。
配套同目录 `CODEX_CAI_V3_W2_REPLAY_REVIEW.md`、`W2_REPLAY_SOURCE_BINDINGS.md`、`W2_REPLAY_AUTHORIZATION.json`、`basis/`；原v3科学定义仍由 `docs/cai/v3/` 约束。
分支不变，入口HEAD `292b1c74b27bb344af370f6738ac9c493cfd137d` 是原00ac8fb后的恢复交付；已fetch核对，无reset或覆盖。
本轮ID `W2_EXACT_COST_REPLAY_R1_292b1c74`；RUN=`results/cai_agent_v3/w2_replay/r1_292b1c74/`，ART=`artifacts/cai_agent_v3/w2_replay/r1_292b1c74/`。
W2-A三个共同预测器各≤2000；A准备通过才W2-B固定三个OOF各≤2000。总新增≤12000、累计≤34264（纠正旧交接34464），旧未知750保留；新增GPU活动≤21600秒，物理GPU1/logical cuda:0、CPU≤4。不做独立真实优化预检，不自动重启未完成job。
必须训练前commit，首250步真实归档核实；全部实际参选权重/同次预测/选中快照/最小optimizer-RNG状态保留。科学参数、固定VALID和OOF组不变。任何科学未就绪也如实交付，不改门槛。
完成条件：A真实完成，条件B完成或明确跳过/资源停止；A1–A8有限审查；实际结果、最多2PNG、全部模型、交接及任务状态commit/push同分支，三方SHA一致。W3/GDFS/Actor任何前向/VLM新增/TEST仍禁止，旧W2–W4失效报告保持。


## 当前任务：W3 seed1受控实验

用户指定 `/home/ww/diff/docs/W3 seed1受控实验/CAI_V3_W3_VALID_PILOT_CODEX_PACKAGE.zip`。主规范 `/home/ww/diff/.worktrees/cai-vlm-agent-v3-controlled-reuse/docs/cai/w3_valid_pilot/CODEX_CAI_V3_W3_PILOT_EXECUTION.md`；同目录AUTHORIZATION、REVIEW、SOURCE_BINDINGS、HAND_COMPUTED_CASES和basis为配套。实际分支research/cai-vlm-agent-v3-controlled-reuse；入口HEAD=0e11452ac6590b3b2b694364bd4d1cef7c9315cf，fetch后无前进，保留历史。
任务ID=W3_VALID_PILOT_R1_0e11452a。DATA=new_protocol只读，W2=w2_replay/r1_292b1c74只读，RUN=results/cai_agent_v3/w3_pilot/r1_0e11452a，ART=artifacts/cai_agent_v3/w3_pilot/r1_0e11452a。根均相对实际工作树。
本轮五个固定seed1：主/无VLM/开环/均值各1250，真实静态750，总新增≤5750（包括失败/重试），单GPU≤21600秒，CPU≤4；累计≤40014，旧W2余额1000不挪用，历史已用33264和未知750保留。W2训练/VLM新调用/编码/GDFS/STOP/seed2、3/TEST均0。
阶段P0接线和有限检查→训练前commit→P1固定四方法→P2五个seed1（每job首250即时核验，同次真实轨迹全部保存）→P3只读汇总与最多3个固定VALID案例→P4最终八项review、全部真实权重/轨迹/图稿/交接commit/push。正常650最终episode；正负结果均交付，不自动扩展。缺原图只阻塞图稿，资源终止保留真实未完成状态。


## 当前生效任务：W2 零训练恢复复核（入口 0f18001）

本轮以用户重新明确的 W2 exact-cost checkpoint 任务为准；上方 W2 重放和 W3 任务卡保留为既往记录，不据其启动新计算。
实际工作树 `/home/ww/diff/.worktrees/cai-vlm-agent-v3-controlled-reuse`；分支 `research/cai-vlm-agent-v3-controlled-reuse`；入口 HEAD `0f18001c8fa25e83b6fd3ece2fa6c7cd74e6c5ff`，fetch 后 upstream `0e11452ac6590b3b2b694364bd4d1cef7c9315cf`，均为用户指定 `00ac8fb8800e3a3b908532d077af37f09aee90da` 的后代。两笔本地 W3 接线提交和现有未提交 W3 产物保留，不 reset、切换或覆盖。
主规范：`/home/ww/diff/.worktrees/cai-vlm-agent-v3-controlled-reuse/docs/cai/v3/CODEX_CAI_AGENT_V3_EXECUTION.md`。
配套：同目录 `CAI_AGENT_V3_SOURCE_BINDINGS.md`、`CODEX_CAI_AGENT_V3_REVIEW.md`、`metric_reference.py`、`golden_cases.json`；五份已与 `/home/ww/diff/docs/CODEX_HANDOFF_CAI_AGENT_V3/` 实际来源逐字核对。未重建规范。
阶段：已有 W2 恢复实现的零训练复核与交付。完成条件：五类定向检查、历史/新增候选分离、实际账目与复用清单更新、保留旧失效报告、本轮文档提交并实际 push 同分支核对三方 SHA。
训练更新/VLM/TEST/W3/GDFS/扩 seed 新执行均为 0。不得因后续正式重放已有完整归档而声称找回旧 checkpoint；不得以本任务追认既有未交付 W3 科学结论。证据及当前资源见 `artifacts/cai_agent_v3/W2_RECOVERY_RECHECK.json` 和 `W2_RECOVERY_HANDOFF.md` 的本轮增补。
