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
