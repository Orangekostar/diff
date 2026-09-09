# C-scan 人工评审工具：源码绑定与任务清单

## 仓库身份

- 仓库：`git@github.com:Orangekostar/diff.git`
- 开发分支：`research/cscan-human-review-html`
- 工作树：`/home/ww/diff/.worktrees/cscan-human-review-html`
- 唯一证据基点：`2102cc4a1726910931dfaaf20e29ad29a20eaf2e`
- 外部数据根：`/home/ww/paper3/cmc_damage_inference`
- Hasebe 数据根：`/home/ww/paper3/cmc_damage_inference/data/public/hasebe`
- 本轮 registered C-scan 路径模式：`data/public/hasebe/processed/<dataset_id>/internal_cscan/<specimen_id>.png`

## 冻结输入

| 输入 | SHA-256 |
|---|---|
| `paper_v3/configs/bc_cscan_frozen_process_analysis.yaml` | `6fc7d88e72bec9a37f3527a0a137b3407fb11803d5f90547183f56c82db4f5d7` |
| `results/bc_cscan_frozen_process_analysis/first_stop_decomposition.csv` | `1a716518ad7db79b665f00e9dba038ac26df1620527cef0d63b984e03b5dab72` |
| `results/bc_cscan_path_b_supplement/trajectories.parquet` | `5bf8e4ba677be063f3d434ce947c541e8dc213bf5d017c07b893d3d74a9b50c1` |
| `results/bc_cscan_path_b_supplement/cohort_and_reference_coverage.csv` | `2b5938e93d8df8b738a527dba17127bdcc8ab0876e03001e7de6eb736471193f` |

配置加载时逐项核对上述哈希；不匹配即拒绝打包。

## 复用接口

- `vlm_cscan.runtime.load_input_records` / `InputSpecimen`：以 `dataset_id:specimen_id`、registered crop、原尺寸和源 SHA 绑定图像。
- `learned_cscan.runtime.load_study_config`、`load_study_roster`、`load_study_context`、`open_study_specimen`：读取冻结 TEST cohort，并按需打开 24 件试样。
- `inspection_agent.state.zero_state`、`InspectionCellAction`、`action_added_positions_from_mask`、`apply_action`：只重放已存动作前缀。
- `learned_cscan.frozen_process_recovery.FrozenVisibleReportReader`：恢复冻结首次 STOP 的可见报告。
- `learned_cscan.benchmark._report_digest`：核对恢复报告与冻结摘要一致。
- `vlm_cscan.references.reference_from_payload`：验证浏览器导出的 canonical polygon 参考。
- `learned_cscan.frozen_evidence_finalize.summarize_blind_reviews`：验证 `report_id` 配对和盲评汇总语义。

## 真实任务清单

- 参考标注：24 件 TEST 物理试样、6 域、每域 4 件；缺图 0；分为 4 个包，每包 6 件。
- 源 registered crop 文件总字节数：9,773,648。
- 首次 STOP 候选：192 个冻结运行组合。
- 可盲评 STOP 报告：187；未停止运行：5；分为 6 个包，前 5 包各 32 份，末包 27 份。
- 恢复：187 个终点、187 次 Reader 报告、30,111 个已存动作转换、24 件试样；报告摘要和成本不匹配均为 0。
- 正式人工返回：参考 0、盲评 0；科学重评未运行。

## 不改范围

本轮没有训练、VLM 调用、Actor/STOP 前向、阈值搜索、动作生成或科学结果重算。Reader、动作、STOP 点、split、任务成功标准、原分析结果和历史 proxy 决定均未修改。
