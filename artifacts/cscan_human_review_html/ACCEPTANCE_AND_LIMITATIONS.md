# C-scan 人工评审工具：验收与限制

## 验收结果

| 项目 | 结果 |
|---|---|
| `tool_status` | `READY` |
| `reference_packet_status` | `PREPARED` |
| `blind_packet_status` | `PREPARED` |
| `browser_smoke_status` | `PASSED` |
| 正式参考返回 | 0 |
| 正式盲评返回 | 0 |
| 科学重评 | `NOT_RUN_NO_HUMAN_INPUT` |

实际 Chromium 143.0.7499.4 在 1440x900 视口完成两条闭环：导入、缩放/平移、绘制和编辑 polygon、填写中文与换行备注、确认、导出、重新打开、恢复。盲评闭环外部网络请求 0，页面错误 0，控制台错误 0。最终截图：

- `artifacts/cscan_human_review_html/screenshots/reference_workflow.png`
- `artifacts/cscan_human_review_html/screenshots/blind_workflow.png`

测试用参考会话通过原 `reference_from_payload` 回栅格；测试用盲评会话通过原 `summarize_blind_reviews`，逗号、中文、换行和 `UNABLE_TO_JUDGE` 均保留。所有 TEST_ONLY 输出只位于本地测试目录，没有导入正式研究结果。

## 数据与恢复完整性

- TEST 清单为 24 件、6 域、每域 4 件，源图缺失 0。
- 192 个候选运行中 187 个有首次 STOP 报告，5 个无 STOP，均显式记录。
- 187 个恢复报告的原 digest 和 STOP 成本不匹配数均为 0。
- 恢复只重放 30,111 个已存动作转换；`world.step`、训练更新、VLM、Actor/STOP 前向调用均为 0。
- 盲评公开包仅包含白名单字段；方法、试样、seed、成本和私有匹配保存在独立本地索引。
- 4 个参考包和 6 个盲评包已逐包通过嵌入 PNG 哈希、尺寸、模式、包 ID 与公开字段完整性校验。
- 未测图像像素由不透明棋盘遮罩替换，任务包不携带完整 C-scan 参考。

## 验证命令与结果

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_cscan_human_review_tool.py
# 18 passed

PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  tests/test_vlm_cscan_evaluation.py::test_unreviewed_proxy_never_becomes_formal_success \
  tests/test_vlm_cscan_evaluation.py::test_polygon_import_requires_real_human_review_provenance \
  tests/test_learned_cscan_runtime.py::test_algorithm_proxy_never_becomes_formal_success \
  tests/test_bc_cscan_frozen_process_analysis.py::test_stop_prefix_excludes_action_recorded_on_stop_row \
  tests/test_bc_cscan_frozen_process_analysis.py::test_two_blind_reviewers_do_not_become_two_physical_specimens \
  tests/test_bc_cscan_frozen_process_analysis.py::test_blind_reviews_pair_p8_and_bc_by_specimen_task_stop_and_reviewer \
  tests/test_bc_cscan_frozen_process_analysis.py::test_blind_reviews_do_not_pair_across_reference_versions \
  tests/test_bc_cscan_frozen_process_analysis.py::test_reviewed_reference_without_certain_region_is_not_formally_scored
# 8 passed

python -m ruff check scripts/cscan_human_review.py \
  src/cmc_bbdm/learned_cscan/human_review_tool.py \
  tests/test_cscan_human_review_tool.py \
  tests/browser_smoke_cscan_human_review.py
# All checks passed

node --check web/cscan_human_review/app.js
git diff --check
```

提交前将上述新增测试与 8 个直接相关既有测试合并重跑，结果为 `26 passed in 72.57s`。

实现前基线还运行了三个相关既有文件，共 `42 passed`。未运行全仓库测试、W0-W5、全量 reviewed 评分或正式 finalize，因为本任务没有修改共享科学运行时，也没有真实人工输入。

## 能力边界

- 这是本地单用户工具，不包含账户、数据库、并发协作、移动端或浏览器矩阵。
- 参考标注只支持多个简单 polygon、顶点编辑和整区域删除，不支持洞、画刷、自动轮廓或模型预标注。
- 盲评只覆盖冻结首次 STOP 可见材料，不是人工扫描动作规划界面。
- 浏览器缓存不是正式备份，必须显式导出会话。
- 工具可用不等于专家验证完成；当前没有独立参考正确率、盲评可交付性结论或人机比较结果。
- 既有数值仍是 `PROXY_LEGACY` 冻结结果；本轮没有改变其科学状态或 Path B 决定。
