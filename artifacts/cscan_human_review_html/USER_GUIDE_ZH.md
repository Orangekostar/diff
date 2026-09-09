# C-scan 人工评审工具操作说明

## 打开工具

```bash
xdg-open /home/ww/diff/.worktrees/cscan-human-review-html/dist/cscan_human_review/index.html
```

页面完全离线运行。显式导出的会话 JSON 才是可移动备份；浏览器本地缓存仅用于辅助恢复。

## 流程 A：参考区域标注

1. 点击“导入任务包”，依次选择 `.local/cscan_human_review_html/reference_packets/reference_packet_001.json` 至 `004.json`。
2. 填写真实评阅者别名、参考类型、标注方式，并如实勾选是否参与方法开发、是否提前看过模型输出。
3. 在“确定区”或“不确定区”模式中逐点绘制，多边形最后一点双击闭合；允许同类多个不相连区域。
4. 用“选择”编辑顶点，用删除、撤销、重做和取消按钮修订；缩放、平移和重置只改变视图，不改变原图坐标。
5. 未完成时点“保存草稿”并“导出会话”。完成后选择“确认完成”“确认无确定区”或“无法判读”，再导出会话。
6. 暂停后重新打开页面，先导入同一任务包，再点“恢复会话”选择之前导出的 JSON。

确定区与不确定区重叠时，既有 Python 解析规则以确定区优先。页面只支持多边形及整区域删除，不支持画刷、自动分割或任意洞编辑。

练习包 `dist/cscan_human_review/reference_practice_TEST_ONLY.json` 只能用于熟悉操作，不得作为正式参考。

参考会话回收后执行：

```bash
PYTHONPATH=src python scripts/cscan_human_review.py validate-return \
  --session <reference-session.json> \
  --packet .local/cscan_human_review_html/reference_packets/reference_packet_001.json

PYTHONPATH=src python scripts/cscan_human_review.py export-references \
  --session <reference-session.json> \
  --packet .local/cscan_human_review_html/reference_packets/reference_packet_001.json \
  --output-root .local/cscan_human_review_html/converted
```

每个 reviewer/export 会写入独立目录；`references/` 顶层只含逐试样 canonical JSON，原会话和审计 CSV 位于其旁侧。不要手工编辑 JSON、SHA 或试样身份。

## 流程 B：首次 STOP 报告盲评

1. 导入 `.local/cscan_human_review_html/blind_packets/blind_packet_001.json` 至 `006.json` 中的一包。
2. 填写真实评阅者编号。页面不会显示方法、试样、seed、成本或完整参考。
3. 只依据首次 STOP 时已测证据、未测区遮罩、原报告区域和支持点，选择“可交付”“需要继续检查”或“无法判断”。
4. 填写问题类型、判断依据和备注，确认判断后导出会话。未评条目不会自动变成“无法判断”。
5. 恢复时先导入同一任务包，再导入该会话 JSON。

盲评会话回收后执行：

```bash
PYTHONPATH=src python scripts/cscan_human_review.py validate-return \
  --session <blind-session.json> \
  --packet .local/cscan_human_review_html/blind_packets/blind_packet_001.json \
  --private-index .local/cscan_human_review_html/private_report_index/report_index.json

PYTHONPATH=src python scripts/cscan_human_review.py export-blind-reviews \
  --session <blind-session.json> \
  --packet .local/cscan_human_review_html/blind_packets/blind_packet_001.json \
  --private-index .local/cscan_human_review_html/private_report_index/report_index.json \
  --output-root .local/cscan_human_review_html/converted
```

输出 UTF-8 `blind_reviews.csv`、原始会话和既有汇总函数生成的校验摘要。不同评阅者分别保存，不自动形成共识。

## 重新生成任务包

```bash
PYTHONPATH=src python scripts/cscan_human_review.py build-ui

PYTHONPATH=src python scripts/cscan_human_review.py prepare-references \
  --source-root /home/ww/paper3/cmc_damage_inference

PYTHONPATH=src python scripts/cscan_human_review.py prepare-blind \
  --source-root /home/ww/paper3/cmc_damage_inference
```

真实人工返回齐备后的正式 reviewed 重评是单独步骤。它依赖完整旧分析输出，必须写入版本化的新输出位置或先保留旧输出快照；不要覆盖历史 proxy 结果。
