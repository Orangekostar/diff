# C-scan 人工评审工具

入口：`dist/cscan_human_review/index.html`。该文件无网络依赖，可直接双击打开，或运行：

```bash
xdg-open /home/ww/diff/.worktrees/cscan-human-review-html/dist/cscan_human_review/index.html
```

真实参考标注包位于 `.local/cscan_human_review_html/reference_packets/`，共 4 包、24 件 TEST 试样。真实首次 STOP 盲评包位于 `.local/cscan_human_review_html/blind_packets/`，共 6 包、187 份报告。任务包含完整图像，因体积和数据治理要求保留在本机，不提交 Git。

页面右上角依次用于导入任务包、恢复会话和导出会话。正式工作必须填写真实评阅者编号与来源信息；`reference_practice_TEST_ONLY.json` 仅供练习，不得导入研究结果。完整步骤见 `artifacts/cscan_human_review_html/USER_GUIDE_ZH.md`。

当前正式参考返回数和正式盲评返回数均为 0，尚未进行基于人工输入的科学重评。
