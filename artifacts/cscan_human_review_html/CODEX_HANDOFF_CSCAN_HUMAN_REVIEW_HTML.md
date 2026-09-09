# Codex Handoff: C-scan 本地人工标注与盲评工具

## 仓库与入口

- 仓库：`git@github.com:Orangekostar/diff.git`
- 基点：`2102cc4a1726910931dfaaf20e29ad29a20eaf2e`
- 分支：`research/cscan-human-review-html`
- 工作树：`/home/ww/diff/.worktrees/cscan-human-review-html`
- 工具入口：`dist/cscan_human_review/index.html`
- 工具 stage/schema：`CSCAN_HUMAN_REVIEW_HTML` / 1
- 本文件所属提交：在分支上运行 `git rev-parse HEAD` 获取，避免文档自引用 commit。

## P0-P7 完成状态

- P0：已核对证据基点、冻结输入哈希、Hasebe registered crop、TEST roster 和现有解析/恢复/汇总接口。
- P1：已生成 4 个真实参考包，覆盖 24 件、6 域、每域 4 件，缺图 0；另生成非对称 `TEST_ONLY` 练习包。
- P2：已实现中文三栏参考标注页，包括多 polygon、确定/不确定区、顶点编辑、删除、撤销/重做、取消、缩放/平移、状态确认、导出和恢复。
- P3：已从 192 个冻结候选中准备 187 份首次 STOP 匿名报告，5 个无 STOP 显式列入覆盖表；公开材料与私有身份索引分离。
- P4：已实现三类盲评判断、说明字段、草稿/确认、逐包导出和恢复；未评与无法判断分离。
- P5：已实现六个 CLI 命令；TEST_ONLY 会话已分别通过原参考解析器和原盲评汇总函数。
- P6：新增测试 18 个通过；直接相关既有测试 8 个通过；全部真实任务包完整性校验通过；Chromium 双流程通过；Ruff、JavaScript 语法和 `git diff --check` 通过。
- P7：代码、单文件 HTML、小型摘要、说明和截图已准备提交；大任务包留在本地。推送命令为 `git push -u origin research/cscan-human-review-html`，最终 SHA 由终端和交付消息记录。

## 实际数据状态

- 参考：24 件 physical TEST specimens；6 domains；4 packets；源图总计 9,773,648 bytes。
- 盲评：192 candidate runs；187 eligible first-STOP reports；5 no-stop runs；6 packets；96,532,215 packet bytes。
- STOP 恢复：187 endpoints；187 Reader calls；30,111 stored-action transitions；24 opened specimens。
- 完整性：report digest mismatch 0；cost mismatch 0；`world.step` 0。
- 资源：training 0；VLM 0；Actor/STOP forward 0；CPU 单进程。
- 人工输入：正式参考返回 0；正式盲评返回 0；科学重评状态 `NOT_RUN_NO_HUMAN_INPUT`。

这意味着工具与真实任务材料已经就绪，但专家验证尚未发生。浏览器测试中的一个 polygon 和一个 `UNABLE_TO_JUDGE` 判断均标记 `TEST_ONLY`，只在 `.local` 测试目录中使用。

## 页面与真实任务包

```bash
xdg-open /home/ww/diff/.worktrees/cscan-human-review-html/dist/cscan_human_review/index.html
```

- 参考包：`.local/cscan_human_review_html/reference_packets/reference_packet_001.json` 至 `004.json`，精确字节数合计 13,052,922。
- 盲评包：`.local/cscan_human_review_html/blind_packets/blind_packet_001.json` 至 `006.json`，精确字节数合计 96,532,215。
- 私有索引：`.local/cscan_human_review_html/private_report_index/report_index.json`。
- Hasebe 数据根：`/home/ww/paper3/cmc_damage_inference/data/public/hasebe`。
- 详细操作：`artifacts/cscan_human_review_html/USER_GUIDE_ZH.md`。

## CLI 与既有研究接口

```bash
PYTHONPATH=src python scripts/cscan_human_review.py build-ui --help
PYTHONPATH=src python scripts/cscan_human_review.py prepare-references --help
PYTHONPATH=src python scripts/cscan_human_review.py prepare-blind --help
PYTHONPATH=src python scripts/cscan_human_review.py validate-return --help
PYTHONPATH=src python scripts/cscan_human_review.py export-references --help
PYTHONPATH=src python scripts/cscan_human_review.py export-blind-reviews --help
```

`export-references` 逐项调用 `reference_from_payload`，生成目录可对应旧 finalize 的 `--references`。`export-blind-reviews` 用私有 `report_id` 索引配对并调用 `summarize_blind_reviews`，输出 CSV 可对应 `--blind-reviews`。原会话和审计文件与 `references/*.json` 分开，避免旧 `_reference_inputs` 误读元数据。

正式人工返回后再执行 reviewed finalize；它不是保存接口，会重算并写多个分析目录。执行前应保留旧输出快照或使用版本化的新输出位置，不覆盖历史 proxy 数值。没有人工规划会话时不传 `--human-sessions`。

## 文件与接口对应

- `paper_v3/configs/cscan_human_review_tool.yaml`：证据 SHA、路径、包大小、盲法和零模型调用约束。
- `src/cmc_bbdm/learned_cscan/human_review_tool.py`：配置核验、两类打包、已存动作 STOP 恢复、会话校验和既有接口转换。
- `scripts/cscan_human_review.py`：六个离线 CLI 入口。
- `web/cscan_human_review/`：HTML 模板、CSS 和浏览器状态/交互逻辑。
- `dist/cscan_human_review/index.html`：内联生成的无网络单文件交付版。
- `tests/test_cscan_human_review_tool.py`：数据身份、坐标、状态、盲法、恢复、CSV 和真实清单合同。
- `tests/browser_smoke_cscan_human_review.py`：Chromium 导出/恢复及视觉布局闭环。
- `results/cscan_human_review_html/`：构建、清单、无 STOP 覆盖和验证摘要。
- `artifacts/cscan_human_review_html/`：绑定、指南、验收、截图和本交接。

## 验证证据

```text
新增 pytest：18 passed in 68.78s
直接相关既有 pytest：8 passed in 26.12s
最终合并重跑：26 passed in 72.57s
实现前相关基线：42 passed in 163.72s
真实任务包完整性：4 包/24 个参考项，6 包/187 个盲评项，PASSED
浏览器：Chromium 143.0.7499.4，1440x900，PASSED
页面错误：0；console error：0；外部网络请求：0
Ruff：All checks passed
JavaScript：node --check passed
```

截图：

- `artifacts/cscan_human_review_html/screenshots/reference_workflow.png`
- `artifacts/cscan_human_review_html/screenshots/blind_workflow.png`

没有运行全仓库测试、W0-W5、完整 reviewed 评分或正式 finalize；本轮没有修改共享科学运行时，也没有真实人工输入。

## 未提交数据及原因

真实任务包、STOP 恢复缓存、私有报告身份索引和 TEST_ONLY 浏览器会话位于 `.local/cscan_human_review_html/`，均由 `.gitignore` 排除。它们包含完整 C-scan 图像或私有匹配且总量较大，不应仅为 GitHub 交付复制进仓库。代码、源图/包清单、状态摘要、练习包和截图已提交，可用 `prepare-references` / `prepare-blind` 从绑定数据重建。

## 科学边界

No scientific result changed. No model was trained or rerun. No action, STOP point, threshold, split, Reader rule, proxy label, frozen result or historical Path B decision changed. Existing results remain `PROXY_LEGACY`; independent reference correctness, blind deliverability and human comparison remain pending real human input.

## GitHub 核验

```bash
git rev-parse HEAD
git rev-parse '@{upstream}'
git ls-remote origin refs/heads/research/cscan-human-review-html
git status --short
```

推送后上述前三个 SHA 必须一致，工作树必须为空。未创建 PR，未合并其他分支，未 force push。
