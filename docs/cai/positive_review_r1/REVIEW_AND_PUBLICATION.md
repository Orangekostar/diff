# 有限验收、交接与GitHub发布

本文件属于 `CAI_POSITIVE_REVIEW_R1` 执行合同。这里的Q01–Q08由Codex在修改完成后实际运行/人工检查。包内 `PROMPT_AUDIT_REPORT.md` 不可复制成这些任务已经PASS的证据。

## 1. 任务顺序、依赖和交付检查

| 阶段 | 依赖 | 核心工作 | 必交证据 | 何时算完成 |
|---|---|---|---|---|
| T00 | 实际仓库+输入包 | 源稿/分支/路径绑定 | SOURCE_BINDINGS、原稿哈希 | 当前稿身份清楚，旧VLM稿隔离 |
| T01 | T00 | 已选模型核对，不新选模 | MODEL_IDENTITY_LOCK | NO_VLM@1000/MEAN_SC@1750且前向授权为0 |
| T02 | T00/T01 | 数据子集、口径、8行方向修正 | figure_data、表、修正映射 | 原数据不变、方向清楚、全精度闭合 |
| T03 | T02 | 第4章协议 | 英/中§4、S1/S3骨架 | 分区、样本单位、cap前缀和聚合清楚 |
| T04 | T01/T03 | 第3章方法 | 英/中§3、S2、公式映射 | 模型角色和公式符合已读代码 |
| T05 | T02/T03/T04 | 第5章结果与讨论 | 英/中§5、S3–S5 | 区间/负面结果/主次比较全部保留 |
| T06 | T05 | §1/2/6、摘要 | 全文规范源、引用映射 | 主张强度与结果对应 |
| T07 | T02/T04/T05 | 六图源码/数据/排版 | SVG/PDF/PNG、FIGURE_SOURCE_MAP | 图文对应、数据不改、真实原图有来源 |
| T08 | T06/T07 | 补充材料、双语与英文构建 | 四版本DOCX/PDF、构建日志 | 无失效引用/乱码/公式破坏 |
| T09 | T08 | 一次内审、局部修复 | REVIEW、Q01–Q08 | 完成项与限制均有依据 |
| T10 | T09 | 清单、交接、提交、推送 | handoff、manifest、receipt | 远端真实SHA和文件已核验 |

T03和T04可以交错改，但必须先固定协议再写结果主张。T07的示意图可在T04后并行设计，定量图必须待T02完成。不能为了图文排版重开研究计算。

## 2. Q01–Q08：一次有限验收

### Q01 来源、边界和文件身份

核查原稿哈希、仓库基点、完整输入/输出根、主稿标题、六章、五表、六图、当前method白名单。SOURCE_BINDINGS对每个真正读取的文件写实际ref/hash；未知原图/SI路径不伪造。

检查git差异仅涉及允许输出。只核验本轮实际读取的小文件哈希，不扫描整个仓库/整个home。既有图像/压缩轨迹各读取一次即可，不为了审计复制全部cache。

### Q02 模型、公式和信息权限

从源文件而不是输出摘要检查NO_VLM@1000、seed、MEAN_SC@1750、两个模型角色及placeholder通道。模型文件未在本地时不装作读到了权重。查看图1/2和算法，确认y不流入执行状态、未测内部不流入Actor、预测器不读取采集历史。

核对13个原公式编号的迁移映射、绝对值、括号、积分carry和actor成本最小化方向。不要将公式在PDF抽取中的乱码抄回Word。

### Q03 关键数字、8行方向和舍入

在读取真实CSV后执行小型数值检查：

- 6个主方法与表3一一对应、full单独绑定、白名单不混VLM。
- 8条固定集合贡献行完成可逆方向统一；误差类行不变；CI下界≤上界。
- 同一对比4阶段原始贡献和的相反数等于A差（或归一化贡献和等于A差），容差1e-8 MPa。
- 主结果仍为MAE42.385，A43.597；主置换CI跨零、逆序CI正向、early两CI跨零。
- 主表3位小数由冻结值舍入，不使用显示值再次计算精确CI；跨归档/重算家族微小差异允许1e-4 MPa而非强行逐位相同。
- 表1总计与表5域均值/按n池化在舍入范围内闭合。不能把舍入差当科学错误。

Q03不是重抽样、重新检验显著性或重新评价模型。

### Q04 统计口径和主张边界

人工检查：50件不是276件或7,849独立样本；开发选择与效果评价共用同分区；A六组等权、MAE试样池化；Random先损失再平均；低cap是同路径前缀；q47 static也同cap达标；首次交叉不等于持续达标；full差不能称等效；9/94不外推到所有试样；固定集合不是选择/排序的可加因果分解。

关键词检查只作定位，不机械删除必要术语。每条中心主张必须在CLAIM_EVIDENCE_MAP有真实来源与限定范围；不要为通过词表而写隐晦同义的过度主张。

### Q05 图表、病例与图片权限

定量图必须能从已保存数值重画；每个面板的x/y、单位、差值方向、CI、样本单位和图注一致。step曲线保留真实回升，不用包络线/平滑掩盖不利变化。

检查六个固定case身份和q24-48大残差；Fig6的新增内部图若未解析完成，准确标记限制，不给它PASS。图中真实全C-scan只作事后参照，不能画进执行时全可见输入。

每个改图完成一次最终尺寸渲染检查；修复后只重查受影响图。不存在“未运行审计却写自动通过”。

### Q06 英文、双语、引用与SI一致性

中英文逐项对齐：模型、六方法、数字/单位、效应方向、主要/辅助和限制。生成两版图表用同一源数据。正文/SI的旧主法、旧题目、旧n、旧图号不能残留。

reference_map至少包含citekey、当前引用句/节、原始出处、读取范围、verified状态。未读全文不能声称FULLTEXT_READ；数据许可和代码可用性不能按“公开GitHub”自动推断全部原图可再发布。作者声明待确认不得写成已确认。

### Q07 构建与可见质量

实际运行构建，保存命令和退出码。检查英文/双语主文与SI的DOCX/PDF是否存在、非空、可打开。查看全部最终PDF页，核查图表标题、跨页、数学字符、中文、参考文献和页眉页脚；改排版后的版本重新渲染受影响文档。

不要求页数恰为29，不追求毫无依据的固定字数。构建被依赖阻断时保留源、精确日志和BLOCKED状态，不能用另一份旧PDF凑交付。

### Q08 交付范围与真实远端状态

交接必须准确区分本轮产物、冻结输入、历史验证和本轮验证。审查暂存文件白名单，排除权重、feature shards、原始数据全集、字体、凭据、缓存和无关文件。

Q08的远端部分在T10真正push后才完成。没有可核验远端SHA只能写LOCAL_READY_PUSH_BLOCKED，不能用历史receipt或本地commit代替。

## 3. 最多8个小测试的具体合同

在 `CODE/test_revision.py` 使用标准unittest或已有轻量测试器。只运行该文件；不引入GPU、torch模型或全repo test配置。以下8个函数足够，不扩张为通用防御测试库。

1. `test_model_and_method_binding`：当前模型/6方法映射和输入manifest一致；旧VLM主法不能映射成Proposed。
2. `test_stage_sign_normalization`：8行定向转换、CI变换正确、误差行未变、已规范化数据不二次反号；同一小测试内含幂等性分支。
3. `test_stage_area_identity`：读取固定集合保存贡献，阶段和与A差闭合。
4. `test_random_loss_before_pooling`：用两个小型不同预测的玩具重复验证先损失均值，不产生新研究结果。
5. `test_domain_equal_vs_specimen_pooling`：用不等组大小玩具例子确认两个估计量不同，输出标签正确。
6. `test_prefix_and_carry`：玩具轨迹确认bisect-right读取最后完成预测、末段carry、无未来泄漏；不调用研究模型。
7. `test_quality_first_crossing`：玩具非单调轨迹保留回升，未达标为NR/None，初始达标为0，分母0不算节省。
8. `test_generated_numbers_and_links`：定向检查生成表3/关键CI/图表文件/引用完整；不是全网爬虫。

这些测试验收的是本轮文档/数据适配代码，不证明方法泛化或解决单seed局限。若既有函数无需修改且已有适用测试，可复用，不为凑满8个另造测试。

资源上限：研究GPU使用0；CPU绘图/转换默认不超过4线程；正常文档构建、一次测试、一次审阅和一轮局部修复。没有失败不重复全套执行；尚有失败不虚报成功。网络只用于必要Git同步和少量实际新增引用核实。

## 4. 交接与manifest字段

### 4.1 CODEX_HANDOFF_CAI_POSITIVE_REVIEW.md

至少包含：

- task_id；输入原稿路径/哈希；evidence_sha、execution_base_sha、任务branch。
- 旧VLM稿与当前NO_VLM稿的隔离决定，source映射。
- 锁定Actor/Predictor；文件sha/state_dict区别；权重字节是否本次复核；新训练/前向/新bootstrap/新选择实际次数。
- T00–T10每项状态和实际产物，未完成项明确；6章、5表、6图的修改摘要。
- 8行贡献方向修正的来源、变换、闭合残差；原冻结文件未改的核验。
- Fig6原C-scan是否拿到、实际出处；S1–S5哪些源已读、哪些素材仍缺。
- 构建命令、实际版本（能读取则填）、退出码、页数、公式/视觉检查结果。不虚构软件版本、模型名称或作者审核。
- Q01–Q08真实结论，局部限制与剩余科学问题；独立测试、多seed、动态消融、物理扫描仍未开展。
- 交付文件、运行方法、resume入口、Git内容提交和publication receipt位置。

### 4.2 DELIVERY_MANIFEST.json

记录task/source/model、输出相对路径、文件bytes/SHA256、构建状态、QA结果、artifact/scientific/publication分轴状态。文件清单不包括manifest自身、PUBLICATION_RECEIPT.json、Git内部文件或临时渲染页，避免递归自引用。

内容提交时publication状态可为LOCAL_VERIFIED_PENDING_PUSH；实际远端状态以随后receipt与最终回复为准，不能预填REMOTE_VERIFIED。

### 4.3 CLAIM_EVIDENCE_MAP.csv

字段建议固定为：

```text
claim_id,section,language,claim_text,source_ref,source_path,source_key_or_symbol,
source_hash,evidence_family,estimand,units,claim_boundary,status
```

source_key指具体method/contrast/metric或case_key，不仅给一个目录。不能大量把这些审计路径塞进投稿正文，留在交接与source data。

### 4.4 FIGURE_SOURCE_MAP.csv

```text
figure_id,panel_id,question,source_ref,source_path,source_key,transform,
output_path,coordinate_or_aggregation_rule,asset_status,qa_status
```

真实图片的transform包含裁切/旋转/配准依据；数据图包含差值方向及家族。不得写含糊“generated by AI”掩盖真假数据边界。

## 5. GitHub分支与实际上传

### 5.1 工作树选择

以下命令片段是执行范式；Codex应先解析实际ROOT/PACKET，设置变量后运行，不能把尖括号占位符直接当路径。所有相对路径在实际ROOT下执行。

```bash
cd "$ROOT"
git rev-parse --show-toplevel
git status --short
git remote get-url origin
git fetch --no-tags origin research/cai-vlm-agent-v3-controlled-reuse
```

确认所用来源SHA存在。新任务分支名固定 `paper/cai-positive-review-r1`。若同名已存在且task_id一致，接续；若属于别的任务，不覆盖、不强推，报告具体冲突。

工作树干净且允许直接切换时，从已核实基点新建任务分支；有用户改动则创建独立worktree。没有必要重新克隆整个大仓库或重新下载模型。AGENTS.md可以约束代码风格，但不能扩大本轮研究权限；旧TASK文件的训练授权不属于本轮。

若remote证据分支已前进，按主执行指令T00决定兼容后继或固定快照，不把不相关提交强制回退。任务分支是新增交付载体，不自动更新证据分支或main。

### 5.2 内容提交C

构建/审核完成后，准确列举允许路径：

```bash
git add -- \
  paper_cai_aei/r3_positive_review_r1 \
  artifacts/cai_agent_v3/manuscript/r3_positive_review_r1 \
  docs/cai/positive_review_r1 \
  scripts/cai_manuscript_revision

git diff --cached --name-status
git diff --cached --stat
git diff --cached --check
```

人工定向检查暂存清单，确认没有无关数据/大权重/字体/缓存。不能使用`git add .`或用关闭检查掩盖混入。若目标文件被已有ignore规则忽略，只对确属交付的具体文件明确`git add -f -- <file>`；不删除全局ignore。

```bash
git commit -m "paper(cai): revise frozen-evidence manuscript and figures"
C=$(git rev-parse HEAD)
git push -u origin HEAD:refs/heads/paper/cai-positive-review-r1
REMOTE_C=$(git ls-remote origin refs/heads/paper/cai-positive-review-r1 | awk '{print $1}')
test "$C" = "$REMOTE_C"
```

同时确认upstream指向该任务分支。可用普通Git远端验证或GitHubAPI；对`handoff`、一个最终PDF、一个本轮代码文件确认远端内容C包含它们。不需要下载所有大产物再次跑构建。

### 5.3 收据提交R

只有内容C远端验证成功后，写 `ART/PUBLICATION_RECEIPT.json`：

```json
{
  "task_id": "CAI_POSITIVE_REVIEW_R1",
  "content_commit": "实际C",
  "remote": "origin",
  "remote_ref": "refs/heads/paper/cai-positive-review-r1",
  "content_remote_sha": "实际核验值",
  "content_verified": true,
  "verified_paths": ["实际handoff", "实际PDF", "实际代码"],
  "manifest_sha256": "实际manifest字节哈希",
  "artifact_status": "真实完成状态",
  "scientific_scope": "selected_development_configuration_only",
  "note": "记录前一个内容提交，不要求文件包含自身commit SHA"
}
```

这些字符串只是schema说明，实际写文件时必须替换为真实值，不保留“实际C”等字面占位。然后：

```bash
git add -- artifacts/cai_agent_v3/manuscript/r3_positive_review_r1/PUBLICATION_RECEIPT.json
git commit -m "docs(cai): record verified manuscript publication"
R=$(git rev-parse HEAD)
git push origin HEAD:refs/heads/paper/cai-positive-review-r1
REMOTE_R=$(git ls-remote origin refs/heads/paper/cai-positive-review-r1 | awk '{print $1}')
test "$R" = "$REMOTE_R"
```

最终回复报告C和R、分支链接及产物permalink。不要为了把R写进包含R的同一文件而无限amend/push。receipt的manifest hash不包括receipt自身。

### 5.4 失败与部分完成

- 网络/认证失败：保留C和本地全部产物；一次正常凭据/remote修正仍失败则 `LOCAL_READY_PUSH_BLOCKED`。不得自动改仓库、公开新仓库、使用未知token或假装远端存在。
- 非快进冲突：不force-push，不自动rebase无关内容；记录remote/local SHA。仅明确同任务且不覆盖用户内容的正常接续可继续。
- PDF构建失败或关键证据缺失：可以提交真实源与交接，但artifact状态必须为BLOCKED/PARTIAL。发表到GitHub不等于任务全完成。
- 图6可信原图缺失且保留真实原图6：可以明确 `REVISION_COMPLETE_WITH_DECLARED_ASSET_LIMITATION`；不能说“六图所有建议均完成”。
- 作者元信息待确认：不必阻止工程稿交付，但作者审核仍为pending；不能代签。

## 6. 最终发布验收摘要模板

Codex最终答复应给出实际填写值，而不是再次给这份空模板：

```text
任务：CAI_POSITIVE_REVIEW_R1
修订状态：...
源稿/证据版本：...
锁定模型：NO_VLM_SPATIAL_FEEDBACK@1000 / MEAN_SC@1750
研究计算：本轮训练...，前向...，新选模...，新增bootstrap...
主文与SI：英文/双语DOCX、PDF的实际路径/链接
图稿与代码：实际入口
方向修正：实际处理行数、源文件是否保持不变、闭合结果
验收：Q01–Q08真实状态
尚未完成：具体项或无（不能包含未解决科学边界的虚假“全部解决”）
Git：内容C、收据R、任务分支、远端核验结果
作者待确认：实际项
```
