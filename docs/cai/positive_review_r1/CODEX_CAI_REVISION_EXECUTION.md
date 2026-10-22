# Codex 最终执行指令：CAI / C-scan 论文冻结证据修订、构建与发布

版本：R1。任务 ID：`CAI_POSITIVE_REVIEW_R1`。

**执行目标：完成修改后的论文和图表，以及可复现的轻量构建代码、证据对应表、审核结果、交接文件，并实际提交和推送到 GitHub。不要只输出下一轮计划。**

配套必读：`FILE_TASK_BINDINGS.md`、`REVIEW_AND_PUBLICATION.md`。本文件定义执行范围；配套文件提供定位和验收细则。`PROMPT_AUDIT_REPORT.md` 只记录提示词生成时的审核，不代替运行期验收。

---

## 0. 研究身份与本轮权限

### 0.1 本轮修订的是哪篇文章

保留题目：

> Learning region selection and acquisition order in C-scan images for efficient compression-after-impact strength assessment of composite laminates

方法名：**Strength-assessment-driven multimodal sequential observation / 强度评估驱动的多模态序贯观测方法**。正文后续称 Proposed / 本文方法，不新造缩写。

本轮主要工作是将已冻结的科学结果组织成更准确的文章，不评价新模型，不将旧研究重新包装成另一个实验。研究模型对应关系为：

| 对象 | 本轮必须使用的既有身份 |
|---|---|
| Proposed | `NO_VLM_SPATIAL_FEEDBACK` |
| 策略 | `SpatialCAIActor(use_vlm=False, use_feedback=True)`，已选 update 1000，seed 2026091302 |
| 共同强度预测器 | `MEAN_SC` / `MeanSCPredictor`，已选 update 1750 |
| 图像编码器 | 既有冻结 ResNet18，逐格先裁切后编码，512 维 |
| 数据评价 | 50 件开发验证试样、48 个捕获组、6 个来源组 |
| 采集 | 8×8 网格，原生像素面积，整格请求，总 cap 0.25 |

这里的“完成选模”指 **核对并锁定已经完成的选择**。禁止重新训练、搜索候选、比较新 seed 或凭新文案重选 checkpoint。[E03–E06]

### 0.2 允许与禁止

允许读取既有正文、源代码、CSV/JSON/已保存轨迹和原始案例图片；允许对保存值进行确定性重排、舍入、已有差值的符号统一、图表排版、语言修订和文档构建。

本轮固定：训练更新 = 0；研究模型前向 = 0；新增 bootstrap 抽样 = 0；新增 seed = 0；新增 TEST/预留数据评价 = 0；重新选 checkpoint = 0。历史顺序实验曾允许 predictor 前向，其授权不沿用到本任务。

不得运行旧研究入口的 `all`、`infer`、`train`、`evaluate`、重生成归因/attention、新反事实选点或新物理成本评测。不得重新产生 350 个顺序条件。不得为满足审稿意见补齐本轮未授权的新对照。可以明确保留这些研究边界，但不能通过改写文字宣称它们已经解决。

本轮不上传模型权重、完整 feature shards、整个原始数据集、字体、缓存、凭据；不创建 PR、不合并、不 force-push、不创建 tag/Release、不实际向期刊投稿。

### 0.3 证据优先级与冲突处理

1. 用户本轮修改范围及本执行包的明确权限。
2. 当前上传 PDF / 内容核对一致的 DOCX，定义要修改的文章与现有主张。
3. 固定提交下的模型锁、已保存预测/结果及对应代码，定义可支持的事实。
4. 旧大纲、旧写作规范、旧论文目录，仅在与前述一致时复用。
5. Nature-skills 提供写作/图形/审核方法，不扩大实验权限，也不是 AEI 官方投稿规则。

正文与原始证据真正冲突时，记录具体位置并核对，不能为了“保留数字”改造源数据，也不能为了新故事换模型。修正文稿或派生显示，冻结研究文件保持原字节。

---

## 1. 工作根、来源和输出合同

### 1.1 实际路径

`ROOT` 由实际 `git rev-parse --show-toplevel` 得到；`PACKET` 为本包实际解压路径。不要假定 `/home/ww`、服务器节点或聊天沙箱路径存在于 Codex 环境。

```text
REPO = Orangekostar/diff
SOURCE_BRANCH = research/cai-vlm-agent-v3-controlled-reuse
EVIDENCE_SHA = 6082b69f2792130a321c74b0efcb198f5d65cefb
TASK_BRANCH = paper/cai-positive-review-r1

ORDER = results/cai_agent_v3/order_mechanism/r1_37b3c404       # 已核实
EVID  = results/cai_agent_v3/c_render_retrain/r1_331f5295/evidence # 已核实相关文件
PAPER = paper_cai_aei/r3_positive_review_r1                  # 本轮新增
ART   = artifacts/cai_agent_v3/manuscript/r3_positive_review_r1 # 本轮新增
DOCS  = docs/cai/positive_review_r1                         # 本轮新增
CODE  = scripts/cai_manuscript_revision                    # 本轮新增
```

不能把默认 `main` 当作本文证据分支。不能把旧 `paper_cai_aei/r2_c_331f5295/README.md` 所称“current C manuscript”当作当前稿源；它对应另一套 VLM 主法与 checkpoint。[E01,E07]

若本地有同任务分支与同任务输出，先核对任务 ID 和输入哈希，接续已完成阶段，不另建重复项目。若现工作树有无关未提交修改，保留原工作树，使用单独 worktree。禁止 `reset --hard`、`git clean`、自动 stash 或覆盖用户文件。

### 1.2 输出目录

在新 PAPER 下建立以下最小结构。可以复用已存在的兼容构建函数，但不得另造大型任务管理框架。

```text
PAPER/
  source/
    manuscript_en.md
    manuscript_zh.md
    supplementary_en.md
    supplementary_zh.md
    references.bib
    figure_captions.json
  figures/                    # 图1–6最终SVG/PDF/PNG，真实影像部分保持栅格
  figure_data/                # 每幅定量图用到的冻结数值子集及来源键
  tables/                     # 表1–5与必要SI表；保留机器可读形式
  build/
    manuscript_en.docx
    manuscript_en.pdf
    manuscript_bilingual.docx
    manuscript_bilingual.pdf
    supplementary_en.docx
    supplementary_en.pdf
    supplementary_bilingual.docx
    supplementary_bilingual.pdf
  reference_map.csv
  change_log.md
  README.md
ART/
  SOURCE_BINDINGS.json
  MODEL_IDENTITY_LOCK.json
  CLAIM_EVIDENCE_MAP.csv
  FIGURE_SOURCE_MAP.csv
  REQUIREMENT_MATRIX.csv
  REVIEW.md
  CODEX_HANDOFF_CAI_POSITIVE_REVIEW.md
  DELIVERY_MANIFEST.json
  PUBLICATION_RECEIPT.json     # 仅在真实发布验证后写入
CODE/
  run.py
  evidence.py
  figures.py
  build.py
  verify.py
  test_revision.py
```

这是**待实现的接口及产物合同**，不是声称这些文件已经存在。局部实现可合并少量模块，但主入口、产物名称和验收能力不得缺失。研究源码 `src/`、旧论文、ORDER、EVID 均只读；派生修正写新 PAPER/ART。

可将 `PACKET/inputs` 按原字节复制到 `DOCS/inputs`，作为本轮追溯源稿；本包三份主规范、事实摘要和审核报告同步放入 DOCS。不要把聊天全部历史或无关旧规范作为可执行指令导入。

---

## 2. T00：绑定源文件与版本，不在旧论文上误改

**输入**：本包两个稿件原件；仓库分支/HEAD；[E01–E12]。

**操作**：

1. 一次查看仓库根、分支、HEAD、`git status --short`、remote。确认 origin 对应 `Orangekostar/diff`。
2. 对来源分支做一次正常 fetch。固定证据快照为 EVIDENCE_SHA。若远端已有后续提交，只定向比较本文相关路径；源证据未变时可从兼容后继建任务分支，并记录 `execution_base_sha`。若科学文件变化或旧稿身份不同，使用已核实快照建立独立任务分支，不替换用户当前分支、不自行合并研究变更。
3. 验证两个输入哈希：
   - PDF：`6e7369955e378592c04758d5b8297391eac0a4fa89babbed255061cee47188c3`
   - DOCX：`277a673c2235f2196781fa1b78cad90310fd43cc8820b15fca1e45bc36264160`
4. 读取原稿全部六章、五张表、六幅图及图注。原稿 DOCX 有 275 个直接 body 段落、5 个表格、7 个嵌入图对象、188 个 OMML 数学对象，其中13个陈列公式段；7个图对象是因为图6分两张，不代表七幅主图。
5. 原稿定位用 `reference/docx_block_anchors.json` 的“原始零起始段落序号＋标题文本”。修改后序号会变化，不得继续按旧序号无条件替换。
6. 建立 SOURCE_BINDINGS：每条来源记录 repo/ref/path/blob SHA 或本地 SHA256、读取范围、用途、是否原始/派生、对应任务。未读取的补充材料或案例原图记录 `NOT_YET_READ`，不能写“完整核验”。

**输出**：SOURCE_BINDINGS、源稿副本、模型和数据文件入口绑定。

**验收**：能说明新正文来自当前双语稿，不是旧 VLM 英文稿。没有把旧模版 title/abstract/selected model 写进新稿；两份原稿未被修改。

---

## 3. T01：核对已选模型与科学对象

读取 ORDER/MODEL_SELECTION_LOCK.json、主法 job JSON、models.py 和 metrics.py 的相关符号，并读 actor_training.py::_actor 的方法映射。[E03–E06,E13,E20]

### 3.1 锁定值

Actor 文件：

```text
results/cai_agent_v3/w3_pilot/r1_0e11452a/models/actor_no_vlm_spatial_feedback_seed1.pt
sha256 = 0dbf45392c8a06e40b86c7275c1ab46e29ef11352016d9feaf0f31db924a646d
selected_update = 1000
seed = 2026091302
```

Predictor 文件：

```text
results/cai_agent_v3/w2_replay/r1_292b1c74/models/predictor_mean_sc.pt
sha256 = f67e912bd5faa26ae5cff8a9a0241439797fccef8cec825f43ebe5dc9f9d57ff
selected_update = 1750
```

只读 job 中已保存的候选成绩，确认当时的选择过程；**不再次运行候选评估**。checkpoint 文件哈希与 `state_dict_sha256` 不是同一种哈希，不能互相替代。已有权重文件可按字节哈希一次；权重不在本地时，使用已核实 manifest 并记录 `checkpoint_bytes_not_reverified`，不要为纯写作下载权重。

### 3.2 方法事实

- 编码器、预测器和策略是不同角色；Actor 的 Transformer 不等于预测器的架构。
- `MeanSCPredictor` 是区域 MLP + 全格均值/已测格均值，内部描述符在仿射投影前置零，投影偏置保留。
- `SpatialCAIActor` 使用未知内部描述符；区域 token 102维输入投影到128，global输入6维。当前版本两处各有2个恒零 VLM 占位通道；有效输入与存储宽度都要据实描述，不能删通道后声称仍是同一权重。
- 三份分组交叉拟合预测器用于训练侧代价；评价使用共同冻结 MEAN_SC，不是三模型集成。
- 策略读获取历史，预测器不直接读历史。固定最终集合与累计面积产生同终点输入，这是结构性质。
- 正文不要重新加入“VLM每步规划”“BC模仿”“STOP判断”“重建完整损伤图”等其他历史任务内容。

**输出**：本轮 MODEL_IDENTITY_LOCK，包含 `new_model_selection=0`、`research_model_forwards=0` 和 manifest/代码依据。

**验收**：正文、图1–2、S2、摘要和交接中的模型身份完全一致。

---

## 4. T02：实现轻量证据适配与方向修正

### 4.1 数据入口与家族隔离

主性能表优先读取 EVID/method_summary.csv 中下列6个 method，另从已保存完整输入参照加入 full 行：

```text
CENTER_FIRST
GEOMETRY_SPREAD
SERPENTINE
RANDOM
LEARNED_STATIC_TRUE
NO_VLM_SPATIAL_FEEDBACK
```

该原 CSV 还有3行 VLM 方法，不能混作当前 Proposed，也不要删除或覆盖它们的历史源文件。表格数据选择必须按 method 字段白名单，不按“最优几行”或行号。[E09]

ORDER 中区分：

- `ARCHIVED` 家族：6方法、500条轨迹、7,849事件，原保存面积时钟。
- `FIXED_SET_ORDERS` 家族：350条件，NATIVE_REPLAY/REVERSE/PERMUTED，整数累计像素时钟。

原归档主法 A 为 43.59708783510903，固定集合原序重算 A 为 43.59708700243914，三位小数均为43.597。允许已报告的浮点级差异，不能跨家族要求逐位相同，也不能拼接两个家族的中间轨迹。

### 4.2 必修：8条阶段贡献的差值标签

固定快照的 `ORDER/order_paired_contrasts.csv` 对所有行统一标记“comparator−native，正值有利于native”。这对误差类量成立，但对数值越大越有利的 `weighted_stage_1_mpa` 至 `weighted_stage_4_mpa` 不成立。[E10]

只在新派生表 `PAPER/tables/order_paired_contrasts_normalized.csv` 中处理：

```python
# 不是重估效应；只是把已经保存的同一效应转换显示方向。
if metric matches r"^weighted_stage_[1-4]_mpa$":
    new_estimate = -old_estimate
    new_ci_low = -old_ci_high
    new_ci_high = -old_ci_low
    difference_definition = "NATIVE_MINUS_COMPARATOR_POSITIVE_FAVORS_NATIVE"
    positive_favors = "NATIVE_REPLAY"
else:
    # A / early A / objective / final error 等误差类行完全保留
    keep_saved_values_and_definition()
```

保留 `source_estimate/source_ci_low/source_ci_high/source_difference_definition` 和源 blob SHA。只应修改8条贡献行；其余8条误差行保持数值不变。修复后的4阶段贡献和应与对应 A 差闭合（1e-8 MPa内）。记录在 change_log 和 S4 的数据说明。

**不要对 `stage_paired_contrasts.csv` 全局反号。** 那是归档方法间阶段比较，其原始/加权贡献方向与这里的固定集合行不同；原稿图5a/b使用这份正确来源。不能为了“统一正值”把已经正确的表反转。

若执行时源分支存在已修正版本，先验证定义、数值与 blob，记录已满足；不得二次反号。对未知方向明确报错，不根据效应正负猜差值定义。

### 4.3 统计口径不变

MAE、MSE先对随机重复算损失再试样内平均，随后按物理试样池化；不能先平均预测再算绝对误差。过程 A、early A 和阶段贡献按重复→试样→来源组→六组等权。不能因图3好看而换聚合。

每个报告 cap 读取总 cap0.25 路径中不超过该值的最后完成预测；事件间保持当前预测，使用阶梯曲线，不插值、不平滑、不使用未来状态。结束未满 cap 时末预测延续到 cap。

质量目标保持41–61 MPa整数范围及既有完整输入锚点；未达标为 null/NR，分母0时比率未定义；保留首次达标后回升及负节省。读取已有区间，不重新抽样。

**验收**：表1–5的原有科学数值、样本、method身份与全部不利结果保留；仅显示方向错误被可追溯修正。

---

## 5. 开发接口与执行边界

在 CODE 实现一个轻量 CLI，以下命令是**待实现的合同**。执行前先完成代码，不能把不存在的命令当成已运行。

```bash
python scripts/cai_manuscript_revision/run.py bind \
  --repo "$ROOT" --packet "$PACKET" --paper-root "$PAPER" --artifact-root "$ART"
python scripts/cai_manuscript_revision/run.py prepare-evidence --paper-root "$PAPER" --artifact-root "$ART"
python scripts/cai_manuscript_revision/run.py render-figures --paper-root "$PAPER" --artifact-root "$ART"
python scripts/cai_manuscript_revision/run.py build --paper-root "$PAPER" --artifact-root "$ART" --languages en,bilingual
python scripts/cai_manuscript_revision/run.py verify --paper-root "$PAPER" --artifact-root "$ART"
```

接口要求：

- `bind` 只验证来源、版本、路径、必要 schema；不触发研究准备/下载/推理。
- `prepare-evidence` 读取冻结结果，导出主表/图的数据子集、修正方向及事实映射；不运行旧研究 CLI。
- `render-figures` 只读 figure_data 与已绑定图片，输出新目录；可单独重画受影响图。
- `build` 从规范源文本生成 DOCX/PDF，生成后不得再手工修改导出稿形成第二事实源。
- `verify` 检查本轮产物和来源关系，输出机器状态与可读 REVIEW；不执行任何训练代码。
- 路径解析后限制写入新 PAPER/ART/DOCS/CODE；已有文件同任务续跑可复用，输入变化只重建受影响输出。异常信息包括任务、文件、字段及缺口。
- 运行期处理 CSV 的编码、重复 key、必需列和 method 缺失；这些是必要数据验收，不扩展为通用安全框架。
- 不创建 Web UI、数据库、后台 worker、远程监控或庞大依赖管理系统。

---

## 6. T03–T06：逐章修改任务与证据绑定

顺序固定：第4章事实/协议 → 第3章方法 → 第5章结果 → 第1、2章 → 第6章/摘要。保留六章，不以重写为名把旧稿内容自动补回。

### T03：第4章，建立一次性且完整的评价协议

位置：原稿§4.1–4.4；DOCX P132–P169（原始0起始索引）。

修改动作：

1. 明确276件总队列分为161训练、50开发验证、65预留；捕获组对应152/48/59。所有主效果使用同50件，checkpoint也由该开发验证分区选择。
2. 试样、捕获组、来源组、随机轨迹、事件分别定义；500/350/7,849都不是独立试样数。
3. 将六种序贯方法共享的编码器、预测器、表面输入和原生面积规则写清。full input单列成本1.0，不赋予虚构过程A。
4. 说明较小cap是原总cap0.25轨迹的前缀，不是重新规划的低预算策略。
5. 同一段区分池化终点误差与六域等权过程量；统计区间条件于已选checkpoint，未校正模型选择/多重比较。
6. 表1保留实际配置与分区；表2保留选点信息和共同预测条件。

验收：摘要/正文/图注不再出现把当前50件称为独立test的表述。无需在每段重复免责声明，但协议定义必须可见。[E02,E04,E09]

### T04：第3章，方法角色与目标解释

位置：§3.1–3.6；DOCX P40–P131。

- §3.1先交代固定区域编码、共同强度读出和区域策略的职责；训练侧实测y只用于任务代价，不进入执行状态。
- §3.2说明固定位置、整格请求、测量掩膜、未知表示、原生面积；B=.25允许15或16格。图中编号与实际位置一一对应，获取序号不能代替空间位置。
- §3.3解释全格/已测格汇总、仿射偏置、训练归一化与MPa输出；OOF和共同读出不混淆。
- §3.4用输入→上下文化→区域分数→可行动作解释Actor；102/6维存储输入中的恒零占位通道在S2披露，不需要以VLM叙事扩张正文。
- §3.5保留 A(B)、J、事件恒等式与更新方向。恒等式是既有目标的等价展开，不称新定理或充分机制证明。完整推导与实现参数移S2，主文不重复三次“更早更有价值”。
- §3.6算法区分历史训练和固定权重执行。算法的train文字是方法描述，不是本轮执行授权。

公式重排可读性修正必须保留运算结构：

```text
e_t = |y_hat_t - y|
A(B) = [sum_{t=0}^{T-1}(c_{t+1}-c_t)e_t + (B-c_T)e_T] / B
J = A(B) + 0.25 e_T
Delta e_t = e_{t-1} - e_t
A(B) = e_0 - sum_{t=1}^T (1-c_t/B) Delta e_t
G_t = (1/sigma_y) [sum_{k=t}^{T-1} ((c_{k+1}-c_k)/B)e_k
                   + ((B-c_T)/B + lambda_T)e_T]
```

把括号、求和上下标和绝对值符号作为真实公式核对，不能把PDF提取时的错序当作要复制的LaTeX。保留原稿13个编号公式的引用关系；若一部分移S2，更新所有引用和编号映射，不留悬空“Eq.(11)”。[E05,E06,E13]

验收：图1–2、算法、公式与已选模型一致，没有“全部模块端到端联合训练”“在线更新权重”或读到未测内部信息的错误。

### T05：第5章，按证据强度重排结果

位置：§5.1–5.6；DOCX P170–P250。

**§5.1 终点→早期→过程**：先写42.385及相对geometry的4.525 MPa改善和CI[0.670,8.276]；再写原总cap路径的0.0625前缀MAE43.324；最后写A改善3.714和CI[-0.337,7.283]。过程CI跨零不能被终点CI替代。保留五对照和full行。[E02,E09]

**§5.2 等质量**：47 MPa五点网格上Proposed .0625、geometry .125、Random .25、learned-static .0625。50%只针对geometry和该网格；75%只针对Random。保留更严格目标、事件网格、首次交叉后的回升以及未达到full anchor。不把首次交叉解释为自动停止规则。

**§5.3 事件与顺序**：区分未加权净下降与时间加权贡献；阶段求和是各事件变化之和，不是“每步平均”。固定集合先报告主要置换A差1.486，CI[-0.658,3.384]，再报告辅助逆序3.929，CI[0.145,7.405]；早期两项区间均跨零。终点相等为集合型读出的结构性质及实现一致性检查，不是新的准确性提升。

删除/合并§5.3.3中与前两小节重复的总结，但保留“目标解释计价，事件显示实际变化，固定集合分析区分内容与到达过程”的一句收束。不把完整方法差减去顺序差计算贡献百分比。

**§5.4 行为诊断**：把9/94限定为6个既定案例中的相对初始可行排序变化。描述“试样相关初始化＋选择性重排序”，不把其变化完全归因为新C-scan内容，也不说已通过动态性消融。主图仍用c8-16和q24-48，六个案例都保留在S5；不择优换掉q8-22或大残差案例。

**§5.5 组间与全输入**：表5主要行标签改为层数/铺层，编号作为追溯。保留8层准各向同性不利结果、c24终点不利结果、29/50终点改善与33/50过程改善。部分与full的差0.695、CI[-4.009,5.327]只作差异估计，不称等效/非劣/无损替代。

**§5.6 讨论**：依次说明观测组织为何成为优化对象、现有三类分析各支持什么、区域请求接口的工程意义。面积是信息规模，不等于设备时间；实际扫描延迟/移动和独立泛化留作明确边界，不另造硬件实验数据。

验收：正反向结果都可见，主要/辅助比较不互换，结论不强于证据；数字与引用位置写入 CLAIM_EVIDENCE_MAP。

### T06：第1、2、6章和摘要

**引言**：围绕“选择什么”和“何时进入预测”提出任务，不将普遍资源短缺或工业只能固定路径作为未经证实前提。解释中间估计为什么是本文研究对象，而非声称已存在经验证的实时停止需求。

三项贡献固定为：任务表达（试样级CAI监督区域请求）、方法（可见多模态状态与过程/终点联合目标）、评价证据（同cap准确性、同质量观测需求、事件与固定集合分析）。网络名、缓存和代数恒等式不独立包装成通用算法创新。

**相关工作**：保留2.1图像CAI、2.2序贯检查、2.3主动特征获取/部分观测预测这条问题链。将原2.3中具体token、位置和mask实现描述移方法。只沿用已核实引用；若增加新的文献差异主张，必须阅读支持它的原始论文段落并登记。没有读全文不能写“对方不使用某机制”，不同协议成绩不并入主表。文献检索限定于实际新增/修改句，不启动大规模新综述。

**摘要**：默认200–250英文词是本轮编辑目标，不称AEI硬限。保留离线区域回放和50件开发验证条件；主要数值为B=.25 MAE42.385、相对geometry9.65%、同一路径.0625前缀MAE43.324。47 MPa的50%留结果章。顺序结果用与区间相称的短句，不能只突出reverse、隐去主要置换不确定性。

**结论**：两段，方法/主要发现→有边界的工程意义；不重列所有表格。不得新增效果、文献或泛化主张。

**语言**：先陈述研究动作和发现，再给必要条件；不用通篇否定句掩盖主线，也不能用positive framing删掉跨零区间或反向结果。中英文对齐模型、数字、方向、主次比较和适用范围，不要求逐字直译。

验收：标题和六章结构保留，完整主线与原有证据一致，没有新增科学事实。

---

## 7. T07：六幅图的开发合同

复用仓库既有 Python 绘图路线。不要启动图像生成服务；不得生成或修饰成“真实数据”的试样、C-scan、损伤GT、attention或采集轨迹。示意图必须标schematic；真实图片必须绑定原文件和试样key。

| 图 | 数据/代码入口 | 必做修改和验收 |
|---|---|---|
| Fig1 | 已选模型代码、原图1、固定案例 | 保留表面表示→策略→观测/状态更新→强度读出四模块；区分训练/执行；用真实已保存观测或明确示意说明一次更新，不再只有四个文字框。y只连训练代价；输出区域编号与CAI/MPa；权重冻结。 |
| Fig2 | models.py的MeanSCPredictor/SpatialCAIActor、原图2 | 可见/未知格→区域/全局token→空间上下文化→64分数→可行性掩膜→选格；共同预测器作为独立读出。未知区域不能被画成内容已知。原生面积与格数不当作完全等价。 |
| Fig3 | 已保存归档cost-error曲线/episodes | 六方法阶梯MAE；标.0625与.25关键点；保留局部反弹和full1.0水平参照。x轴是区域观测比例，不是时间；图注说明MAE池化，表3 A为六域等权。 |
| Fig4 | 已保存matched-quality结果/质量目标表 | 同一组预设目标的五点/事件双网格；明确NR和初始零成本；保留learned-static在q47五点同样.0625，不只挑有利阈值。 |
| Fig5 | stage_paired_contrasts、order_summary、order_paired_contrasts、fixed-set曲线 | a未加权早期净变化；b阶段加权变化；c同最终集合轨迹；d主要置换与辅助逆序A差。标单位、零线、方向及不同聚合；固定集合贡献若进入SI，使用T02归一化派生表。 |
| Fig6 | c8-16、q24-48对应原图、已保存动作/预测与原图6 | 真实表面、内部C-scan及采集格叠加、顺序、CAI轨迹构成案例链；保持q24-48残差。不能把softmax当损伤概率；不能推断未有的真实损伤标注。 |

### 7.1 Fig6 原图解析与缺失处置

当前已核实的cohort manifest提供试样key、原生尺寸与格面积，但**还未确认该运行环境中每件原始C-scan的实际图片路径**。按以下确定顺序解析：

1. 在 ORDER/input_bindings.json、ORDER/cohort_manifest.csv、feature index及其明确引用的配准/源图manifest中找到这两件试样。
2. 用已确认的配准图路径、裁切方向和原生尺寸，保留现有8×8边界；按实际保存动作序号叠加。禁止临时重新配准/重新分割以凑图。
3. 检查源文件存在、试样key一致、尺寸/方向可对应、发布许可/来源记录无明确障碍；只能称“原始二维C-scan图像展示”，不能改称恢复的物理幅值或损伤GT。
4. 若定向查找后无法取得可信原图，保留并重新排版当前DOCX中的原图6及其真实轨迹，不新增内部图面板；FIGURE_SOURCE_MAP记录 `OPTIONAL_CSCAN_OVERLAY_UNRESOLVED` 和缺失文件/元信息。继续其余改稿，不更换成网上图片或伪造热图，也不声称这项增强已完成。

真实C-scan面板可作为事后解释，但若展示未采区域必须明确“retrospective full-image reference，未提供给策略”；运行侧示意图只揭示合法可见格。

### 7.2 图稿输出与有限质量要求

图1–2使用原生可编辑矢量文字/形状；图3–5曲线/标签矢量化；图6的原始影像保留栅格，不声称所有像素可编辑。输出SVG/PDF及PNG预览，保留源码、图数据和来源。不得描摹图片曲线代替读取数值。建议最终物理尺寸下标签不小于8pt，这是本项目排版目标，不是期刊政策声明。

对每个修改过的最终图导出后检查一次文字、碰撞、面板对齐和CI方向；已有可用nature-figure检查脚本可定向运行。脚本不可用时进行渲染人工核查并标明自动检查未执行，不安装整个绘图/审稿生态，也不声称自动PASS。

---

## 8. T08：补充材料、引用与构建

### 8.1 S1–S5 的内容合同

- S1：数据、配对、capture group、划分与50件评价清单。只使用现有metadata；不读取预留标签做评价。
- S2：模型参数/占位通道、部分掩膜训练、三折OOF、已选模型与目标推导。参数来自当前模型，不复制旧VLM方法说明。
- S3：全部cap、实际面积、重复损失、已有配对区间、全部质量目标和未达标/回升。
- S4：事件/阶段定义、350条件、原序核验、主要和辅助比较；保留T02符号映射说明及原始来源。
- S5：六个固定案例及已存在的信息使用诊断，区分softmax、归因、attention。既有素材缺失不能新前向生成。若不能恢复某附件，精确列缺口，不写“all data available”。

旧supplementary只能作为逐项核对后的素材，不整篇复制为当前SI。正文里引用S1–S5时必须有实际交付内容/具体数据链接，不能留空壳。

### 8.2 单一源与构建方式

先以当前DOCX提取并对照PDF生成规范英文/中文源文本，保留公式运算结构和真实图表。使用稳定段落ID或对应表维持双语关系；不靠“偶数段英文、奇数段中文”直接删除，因为表题、公式和算法会破坏此规则。

原仓库 `build_manuscript.py` 已核实使用Pandoc，假定英文abstract.md恰有3段并重写所在目录。这一脚本**不能在旧目录直接运行，也不能直接喂当前双语稿**。只在新 CODE/PAPER 复用其转换思路，改成显式输入/输出根和双语构建。[E08]

优先使用环境现成Pandoc＋DOCX/LaTeX或LibreOffice工具链；将公式输出为可编辑Word公式/LaTeX，不用低清公式截图。字体使用本地可用字体，不复制字体文件。正文图表由共享数据生成，不在英文和双语稿分别手填数值。

编译主文与SI的英文、双语四个版本，生成对应DOCX/PDF。一次正常安装/路径修正可解决的缺依赖允许处理；不能大规模升级系统。确实缺少编译能力时交付完整可编辑源和 BUILD_BLOCKED 证据，不把缺PDF标为完成。

构建后查看全部最终PDF页，重点核对公式、五表、六图、中文字符、段落跨页、链接与交叉引用；页数可以合理变化，不以强行保持29页破坏排版。

作者、基金、通讯信息、CRediT、利益冲突、投稿批准等未知内容保持待作者确认。不要声称已获Nature/AEI正式认可。当前包不授权替作者作实质声明。

---

## 9. T09：一次有限审核与局部修复

执行 `REVIEW_AND_PUBLICATION.md` 的Q01–Q08：来源身份、模型/方法、数值/方向、统计/主张、图像与案例、双语/引用、构建视觉、交付范围。

审核实际修改后的源文件和产物，不只比较配置与自己写的测试。最多8项小型数值/结构测试加一次完整文稿审阅；只复核修复影响的地方。不运行全仓库pytest，不做渗透/fuzz/压力/硬件负载/重复GPU监控，不因模拟审稿分数继续无限重写。

可用nature-reviewer做**一次明确的AEI定向内部审核**，不是伪造三位独立互盲审稿。重要缺口保留准确严重程度，不要求所有科学限制在本轮“消失”。

失败项必须记录，必要时局部修复。有限资源上限不是掩盖失败的理由：达到上限仍有问题则交付可检查的部分结果和 BLOCKED 状态，不声称全通过。

---

## 10. T10：交接、GitHub提交推送和发布核验

具体命令和两提交协议见 `REVIEW_AND_PUBLICATION.md`，执行，不只描述。

交接必含：源PDF/DOCX哈希、证据SHA与实际执行基点、锁定模型、每项修改/文件映射、方向修正8行、图6原图状态、构建命令、QA真实结果、作者待确认、未解决的科学边界、运行消耗、提交与远端验证。

提交范围只包括新PAPER/ART/DOCS/CODE和定向测试。不得`git add .`把整个研究仓库、原始数据/权重/缓存带上。不得把原输入文件标成“已修订稿”。

先提交内容C并push，验证origin任务分支=C；然后写记录C的PUBLICATION_RECEIPT并提交元数据R、push，再验证远端=R。最终回复给出C和R。收据不要求包含自身最终commit SHA，manifest排除自身与收据，避免自引用循环。

本任务“发布”仅指GitHub任务分支上可访问的结果，不等于合并main、创建公开release、正式投稿或外部独立验证完成。

---

## 11. 完成状态与阻塞决策

同时记录三条状态轴，不能混成一个PASS：

1. `artifact_status`：`REVISION_COMPLETE` / `REVISION_COMPLETE_WITH_DECLARED_ASSET_LIMITATION` / `BUILD_BLOCKED` / `REVISION_PARTIAL`。
2. `publication_status`：`REMOTE_VERIFIED` / `LOCAL_READY_PUSH_BLOCKED` / `NOT_ATTEMPTED`。
3. `scientific_status`：固定为有边界的开发验证结果；不得写 `INDEPENDENTLY_VALIDATED` 或 `ORDER_OPTIMALITY_PROVED`。

若仅Fig6新增内部图面板缺失，而原真实案例图保留、其余必交项目完成，可用 `REVISION_COMPLETE_WITH_DECLARED_ASSET_LIMITATION`，明确这项增强未完成。模型身份冲突、关键主表无来源、伪造案例、构建失败或数字矛盾不属于这种可降级项。

身份/凭据或缺文件问题只做一次正常解析/修复尝试；仍不可解决时完成可独立进行的任务，精确记录阻塞。不要重复向用户询问已经明确的六章结构、模型选择、是否训练或是否推送。

## 12. 最终回复合同

用中文交付：修改完成度、英文/双语主文与SI路径、图稿和源码入口、已锁定模型、8行方向修正结果、有限验收结果、内容C/收据R提交、远端任务分支和可访问文件链接，以及仍需作者处理的真实事项。报告研究训练/前向/新选模均为0时，必须与实际命令记录一致。

禁止用“代码已准备”“建议自行运行”“下一步再推送”替代用户已授权且环境允许完成的执行。若真实环境阻塞，则明确是哪一步没有完成，而不是虚构成功。
