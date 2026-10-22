# 文件理解、证据等级与修改任务绑定

本文件把已读取的资料、由其支持的判断与待实施动作分开。`VERIFIED_READ` 表示本次读到了相关内容；`MANIFEST_POINTER` 表示已核实的manifest/源码指向该文件，但本次未读取完整字节；`RUNTIME_UNRESOLVED` 表示必须由Codex在实际环境解析。禁止将三者混称“已复现”。

所有GitHub研究路径的证据快照为：

```text
https://github.com/Orangekostar/diff/tree/6082b69f2792130a321c74b0efcb198f5d65cefb
```

以下Git blob SHA标识Git对象，不是文件SHA256。两类哈希在记录中分栏。

## 1. 已核实来源登记

| ID | 来源/状态 | 实际读取的内容及支持范围 |
|---|---|---|
| E01 | GitHub仓库/分支元信息，VERIFIED_READ | 仓库Orangekostar/diff；证据分支research/cai-vlm-agent-v3-controlled-reuse，读取时HEAD=6082b69f…；默认main不是本文证据基点。本轮执行仍要记录实际remote状态。 |
| E02 | `inputs/current_manuscript.pdf` 与 `.docx`，VERIFIED_READ | 当前29页中英稿全部文本、表1–5、图1–6；DOCX内容与PDF逐页归一化文本匹配。论文陈述是修订对象，不自动等于所有底层实验均被本次复现。 |
| E03 | `results/cai_agent_v3/order_mechanism/r1_37b3c404/MODEL_SELECTION_LOCK.json`，VERIFIED_READ | Actor=NO_VLM_SPATIAL_FEEDBACK@1000/seed2026091302；predictor=MEAN_SC@1750；文件路径和SHA256；new_model_selection=0。blob `0dfbbcf923ebd7b8a5cc4760cd6f10b40dcba05a`。 |
| E04 | 同目录 `ORDER_MECHANISM_SCOPE.json`，VERIFIED_READ | 六方法、50件/48组/六域、500轨迹、350条件、历史数据入口、聚合规则、两个时间时钟和历史允许的predictor前向。blob `a44077bccf655ba481132d978d68daa14a7fbe4f`。 |
| E05 | `results/cai_agent_v3/w3_pilot/r1_0e11452a/jobs/NO_VLM_SPATIAL_FEEDBACK.json`，VERIFIED_READ | 1250训练更新、1000选择点、候选成绩、架构、378978参数、checkpoint文件哈希/state_dict哈希及selected episodes路径。blob `4ee37093285431447a01b77a6a1680f940c514ba`。 |
| E06 | `src/cmc_bbdm/cai_agent_v3/models.py`，VERIFIED_READ（1–397行相关区段） | MeanSCPredictor与SpatialCAIActor定义；真实mask位置、池化、token宽度、零占位输入；没有执行forward。blob `d98fdd0f8431de13a4b463ee7a2361a73d67bad0`。 |
| E07 | `paper_cai_aei/r2_c_331f5295/README.md`，VERIFIED_READ | 明确写C_P0_R1_GLOBAL_V1、VLM_SPATIAL_FEEDBACK@250、VLM_SPATIAL_OPEN_LOOP@1250、VLM_MEAN_FEEDBACK@250；因此它的“current”不能覆盖当前上传稿。blob `0c907ef9aa6b1e8dfa0a18ec548b3858348eb772`。 |
| E08 | 同目录 `build_manuscript.py`，VERIFIED_READ | Pandoc→MD/TeX/HTML，读取abstract.md三段与sections/0[1-6]_*.md，输出到脚本所在目录。适合作转换基础，不可直接运行旧目录或直接输入双语全文。blob `6dffc662d74029bf3292c79e94f3736f6c3a5747`。 |
| E09 | `results/cai_agent_v3/c_render_retrain/r1_331f5295/evidence/method_summary.csv`，VERIFIED_READ | 含9方法；当前稿用前述6方法子集。NO_VLM一行与本文43.597/42.385等匹配，另3行VLM属于历史方案。blob `dd1705aa3387ea160699dbf51d952b5bbe9bf2ce`。 |
| E10 | `ORDER/order_paired_contrasts.csv`，VERIFIED_READ（全部16行） | 两个预设对比各8行；其中4个阶段贡献行方向标签不适合higher-is-better，总计8行。A/early A/endpoint/objective类行没有此问题。blob `23a65f8d481797298b25bb228d1b2fd91181c65f`。 |
| E11 | `ORDER/order_summary.csv`，VERIFIED_READ | Native/Reverse/Permuted mean的A、early A、endpoint与阶段贡献全精度值。blob `21242f7ea8cff471a2495e8c04d35a67878be019`。 |
| E12a | `ORDER/acceptance_results.json`，VERIFIED_READ | 历史程序报告10项通过、350终点、原序预测最大差6.103515625e-05 MPa、既有资源/缓存计数。本次没有重新运行这些检查。blob `a0e68412965c27488a72339f9ae7ba6c4b547bb6`。 |
| E12b | `artifacts/cai_agent_v3/order_mechanism/r1_37b3c404/PUBLICATION_RECEIPT.json`，VERIFIED_READ | 历史内容提交738e79e7…与远端核验；文件说明收据记录前一提交而不自引用。不能把该历史push当成本轮push。blob `78e13e73ced7533e0941fba4594aca9bd7b3bffd`。 |
| E12c | 同artifacts目录 `CODEX_HANDOFF_CAI_ORDER_MECHANISM.md`，VERIFIED_READ | 历史主置换CI跨零、次逆序正向、终点一致和不重选模边界。blob `c7d3afa7942fb1a6dd0f285c8390fc4c2f134dc3`。 |
| E13 | `src/cmc_bbdm/cai_agent_v3/metrics.py`，VERIFIED_READ | prediction_at_budget、left_error_area_mpa、trajectory_objective_mpa、policy_cost_to_go；末状态保持、cost-to-go含当前后续区间。只读，模块导入torch，不为算术检查加载整个研究环境。blob `e03de785d7246cb3a3b163b7dc19002281379343`。 |
| E14 | `scripts/cai_order_mechanism/reporting.py`，VERIFIED_READ（1–190行） | 现有Python/matplotlib绘图、stage_paired_contrasts、cost_error_curves.csv.gz及FIXED_SET_ORDERS分组；可复用绘图思路，不在冻结根执行report。blob `3d4109716ac36a4029385a157441abf49bb0b51f`。 |
| E15 | `scripts/cai_order_mechanism/run.py`，VERIFIED_READ（1–180行） | all会串联prepare/derive/plan/infer/analyze/report/verify；运行旧all会突破当前零前向范围。因此本轮新入口不得调用它。blob `ae70c43fa4f76003a384948fbe619bbdc35a4295`。 |
| E16 | 文件库 `AEI_CSCAN_FINAL_WRITING_OUTLINE_V6.md`，VERIFIED_READ | 已明确当前无VLM主法结果、完成的顺序分析、同一50件和不重选；F2已提示8行方向标签问题、F3每件应15或16格。本轮对问题做了原CSV独立读取确认。此大纲基于25页旧稿，不能沿用其旧页码。 |
| E17 | 文件库 `CODEX_AEI_MANUSCRIPT_EXECUTION.md`，VERIFIED_READ | 既往允许写作、禁止新研究计算、有限审核/推送等边界；其VLM主法/九方法/旧题目已被当前稿与E03替代，不整体执行旧指令。 |
| E18 | `ORDER/cohort_manifest.csv`，VERIFIED_READ（前20行） | 真实字段包含specimen_key、capture_group_id、native_height/width、cell_pixel_counts、feature shard路径；c8-16=338×352。该表不是已确认的原始C-scan图片地址表。本次没有读完所有50行。 |
| E19 | Nature-skills三个入口，VERIFIED_READ | nature-reviewer、nature-polishing、nature-figure的实际规则，见第7节。用于当前任务的写作/图稿/有限审核，不作为期刊政策。 |
| E20 | `src/cmc_bbdm/cai_agent_v3/actor_training.py`，VERIFIED_READ（1–190行） | `_actor` 显式把NO_VLM_SPATIAL_FEEDBACK构造为SpatialCAIActor(use_vlm=False,use_feedback=True)；不是凭名称猜测。blob `80e73c097c6adedcd1bbbea34263e7c67e161798`。只读，不调用训练/seeded_actor。 |

## 2. 对三项关键判断的推理与边界

### J01 不能从仓库旧“current C”稿直接改

**证据**：E02当前主结果42.385；E09对应NO_VLM行；E03锁定NO_VLM@1000、E20显式确认构造参数；E07却明确旧默认稿为C/VLM@250。

**判断**：同名项目和“current”字样不足以确定当前稿身份。新正文基于输入DOCX，研究事实取E03/E09；旧稿仅复用已核对的构建代码和参考条目。

**落实**：T00绑定、T01锁模、T04图1–2审查、T08构建隔离。

### J02 顺序阶段贡献8行需要显示方向修正，不需要重算实验

**证据**：E11 native第一阶段贡献13.069448880203744，permuted=9.995460713661215；E10差为−3.073988166542532，定义comparator−native却写positive_favors=native。贡献越大表示更多带符号过程改善，因此此标签与数值方向不一致。

**判断**：是8条贡献行的显示方向/标签问题，不是主置换A=1.486错误，也不是原稿图5a/b来源全错。可以对已保存效应作可逆符号变换，CI同时反向并交换上下界。

**落实**：T02派生表归一化；Q03精确检查8行、误差行不变、阶段和闭合。冻结CSV保持原哈希，不能覆盖来“证明原来正确”。

### J03 可编辑源稿已找到，但不是已经改好的稿

**证据**：PDF哈希6e736995…；DOCX哈希277a673c…；DOCX有5表/7图对象/188数学对象。重新导出29页后，与原PDF逐页去空白文本29/29一致，结构与页面视觉检查对应。

**判断**：它是可以用于本轮编辑的匹配源稿，不需要从PDF逐页OCR或从旧VLM论文重建。没有声称逐像素一致，也没有声称原稿现有图表已经达到本轮修订目标。

**落实**：T00输入绑定，T04保留/正确转换公式，T08双语单一源构建。

## 3. 代码对象 → 文中功能 → 实施动作

| 已读代码/记录 | 实际功能 | 本轮允许动作 | 不允许的误读/操作 |
|---|---|---|---|
| models.py::MeanSCPredictor._forward_weighted | 先mask内部，再仿射投影；全格/已测格池化输出MPa | 改§3.3与图2读出块；保持bias与归一化 | 换成SpatialPredictor，或改成“所有缺失投影严格为零” |
| models.py::SpatialCAIActor.forward | use_feedback使用已测内部与当前预测；use_vlm=False产生零通道 | 改§3.4、S2；标有效字段和存储宽度 | 为简化图删除权重通道、加入新VLM调用 |
| metrics.py::prediction_at_budget | bisect_right读最近已完成预测 | 绘制已存轨迹或按同规则查询 | 插值未来预测、把未完成格贡献提前 |
| metrics.py::left_error_area_mpa | 左常值积分，归一化到评价区间 | 解释A和carry；小数组核对 | 将每步误差简单平均等同面积积分 |
| metrics.py::policy_cost_to_go | interval累计+tail+terminal | 核对G_t符号、尺度与末项 | 重训actor验证公式 |
| E05 job.selected_episodes | 既有update1000保存路径 | 绑定动作/预测来源 | 搜索一个更好checkpoint |
| E08 build_manuscript.py | 英文MD源拼接生成TeX/HTML | 复制必要逻辑到新代码，显式输出根 | 在旧目录执行覆盖旧稿；split三段读取整个双语摘要 |
| E14 reporting.py | 定量图从保存CSV读数，step曲线 | 新目录绘图，引用数据键 | 调旧render_report改冻结resource_usage/旧图 |
| E15 run.py::ordered_phases | all包含infer | 明确禁用旧总入口 | 因脚本已有缓存便认为重跑all没有前向风险 |

## 4. 原稿章节/图表定位 → 修改事项 → 产物

P为DOCX直接body段落的**原始零起始编号**，以提供的哈希版本为准。表格不计入P序列。

| 任务 | 当前位置 | 输入依据 | 输出位置与验收关键词 |
|---|---|---|---|
| T03 数据/划分 | PDF13–14；P132–141；表1 | E02/E04 | §4.1、S1；276/161/50/65与捕获组、开发选择依赖明确 |
| T03 基线/评价 | PDF14–17；P142–164；表2 | E02/E04/E09/E13 | §4.2–4.3、S3；6方法、full1.0、前缀与两聚合 |
| T01/T03 选模 | PDF17；P165–169 | E03/E05 | §4.4、S2；1000/1750、单seed、零新选模 |
| T04 方法总览 | PDF5–6；P40–48；图1 | E03/E06 | §3.1、Fig1；固定编码/读出、学习策略、y仅训练 |
| T04 状态 | PDF6–8；P49–64；图2 | E02/E06/E18 | §3.2、Fig2；64固定位置、未知/测量掩膜、15或16格 |
| T04 预测器 | PDF8–9；P65–76 | E06 | §3.3；零mask在投影前、bias、均值池化、set-only |
| T04 actor | PDF9–10；P77–86 | E05/E06 | §3.4、S2；65tokens、2层4头、恒零通道 |
| T04 目标/算法 | PDF10–13；P87–131 | E13/原13公式 | §3.5–3.6、S2；A/J/G、代数身份、算法不等于本轮执行 |
| T05 准确性 | PDF17–19；P170–184；表3/图3 | E09、原配对CI | §5.1；终点、早期、过程分开，CI不串用 |
| T05 等质量 | PDF19–21；P185–196；表4/图4 | 原结果、既有质量目标结果 | §5.2；q47 static也达标，五点/事件与回升 |
| T05 时机 | PDF21–23；P197–217；图5 | E10/E11/归档阶段对比 | §5.3、S4；主要/辅助及原始/加权/家族隔离 |
| T05 案例 | PDF24–26；P218–232；图6 | 原六案例、E18、既有动作记录 | §5.4、S5；9/94仅六案例，q24-48大残差保留 |
| T05 分组 | PDF26–27；P233–241；表5 | 原表5、已存逐域结果 | §5.5；配置名替代难读ID、不利域保留 |
| T05 讨论 | PDF27–28；P242–250 | 已有结果与协议 | §5.6；三分析不同解释能力、无物理时间主张 |
| T06 引言/相关 | PDF2–5；P7–39 | 当前问题与已核实文献 | §1–2；task/design/evidence；2.3不重复具体网络 |
| T06 结论/摘要 | PDF1/28；P3–4、P251–255 | E02/E09/E11 | 摘要/§6；42.385与43.324，开发验证条件明确 |
| T08 可用性/参考 | PDF29；P256以后 | 当前15参考+真实SI | statements/S1–S5；无空承诺、作者项单列 |

## 5. 数字与科学表述的锁定表

这里主要用原稿显示精度；全精度来源保留在真实CSV。`reference/verified_facts.json`提供机器读摘要。

| 主题 | 固定值/规则 | 禁止变换 |
|---|---|---|
| Proposed主表 | A43.597、early A48.771、MAE42.385、RMSE56.534、R²0.682 | 换VLM行、把MAE当准确率 |
| 相对geometry | MAE减少4.525/9.65%，CI[0.670,8.276]；A减少3.714/7.85%，CI[-0.337,7.283] | 将MAE的正向CI写到A |
| 相对static | MAE减少4.608，CI[0.656,8.628]；A减少4.380，CI[0.199,8.405] | 用静态全局排序证明动态反馈独立有效 |
| 初始与前缀 | 58.551；.0625→43.324；.125→43.164；.1875→42.928；.25→42.385 | 将前缀当重新规划的低cap实验 |
| q47五点 | Proposed6.25%、geometry12.5%、static6.25%、Random25% | 对所有方法声称统一节省50% |
| q47事件 | Proposed3.1203%、geometry9.3426%、static6.2407%、Random24.9630% | 粗/细网格混填一列 |
| q42事件 | 曾在.078844跨越，终点42.385；五点未记录达标 | 将瞬时交叉称持续满足/可部署停止 |
| 主顺序对比 | permutation−native A=1.486，CI[-0.658,3.384] | 删除跨零区间，改主要为逆序 |
| 辅顺序对比 | reverse−native A=3.929，CI[0.145,7.405] | 推到普遍最优排序或早期A显著 |
| early顺序 | perm1.628[-2.190,5.180]；reverse3.328[-1.527,7.979] | 换入全过程A区间 |
| 原始早期变化 | 相对geometry6.865[0.362,12.502]；Random9.707[4.862,14.131] | 改称每一步平均下降或加权贡献 |
| 加权早期贡献 | geometry5.926[-0.117,11.095]；Random8.617[4.194,12.678] | 隐去前者跨零 |
| full输入 | MAE41.690、RMSE53.944、R²0.711，成本1.0 | 在25%曲线上画成真实已获得full预测 |
| 部分/full差 | .695，CI[-4.009,5.327]，实际平均面积.247174 | 等效、非劣、扫描时间减75% |
| 行为 | 6固定案例94次决策9次改初始可行排序 | 当50件全部动作统计；因果归于新内部图 |
| 案例 | q24-48真值136.680、终点258.932、残差122.252 | 删除高残差案例、换更好试样 |

六个来源组配置应从原表1映射，不按缩写猜：

| source | 配置 | train/dev/reserved | dev表5 Proposed A / MAE | gain A / gain MAE（geometry−Proposed） |
|---|---|---|---|---|
| 74t7kcdgkr | 8-layer cross-ply | 25/9/11 | 31.330 / 28.655 | +0.893 / +5.634 |
| cgtnjyggtm | 24-layer quasi-isotropic | 29/9/11 | 68.749 / 64.067 | +13.061 / +16.440 |
| w68dtmpfyf | 16-layer quasi-isotropic | 25/8/10 | 47.030 / 44.088 | −0.105 / +1.615 |
| xcmzfsbd9t | 24-layer cross-ply | 35/9/15 | 48.163 / 46.433 | +0.760 / −2.097 |
| yfxyg8jm46 | 16-layer cross-ply | 25/8/9 | 48.059 / 48.408 | +8.157 / +5.875 |
| ykhs7s2dck | 8-layer quasi-isotropic | 22/7/9 | 18.250 / 18.124 | −0.485 / −1.922 |

## 6. 已知入口但仍需实际绑定的文件

以下路径来自已读scope、源码或大纲。Codex必须先读实际文件，不能把本表的“指向存在”替代内容审核。

| 入口 | 级别 | 用途/注意 |
|---|---|---|
| `ORDER/archived_cohort_episodes.csv.gz`、`archived_events.csv.gz` | 文件目录已见，完整数据未读 | 归档路径、动作、预测、事件；读取一次，不重新生成 |
| `ORDER/reorder_trajectories.csv.gz` | MANIFEST_POINTER（run.py） | 350固定集合已算轨迹；不可执行infer补齐 |
| `ORDER/cost_error_curves.csv.gz` | MANIFEST_POINTER（reporting.py） | family/grid/series分组后绘图；不跨家族拼接 |
| `ORDER/stage_paired_contrasts.csv` | MANIFEST_POINTER（reporting.py及V6） | 原始/加权阶段比较；不是8行问题表，先确认定义 |
| `ORDER/input_bindings.json`、`release_manifest.json` | MANIFEST_POINTER（历史handoff） | 真实来源路径与文件身份 |
| `results/cai_agent_v3/new_protocol/feature_bank_index.csv` | MANIFEST_POINTER（scope） | 区域/试样对应；不能把特征bank当可直接显示的C-scan图片 |
| `artifacts/cai_agent_v3/w3_pilot/r1_0e11452a/INPUT_BINDINGS.json` | MANIFEST_POINTER | 原图/处理来源追溯，按引用继续读取，不遍历全部home |
| `EVID/full_scan_predictions.csv` | MANIFEST_POINTER（scope） | full1.0参照，仅当前50key，不新增前向 |
| `results/cai_agent_v3/w3_pilot/r1_0e11452a/p_all_saved_reference.json` | MANIFEST_POINTER | full预测身份 |
| `EVID/bootstrap_group_weights.npz` | MANIFEST_POINTER；ORDER内同名文件目录已见 | 已有抽样，默认无需加载/重算，缺区间不得新抽 |
| 现有matched-quality完整表、6案例记录、参考文献ledger | RUNTIME_UNRESOLVED具体路径 | 在ORDER/EVID/manifests/旧稿references下定向发现，按字段/哈希验证；不得凭名称创建假的输入文件 |
| c8-16与q24-48真实C-scan配准原图 | RUNTIME_UNRESOLVED | 只能按真实manifest和试样身份找到；缺失走主指令Fig6降级条款 |
| 当前29页稿的独立S1–S5源文件 | RUNTIME_UNRESOLVED | PDF提到它们不等于本次已读到；从既有真实资料整理，不伪称已给全 |

扫描限制：只列相关根和manifest引用路径；必要时对这些根的文本做一次精确关键词查找。不要全盘`find /`、全部Git历史重扫或下载全部大数据。

## 7. Nature-skills 的实际使用与边界

已读取下列入口，不声称读取了该仓库所有skill或其全部引用：

```text
https://github.com/Yuan1z0825/nature-skills/blob/main/skills/nature-reviewer/SKILL.md
  observed blob: 04938b9c002d929365a84ddcedff577a39713178
https://github.com/Yuan1z0825/nature-skills/blob/main/skills/nature-polishing/SKILL.md
  observed blob: 23b312f9ddf0a03971df0d11cbd2b42fed4e1821
https://github.com/Yuan1z0825/nature-skills/blob/main/skills/nature-figure/SKILL.md
  observed blob: 72a1c90240e74ec4b0690d61bba1d0b73ccbff22
```

Codex优先用现有安装，记录实际路径/版本；只读所需manifest与引用片段。任务轴为 methods、全文修订、zh-to-en、journal=generic（目标AEI），不要套Nature主刊章节/显示数量限制。现有Python绘图代码已确定Python路线，不再询问R/Python。

nature-polishing要求先修段落论证和证据边界，再处理句子；nature-figure要求每幅图有明确问题、真实数据与渲染QA；nature-reviewer要求来源与主张可核实，不能假造独立评审。当前工作是一次执行验收，不启动默认三报告循环，也不声称互盲。

当前公开skill可能更新，执行时读取到不同版本记录差异即可；不因此升级全部依赖或改变研究协议。没有当前官方出处时，不把编辑建议写成AEI强制政策。禁止把未发表全文发送给额外外部审稿/绘图API。

## 8. 包内小型原始快照的可验证性

`reference/snapshots/` 附带 E03/E09/E10/E11 的四个小文件。本包将连接器读到的文件内容保存后，按Git对象算法 `SHA1("blob " + byte_length + NUL + bytes)` 校核，四份全部匹配远端已读blob。具体文件SHA256/字节数见 `reference/snapshot_provenance.json`。

它们因此不是根据表格手填的仿真原始结果，可以作为离线交叉核对。冻结快照不可修改；当前任务的零前向授权仍由主指令控制，MODEL_SELECTION_LOCK历史字段`predictor.forward_allowed=true`不是本轮权限。它们也不包含所有轨迹/原图，不能替代运行期解析完整图源。
