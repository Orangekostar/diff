# CAI_POSITIVE_REVIEW_R1 交接

- artifact_status：REVISION_COMPLETE；scientific_status：SELECTED_DEVELOPMENT_CONFIGURATION_ONLY。
- 内容交付的publication_status：LOCAL_VERIFIED_PENDING_PUSH；实际远端状态只由随后PUBLICATION_RECEIPT.json和最终回复确认。
- 任务分支：paper/cai-positive-review-r1。
- evidence_sha / execution_base_sha：6082b69f2792130a321c74b0efcb198f5d65cefb。
- 隔离工作树：/home/ww/diff/.worktrees/cai-positive-review-r1；旧研究工作树的用户修改未纳入。

## 输入和模型身份

原稿为输入包中的当前NO_VLM稿，原题保持不变。历史r2目录仅取已核实BibTeX/构建思路，旧VLM主法和成绩不映射为本文方法。六个主方法见table3.csv；full-input单独绑定，不混入顺序轨迹家族。

| 输入 | 本轮原样保存位置 | SHA256 |
|---|---|---|
| current_manuscript.pdf | docs/cai/positive_review_r1/inputs/current_manuscript.pdf | 6e7369955e378592c04758d5b8297391eac0a4fa89babbed255061cee47188c3 |
| current_manuscript.docx | docs/cai/positive_review_r1/inputs/current_manuscript.docx | 277a673c2235f2196781fa1b78cad90310fd43cc8820b15fca1e45bc36264160 |

Actor锁定SpatialCAIActor，NO_VLM_SPATIAL_FEEDBACK，use_vlm=False、use_feedback=True；seed 2026091302；选定1000/已完成1250次；参数378,978。文件sha为0dbf45392c8a06e40b86c7275c1ab46e29ef11352016d9feaf0f31db924a646d；state_dict身份554128e580590e93efb3d60c974476caeec0c7a1f96f49cc2ba582465f070dfc来自冻结manifest，未反序列化张量重算。共同预测器MeanSCPredictor（MEAN_SC）选定1750/完成2000次，文件sha f67e912bd5faa26ae5cff8a9a0241439797fccef8cec825f43ebe5dc9f9d57ff。

T00对本地已存在的Actor/Predictor文件均完成原始字节SHA256核验，实际路径和字节数见MODEL_IDENTITY_LOCK；未加载研究模型。运行计数：optimizer_updates=0，research_model_forwards=0，new_bootstrap_draws=0，new_seeds=0，new_test_evaluation=0，new_checkpoint_selection=0。CPU仅作既有数值整理、图稿、文档转换和八项适配器测试，线程上限4。

## T00–T10

| 阶段 | 内容状态 | 产物/改变 |
|---|---|---|
| T00 | COMPLETE | 原稿、69条实际读取文件绑定、固定基点及task_id |
| T01 | COMPLETE | MODEL_IDENTITY_LOCK、模型字节核验与零研究计算边界 |
| T02 | COMPLETE | 6方法白名单、table1–5、完整精度子集、8行方向修正 |
| T03 | COMPLETE | 第4章：276/50、152/48/59组、前缀、Random损失与估计量 |
| T04 | COMPLETE | 第3章：合法可见状态、独立读出、102/6维、13原公式及代价最小化方向 |
| T05 | COMPLETE | 第5章：终点→早期→过程、等质量、事件/固定集合、六例及不利组别 |
| T06 | COMPLETE | 第1/2/6章与200–250词摘要；贡献对应任务、方法、评价证据 |
| T07 | COMPLETE | 6主图与6病例板SVG/PDF/PNG，真实C-scan已解析 |
| T08 | COMPLETE | S1–S5、四版DOCX/PDF；稳定双语块173/67；共享公式/表格只出现一次 |
| T09 | COMPLETE | 一次主代理内部审核与局部修复；八项适配器测试8/8通过，见LOCAL_VALIDATION |
| T10 | CONTENT_COMPLETE_REMOTE_PENDING | 交接/manifest；C与R须按下述真实远端协议生成 |

六章结构、五张主表、六幅主图保留；第3章澄清角色和信息权限，第4章先固定协议，第5章限制主张，其余章节围绕任务主线压缩调整。change_log.md及TEXT_CHANGE_MAP提供逐块理由；equation_map.csv保留1–13编号及位置。两份主文DOCX均为5表、6图、13个可编辑编号公式。

固定集合源order_paired_contrasts.csv的Git blob为23a65f8d481797298b25bb228d1b2fd91181c65f，SHA256为98786ed1638ac3372ffa274257a43dc80497be1d2a17c6d648536a955570a26b。8条weighted_stage估计反号、区间[lo,hi]→[−hi,−lo]；8条误差/目标行不变。派生表保留source_*，可重复规范化而不二次反号。主要/辅助四阶段闭合残差分别2.220446049250313e−16 / 4.440892098500626e−16 MPa，低于1e−8。源文件保持不变，最终verify核对69条绑定哈希。

## 图像与补充材料

Fig6的c8-16、q24-48真实表面/C-scan已取到，路径由candidate_queue与feature_bank_manifest解析至/home/ww/paper3/cmc_damage_inference。assets.json包含精确原路径、哈希、尺寸、CC BY 4.0与Hasebe等署名。官方版本页：https://data.mendeley.com/datasets/74t7kcdgkr/1 和 https://data.mendeley.com/datasets/cgtnjyggtm/1 。C-scan沿用登记裁切和rint(linspace)原生8×8边界，表面沿用冻结90°方向。完整C-scan只作事后参照；新增内部面板没有缺失。

S1交付50件评价清单、捕获组和276件划分metadata；未对65件预留集评价。S2披露模型、恒零通道、掩膜/OOF与目标推导，保留既有选模历史。S3交付全部cap、原生面积、已保存配对区间、全部目标/NR/零成本/回升。S4交付500条归档轨迹及350条件完整数值台账、40项阶段对比和16项规范化顺序对比。S5保留六个固定病例的动作、CAI、初始softmax、已保存L1归因与第一层平均头attention；不新求梯度、不新前向。未恢复缺件：无本轮必交素材缺件；不打包原始数据全集、权重或feature shards。

## 构建和验收

实际Python 3.13.13、pypandoc 1.17、Pandoc 3.9、XeTeX 0.999998（TeX Live 2026）；字体Liberation Serif/Sans及WenQuanYi Zen Hei。最终11项构建/版本命令exit_code均为0，完整参数见BUILD_REPORT及build/*.log。最终页数：英文主文21、双语主文30、英文SI13、双语SI15。四份PDF共79页逐页查看；无页外文本、缺字或最终构建警告。四份DOCX可打开且ZIP CRC通过，公式为OMML而非截图。

12组图稿最低导出字体9 pt，正文缩放后至少8.125 pt；40个面板人工核查。自动碰撞/文字/对齐无FAIL；F6七项跨相邻序号文本框WARN已人工核对。源审计19 PASS、2 WARN、0 FAIL：TIFF未要求交付，PNG预览300 dpi；已交SVG/PDF/PNG。详见REVIEW、FIGURE_PANEL_REVIEW及figure_qa原始报告。

Q01/Q02/Q04/Q06/Q07 PASS，Q05 PASS_WITH_REVIEWED_WARNINGS；Q03 PASS（八项测试8/8通过，exit_code=0）；Q08本地范围通过，远端待真实receipt。最终八项轻量测试执行结果由TEST_RESULTS.log和LOCAL_VALIDATION记录，不重复使用历史研究acceptance冒充本轮测试。

## 文件与继续入口

论文根paper_cai_aei/r3_positive_review_r1：source四份规范Markdown、references.bib、alignment/figure_captions，build八份DOCX/PDF，figures六主图/六病例板，figure_data和tables全精度数据、reference_map及change_log。审计根artifacts/cai_agent_v3/manuscript/r3_positive_review_r1保存来源、模型、claim/figure/requirement映射、审核、清单和最终receipt。docs/cai/positive_review_r1在本地保存原输入及提取副本；远端仅保存合同和路径/哈希元数据，scripts/cai_manuscript_revision保存适配、绘图、构建、验收代码。

从仓库根执行（不要在旧稿目录运行旧build脚本）：

```bash
python scripts/cai_manuscript_revision/run.py build --paper-root paper_cai_aei/r3_positive_review_r1 --artifact-root artifacts/cai_agent_v3/manuscript/r3_positive_review_r1 --languages en,bilingual
python scripts/cai_manuscript_revision/run.py verify --paper-root paper_cai_aei/r3_positive_review_r1 --artifact-root artifacts/cai_agent_v3/manuscript/r3_positive_review_r1
```

prepare-evidence、render-figures与build入口同上，见论文README和代码README。bind仅用于固定基点的首次来源绑定，需原输入包和精确本地图像路径；发布分支继续编辑无需重绑或新研究计算。图稿自动审计依赖本机nature-figure脚本；缺失时必须声明自动审计不可用并重新人工审核。重新构建PDF会使人工视觉哈希失效，必须重查修改影响的页后才能verify。

内容提交C的消息为paper(cai): revise frozen-evidence manuscript and figures；先push任务分支并核验远端C、upstream及handoff/PDF/代码Git blob，再生成PUBLICATION_RECEIPT，单独提交R（docs(cai): record verified manuscript publication）并正常push。C和R实际SHA在receipt/最终回复，不通过amend构造自引用；manifest不包含自身或receipt，也排除Git/cache/临时渲染页。

## 保留的边界与作者事项

效果基于同一开发分区和单一主初始化；条件区间没有选择/多重性校正。主要置换不确定、两个early顺序区间跨零、部分组别不利、六例中存在大残差。独立评价、多seed、动态/反馈组件干预、物理扫描延迟与停止准则均未开展，不能以此稿宣称部署、普适顺序优势或完整输入等效。参考文献继承原稿，除两图像数据版本/许可页外未新读全文；reference_map明确读取范围。作者身份、单位、基金、通讯信息、CRediT、利益冲突及正式投稿批准待作者确认，未代签。GitHub任务分支发布不等于正式投稿或合并main。

## 原稿仅本地保存

按用户2026-10-22的指令，原始PDF/DOCX、提取全文、原图对象和原表格仅保留本地。docs根的.gitignore阻止常规暂存，交付清单排除其内容。为撤回此前上传，只重写本代理在任务分支创建的两个提交，基点及其他分支不变。后续receipt记录清理后的内容提交与远端核验；GitHub旧提交缓存是否清除未获确认。
