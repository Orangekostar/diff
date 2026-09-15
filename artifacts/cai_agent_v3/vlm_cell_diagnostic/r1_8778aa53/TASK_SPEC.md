# Codex执行提示词：VLM选格前注意力与cell坐标链诊断

## 0. 本轮目标和授权

用户要亲自查看：VLM在给出格号前对图像的视觉注意是否位于表面线索附近，以及后续格号解析、旋转、缩放或overlay是否把位置画错。**不要把现有橙框平滑成热图交差，也不要先假定坐标有错。**

仓库 `Orangekostar/diff`；分支 `research/cai-vlm-agent-v3-controlled-reuse`。
已核对基点 `8778aa53875c75f3ca681915e322a9c02cbc1ac3`。
任务ID `VLM_CELL_DIAG_R1_8778aa53`。

本轮是一个小范围可视化诊断，不修改论文和历史科学结果，不重训、不改主方法、不重新选动作。

- A阶段：CPU读取冻结文件、恢复输入图、导出原始回答/置信分布/坐标对应，**研究模型前向0**。
- B阶段：为回答真正“选cell前”的问题，**只对确认身份后的当前一个案例，允许最多2次Qwen诊断性前向**，包含失败尝试；不调用generate产生新完整回答，不创建新的候选区域实验。只读现有权重，参数不更新。优先一次前向提取所需query行。
- B阶段采用单GPU、batch=1，新增GPU诊断时间上限15分钟，CPU最多4线程。现有环境足够才执行，不全局升级Transformers/PyTorch、不重新下载大型模型、不挤占用户运行中的训练。
- Actor/CAI预测器/ResNet/Reader/STOP的新前向均为0；训练、TEST接入、新bootstrap、性能重评均为0。
- B是**本次新授权的单例诊断计算**，必须记录实际Qwen前向数和attention实现，不得仍汇报“全部模型前向0”。它不替换历史VLM缓存，不计作旧实验的原始attention。
- 若只完成A，准确标记`COORDINATE_EXPORT_COMPLETE / ATTENTION_NOT_EXPORTED`，不能声称选格前热图已经生成。B的依赖/显存阻塞不妨碍交付A。

## 1. 先辨别三种不同东西

实际已核对调用链：

```text
原始表面图
  → render_surface_inputs：RGB、顺时针90°、必要时最大边1024缩放
  → clean图 + 带0..63行优先编号的gridded图
  → 冻结Qwen + SURFACE_PERCEPT_PROMPT
  → raw_text JSON中的regions[].cells及confidence
  → parse_surface_percept
  → _features：64维indicator和序数confidence
  → 首步合法集/C0约束
  → Actor选择实际first action
  → 绘图按格号覆盖原图
```

现有backend和cache返回/保存文本、调用计数等，**没有保存像素级热图、attention或64格连续效用logits**。不要将这种接口误写成“原生heatmap→argmax cell”。

输出须分别命名：

1. `Cached VLM cell confidence — ordinal, after VLM decoding / before Actor selection`：原回答导出的8×8序数图。
2. `Pre-first-cell visual attention — diagnostic replay`：本次B阶段真实前向所取的注意力。
3. `Historical Actor first action`：原轨迹已执行的动作。

这三者不能互换。注意力权重不是损伤概率、不是CAI信息价值，也不是原系统用于argmax的选格分数。

## 2. 绑定案例与只读来源

首先匹配用户本轮上传的原图/overlay与既有case manifest，确认`specimen_key`。**不能只凭标题cell 36认定是c8-16**；不止一个案例首步是36。

当前已核对的优先匹配候选：`cgtnjyggtm:q24-48`。
该行的region_indicator对应格36、37、59、60，confidence均为medium（float32约0.6666666865），与用户图的候选布局一致；最终仍须确认源图/既有overlay身份。

```text
cache_key: d11f57f3d3ba2e96339d1857e31f3f9192074a624682d7bf4faa29cd04345773
clean_image_sha256: c4d3d9badaaa6b9ff0b351685a1833a8dab13443b79642d4f6353ef3f6fdf0c9
gridded_image_sha256: 27d40ac5937960d91439d15a6f928dde9e834da4fcd826f540678530e314da8b
call_count=1; repaired=False（现有CSV记录）
```

上述不是已读取raw_text的替代；要从cache按key提取真实原回答。若身份匹配失败，报告准确的匹配缺项，不拿另一个好案例替换。CPU可查看原先3个固定案例以定位，但B只运行确认后的1件。

相对于实际ROOT：

```text
DATA = results/cai_agent_v3/new_protocol/
W3   = results/cai_agent_v3/w3_pilot/r1_0e11452a/
EVID = results/cai_agent_v3/paper_evidence/r1_e2a11154/
OUT  = results/cai_agent_v3/vlm_cell_diagnostic/r1_8778aa53/
ART  = artifacts/cai_agent_v3/vlm_cell_diagnostic/r1_8778aa53/
```

只读：
- DATA/candidate_queue.csv：仅提取目标试样路径、split、尺寸及哈希，不接入其他TEST内容。
- DATA/feature_bank_manifest.json：解析真实`encoder_execution_root`，不猜外部数据路径。
- DATA/vlm_actor_features_fit.csv、vlm_surface_percepts.jsonl、vlm_manifest_fit.json。
- W3/case_manifest.csv、figure_manifest.json、policy_validation_episodes.csv.gz中目标案例行。
- EVID/case_figure_reuse.csv；已有论文图只作显示对照。

优先读取原稿件基点后的实际HEAD，记录差异，保留用户未提交工作；不reset、不改分支历史。

## 3. 按任务定向读代码

| 任务 | 必须核对的实际文件/函数 |
|---|---|
| 原始提示词、JSON与缓存 | src/cmc_bbdm/learned_cscan/perception.py：SURFACE_PERCEPT_PROMPT、parse_surface_percept、SurfacePerceptCache |
| VLM特征转换与输入身份 | src/cmc_bbdm/cai_agent_v3/vlm_perception.py：_features、run_vlm_perception（只读，不运行入口） |
| 旋转、缩放和文字网格 | src/cmc_bbdm/vlm_cscan/runtime.py：render_surface_inputs、_image_sha256 |
| 当前推理与processor参数 | src/cmc_bbdm/vlm_cscan/vlm.py：QwenVLBackend.load/infer |
| 原生格子边界 | src/cmc_bbdm/cai_active_image/environment.py：NativeCellGrid.from_shape |
| 图格裁剪 | src/cmc_bbdm/cai_agent_v3/feature_bank.py：_cell_crops及surface_render生成 |
| 实际显示与动作 | cai_agent_v3/w3_results.py：export_figures内部绘图逻辑；diagnostics.py::_draw_cells；actor_training.py::_evaluate_one；policy.py::vlm_first_action_mask |

现有`perception.py`还定义display_cell_id/physical_cell_id/map_percept_to_physical。**函数存在不等于当前流程调用。**核对实际引用：当前v3的render明确画row*8+column，不得为了“修坐标”擅自套用置换映射。

## 4. A阶段：导出历史回答及完整坐标证据（先交付可看图）

### A1. 还原VLM实际输入

从绑定原图调用相同纯render函数，导出：
- `01_source_surface.png`：原始方向的表面图，注明`source orientation`。
- `02_vlm_clean_input.png`：旋转/缩放后、实际送给processor之前的clean图。
- `03_vlm_numbered_input.png`：实际gridded图，保留原始小字号数字和网格；不能换成更好看的版本冒充历史输入。
- `04_readable_grid_reference.png`：另外生成清晰的0..63编号与行列坐标图，明确标`diagnostic reference, not historical VLM input`。

使用原 `_image_sha256` 的PNG编码方式核对02/03与request中的hash。若PNG编码版本造成文件hash差异，另报告解码RGB差异，**无历史像素可比时不能仅凭视觉相似认证一致**。找出输入渲染或历史版本不一致再决定B是否能忠实重建。

原raw_text原样导出，并另存解析后JSON与prompt；记录region的cue/alternative/confidence、call_count/repaired。修复过的缓存默认仅保存最后一次回答，初始invalid回答未存就标`NOT_RECORDED`，不编造第一次输出。

### A2. 画两种原回答可视化，而非假像素热图

- `05_cached_confidence_8x8.png`：8×8未平滑序数分布，每格标ID及等级。0/1/2/3分别对应unknown/low/medium/high仅作标注；原数值仍保留0、1/3、2/3、1。未被选中不是“确定正常”。
- `06_raw_ids_on_numbered_input.png`：将raw JSON的格号直接画回03的坐标，不经过Actor或论文绘图函数。
- `07_parsed_and_feature_cells.png`：将解析后的cells与CSV中indicator/confidence画回02，并报告二者是否逐格一致。
- `08_c0_and_historical_action.png`：分开显示所有候选、最高可靠C0、原Actor首动作。实际action由历史trace读取，不新前向。置信相同的格子同色，不能凭区域大小或中心距离创造差别。

以上每张独立PNG。保存原64维数组和CSV，杜绝对离散候选做高斯平滑/插值并声称得到VLM原始注意力。

### A3. 格号到像素的坐标表

输出 `coordinate_trace.csv`，每格一行，至少包含：cell_id、row、col、原输入尺寸、render尺寸、每格像素边界、归一化边界、显示矩形边界、raw/parsed/feature/C0/action各字段。

必须核对：
- 当前编号为`row=cell_id//8; col=cell_id%8`，原点左上；cell36是零基(4,4)，即第五行第五列，不是一个GT标签。
- 原始表面→render的顺时针90°只发生一次；不要对已render图再旋转。区分表面坐标、VLM输入坐标、C-scan注册坐标；不要用C-scan尺寸直接在表面图画矩形。
- PIL为(width,height)，NumPy为(height,width)；索引为[y,x]；裁剪为半开区间。
- NativeCellGrid使用`np.rint(np.linspace(0,size,9))`；与render文字/线条所用round比较实际边界。奇数尺寸的最多像素取整差异与整格错移分开报告。
- Matplotlib使用`origin='upper'`时核对像素中心和半像素边界。统一extent或精确使用边界−0.5，禁止同时翻转轴又反转数据。
- 构造一张64格唯一ID的合成图，核对0/7/56/63和27/28/35/36的中心、裁剪及overlay往返；不做全图旋转搜索来挑最像损伤的版本。

对原图中的表面痕迹位置，仅让用户目视判断。不要自动把最亮点、圆圈中心或no-VLM首动作当作正确答案。

## 5. B阶段：真正输出首个格号前的视觉注意（新诊断，不是旧缓存）

### B0. 定义清楚提取对象

当前模型是decoder-only多模态语言模型的生成接口，不假设存在一个独立的二维分割头。

要取的是：**给定原始双图、原提示词和已记录回答中首个cells数字之前的文本前缀，预测这个数字的query位置对视觉tokens的注意力。**

模型/processor使用实际冻结Qwen2.5-VL-7B-Instruct revision：
`cc594898137f460bfe9f0759e9844b3ce807cfb5`。
从现有配置读取路径；保持clean、gridded顺序、chat template、min_visual_tokens=256/max_visual_tokens=1280对应的processor范围及其余实际设置。确认代码后使用，不机械套用最新版文档默认值。

### B1. 严格截断回答，避免答案泄漏

1. 在原raw_text定位首个`"cells": [`中的首个数字字符位置，保留之前的真实回答前缀，不包含该数字或之后的内容。
2. 用原chat template和tokenizer构造输入，记录token序列、前缀、query索引、原首个cell值和对应的下一个token预测。检查token边界，不能将目标数字误含进前缀。
3. 注意力应取**用于预测下一个cell数字token的最后一个已输入token位置**，不是已经读入cell36后的那个位置。若格号分为多个token，本次主图只解释首个token的预测时点，不声称覆盖整个数字解码。
4. 禁止将完整历史回答先送入模型再把末尾attention命名为“选cell前”。
5. 若原记录是格式修复结果且缺失原repair上下文，不能声称重建原调用；报告阻塞/不同条件。本案例CSV当前标repaired=False，仍以实际cache为准。

### B2. 提取规则和资源

- 先阅读本机已安装版本的Qwen建模代码，确认attention返回机制、图像token位置和空间排序，不全局升级依赖。
- 预先固定取语言解码器最后4层，各层全部attention heads平均，再跨这4层平均。另保存这4层各自的head-mean向量供检查，不根据图案挑层/挑head。
- 优先只捕获目标query行；不要保存全部层完整N×N矩阵。若需诊断进程中切换eager获得attention，记录与原sdpa实现的区别，不改生产代码/历史配置。
- 先估算内存；原输入不变且能运行才前向。不为拿图偷偷缩小输入、换模型、量化或换prompt。OOM/不支持时记录准确错误，最多2次总前向后停止；不得开展长时间兼容性工程。
- 只读权重、eval、无梯度，不generate完整新回答。可以直接forward，读取真实attention和下一token logits；保存top-k下一token供判断前缀是否忠实，但不据此改变历史候选。
- 如诊断下一token不支持历史原数字，标`REPLAY_TOKEN_MISMATCH`，保留图并明确为同权重条件诊断，不称精确恢复历史推理。

### B3. 视觉token到图像的映射

原调用有**clean和gridded两张图**。分别提取各自视觉token段，不能把两段拼成一个平方网格。

从实际processor的`image_grid_thw`、模型`spatial_merge_size`及当前版本的输出顺序建立每段二维网格，核对每图视觉token总数。不是8×8，也不能用sqrt(token_count)猜尺寸。保留processor resize、窗口顺序恢复、merge后的索引关系；这些必须据本机实现核对。

分别导出：
- `09_pre_cell_attention_clean.png`：clean图上的真实注意力。
- `10_pre_cell_attention_numbered.png`：gridded图上的真实注意力，可以看是否主要落在文字编号/边界。
- 每张同时保存纯热图数组`.npy`、token空间映射CSV及一个原始分辨率未平滑版本；上采样仅用于显示，不能声称产生了像素级定位精度。
- 保存分配到clean视觉tokens、gridded视觉tokens及非视觉tokens的原始attention mass。展示条件归一化时在色条写清；两图强弱比较用相同标度，禁止各自自动拉满后比较注意力总量。
- `11_attention_with_raw_cells.png`：在注意力显示上叠加历史raw候选轮廓和历史首动作，用于人工对照，不能由热图反算候选再冒充历史输出。

图题含`diagnostic replay / pre-first-cell token attention`。热图峰值所在格可以报告为观察描述，但不命名`best cell`或`GT`。

若本机提取到的是token相关性、梯度归因或别的量，必须另外命名并说明定义；本任务不自动扩展到Grad-CAM/遮挡扫描/新VLM框选实验。

## 6. 便于用户查看，不要只交日志

输出一个轻量本地`index.html`，按原图→真实编号输入→原始回答→置信图→C0与真实动作→两张注意力图→坐标表的顺序展示，图片可点开原图。无需后端、登录或新标注平台；B未完成的位置显示原因，不显示假热图。

所有PNG同时独立保存。不同视图保持相同方向和长宽比；有网格和无网格各保留，透明叠加不能遮住浅表线索。当前是诊断图，不追求论文卡通美化。

可选一个很薄的鼠标坐标读数：在诊断clean图上显示原生(x,y)、归一化(u,v)及按真实边界求出的cell_id。标记只停留浏览器，不自动变成专家GT或改写cache；不为该功能拖延主要导出。

## 7. 判断框架：先定位断在哪一环，不提前宣布因果

| 实际看到什么 | 可作的判断 |
|---|---|
| raw IDs→parsed IDs或CSV不一致 | 解析/缓存身份/特征转换问题，定位具体行 |
| raw ID在原编号输入对应区域，与论文overlay不同 | 显示或坐标变换问题；指出是哪一级、差几个像素/格 |
| 原编号图与overlay一致，但标出的格子不贴合用户看到的线索 | 没有找到后处理坐标错误；进一步考虑VLM视觉定位或格号输出问题 |
| 新attention集中在线索附近，但历史格号指向别处 | 生成格号/视觉—数字对应值得关注；**不是证明后处理错，也不是CAI最优格证明** |
| 输入渲染/processor条件不同或提取不可重建 | 标为不能确认，不以漂亮热图替代证据 |

首动作属于C0只是受约束选择的一致性检查，**不是VLM定位命中率**。no-VLM选点不是空间真值；即使发现一张图的坐标错误，也不能自动解释群体MAE差异，更不能修改已冻结结果宣称改善。

若确实发现bug，本轮只给出精确位置、最小复现和拟修方案；展示用坐标修正图另命名，不覆盖历史图、VLM候选、论文数字或模型。待用户看图后决定后续。

## 8. 有限验收及实际交付

只做4项相关核对：
1. 目标试样、输入哈希、cache_key与raw/parsed/CSV来源连通；
2. 64格编号与像素边界往返正确，历史overlay差异有明确结果；
3. 热图数据类型标注准确；B的query在首数字之前、两图token分开、真实数组可读；
4. PNG/HTML能打开，原研究文件未变，新增计算和git状态如实记录。

不重跑W2/W3、整库pytest、全模型checksum、bootstrap、压力测试或多轮模拟审稿。

新增脚本放`scripts/`或独立诊断子目录；不要将新attention后端写回正式QwenVLBackend。输出`ART/VLM_CELL_DIAGNOSTIC_HANDOFF.md`：源码依据、实际案例、原回答、坐标结论、热图类型/提取定义、实际前向计数、阻塞项、用户先看哪些图。仅追加一条实际诊断账目，不重置历史训练额度。

将相关脚本、单张图、HTML、小数组/CSV/JSON与交接实际commit并push同一研究分支，核对local/upstream/remote。不要PR、merge、force push，不提交模型、环境、大矩阵、原始全数据或字体。原论文、src生产路径和冻结结果不改。

最终回复给用户一个可打开的HTML位置和关键PNG位置，简述：
- 有没有历史“选格前热图”（不能把无缓存写成模型完全没有注意力）；
- B是否真的提取了新attention，是否与历史条件一致；
- 原回答→格号→实际显示有没有查到不一致；
- 需要用户目视判断的图及实际Git SHA。

## 9. 已核对依据（执行时以绑定版本及本机实际依赖为准）

基点代码链接前缀：
https://github.com/Orangekostar/diff/blob/8778aa53875c75f3ca681915e322a9c02cbc1ac3/

- src/cmc_bbdm/learned_cscan/perception.py：直接cells JSON；raw_text/cache字段；编号映射辅助函数。
- src/cmc_bbdm/cai_agent_v3/vlm_perception.py：_features、render输入、cache_key、模型revision和实际CSV路径。
- src/cmc_bbdm/vlm_cscan/runtime.py：ROTATE_270、max_edge缩放、row*8+column、PNG hash。
- src/cmc_bbdm/vlm_cscan/vlm.py：Qwen2.5-VL sdpa generate接口只返回文本/计数，min/max视觉token范围。
- src/cmc_bbdm/cai_active_image/environment.py：np.rint原生8×8格边界。
- src/cmc_bbdm/cai_agent_v3/feature_bank.py：先裁格、同表面render和内部注册crop。
- src/cmc_bbdm/cai_agent_v3/w3_results.py：原surface_cues图按region_indicator涂色，不是attention图；禁止直接调用export_figures覆盖旧目录。
- results/cai_agent_v3/new_protocol/vlm_actor_features_fit.csv：q24-48行的cache身份与36/37/59/60候选。

外部仅作API/坐标参考，不替代本机依赖核对：
https://huggingface.co/docs/transformers/main/en/model_doc/qwen2_5_vl
https://matplotlib.org/stable/users/explain/artists/imshow_extent.html

A阶段是现有记录恢复；B的截断query/固定层平均是本次提出的诊断方法，不是旧实验已经实现或已经保存的内容。
