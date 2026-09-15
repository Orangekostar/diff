# Codex执行提示词：VLM表面线索定位的提示词×编号呈现有限对照

## 0. 本轮具体任务与授权

- 仓库：`Orangekostar/diff`。
- 分支：`research/cai-vlm-agent-v3-controlled-reuse`。
- 已核实依据提交：`e8d9ef0171ab6d0e1d14187a56cdbd7ecc579750`。
- 任务ID：`VLM_GROUNDING_PILOT_R1_e8d9ef01`。
- 交付目标：**实现并实际执行小规模对照，输出准确的输入图、完整VLM回答、格子叠图、C0诊断和本地HTML，让作者判断提示词与编号呈现是否改善“表面线索—格号”对应。**不是只写计划，也不是更新论文方法。

本规范明确授权有限的新Qwen完整回答生成：最多6件×4配置=24个主调用，每项最多一次格式修复，总尝试不超过48；训练、Actor/CAI/ResNet/Reader/STOP、新注意力诊断、新TEST均为0。它覆盖此前“单例最多2次前向”的旧任务范围，但只覆盖本轮列出的VLM定位对照，不能据此重跑整套CAI实验。

完成不要求得到正结果。负结果、无可靠线索、解析失败和与历史回答不一致都是真实结果，需保留并交付。任何新配置不得直接替换生产提示词、旧VLM缓存或已发表述的CAI数字。

### 0.1 要回答的四个问题

1. 在同一模型和同一图像下，补充坐标/图像身份/定位置信规则，是否改变线索与格号的对应？
2. 只改善编号可读性和所属格内的位置，结果是否变化？
3. 两项改变组合的结果如何，是否只在已反复查看的q24-48上有变化？
4. 这些输出如果经过既有C0规则，会怎样改变首步允许集合？只分析集合，不执行Actor或CAI预测。

**定位的目标是可见表面线索，不是内部损伤GT，也不是最优CAI采样位置。**没有人类位置参考时，自动输出仅能证明编号/解析一致、候选改变等，不能自行宣称定位精度提高。

### 0.2 不得扩展到以下任务

不微调Qwen；不切换更大模型；不搜索temperature、beam、分辨率、语言或提示词多版本；不进行点/框输出路线、编号置换/图像遮挡、多图换序或注意力因果研究；不接入真实超声幅值；不新建专家标注平台；不重开旧BC、LOCATE、CHARACTERIZE、STOP；不训练新Actor，不让作者说过的23/31或27/28成为模型提示或自动真值。

## 1. 已核实事实、依据与本轮设计的区别

| 已核实事实 | 实际来源 | 对开发的直接影响 |
|---|---|---|
| VLM直接生成`regions[].cells`，不是先输出连续热图再换算格号 | `learned_cscan/perception.py`的schema/prompt/parser | 比较完整JSON和物理位置；不把注意力峰值当正确答案 |
| 原表面先顺时针90°一次，再缩放，再复制出编号图 | `vlm_cscan/runtime.py::render_surface_inputs` | 保留已有方向；R1必须从已旋转clean复制，不能再次调用旋转 |
| 生产接口是clean、gridded同向双图；只传最终prompt | `cai_agent_v3/vlm_perception.py`和`vlm_cscan/vlm.py::infer` | 所有对照保持图像顺序和chat template，不夹入案例答案 |
| 原编号在每格左上角+2/+1，未显式设置字体大小 | `runtime.py` | R1只改变标签呈现，不改几何、原图或grid边界 |
| `SurfacePerceptCache.resolve()`内部写死`infer(SURFACE_PERCEPT_PROMPT)` | `learned_cscan/perception.py` | **不能只改hash仍调用resolve；必须显式将本配置prompt传入新诊断调用** |
| `run_vlm_perception()`写旧`new_protocol`，还依赖旧gate | `cai_agent_v3/vlm_perception.py` | **不可直接运行该入口**；新对照独立目录、独立缓存，无旧gate前置阻塞 |
| C0取最高medium/high等级；之后解除硬限制 | `cai_agent_v3/policy.py::vlm_first_action_mask` | 沿用函数做CPU集合诊断；不改阈值、策略规则或Actor输入 |
| q24-48已确认输入和overlay链一致，旧回答为36/37与59/60 | `vlm_cell_diagnostic/r1_8778aa53`交接和identity/cache | 不重复整套旋转排查；仍不能断言格号定位正确 |
| 旧诊断只有一个首数字之前的attention前向，没有完整新回答 | 同交接、`attention_preflight.json` | 本轮生成完整回答；不拿首token“3”匹配当完整复现 |

以下的6件选择、2×2配置、R1绘字尺寸、P1文字、生成上限和人工核对方式是**本轮设计**，不是已完成实验或公认最优设置。仓库现状与固定来源见`VLM_GROUNDING_SOURCE_BINDINGS.md`。

## 2. 路径、版本和最短阅读顺序

相对当前实际工作树ROOT：

```text
DATA    = results/cai_agent_v3/new_protocol/
DIAG    = results/cai_agent_v3/vlm_cell_diagnostic/r1_8778aa53/
DIAGART = artifacts/cai_agent_v3/vlm_cell_diagnostic/r1_8778aa53/
W3      = results/cai_agent_v3/w3_pilot/r1_0e11452a/
OUT     = results/cai_agent_v3/vlm_grounding_pilot/r1_e8d9ef01/
ART     = artifacts/cai_agent_v3/vlm_grounding_pilot/r1_e8d9ef01/
CODE    = scripts/vlm_grounding_pilot/
```

OUT/ART/CODE为本次建议新增路径，不声称已经存在。正文工程`paper_cai_aei/`、旧DIAG、旧W2/W3与生产`src/`保持只读。允许更新本次明确绑定的`docs/stagetask/`任务/状态文件，不修改其他任务记录。

先查看分支、HEAD、upstream和`git status --short`。若HEAD在上述基点之后且包含相关工作，保留既有成果，定向查看差异；不reset、不checkout覆盖用户更改、不自动追随main。规范路径、范围和阶段写到当前任务卡。本文件就是有效任务授权，不得反复返回“缺具体任务”。

最短阅读：
1. `DIAGART/VLM_CELL_DIAGNOSTIC_HANDOFF.md`；`DIAG/identity.json`、`cache_record.json`、`prompt.txt`。
2. 上表五个生产源码文件，只读相关函数；确认`SurfacePerceptCache.resolve()`与backend实际参数。
3. `scripts/vlm_cell_diagnostic/export_cpu.py`的来源连接/导图逻辑；不直接执行其main，因为它写旧DIAG。
4. `DATA/candidate_queue.csv`必要元数据、`vlm_actor_features_fit.csv`、`vlm_surface_percepts.jsonl`及`feature_bank_manifest.json`；`W3/case_manifest.csv`。
5. 本机成功执行上轮诊断的环境记录。外部数据根优先读`feature_bank_manifest.json::encoder_execution_root`，并确认源文件存在；不机械假设另一工作树就是数据根。

已经附带的2026-09-09 BC交接不是当前方法或训练授权，不能从中恢复192步扫描/STOP。Hasebe作者损伤面积及旧RGB形态学proxy也不是本轮表面定位参考。

## 3. 锁定6个案例，禁止按新结果更换

### 3.1 三个固定诊断案例

```text
cgtnjyggtm:q24-48   # 当前问题案例，先执行，但不可据其结果改P1/R1
74t7kcdgkr:c8-16
w68dtmpfyf:q16-29
```

三者在当前W3 case manifest中均为VALID，已经用于诊断，不称新测试。

### 3.2 另外三个域，各取一个TRAIN试样

从`xcmzfsbd9t`、`yfxyg8jm46`、`ykhs7s2dck`三个域的TRAIN元数据中分别取一件。固定排序键：

```python
(hashlib.sha256(("VLM_GROUNDING_PILOT_R1|" + specimen_key).encode("utf-8")).hexdigest(), specimen_key)
```

每域取最小项，不读取CAI数值、C-scan内容、旧误差、先前VLM质量、热图强弱或候选位置来选择。**不能仅挑旧VLM成功、有明显线索、或新结果好看的试样。**这三件的具体key须由当前CSV按规则解析并记录，不能由聊天内容编造。

只保留读取所需列：`specimen_key,dataset_id,split,capture_group_id,impacted_surface_path,surface_sha256,identity_status,p0r_roster_status`以及用于C0首步可负担性计算的已登记C-scan宽高。CSV同文件里的标签列不进入模型、报告、选择或评估，不打开任何TEST图或CAI工作簿。登记宽高不是读取内部观测。

在任何新生成前保存`cases.csv`与`experiment_lock.json`，包含6件身份、来源、旧缓存可用性、既有/规则选择标志和四配置hash。保存后不替换无cue、错误回答或资源失败案例。文件缺失只标明该案例输入缺项，其他案例继续；不选替补凑好结果。没有历史可用回答不阻止本轮新调用。

本轮是6件开发诊断，不是六域泛化试验，不计算宣称代表总体的显著性。

## 4. 实验矩阵：提示词和编号呈现分开改变

| ID | 提示词 | 编号图 | 作用 |
|---|---|---|---|
| A_P0_R0 | P0原文 | R0原渲染 | 当前环境的完整基线回答 |
| B_P1_R0 | P1定位增强 | R0原渲染 | 相同图像下看提示词包的作用 |
| C_P0_R1 | P0原文 | R1清晰标签 | 相同文字下看编号呈现的作用 |
| D_P1_R1 | P1定位增强 | R1清晰标签 | 看两项组合 |

另列`H00_CACHED`：只读取历史原回答，不生成、不替代A。A在当前同一环境重新调用一次，是为了避免将“历史环境不同”混入新旧提示词比较。历史与A不一致须记录文本、格号、cue、confidence差异；不重复运行A直到一致，也不删除历史。

所有配置：同一模型/revision、同一clean像素、同一方向、同一图像尺寸、clean→numbered顺序、同一processor、同一生成参数、相同schema与一次格式修复规则。**chat template保持原样，不额外切换`add_vision_id`、system消息或多轮上下文**。P1通过正文明确第一/第二图的职责。

P1是“角色/坐标/输出位置/置信定义”组合改动，R1是“字体/内缩/可读性”组合改动；本2×2不能再分解到某一句话或某个字号的因果贡献。

### 4.1 R0：逐字节还原原输入

对每件原surface运行一次`render_surface_inputs(image,max_edge=1024)`，获得clean与R0编号图。用旧PNG编码函数核对能核对的历史输入哈希。q24-48直接对照已成功的DIAG identity。source只是来源对照，**绝不把未旋转source作为第一图**。

原P0读取`SURFACE_PERCEPT_PROMPT`实际字符串，不手抄或简化；所有JSON schema顺序/空格按当前原实现。实际prompt另保存文件。

### 4.2 R1：仅改善编号标签

- 输入为本步骤已经生成的clean，复制后绘制；禁止把clean再次送入`render_surface_inputs()`造成二次旋转。
- 网格边界、线条位置、颜色/透明度、图像尺寸及0—63 row-major含义与R0相同，不旋转/翻转/裁切/拉伸/锐化原内容。
- 明确使用本机TrueType字体（优先已安装DejaVu Sans），记录字体名称、大小、路径和hash；**不上传或打包字体文件**。
- 建议固定字号：`max(12, round(24*min_cell_side/128))`；标签内缩：`max(3, round(10*min_cell_side/128))`。1024方图约为24px和10px。
- 数字白色，紧凑半透明深色底，矩形完全位于所属格内；以真实`textbbox`修正字体基线偏移，不把anchor当实际字框。
- 所有64个标签用同一几何规则。若尺寸较小导致越界，仅按几何检查统一缩小字体直到标签落在格内；在模型输出前锁定该决定。不得为避开某个已知线索单独移动那一格的标签。
- 保存64个实际字框与所属cell表，导出标签改变区域的mask；R1与R0在这些声明的标记区域之外应保持相同。标记会遮挡部分表面，clean始终保留；不宣称R1必然更好。
- 模型读取的是实际R0/R1编号图。报告里额外做的大字号参考图必须标为reference only，不能混作模型输入。

### 4.3 P1：只换下列固定文字，沿用同一schema

P1正文取包内`prompts/P1_SPATIAL_GROUNDING_ZH.txt`，若只收到本主文件，按下段原样生成。随后用与P0相同的`SURFACE_PERCEPT_SCHEMA`和`json.dumps(...,sort_keys=True,separators=(",",":"))`追加`JSON schema:\n`。不新增bbox/row/column/localization_confidence等键，不改字数或cell数限制。

```text
你是材料试样的表面观察与位置标注模块。输入依次为两张图。
图1是干净表面图，用于辨认可见的材料表面线索。
图2是同一张图、同一方向、同一范围的8×8编号图，用于将图1中的线索对应到格号。图2除了程序添加的网格和数字外，与图1对齐。

位置必须按当前输入图确定，不自行旋转、翻转或恢复原文件方向。左上角为第0行第0列，行向下增加、列向右增加；行列均为0到7，cell_id = 8 × row + column，格号为0到63。
编号文字、文字底色和描边、网格线均为程序叠加的定位标记，不是材料表面线索。先以图1辨认可见内容，再利用图2确定该内容实际覆盖的格子；不要因某个数字靠近线索就把线索归入该数字的格子。

每个区域的cue必须描述该区域所报格子中真实可见的同一线索；alternative给出一个合理的非损伤替代解释。仅依据表面外观，不推断内部超声、损伤深度、材料铺层、冲击能量、CAI或检测完成情况。不要推荐扫描动作、工具、策略或后续操作。

confidence使用unknown、low、medium、high，表示“线索可见性以及线索与所报格号对应”的综合把握，不是损伤概率或CAI信息价值。线索清楚但格号对应有疑问时，不能仅凭线索醒目给medium或high；位置有疑问应使用low或unknown，不要编造位置。medium或high要求所报格子内确有可辨认线索，而且格号对应清楚。区分表面痕迹与反光、灰尘、标记等不确定性，写入alternative。

允许没有可靠线索，不要求凑满区域或格子。最多两个区域，每个区域1到4个格子，所有区域合计最多8个不重复格子。线索横跨更多格子时，仅报告其中可清楚对应的局部，不猜测不可见的延伸范围。没有可可靠描述并定位的区域时，返回regions为空数组、no_reliable_cue为true；有区域时no_reliable_cue为false。

只返回符合所附JSON schema的一个对象，不添加解释段落、坐标字段或额外键。
```

此处medium/high明确包含位置对应把握，会影响候选数量和C0。这是本轮提示词定义改变的一部分，不是已经校准的概率。报告不能把“更少候选/更多no-cue”直接叫作精度提高。

**禁止进入任何模型prompt：**案例ID、文件名中的试样信息、用户说的23/31或27/28目标、旧回答36/37/59/60、旧注意力图、C-scan、CAI标签、哪组应当获胜。0—63编号范围和公式属于通用坐标规范，不属于答案。唯一允许把先前回答用于另一调用的场景，是同一case/variant的一次格式修复。

## 5. 开发范围：复用纯函数，隔离调用和缓存

建议实现以下小模块，可以合并文件但要保留明确职责：

```text
scripts/vlm_grounding_pilot/
  prepare.py          # 选样、源连接、R0/R1、prompt与配置冻结
  run.py              # 单模型顺序执行四配置，独立缓存与有界修复
  report.py           # 从保存结果导出叠图、数表、index.html
  validate.py         # 少量合成/静态检查，不执行研究模型
```

### 5.1 真实源码绑定

| 工作 | 复用什么 | 不能怎么做 |
|---|---|---|
| 源图定位/哈希 | `export_cpu.py`已验证的manifest字段及`_image_sha256` | 不执行其旧main写回DIAG，不再全目录搜图 |
| R0渲染 | `render_surface_inputs` | 不在生产函数加入R1开关，不篡改旧render_version |
| 解析 | `parse_surface_percept`与当前schema | 不把语义不喜欢的输出改为格式失败重试 |
| 特征 | `cai_agent_v3/vlm_perception.py::_features` | 不改ordinal数值，不直接写`vlm_actor_features_fit.csv` |
| C0 | `policy.py::vlm_first_action_mask`的纯CPU调用 | 不执行Actor，不能用“历史动作属于C0”作为定位准确率 |
| Qwen加载/生成 | `QwenVLBackend.load/infer`的实际配置 | 不执行`run_vlm_perception`或`SurfacePerceptCache.resolve` |
| 导图 | 已有格子边界与Rectangle画法、HTML静态布局 | 不把候选平滑成“选格前热图”，不调用生成式绘图 |

当前环境曾有另一个editable `cmc_bbdm`安装。启动时记录这些关键模块的`__file__`并确认落在当前ROOT；使用已成功的隔离入口/PYTHONPATH方式，避免静默导入旧工作树。不要为了导一个解析器重装整个仓库。

### 5.2 新调用必须真的收到当前prompt

`SurfacePerceptCache.resolve()`即使request里换了hash，内部仍用原全局提示词。因此新runner应显式：

```python
response = backend.infer((clean, numbered_for_variant), actual_prompt_text)
```

旧cache只可`get`或只读按key读取，不可`resolve`。新cache用独立schema和签名：至少含模型revision、实际prompt/repair文本hash、clean与numbered内容hash、render配置/字体hash、图像顺序、chat-template摘要、生成/processor参数、实际库版本。case/variant也作为外层主键。相同字节不重复新调用；签名不同不复用错误结果。

在正式生成前，用fake backend的单元例验证四配置接收的prompt与图像hash，不能只验证文件名。

### 5.3 完整回答，而不是只看首token

使用成功的本机环境，不升级Transformers/torch/Pillow或换注意力实现。冻结值：

```text
Qwen/Qwen2.5-VL-7B-Instruct
revision cc594898137f460bfe9f0759e9844b3ce807cfb5
bfloat16, sdpa, eval, requires_grad=False
processor use_fast=False
min_pixels=256*28*28=200704
max_pixels=1280*28*28=1003520
clean first, numbered second
do_sample=False, temperature=None, use_cache=True
max_new_tokens=500, batch_size=1
```

文档示例不是生产设置的替代。本机版本与上轮不同则记录差异，优先用已有成功环境；不可声称历史逐token精确复现。保持同一环境完成全部新配置。

保存真实prompt、实际chat text、两张输入图hash、image_grid_thw、输入/输出token数、完整raw text及generated token IDs。允许在**独立runner的模型实例**上用薄包装截取`generate`参数与返回ID，或复用等价生成调用；不得更改生产`vlm.py`。不启用`output_attentions`、保存全词表scores或梯度；不再运行上轮`replay_attention.py`。

一次`generate`通常包含多次自回归forward，**不得将24个完整回答写成24次forward**。计数优先为generation attempts/repair calls/output tokens；能用轻量forward hook真实计数则记录，不能确定时写未单独计数而不是0。无额外前向的token-ID保存应作为本次例行输出。

### 5.4 有界格式修复与失败

- 每个case/variant恰有一次主生成机会；不是生成多个候选再选最好。
- 仅在JSON或schema/既有输出约束不满足时，可用原`FORMAT_REPAIR_PROMPT`、原context包装和本次raw text进行一次修复。同配置图像不变，四组同一修复规则。
- 原parser会检查许多约束；它未显式拦截“两个region之间重复cell”时，新runner以局部合同校验单独记录并按原提示词的不重复要求判失败。**不修改生产parser，不自动去重改变回答。**报告区分parser-valid与contract-valid。
- 保存初次和修复后回答、各自的格式状态和内容差异。修复不是语义保持保证；第一遍结果和repair-dependent结果分栏，不仅展示修复后成功者。
- 格号看着不对、置信太低、no-cue或没有覆盖作者想看的位置，不是修复理由。
- 最多两次后仍无效则`SCHEMA_INVALID_AFTER_ONE_REPAIR`；不可用与有效no-cue分开。中断/超时记失败，不再重新找一个好答案。
- 运行状态原子写入。已完成请求不重跑，`STARTED`但没有结果的请求记`INTERRUPTED`并占用尝试，不隐性释放额度或自动重发。

## 6. 运行安排与资源上限

### 6.1 顺序

1. CPU准备、案例/配置冻结和4类必要小检查完成。
2. 加载一次模型；先对q24-48依次执行A/B/C/D；只检查输入签名、调用完整性和文件落盘，不按语义“好坏”决定是否继续。
3. 若运行链无错误，保持所有参数不变，完成剩下5件各4配置。**不能看q24-48结果后改P1，再把后五件算作同一冻结配置。**
4. 每次回答即时落盘，退出后仅CPU汇总和导图。人类评价未到不阻塞结果交付与push。

### 6.2 额度

| 资源 | 本轮上限 |
|---|---:|
| 物理试样 | 6 |
| 主case×variant项 | 24 |
| 全部Qwen生成尝试，含格式修复和失败 | 48 |
| 每个case×variant尝试 | 2；第2次仅格式修复 |
| 每次最大新token | 500 |
| 理论最大输出token | 24,000 |
| GPU累计占用时段 | 1,800秒，含加载、全部生成、修复和失败 |
| 每个生成尝试墙钟 | 120秒；超时记失败，无语义重试 |
| GPU并发 / CPU线程 | 1 / 4 |
| 训练、其他研究模型、新attention、新TEST | 0 |

48是硬上限，不是必须用完。GPU计时若触顶，完成已有文件和交接，未执行项标资源截止。可用进程级timeout看门狗限制单次和总时长；超时只终止本runner，不动其他进程，不借此重复加载重试。阶段开始若没有足够GPU资源，完成CPU准备并报告精确缺口一次，不进行无新信息的三轮重复检查。

上轮注意力诊断实际峰值约29.10 GiB，但它不是本轮generate的峰值保证；不再沿用失准的21 GiB估计。建议选择空闲至少34 GiB的可用单卡，并留意其他任务；backend逻辑设备必须`cuda:0`，若指定物理卡，用CUDA_VISIBLE_DEVICES映射并记录。禁止kill/移动他人进程、自动量化或CPU offload改变比较条件。首次正式调用兼作真实内存检查，不另开真实模型smoke生成。

48次与1,800秒为同一experiment_lock的跨重启累计上限，不因新shell或resume重置；每次使用唯一run_id，已记账调用不重复累计。新attempt账本存OUT；全局`results/cai_agent_v3/compute_ledger.jsonl`只在阶段闭合追加本任务真实汇总一行，保留原前缀，不重置旧更新额度。主/repair/failed数要对应文件。结束时不把人工评价PENDING说成实验尚未执行。

## 7. 结果应让作者直接看出“改了什么、有没有对应好”

### 7.1 每件至少导出

- 同一clean图、原编号R0、新编号R1，全部保存真实模型输入版本。
- A/B/C/D各自单张候选叠图：画在同一个clean底图上，保留编号参考、两region分别标识、cue/alternative/序数置信。只画边界或低透明度，不遮住线索；同时可切换本配置真实编号输入。
- 每配置一张未平滑的8×8序数置信图（如合到HTML可不另存重复PNG），明确写`decoded ordinal confidence, NOT attention / damage probability`。
- 每配置的诊断C0与fallback原因；`action_count=0`、空已测集合及登记图像尺寸构造预算0.25的合法集合。它表示既有规则在新候选上的结果，**没有运行Actor，不标为新first action**。
- 历史候选H00只读对照；q24-48可标出已存历史first action=36，但在所有新配置中只能叫historical reference，不随新C0重新解释。
- 完整raw first/repair response下载入口和64格特征/边界CSV。

最终必须有**单张PNG文件**，不能只给不可导出的浏览器画布。报告不复制全分辨率原图、字体或模型；必要来源保留路径和hash，使用≤1024的输入衍生图降低仓库体积。

### 7.2 HTML

生成`OUT/index.html`，静态相对资源、无网络后端/CDN，双击可打开。每个案例上方展示clean和编号输入；下方并排/可切换四配置与H00，显示实际格号、cue、alternative、置信、有效/修复/no-cue状态和C0。

页面必须能隐藏候选/切换真实输入/点开原尺寸，不将“人类可读参考图”误标为模型输入。可提供隐藏配置名的查看模式，但不能称正式盲评。复用已安装浏览器只检查首页+一个案例；没有浏览器时如实记录静态检查，不能把安装浏览器变成主要开发任务。

### 7.3 自动汇总可以说什么

`summary.csv`每个新case/variant一行（最多24），列出：输入身份、prompt/render、主/修复状态、region数、唯一cell数、置信数量、no-cue、C0大小和原因、调用数、时间、raw路径等。

另存`candidate_changes.csv`：B−A、C−A、D−B、D−C、D−A的候选集合Jaccard/新增删除格、ordinal变化、C0变化。双空集合明确`BOTH_EMPTY`，不当作准确定位。分开历史H00与新A的变化。

这些是**输出敏感性/结构一致性描述，不是定位准确率**。不以“更像23、31”“候选变少”“更接近no-VLM首动作”或“热图更集中”自动选赢家。

### 7.4 人类评价与缺失输入

不用开发新专家平台。导出`human_review_template.csv`和简短`HOW_TO_REVIEW_ZH.md`，让作者在现有HTML观察后填：

```text
specimen_key,variant,region_index,reviewer_alias,
visible_cue_match,cell_location_match,annotation_artifact_suspected,
reference_cells_optional,review_status,notes
```

建议值：`MATCH/PARTIAL/MISMATCH/UNCERTAIN`；`review_status=PENDING/CONFIRMED`；可疑标签干扰只能记作者判断，不自动变成模型内部因果结论。模板实际案例/region行自动列出，评价字段默认PENDING/空。

关于用户说的q24-48旋转后线状痕迹在23、31，只可在`USER_HYPOTHESES.md`以用户目视观察记录，**不复制到prompt或自动定为author GT**。首先确认模型的`line`是否指同一条线。对另一个cue区域要单独看，不能只评被讨论的线而忽略圆状痕迹。

未收到真实评分时，最终状态为`RUN_COMPLETE_REVIEW_PENDING`；照常交付图和报告，定位收益结论`PENDING_HUMAN_REVIEW`。Codex可以记录输入/图片可读性，不得替作者造“6/6定位成功”。若作者之后提供评分，读取真实CSV再汇总配对改善/不变/退化计数，不需新增模型调用。

## 8. 有限验收，不另造审计系统

仅六组检查，可用少量assert/单元例和人工看图完成。只运行本目录测试与Ruff已改脚本，不运行全库pytest、不全盘hash、不开多轮模拟review。

| 检查 | 必要内容 |
|---|---|
| Q1 案例与来源 | 6件固定/规则选择；非TEST；同case的四配置clean hash一致；R0和历史能核对的输入一致；无二次旋转 |
| Q2 控制变量与prompt真接线 | fake backend接到A/C的P0、B/D的P1；A/B编号hash同、C/D编号hash同；schema和chat template不漂移；新缓存不命中旧不同请求 |
| Q3 数字与几何 | 一张奇数尺寸合成图核对边界/0与63等角点；新64标签字框在所属格内；同clean位置一致；overlay不翻转 |
| Q4 完整回答与额度 | token IDs、原文、parse/contract状态、repair保留；失败占额度；resume不重复生成；测试用stub，不真实烧调用 |
| Q5 结果与人工边界 | 全24格任务状态有记录；不利/无cue/失败可见；C0不是动作；无GT时不报准确率；HTML/PNG可看且原始回答可读 |
| Q6 冻结与交付 | 生产src/旧缓存/旧统计/论文未改；仅新脚本/结果/交接、任务卡和一行账目；实际commit/push与三方SHA |

P0若发现R0输入hash与历史不一致，先查源身份/实际Pillow/字体渲染版本，不能凭肉眼相似放行并声称历史复现；若不能恢复旧R0，保留其缺口，不偷偷改R0为新图。可交付准备与原因，不能在漂移R0下把差异全部归因P1/R1。

无需再次逐像素重现上轮所有历史图、跑所有64格crop测试或所有旧VLM结果。准确来源只核对本轮6件及新输入。每项检查失败只修相应代码；不因不出现预期语义改善反复改prompt。

## 9. 阶段任务、交接与GitHub

| 阶段 | 操作 | 完成标准 |
|---|---|---|
| W0 | 绑定源码、准备6件、冻结P0/P1/R0/R1 | 真正可调用的实验配置与输入，不是只有文档 |
| W1 | 薄runner、缓存、计数和四类CPU预检 | 24主调用在进入GPU前已具名、输入hash确定 |
| W2 | 固定模型加载一次，先q24-48后其余5件 | 完整raw结果/失败状态保存，最多48次生成 |
| W3 | 纯CPU特征/C0/叠图/HTML/汇总/评价模板 | 作者可直接逐图判断，未评不编造 |
| W4 | 一次定向复核、交接、commit/push | 实际代码和图像已到原分支，三方SHA核对 |

必须交付：

```text
OUT/
  index.html
  experiment_lock.json
  cases.csv
  prompts/                 # P0/P1最终真实prompt与修复prompt
  inputs/<case>/           # clean, R0, R1；必要几何元数据
  runs/<case>/<variant>/   # raw_1、token IDs、元数据；必要raw_2
  overlays/<case>/         # 四配置独立PNG及历史参考
  summary.csv
  candidate_changes.csv
  historical_replay_comparison.csv
  human_review_template.csv
  HOW_TO_REVIEW_ZH.md
  attempts.jsonl
  final_manifest.json
ART/
  SOURCE_AND_METHOD_BINDINGS.md
  RESULTS_AND_NEXT_DECISION.md
  REQUIREMENTS_REVIEW.md
  CODEX_HANDOFF_VLM_GROUNDING_PILOT.md
```

可以合并非核心的小JSON，不为文件数建冗余产物。`RESULTS_AND_NEXT_DECISION.md`分别写：代码/输入检查结论、观察到的候选变化、真实人工评价状态、是否值得之后考虑生产接入。**不能自动应用赢家；不能只凭格式好就推荐部署。**

同分支普通commit/push；只stage本任务范围，不PR、不merge、不force push、不reset。保留旧任务结果和用户无关未跟踪文件。确认MD、PNG、HTML、raw回答都被实际追踪上传，不只推manifest。需要人工评分未到或GPU不足，也推送已实际完成的代码与交接并注明原因。

本轮全局账目只追加真实摘要，不修改旧行；可以只对少量关键冻结文本/缓存文件开始结束核对一次，不对全部模型权重重复hash。字体不分享，模型不上传。

结果提交后记录该真实结果SHA，最终回复提供包含交接的最终HEAD/upstream/remote。不要为把文件自身SHA写入自身反复提交。最终回报：执行案例/配置/调用数、直接可打开的HTML和图片路径、候选变化与待人评项、零训练/其他模型计数、三方SHA。

## 10. 结果的解释规则

- 提示词/标签改变后输出变化，说明当前结果对相应输入设计敏感，不自动说明变化正确。
- 若作者观察P1/R1更好，可报告这6件的真实配对改善；不升级为全50件CAI或新TEST确认。
- 只有q24-48改善，应记为开发案例现象；其余五件也有正向定位材料，才值得进一步讨论通用设置。
- no-cue增多需同时看漏掉了哪些线索；候选更少不是自动更精准。C0变宽/变窄也不是收益大小。
- 新置信含义更明确，但仍不是校准概率。Actor从旧先验分布训练而来，不能只换缓存并沿用旧论文数字。
- 原输入方向/原JSON到绘图链已检查一致。发现新的、具体可复现映射缺陷时记录，不把所有错误都解释为旧旋转问题，也不自动废弃整个旧实验。
- 本轮核心成果是可检验的定位接口对照和完整输入输出证据，而非保证“VLM终于有收益”。
