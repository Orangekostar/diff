# VLM选格前注意力与坐标链诊断交接

任务：VLM_CELL_DIAG_R1_8778aa53。状态：COORDINATE_EXPORT_COMPLETE / ATTENTION_EXPORTED。
基点：8778aa53875c75f3ca681915e322a9c02cbc1ac3；分支：research/cai-vlm-agent-v3-controlled-reuse。执行工作树：/home/ww/diff/.worktrees/cai-vlm-agent-v3-controlled-reuse。

## 先看这些文件

以下OUT为results/cai_agent_v3/vlm_cell_diagnostic/r1_8778aa53/（相对仓库根）：

1. OUT/index.html：可直接在浏览器打开的完整诊断页，无后端、无外链依赖；图可点击查看原尺寸。
2. OUT/03_vlm_numbered_input.png：真实历史编号输入，未替换原小字号。
3. OUT/11_attention_with_raw_cells.png：新注意力与历史候选/首动作的对照。
4. OUT/09_pre_cell_attention_clean.png、10_pre_cell_attention_numbered.png：两图分开、相同原始概率色标。
5. OUT/05_cached_confidence_8x8.png：解码后的序数置信，不能当作选格前注意力。
6. OUT/coordinate_trace.csv：64格的source/render/display/C-scan坐标、候选和动作字段；OUT/raw_response.txt保存原回答。

## 案例身份和历史记录

确认对象是cgtnjyggtm:q24-48，VALID，既定三个案例之一。通过candidate_queue源图路径/哈希、固定case manifest、figure manifest、缓存请求及两张历史输入哈希交叉绑定，未用first action=36单独认定身份。本轮消息没有附新的上传图，因此不声称与未提供的上传图逐像素匹配；本次诊断绑定任务文件明确列出的q24-48及其指定cache身份。

- cache_key：d11f57f3d3ba2e96339d1857e31f3f9192074a624682d7bf4faa29cd04345773。
- clean PNG hash：c4d3d9badaaa6b9ff0b351685a1833a8dab13443b79642d4f6353ef3f6fdf0c9。
- numbered PNG hash：27d40ac5937960d91439d15a6f928dde9e834da4fcd826f540678530e314da8b。
- 原表面3357×3357，顺时针90°一次、LANCZOS最大边1024，得到1024×1024；两张输出按原PNG编码核对精确一致。
- 历史call_count=1、repaired=False；没有待恢复的invalid第一次回答。历史缓存未保存attention、像素热图或64格连续效用logits，不能据此说Qwen没有注意力机制。
- 真实原回答：第一region cells=[36,37]，cue="circle with white speckles"，alternative="dust or debris accumulation"，medium；第二region cells=[59,60]，cue="line"，alternative="scratch or mark on the surface"，medium；no_reliable_cue=false。完整字符含代码围栏原样保存。
- raw JSON→parser→cached percept→CSV的64维indicator/confidence逐格一致。四格均为float32 0.6666666865348816。实际C0也是[36,37,59,60]；历史Actor首动作为36。

## 坐标结论

未发现本案例的raw ID到最终surface overlay链路错位。原surface_cues与first_action两张历史PNG均由当前同一纯绘图逻辑逐像素复现，RGB最大差为0，源文件hash也与复用manifest一致。

编号为左上原点、row-major：cell36=(row4,col4)，即第五行第五列。表面render每格128像素，cell36半开边界[x512,x640)×[y512,y640)；Matplotlib矩形使用(511.5,511.5)起点、128×128。C-scan注册图为675×674，拥有自己的np.rint边界，不能把其像素坐标直接用于1024×1024表面图。coordinate_trace中分别保存两套边界；source逆映射是旋转/缩放后的连续近似，不把重采样像素称为精确可逆。

render的round与NativeCellGrid的np.rint在本表面图九条边界上差0像素。67×83非方形奇数尺寸合成图为64格分配唯一RGB ID，经实际_cell_crops及_draw_cells作全64格往返，包含0/7/56/63和27/28/35/36，全部通过。绘图采用origin=upper与边界−0.5，未同时反转数据/坐标。

display_cell_id/physical_cell_id/map_percept_to_physical虽存在于旧perception模块，但当前v3路径不调用，未套入其置换。既有旋转未重复发生。表面痕迹是否被候选恰当覆盖仍请用户对照03/06/11目视判断；未定义自动GT、亮点真值或no-VLM真值，未由单例推断群体MAE原因。

## B：真实新诊断的精确定义与限制

只执行1次冻结Qwen2.5-VL-7B-Instruct诊断forward；包含失败的forward尝试总数仍为1（无失败forward）。其余准备阶段曾遇到导入路径和slow tokenizer offset API问题，均发生在forward前；没有拿兼容性失败抵扣forward上限。未调用generate，未生成完整回答，未改变候选或缓存。

权重路径从生产vlm_perception.py的_MODEL_PATH读取，revision=cc594898137f460bfe9f0759e9844b3ce807cfb5；本机Transformers4.49.0、PyTorch2.12.1+cu130。clean→numbered顺序、SURFACE_PERCEPT_PROMPT、user chat template、use_fast=False、min_pixels=200704/max_pixels=1003520、bfloat16和SDPA均沿用实际配置。模型eval、requires_grad=False、inference_mode。原prompt输入token数与重建均为2767。

原回答首个cells数字字符位于offset75；保留其前全部真实文本，26个prefix tokens，不含"3"或后续任何数字。原slow tokenizer不提供offset_mapping，所以用原tokenizer逐段token-ID前缀相等、decode前缀逐字节内容相等及加上下一个token后的字符范围核对边界，没有更换tokenizer。完整输入2793 tokens，query是最后已输入位置2792，其输出预测首数字token"3"（id18），不是读完36后的状态。input_token_sequence.csv、answer_prefix.txt和attention_preflight.json保存细节。

固定取语言decoder零基24–27层（最后四层）、每层全部28 heads。诊断包装这四层的SDPA调用，从实际post-RoPE、已repeat_kv的Q/K显式计算最后query的一行FP32 softmax；再将该行概率先平均heads、后平均四层。实际模型仍返回原SDPA结果，未切换eager、未用梯度归因/相关性替代attention。每层只保存[2793] head-mean向量；计算时目标行形状[1,28,1,2793]，无保存N×N矩阵。

该概率行乘V与真实BF16 SDPA最后输出行的RMS差为0.000717–0.001722，最大绝对差0.00777–0.02719，最大误差约为各层输出峰值的0.22%–0.34%，并非逐bit一致；所有head概率行和误差≤3.58e-7。记录在attention_capture_checks.json。这里是由真实本次Q/K显式导出的标准注意力概率，不冒称历史SDPA内部保存的矩阵。

两图image_grid_thw均为[1,70,70]，processor resize为980×980；patch14、spatial_merge_size2，分别形成35×35=1225个merged视觉token。processor把2×2 patch分组；本机视觉模型在window attention/merger之后通过argsort(window_index)恢复merged row-major顺序，再按image-token位置masked_scatter。实际window-index往返也做纯索引核对。没有把两段拼成平方、没有sqrt猜尺寸。

下一token top1="3"，概率0.514893，历史首数字token排名1；第二名"2"概率0.400999。状态REPLAY_FIRST_TOKEN_MATCH只代表首token一致，不证明接下来的"6"或整段回答已复现。历史生成token-ID序列未保存，历史逐token带KV-cache decode也未重跑；本次是严格截断前缀的一次性teacher-forced诊断。不能称恢复了原始历史attention。

## 注意力读图

- clean视觉token总量0.03139775（约3.14%）；numbered总量0.16951552（约16.95%）；非视觉总量0.79908675（约79.91%）。总和1。
- 两图共享vmin0、vmax0.007663713的原始token概率色标，不独立拉满、不作条件归一化。故clean图较暗不表示它没有注意力。
- clean最大token位于(row34,col5)，其中心对应cell57附近。numbered最大token位于(row17,col17)，在render坐标约[497.37,526.63]²，横跨27/28/35/36四格交界。中心归入36只是约定，不能把整个token称为格36专属证据。
- 注意力图同时包含编号、边界和表面内容的可能作用；此单次可视化不能区分哪一种导致了回答，更不能把峰值命名best cell、损伤概率或CAI信息价值。
- 原35×35 .npy、四层向量、token映射CSV、原尺寸nearest未平滑PNG/.npy均保存。上采样只为显示，不增加像素定位精度；透明层固定alpha0.38，未高斯平滑离散候选。

## 实际资源与账目

GPU0单卡、batch1，整个加载/诊断过程168.125秒，小于900秒；外部timeout也设900秒。观察到的最大PyTorch分配29.098GiB。预估21GiB低于实际峰值，原预留8GiB未全保留；执行前空闲33.06GiB，实际成功无OOM。其余现有进程为评估任务而非训练，未停止/移动它们；进程退出后GPU0空闲恢复33850MiB。数值库与torch最多4线程。

实际新增：Qwen forward1；完整回答generate0；Actor/CAI/ResNet/Reader/STOP前向0；训练0；TEST接入0；bootstrap0；性能重评0。compute_ledger.jsonl仅追加VLM_CELL_DIAG_R1_8778aa53这一行，原前缀字节完整保留，未重置历史额度。

## 验收与运行方式

FINITE_VALIDATION.json记录四项有限验收；REQUIREMENT_REVIEW.md按提示词逐项核对。已人工查看输入/序数图/C0图/两注意力图/叠加图，并在Chromium直接打开HTML验证11张图、相对链接、桌面/手机及无页面错误。HTML_OPEN_CHECK.json及浏览器截图在本目录。浏览器原本未安装，补用了/tmp/vlm_diag_browser_cache临时headless shell，仅用于查看，不改研究环境、不提交浏览器/环境。

复核现有导出（不运行模型）：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/vlm_cell_diagnostic/validate_exports.py
python scripts/vlm_cell_diagnostic/build_index.py
```

首次CPU导出入口为export_cpu.py，显示用plot_attention.py只读已保存数组。replay_attention.py已有attempt记录时主动拒绝重跑，防止无意新增前向；本任务不需要再次运行它。没有将诊断attention实现写回QwenVLBackend。

源码依据：vlm_cscan/runtime.py::render_surface_inputs/_image_sha256；learned_cscan/perception.py::SURFACE_PERCEPT_PROMPT/parse_surface_percept/SurfacePerceptCache；cai_agent_v3/vlm_perception.py::_features；cai_active_image/environment.py::NativeCellGrid；feature_bank.py::_cell_crops；w3_results.py::export_figures；diagnostics.py::_draw_cells；actor_training.py::_evaluate_one；policy.py::vlm_first_action_mask；本机transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py和qwen2_vl/image_processing_qwen2_vl.py。

使用本次同分支隔离工作树。只新增独立诊断脚本/输出/交接和一行账目；论文、src生产代码、原缓存/特征/轨迹/统计结果均不修改。实际提交与推送身份见GIT_DELIVERY.json和最终回复；不PR/merge/force push。

Git结果闭合：`5ea9e13ac23e90e1f64e079719a0e2d10db14ed0`已推送；当时local HEAD/upstream/remote三方一致、工作树干净，全部46个OUT文件（含8个.npy）Git blob与本地字节一致。最终交接记录由随后唯一文档提交纳入，最终SHA见最终回复。
