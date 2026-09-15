# 对提示词的最终逐项复核

主代理亲自完成代码语义、attention提取与最终审核。仅build_index.py静态HTML机械实现委派给mechanical_worker（gpt-5.6-luna/max），主代理随后审阅完整文件、补入实际结果摘要/资源链接，并浏览器复核。没有把科学判断或最终review委派。

| 提示词要求 | 实际证据与结果 |
|---|---|
| §0/§2 原分支、基点、保护未提交改动 | 开始HEAD/upstream均8778aa53且工作树干净；复用原研究分支工作树，未reset |
| §0 A纯CPU，B单例≤2次/15分钟 | A模型前向0；B一次，168.125秒，GPU0/batch1；attempt/status/compute记录一致 |
| §1 三种量不互换 | 05标ordinal after decoding；09/10/11标diagnostic replay pre-first-cell token；08标historical Actor；原缓存无attention |
| §2 不能凭cell36认定 | q24-48源图hash、指定cache_key、双输入hash、固定case/figure/reuse manifests共同匹配；无本轮上传文件的限制明确披露 |
| §3 全部指定代码链 | 主代理已读render/parser/cache/features/grid/crops/backend/overlay/actor执行/C0规则及当前依赖；映射辅助函数存在但v3不调用 |
| A1 01—04和实际输入 | 全部独立PNG；02/03与缓存PNG hash精确相同；04明确不是历史输入；原JSON/text/prompt与call1/repairedFalse保留 |
| A2 05—08与64维 | 原数组/CSV、raw/parsed/CSV全64格一致；四格medium同等级；首动作由历史trace；无平滑候选/假attention |
| A3 64行坐标 | coordinate_trace保存source/render/native/normalized/display/C-scan边界与所有候选/动作字段；row-major、一次旋转、width/height和半开边界明确 |
| A3 像素取整/半像素/合成往返 | 本例round与rint差0；实际_draw_cells的−0.5边界；67×83合成64格实际裁剪/overlay全部通过，包含指定八格 |
| A3 历史overlay比较 | 两张历史PNG RGB逐像素相同，max difference0；未自动定义表面GT或旋转搜索 |
| B0/B1 严格截断query | 原数字offset75，prefix26tokens；query2792，总2793；不含目标数字；先前prompt2767tokens与历史相同；slow原tokenizer前缀ID/decode验证 |
| B2 固定层/head/实现 | 最后四层24–27，每层全28 heads；从真实post-RoPE Q/K显式FP32 last-row softmax；SDPA输出未改；保存各层向量及误差记录，不存N×N |
| B2 资源与失败边界 | 本机环境/权重，无量化/换图/换prompt/模型下载；一次forward成功，未第二次；预估21GiB而实测29.098GiB的差异披露，不隐瞒 |
| B2 下一token | top1历史首数字token3匹配，概率0.514893；只声明首token一致，不声称数字36完整复现 |
| B3 两图映射 | 两组[1,70,70]经merge2各35×35；window_index逆序恢复检查；token mapping共2450行，各段1225；无sqrt猜形状 |
| B3 09/10/11、数组、mass、原尺寸未平滑 | 均存在且打开；四层/均值/各图.npy；1024×1024nearest显示；raw mass0.031398/0.169516/0.799087；共享色标，无独立拉满 |
| §6 轻量HTML及PNG | Chromium file://实际打开，11图全加载，所有本地链接有效，桌面/390px手机无溢出/页面错误；图片可单独打开；不开发额外平台。可选鼠标读数未实现，不影响必须项 |
| §7 判断范围 | 当前后处理坐标链未见不一致；编号图峰值跨四格，表面痕迹贴合由用户目视判断；不以attention当GT/价值/因果，也不解释群体MAE |
| §8 四项有限验收 | FINITE_VALIDATION.json四组通过；仅相关检查，无W2/W3重跑、全仓pytest、全模型hash、bootstrap或压力测试 |
| §8 交接、账目、Git | 交接已写；账目仅追加一行，原前缀保留；src/论文/历史结果不改。实际推送闭合在GIT_DELIVERY.json与最终回复 |

最终review未发现需改生产代码的bug。修正的仅是本次诊断脚本：环境已有另一cmc_bbdm editable安装，故绑定当前源码namespace；纯裁剪/绘图函数通过AST加载，避免引入无关缺失旧模块；slow tokenizer不支持offset API，改为原tokenizer的ID前缀/decode边界核对；序数图长标签改为U并解释含义，防止格内文字重叠。以上未触发Qwen前向失败或新增研究执行。

尚不能证明的内容：未提供的上传图与该case的身份；未保存的历史attention/token序列；历史完整数字/回答复现；注意力对表面线索的因果作用；损伤或CAI最优格。这些均未被包装为完成结论，也不是本任务应伪造的产物。

Git结果闭合：`5ea9e13ac23e90e1f64e079719a0e2d10db14ed0`已推送；当时local HEAD/upstream/remote三方一致、工作树干净，全部46个OUT文件（含8个.npy）Git blob与本地字节一致。最终交接记录由随后唯一文档提交纳入，最终SHA见最终回复。
