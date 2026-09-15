# Codex定向review：VLM定位接口有限对照

本文件用于核对主规范实际落地，不用于扩大实验或开展多轮模拟审稿。只做一次开发前静态检查、首件实际四配置接线检查，以及交付时的合并检查；失败仅修对应工程项。

## 1. 必须先确认的任务身份

这是`VLM_GROUNDING_PILOT_R1_e8d9ef01`。基点e8d9ef0171ab6d0e1d14187a56cdbd7ecc579750；6件×4配置；最多48次新Qwen生成尝试；其他研究模型与训练为0。不是旧单例2-forward热图任务，也不是CAI模型性能扩展。

## 2. 六项检查与判定

| ID | 读什么实际证据 | 通过条件 |
|---|---|---|
| Q1 | case manifest、选择代码、输入图/hash、模块__file__ | 三个既有VALID例＋三域TRAIN固定hash例；没有按结果筛选；clean只旋转一次；加载正确工作树；不读TEST图片/标签 |
| Q2 | fake backend调用捕获、冻结prompt与image hash、cache key | A/C真正收到P0，B/D收到P1；A/B同R0，C/D同R1；同schema/chat-template/权重；新prompt不经硬编码旧resolve |
| Q3 | R1绘字函数、label boxes、原尺寸图片、单一合成几何例 | 数字0—63与位置一致、字体明确、全部字框在所属格；R0/R1只在标记区域不同；没双旋转/换底图 |
| Q4 | attempts、raw_1/raw_2、generated IDs、tokens、环境记录 | 至多24主任务和48生成；第2次仅格式修复；无语义挑选；失败保留；resume不重发；完整回答不等同首数字 |
| Q5 | summary、HTML、PNG、C0表、human review模板 | 全部案例/配置状态可见；候选变化不冒充精度；C0不是动作；没有假人工/假GT；未评也完成交付 |
| Q6 | git diff、全局账本增量、交接、远端ref | 旧src/缓存/论文/统计未改；代码/图像/raw/HTML实际推送；无模型/字体/环境上传；事实与未决项分别记录 |

## 3. 特别拦截的错误

- 把新prompt的hash传给旧`SurfacePerceptCache.resolve`，实际上仍运行P0。
- R1把已旋转的clean再旋转一次；将`01_source_surface.png`当clean输入。
- P1中出现q24-48、23/31、27/28或来自旧回答的特定答案，或者把用户上传的overlay/热图传给模型。
- 新旧配置共享缓存key，仅文件夹名字不同。
- 只运行B和D，再将历史回答当同环境A，省掉承诺的新A却仍宣称2×2完成。
- 把schema合格、更多no-cue、更少候选或历史首动作命中C0称为定位更好。
- 没有作者填写参考却输出IoU、命中率或“6/6成功”。
- 把一轮generate报为一个forward；忽略repair、失败或中断资源。
- 使用新候选运行已有Actor并把结果继续算进旧论文，不另行授权与分版。
- 为得到正结果反复调整字体/提示词、换样本/模型，或重新提取attention。

## 4. 有限检查建议

用stub响应检查cache隔离、一次repair与resume；用一个奇数宽高8×8彩色ID小图检查标签边界和绘图方向；字体在实际processor缩放尺寸下肉眼看一次。不要使用Qwen来做这些测试，不需要全库pytest、所有历史图逐像素重建或全部模型哈希。

实际首件四配置完成后，只看输入是否正确、日志/文件是否完整、prompt是否真的传递；即使D没有更好也继续其余冻结案例。任何更改会影响科学配置时，保留已执行版本并停止混表，不能无记录重跑。

## 5. 最終状态表达

- `RUN_COMPLETE_REVIEW_PENDING`：24主任务均有执行/失败记录，导图和HTML已交付，但无真实人评。
- `RUN_COMPLETE_REVIEWED`：作者真实评分已导入，报告对应逐例计数。
- `PARTIAL_RESOURCE_LIMIT` / `PARTIAL_INPUT_MISSING`：保持所有缺项及其原因，仍交付实际代码/产物。
- `RESEARCH_PRODUCTION_UNCHANGED`：旧先验/Actor/CAI/论文未改，不等于新定位已验证。

不得只回复Q1—Q6 PASS。交接必须包含作者可打开的HTML、单张PNG、raw回答入口、实际调用次数和仍待人类判断的问题。
