# W2 exact-cost checkpoint恢复交接

## 当前任务与状态

用户已明确绑定 `research/cai-vlm-agent-v3-controlled-reuse` 的 W2 选择追溯修复，旧“缺具体任务”阻塞已解除。实际工作树 `/home/ww/diff/.worktrees/cai-vlm-agent-v3-controlled-reuse`；入口本地/远端均为 `00ac8fb8800e3a3b908532d077af37f09aee90da`，没有需覆盖的后续提交或未提交工作。未reset、切换分支或重新执行W0。

主规范已实际读取：`/home/ww/diff/.worktrees/cai-vlm-agent-v3-controlled-reuse/docs/cai/v3/CODEX_CAI_AGENT_V3_EXECUTION.md`。配套四个文件为同目录的 `CAI_AGENT_V3_SOURCE_BINDINGS.md`、`CODEX_CAI_AGENT_V3_REVIEW.md`、`metric_reference.py`、`golden_cases.json`。来源为 `/home/ww/diff/docs/CODEX_HANDOFF_CAI_AGENT_V3/` 中五个同名文件的逐字副本，不声称找到了ZIP，不覆盖其他版本。

本轮：真实训练更新0，新VLM调用0，新TEST感知/标签评分0，未启动W3/GDFS/扩seed。临时合成checkpoint只在pytest的tmp目录；训练循环接线测试明确禁用 `optimizer.step`。旧科学产物、旧失效报告均保留；ledger只追加本轮零更新记录。

| 维度 | 结论 |
|---|---|
| 任务绑定 | 已明确，W2零训练恢复准备 |
| 代码恢复准备 | 已补齐逐时点保存、重载、固定VALID身份、完整重评分及关口；定向行为验证见验证日志 |
| 历史checkpoint | 未找回完整候选；43个时点仅6个真实快照，缺37份 |
| 协议/科学结果 | 旧W2–W4仍失效；MEAN_SC只是在保留快照中的暂时最优，不是有效P_all |
| 后续训练 | 未授权；需要明确恢复额度/分配后才可执行 |
| review方式 | 主代理自检＋外部独立数值参照，未声称独立子代理审查 |

## 实际证据和定向查找

已读取原交接、`REQUIREMENTS_REVIEW.md`、`requirements_review_final.json`、`IMPLEMENTATION_BINDINGS.md`、成本审计、扩展状态及ledger。用户列出的 `results/cai_agent_v3/new_protocol/compute_ledger.jsonl` 不存在；真实路径为 `results/cai_agent_v3/compute_ledger.jsonl`。

只检查本study的 `results/cai_agent_v3/` 目录结构及其 `new_protocol/models/`（含可能的子目录）；所读保存代码/manifest未指向另一个W2 checkpoint缓存。该models目录共12个模型文件，其中6个是W2快照。`git log --oneline -- results/cai_agent_v3/new_protocol/models` 仅指向初始保存提交 `7380c4e`，没有后续逐时点保存历史。入口main下 `/home/ww/diff/results/cai_agent_v3` 不存在。没有扫描服务器其他模型库、临时目录或全盘。

`W2_RECOVERY_EVIDENCE.json`记录每个实际参选update、保留update、缺失update、6个文件SHA及逐阶段资源。六个SHA与原ledger的已完成模型记录一致。

| 模型 | 实际参选时点数 | 唯一保留update | 缺失权重数 |
|---|---:|---:|---:|
| MEAN_SC | 8 | 1750 | 7 |
| SPATIAL_SC | 6 | 500 | 5 |
| SPATIAL_C | 6 | 500 | 5 |
| OOF fold0 MEAN_SC | 8 | 1250 | 7 |
| OOF fold1 MEAN_SC | 8 | 1000 | 7 |
| OOF fold2 MEAN_SC | 7 | 750 | 6 |

三个VALID前缀mask曾改变，而旧逐时点评分CSV没有对应全部权重或原始逐状态预测。不能从scalar score恢复权重，也不能凭保留赢家复评分数不变恢复完整选择。本轮未重复六个旧赢家的无效重评分。

## 修复与原已修复项

- `checkpoint_selection.py`：独立run目录、原子写入每个实际参选时点的权重及update、评分、seed/fit组/常量、成本定义、VALID输入身份；所有非赢家一并保留。`selection.json`在训练中为INCOMPLETE，只有完整计划或合法早停时才可完成。
- VALID身份包含试样顺序、域/组、原生尺寸、VALID标签、实际cell特征、路线、mask、state index、精确成本。只绑定VALID所需数据，不读取TEST标签，不以名字或单个READY字段代替输入身份。
- `_train_candidate`在归档前验证输入确为现有float64原生成本和固定前缀；共同预测器及OOF正式调用均传入archive目录。每次VALID后先保存候选，再比较最优；结束时从实际保存文件重载赢家。
- 原checkpoint改善阈值 `1e-12`、每250步、patience4、2000步上限保持不变；P_all仍用既有门槛和`1e-8`平局规则，不放宽预算/误差容限。
- `rescore_predictor_archive`在零训练条件下重新前向所有已归档候选，返回独立结果，不修改旧选择或研究状态。缺任意候选、未正常完成、VALID身份变化均拒绝；它不会将任意新输入自动贴上旧验证身份。
- `_manifest_selection_verified`及OOF加载关口核对完整候选、实际选中权重和当前VALID身份。`refresh-cost-evaluation`不能再只凭history标签或当前mask未变声明验证完成。
- 未来W2任务STARTED以唯一run_id保留≤2000更新上限；只有同run_id完成记录才解除该预留并改计实际更新，保存途中中断也不会漏计。旧记录的求和规则不变，旧750未知上界不释放。
- W2正式入口增加累计资源预检；当前5,836不足以覆盖12,000上限时，在加载特征/拟合/训练之前拒绝执行。本轮未运行正式入口；用临时ledger验证拒绝行为。
- 原生像素格、float64前缀/硬动作合法性、模型输入转float32和左阶梯评价通路均复用。float32只用于神经网络输入，不用于硬预算判断。没有重复修改原修复。

归档用于选择证据和模型重载，不声称新增了中断训练的optimizer/RNG续训CLI。未完成归档保持INCOMPLETE，不从它自动继续，也不清零旧账。未来若执行中断恢复，仍须按主规范处理optimizer/RNG与累计消耗，不能将本轮归档当作续训授权。

## 最小恢复清单

| 输入/产物 | 处理 |
|---|---|
| 276名单、标签来源、capture group、split、原生尺寸 | 原协议未变，可复用；不重做W0或全工作簿审计 |
| 六份冻结特征shard与原图/预处理索引 | 复用现有无标签特征，原图或预处理未变不重编码 |
| 真实VLM fit缓存及6条终止不可用记录 | 复用，失败记录不得被当作可任意重采样的缺口；TEST缓存不新增 |
| 常量、完整Ridge诊断 | 输入未变可复用；部分Ridge如需正式重新报告，须按同一exact库核对/有限CPU重算，不训练新神经模型 |
| 固定exact-cost VALID前缀库 | 复用现有库，并在未来运行前绑定实际输入身份；不得每epoch重采样 |
| 三个共同预测器候选 | 历史选择不可恢复；需按各≤2000的原规则重新训练/选update，不能仅续训旧赢家 |
| 三个OOF回报模型 | 新P_all结构确定且通过后，按原组折重新训练/选update；即使仍选MEAN_SC，旧OOF选择也未验证 |
| W3、GDFS、非自适应路线评分/图 | 等待新有效P_all/OOF后按依赖重新计算；旧策略权重只能作失效诊断，不能只重评分就称合规新pilot |
| seed扩展、新TEST | 仍关闭；不属于本轮，也不由W2恢复自动授权 |

未来正式归档目录为 `new_protocol/models/selection_history/predictor_<name>/` 及 `reward_<name>_fold<k>/`。已有同名目录拒绝覆盖；需要新run位置时先按后续恢复任务保留旧产物。仓库忽略`*.pt`/`models/`，未来交付需明确添加真实归档权重（例如对本run路径定向`git add -f`），不能只提交manifest；本轮没有正式归档或合成模型需上传。

## 累计额度和待授权请求

依据实际ledger（含失败、作废和静态中断上界），已知21,514＋未知≤750＝使用上界22,264，剩余下界5,836。

| 注册类别 | 原上限 | 保守已用 | 原类别剩余额度 |
|---|---:|---:|---:|
| 三个共同预测器 | 6000 | 5000 | 1000 |
| 三个OOF预测器 | 6000 | 5750 | 250 |
| 主/无VLM/开环各三seed | 11250 | 7500 | 3750 |
| 静态三seed | 2250 | 1500（含未知≤750） | 750 |
| 均值Actor诊断 | 1250 | 1250 | 0 |
| GDFS诊断 | 1250 | 1250 | 0 |
| 小型优化预检 | 100 | 14 | 86 |
| 合计 | 28100 | 22264 | 5836 |

完整W2重放上限为6000＋6000＝12,000。仅从总额度看至少缺6,164，但W2两个原类别合计只剩1,250，不能把其他类别剩余4,586默认为W2授权。

建议的后续**W2-only资源请求，当前未获批准**：明确将未用的4,586类别额度转给W2，并新增6,164，总上限变为34,464；W2共同/OOF累计类别上限分别需容纳11,000和11,750。这样可覆盖完整12,000重放，但不会给后续W3/GDFS/TEST留下自动授权。若保留原类别分配不转额度，则W2额外类别额度需10,750，总上限为38,850。两种方案均不清零已经用掉的22,264。

W3新pilot上限5,750、GDFS上限1,250、条件seed扩展9,000，均需另行依据新W2结果决定；不包含在W2-only请求。早停节省量不能事先当作可用额度。

ledger有设备时长的记录累计8,383.222秒（约2.329小时），但部分预检及中断静态任务缺时长，故只能给下界，不能声称12小时预算还精确剩9.671小时。本轮GPU耗时0；未来训练仍需核清原累计GPU时长上界，或在用户明确的新时间授权中处理，不能靠重置时钟扩额。

## 验收与交付

定向命令及实际输出见 `W2_RECOVERY_VALIDATION.md`。覆盖五类：成本/固定前缀一致、保存/重载、完整选择、依赖关口、累计资源。独立常量预测在y=200时，190/198对应A=10/2，完整重评分选第二个；另核对golden的A=8及尾段A=6。非整除9×11格子采用既有nearest-even边界，预算最多24/99像素比例；未擅自换成floor分格。

原失效审查没有改成PASS；历史科学有效性仍为NONCONFORMANT。代码恢复准备通过不证明新模型科学效果，也不解除训练额度缺口。

本轮提交只包含恢复代码/测试、五份规范副本、TASK/STATE、恢复交接/证据/验证日志和ledger零更新追加。不提交入口main的未跟踪`docs/cai/`、`docs/stagetask/`或其他无关文件。完成commit/push后，在最终回复报告local/upstream/remote同一SHA；不PR、不merge、不force push。
