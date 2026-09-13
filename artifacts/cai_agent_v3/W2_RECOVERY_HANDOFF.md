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


## 本轮增补：实际后续工作树的 W2 零训练复核（2026-09-13）

以下更新当前状态；上文为入口 00ac8fb 的历史交接，原失效报告不改写。

- **任务已明确绑定**：当前实际入口 `0f18001c8fa25e83b6fd3ece2fa6c7cd74e6c5ff`，fetch 后远端 `0e11452ac6590b3b2b694364bd4d1cef7c9315cf`。原 00ac8fb 是祖先，其后的 W2 恢复、正式重放及两笔本地 W3 接线提交均保留。工作树已有 W3 未提交 STATE/ledger/结果，未覆盖或继续执行。
- **代码恢复准备 PASS**：本轮复核 28 项定向测试、独立 oracle 7 项通过。逐候选归档、固定 exact-cost VALID 身份、原生成本定义、评分记录、实际赢家重载、缺失非赢家阻断与累计预留机制已存在，不重复修改生产代码。9×11 独立整数像素边界与临时合成权重仍通过；真实训练更新为 0。
- **历史 checkpoint 未找回**：定向检查原 `new_protocol/models/`，仍为 43 个时点仅 6 份权重，缺 37 份；六份 SHA 与旧证据一致。该目录 Git 历史仅有初始保存提交 7380c4e，未指向其他候选缓存。无全盘搜索，未仅重评赢家冒充完整历史选择。
- **后续独立 W2 归档真实存在**：`results/cai_agent_v3/w2_replay/r1_292b1c74/models/selection_history/` 有六份 COMPLETE 归档、44 个实际候选。逐份核对计划/早停、全部候选文件 SHA、固定身份和登记评分排序通过。本轮没有重新进行真实模型前向；它们是后续正式重放产生的新权重，不是旧缺失权重。该重放已在 215df7a/0e11452 交付，P_all=MEAN_SC@1750、OOF@1250/750/1000；本轮不重复训练或追认旧结果。
- **科学边界**：原 `new_protocol` W2–W4 仍为失效诊断，原成本审查、扩展阻断和 requirements_review_final.json 与 00ac8fb 一致。新 W2 单独目录的既有结果不因本轮代码测试自动扩大其科学结论；本轮不审定已有未交付 W3。新 VLM、TEST 感知/标签评分、W3/GDFS/seed 扩展执行均 0。

### 按实际 ledger 更新资源与最小恢复清单

实际 ledger 仍是 `results/cai_agent_v3/compute_ledger.jsonl`，用户列出的 `new_protocol/compute_ledger.jsonl` 不存在。174 行原样保留，00ac8fb 的 ledger 是其字节前缀；无未解除的新 run 预留。已知 38,264 + 旧未知上界 750 = **39,014**，包含历史 21,514、后续正式 W2 11,000、已有 W3 5,750。本轮新增 0，不把旧基线余额 5,836当当前余额。

现有后续登记上限来自 `docs/cai/w3_valid_pilot/W3_PILOT_AUTHORIZATION.json`；仅用于解释既有资源记录，不是本轮训练授权。

| 类别 | 已登记累计上限 | 现有保守已用 | 余额 |
|---|---:|---:|---:|
| 共同预测器 | 11,000 | 10,250 | 750 |
| OOF 预测器 | 11,750 | 11,500 | 250 |
| 主/无 VLM/开环 Actor | 11,250 | 11,250 | 0 |
| 静态 Actor（含未知 750） | 2,250 | 2,250 | 0 |
| 均值 Actor | 2,500 | 2,500 | 0 |
| GDFS | 1,250 | 1,250 | 0 |
| 历史优化预检 | 14 | 14 | 0 |
| 合计 | 40,014 | 39,014 | 1,000 |

原交接曾写 `28,100 + 6,164 = 34,464`，这是历史算术错误，正确为 **34,264**，后续正式重放授权已明确纠正。保留原文供追溯，本轮不按错误数值增加额度。

数据/固定 split、六份特征 shard、VLM 缓存及六条不可用记录、exact-cost VALID 库继续复用，不重新编码或感知。旧六个 W2 模型不能从残存赢家恢复选择；原先所需的三共同+三 OOF 重放已经在独立新目录完成，因此目前无需仅因旧档缺失再训练这些新模型。旧 W3/GDFS 仍不能凭新 P_all 自动变有效，后续重算与训练必须按单独有效任务及额度执行；本轮不启动。

若用户另要求再执行完整六模型 W2 重放，登记上限仍为 12,000；现有类别余额仅 750+250，至少需新增 **11,000**（共同 5,250、OOF 5,750），累计上限至少 51,014，并明确新的 GPU 时间授权。此为条件资源请求，不是推荐重复已完成重放，更不是本轮许可。历史 GPU 时长仍不完整，不声称旧 12 小时还剩精确时长。

### 本轮验收与交付范围

`W2_RECOVERY_RECHECK.json` 保存实际路径、入口 SHA、ledger 身份/汇总、六份新归档候选数及选择时点。本轮定向命令同 W2_RECOVERY_VALIDATION.md 的最终组合，实际输出为 `28 passed, 24 deselected in 5.73s`；`python docs/cai/v3/metric_reference.py --self-test` 为 7 项 PASS。临时检查器首次遇既有 editable-package 路径导入问题，随后按仓库 wrapper 的 `cmc_bbdm.__path__` 方式完成，不修改生产代码。

本次新增提交只含 TASK/STATE 的 W2 绑定增补、本交接和复核 JSON。既有未提交 W3 ledger/STATE 段落/模型/图表保持未提交；主工作树三个未跟踪 docs 目录保留。为保持同分支 fast-forward，push 会包含入口已存在的两笔 W3 接线提交 27ceee5、0f18001；它们不是本轮新增实现，不据此宣称 W3 结果交付完成。不 PR、merge、force push 或 reset。实际最终 local/upstream/remote SHA 在回复报告。
