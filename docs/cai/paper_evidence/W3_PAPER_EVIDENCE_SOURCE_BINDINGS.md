# 冻结论文证据任务：依据与源码绑定

核对日期：2026-09-14。仓库：Orangekostar/diff。
固定证据提交：`e2a1115468da6e8695321204a13fa9a5322ea809`。
本文件记录已实际读到的代码/文本与实施判断；不声称在聊天容器执行了服务器上的新分析。

## 一、信息分级

| 类型 | 内容 | 如何使用 |
|---|---|---|
| 用户决定 | 停止新增实验；不添加均匀稀疏/插值基线；只补完整输入参照；不做硬件开销评测 | 本轮硬范围，覆盖历史扩种子和TEST建议 |
| 已读仓库事实 | W3九方法、650轨迹、50 VALID/48组；选模已完成；单seed | 冻结已有预测，不重选模型 |
| 已读代码事实 | 最终轨迹包含动作前后预测与执行记录；W2 callback保存全输入预测数组 | 允许纯读表分析，不需要新模型前向 |
| 已读代码事实 | 旧W3/W2 summarize写回旧目录和manifest | 新建独立输出入口，禁止运行旧写入函数 |
| 本轮分析设计 | 统一预算网格、全事件断点补充、群体MAE等质量分析、探索性区间 | 明确为事后分析定义，不冒充先于训练的协议 |
| 仍需Codex本地完成 | gzip全部650行及NPZ全输入数组读取、连接核对、等质量完整曲线、真实分析数值 | 不能从摘要构造未读取的逐试样值，不能预报所有阈值的节省量 |

## 二、实际读取的依据

所有blob链接固定到同一证据提交，当前分支引用除外。路径不证明聊天容器存在同名文件；服务器工作树由Codex自行核对。

### [R01] 当前分支

https://api.github.com/repos/Orangekostar/diff/git/ref/heads/research/cai-vlm-agent-v3-controlled-reuse

本轮返回上述e2a11154…；不得回退到W2或旧BC分支。

### [R02] W3结果范围与完成状态

https://github.com/Orangekostar/diff/blob/e2a1115468da6e8695321204a13fa9a5322ea809/artifacts/cai_agent_v3/w3_pilot/r1_0e11452a/RESULTS_AND_CLAIM_BOUNDARIES.md

https://github.com/Orangekostar/diff/blob/e2a1115468da6e8695321204a13fa9a5322ea809/results/cai_agent_v3/w3_pilot/r1_0e11452a/final_manifest.json

确认：650 episodes、50物理样本/48组、单策略seed面板；A/early A六域等权，终点池化。原固定对照选GEOMETRY_SPREAD。VLM早期负向，空间对均值差异很小；未进行独立TEST或多seed确认。累计39,014/40,014，剩余不是自动训练授权。

### [R03] 同成本与配对汇总、原写入副作用

https://github.com/Orangekostar/diff/blob/e2a1115468da6e8695321204a13fa9a5322ea809/src/cmc_bbdm/cai_agent_v3/w3_results.py

本轮读L1–260：`values`解析字符串；`pooled_errors`以prediction_at_budget取当前值，先每key平均重复损失，R²按repeat计算；`method_metrics`并列面积和池化终点；`summarize`写回W3 CSV、gate和final_manifest。

由此得出：可借用纯计算，但不能运行旧summarize生成新论文结果；否则会改变已冻结交付状态。RANDOM不能先平均预测再算误差。

### [R04] 图稿与原始图读取

同文件L260–末：`export_figures`读取case_manifest和真实execution_trace，生成表面线索、首动作、k=1/4/8/末和预测曲线。原函数写入旧W3/figures，故只能复用图片或纯绘图逻辑，不能原样调用。

它通过DATA中candidate_queue的指定图像路径及feature_bank_manifest里的`encoder_execution_root`定位原图；不是凭借外部数据根的历史名称猜路径。

### [R05] 冻结episode读取与面积计算

https://github.com/Orangekostar/diff/blob/e2a1115468da6e8695321204a13fa9a5322ea809/src/cmc_bbdm/cai_agent_v3/actor_selection.py

`read_episodes`是gzip+CSV；`episode_metrics`核对状态/动作长度并按原面积函数计算。归档还支持.pt，但本轮不使用模型载入或重选路径。只读取最终赢家轨迹即可。

### [R06] 统一预算查询与积分

https://github.com/Orangekostar/diff/blob/e2a1115468da6e8695321204a13fa9a5322ea809/src/cmc_bbdm/cai_agent_v3/metrics.py

`prediction_at_budget`以bisect_right取不超过b的最后状态；`left_error_area_mpa`对左端常值预测积分。注意该查询函数没有.25上界，过大b仍返回末值；因此本轮新分析必须另行限制数据覆盖范围，不能把末值外推成.25到1的新曲线。

### [R07] 完整扫描参考真实来源与数组schema

https://github.com/Orangekostar/diff/blob/e2a1115468da6e8695321204a13fa9a5322ea809/results/cai_agent_v3/w3_pilot/r1_0e11452a/p_all_saved_reference.json

https://github.com/Orangekostar/diff/blob/e2a1115468da6e8695321204a13fa9a5322ea809/src/cmc_bbdm/cai_agent_v3/predictor_training.py

本轮读取`evaluate_predictor`details_callback（约L298–435），确认保存字段：`full_specimen_indices`、`full_predictions_mpa`、`full_targets_mpa`，另外的`predictions_mpa`是固定四路线前缀，不应混用。

既有指针：`results/cai_agent_v3/w2_replay/r1_292b1c74/candidate_state_predictions/A_MEAN_SC/update_001750.npz`。

同模型hash：`f67e912bd5faa26ae5cff8a9a0241439797fccef8cec825f43ebe5dc9f9d57ff`。

完整输入汇总：50件、MAE41.69001007080078、RMSE53.94403981050803、R²0.7105359831287097。这里只核对了指针/生成代码/汇总；实际NPZ数组将在Codex本地读出，不能说聊天侧已复算所有50件。

### [R08] 用索引连接既存预测的方法

https://github.com/Orangekostar/diff/blob/e2a1115468da6e8695321204a13fa9a5322ea809/src/cmc_bbdm/cai_agent_v3/w2_replay_results.py

现有代码以 `feature_bank_index.csv`的原顺序将npz的specimen_indices连接dataset_id。新完整输入提取沿用同一映射，但只读full_*。本模块顶层绑定W2输出且summarize会写旧状态，不可直接当新任务入口。

### [R09] 机制分析可用到什么程度

https://github.com/Orangekostar/diff/blob/e2a1115468da6e8695321204a13fa9a5322ea809/src/cmc_bbdm/cai_agent_v3/actor_training.py

本轮读取约L387–630：真实action_callback包含调用号、动作、visible_cells_before、两种合法mask、C0原因、实际像素和动作前后预测；`evaluate_policy`将execution_trace写入episode。新分析可以解释这个过程，但记录没有自然语言思维链或完整动作价值矩阵，不能制作所谓真实推理/注意力解释。

模型与起始规则（已在当前会话固定提交读取）：
https://github.com/Orangekostar/diff/blob/e2a1115468da6e8695321204a13fa9a5322ea809/src/cmc_bbdm/cai_agent_v3/models.py
https://github.com/Orangekostar/diff/blob/e2a1115468da6e8695321204a13fa9a5322ea809/src/cmc_bbdm/cai_agent_v3/policy.py

VLM起始只是一种先验，第一步C0最高可靠置信约束，之后解除；无VLM保留表面CNN和内部观测，不等于纯内部或完全无表面输入。

### [R10] 已冻结案例

https://github.com/Orangekostar/diff/blob/e2a1115468da6e8695321204a13fa9a5322ea809/results/cai_agent_v3/w3_pilot/r1_0e11452a/case_manifest.csv

三件为74t7kcdgkr:c8-16、cgtnjyggtm:q24-48、w68dtmpfyf:q16-29。不能按结果再次选图。

### [R11] 原W3附件只用于解释已完成方法，不提供新训练授权

本轮已通过Files读到上传的 `CODEX_CAI_V3_W3_PILOT_EXECUTION.md` 的起始范围、固定来源、信息权限和方法矩阵，以及 `W3_PILOT_SOURCE_BINDINGS.md`。它们记录五个seed1任务和原规范；本次用户已经进一步收窄到论文证据，不能执行其train-pilots或历史多seed规则。

历史 `AEI_CSCAN_AGENT_HANDOFF_NEXT_CHAT.md`、Hasebe追溯附件属于更早的P4 BC/proxy任务，不能覆盖新W3的CAI监督、数据划分和结果。

## 三、当前可用锚点与必须避免的拼接

| 项目 | 当前表中值 | 只能怎样使用 |
|---|---:|---|
| 主方法A | 45.11037655 | 0–.25六域等权面积，不是终点MAE |
| 固定GEOMETRY A | 47.310645（舍入；精确值以原CSV为准） | 保持原BEST_NONADAPTIVE身份，不逐budget挑有利对手 |
| 主方法MAE@.25 | 44.28579895 | 与同预算方法比较；与full的差属于跨成本权衡 |
| 无VLM MAE@.25 | 42.38468079 | 保留强对照，不隐去 |
| full MAE@1 | 41.69001007 | 一个全输入点，没有0–1的完整路线面积 |
| 原显示点的等质量例子 | q=46.90991699时，主.0625，GEOMETRY .125，STATIC .0625 | 只能作为主网格手算检查；更细最早成本需原轨迹重新计算 |

## 四、本轮新增分析为什么这样设计

1. **同成本上限而非相同步数**：原动作购买不同原生像素数，且会留未用预算；输出实际成本统计能避免单位偷换。
2. **群体MAE先汇总再取最小成本**：避免逐试样使用真实CAI择时、造成不可部署的全知优势。
3. **所有对照查各自最早达标点**：不能将对照.25的质量当目标，却忽略它.125已能达标。
4. **完整扫描仅成本1**：没有保存中间轨迹就不补出曲线，也不为它制造免费预测。
5. **全部q网格与锚点保留**：避免看完结果只挑一个有利质量要求；这些仍为事后经验分析，不是工程合格标准。
6. **探索性区间不构成独立确认**：同50件曾用于选模；重采样不会产生新物理试样或新策略种子。
7. **新增解析、不要重跑旧summarize**：旧入口会改写冻结结果；独立RUN既可复用纯代码，又不影响历史结论。

上述是根据已读代码作出的实施判断，不是外部论文带来的新算法，也不预言等质量曲线处处有利。
