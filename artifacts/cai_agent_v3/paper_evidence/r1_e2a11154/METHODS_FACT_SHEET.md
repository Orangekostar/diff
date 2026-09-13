# 已执行方法事实表

以下为冻结运行事实及本次新增分析定义，不是新训练方案。源码/结果基点e2a1115468da6e8695321204a13fa9a5322ea809，W3实际训练代码0f18001；详细来源为A3/INPUT_BINDINGS.json、actor_manifests.json、原W3 protocol_snapshot.json，本轮INPUT_BINDINGS保存必要输入哈希。

## 任务、输入与冻结预测器

目标为实验CAI强度（MPa），不是BC动作标签、mask成功率或归一化强度比。队列276件/259组，来源关系capture group不跨split；TRAIN161/152组、VALID50/48组、TEST65/59组。本次不接TEST标签。表面/C-scan为冻结ResNet18独立8×8图格特征，每格512维，不重新编码。

有效共同预测器为新W2 MEAN_SC@1750，hash f67e912bd5faa26ae5cff8a9a0241439797fccef8cec825f43ebe5dc9f9d57ff。VALID九方法共用此权重。三折TRAIN回报模型分别选1250/750/1000，按固定query折排除该试样capture组；Actor采样均匀域→域内物理试样，不均匀轮抽折。完整参照从同一共同预测器既存NPZ的full_*读取，不来自OOF平均；完整C-scan仍配原表面输入。

VLM使用冻结真实Qwen2.5-VL-7B缓存：211条fit记录=205可用+6终止不可用，不可用不等于no_reliable_cue；新增调用0。最高medium/high可靠候选并集C0只约束第一步，Actor在可负担候选中选具体格；第二步起开放其余可负担未测格。无可靠/不可用/不可容纳分支保持原记录。

## 策略与已执行训练

| 方法 | 决策允许的信息/结构 | training seed | 参数数 | 所选update | 实际更新 |
|---|---|---:|---:|---:|---:|
| VLM_SPATIAL_FEEDBACK | 表面、VLM、已测内部/当前预测；2层4头128维Transformer | 2026091301 | 378978 | 250 | 1250 |
| NO_VLM_SPATIAL_FEEDBACK | 同表面与内部反馈，去VLM/C0 | 2026091302 | 378978 | 1000 | 1250 |
| VLM_SPATIAL_OPEN_LOOP | 同表面/VLM，不以内部内容/当前预测决定动作 | 2026091303 | 378978 | 750 | 1250 |
| LEARNED_STATIC_TRUE | 仅64共享位置logits，无图像/预测输入 | 2026091304 | 64 | 250 | 750 |
| VLM_MEAN_FEEDBACK | 同主信息，64维均值结构 | 2026091305 | 91650 | 250 | 1250 |

空间结构实际forward调用contextualizer；未知C-scan在全局计算前mask。无反馈与静态只限制决策信息，评价预测器仍读实际采得C-scan。无VLM不是无表面；空间与均值不是严格容量匹配。固定Center/Geometry/Serpentine及Random均不训练；Random已有五repeat seed2026091250..54。

每job构造前独立seed，未加载旧Actor；batch16、AdamW lr3e-4/wd1e-4、clip1、gamma1、value权重.5，entropy从.01线性降至0。损失为CAI误差的cost-to-go最小化，用既有TRAIN尺度归一；J=A+.25终点误差，静态使用zero baseline。每250步固定VALID/eval/argmax选六域等权A最小者，改善阈值1e-12/patience4。五job共5750更新、23参选，所有候选保存；本次只用已经选定的650最终轨迹，不重新选模。

原生rint边界划分8×8；唯一像素成本／全图像素，硬成本float64、预算.25、容限1e-12，模型数值输入可float32。只按可负担完整格终止，不做STOP；各episode独立终止。实际成本均值/范围可能小于共同预算，不用动作步数替代像素分数。

## 本次新分析（事后定义）

- 五预算0/.0625/.125/.1875/.25为主表；599个所有episode公共成本断点为补充。查询取最后一个c≤b的当前预测；.25内末状态保持，.25外拒绝，不单调化或外推。
- MAE/MSE先每试样平均Random重复损失，再池化；RMSE总体MSE开方；R²逐repeat计算后均值。A/early A用原左阶梯，试样内repeat→域内物理试样→六域等权。full成本只有1、A为空。
- 一次5000次、seed2026091401的域内capture-group bootstrap，抽中组的全体成员按重数保留。所有预算和方法共用bootstrap_group_weights.npz；MAE差用池化，四机制面积差用域等权。95%逐点探索区间，条件于已选VALID，不校正选择、不做新gate或p值筛选。
- q为联合所有观测MAE及full范围的完整整数网格41..61 MPa，加20个非自适应方法/预算锚点与full锚点。先同b聚合群体MAE，再取各自最早达标成本；两网格分栏，未达为空，比率零分母为空，保留反弹与负值。full支持集仅{1}，不推论其他工业流程必须全扫。
- 真正事件含调用号/前后预测/像素/合法集，仅分析侧用y算误差变化。summary记录的C0释放reason不自动表示此前真有限制；机制叙述不产生反事实或注意力。

实现入口：scripts/build_cai_agent_paper_evidence.py；新模块paper_evidence.py及paper_evidence_math.py。未调用训练/旧summarize/evaluate/export；不读.pt/特征shard、不访问GPU，全部新前向和优化0。
