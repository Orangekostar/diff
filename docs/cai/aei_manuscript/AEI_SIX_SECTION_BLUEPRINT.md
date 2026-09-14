# 六章正文蓝图与段落—证据绑定

本文件是作者已同意结构的执行细化[D]，不是声称AEI规定必须六章。目标是工程问题牵引的方法论文，英文主稿，中文记录。

## 核心论证与三个贡献

中心问题：有限的内部图像采集应怎样服务于CAI强度评估？

- C1 任务表达：明确表面先验、已测内部证据、未测区域和采集成本，构建CAI导向的顺序信息获取任务。
- C2 方法：冻结表面/图格感知及部分观测预测器，学习每一步状态依赖的采集策略，以全过程和终点CAI误差训练，而非固定动作示范。
- C3 评价解释：共同预测器下比较同成本预测、等质量采集及获取时机，公开各组件与全输入之间的具体权衡。

不把C3写成“保证显著优于所有方法”；主方法身份维持VLM_SPATIAL_FEEDBACK，无VLM较优的对照保留。

## 总体篇幅和写作顺序

规划正文约6,000–7,500英文词，不是期刊硬限；建议各章范围见下表，按实际内容压缩，不强行达到总量。写作顺序3→4→5→2→1→6→摘要/题目，正文编号仍1–6。

| 章 | 建议词数 | 该章唯一主要工作 | 主要产物 |
|---|---:|---|---|
| 1 Introduction | 800–1,000 | 从CAI工程需求引出为什么需要控制采集过程 | 问题、缺口、本文方案、三贡献 |
| 2 Related work | 650–850 | 与最接近图像回归、主动检测、任务采集研究定位 | 三线综述、精简差异表 |
| 3 Framework | 1,650–2,100 | 把工程问题映射为可实现的状态、策略和目标 | 框架图、状态表、算法与方程 |
| 4 Experimental design | 1,000–1,300 | 说明已有实验怎样回答问题 | 数据/权限/对照/指标/选模与统计 |
| 5 Results and discussion | 1,700–2,000 | 从同成本、反馈、时机、等质量到全输入权衡 | 主结果表图与解释 |
| 6 Conclusions | 180–250 | 凝练解决了什么与实际发现 | 贡献、具体发现、适用前景 |

## 1. Introduction

### 段落功能图

| 段 | 论证工作 | 应用材料 | 不能写成 |
|---|---|---|---|
| I1 | CAI评估的工程目标与观测需求 | 实际数据论文、C-scan→CAI最近邻 | 无来源的“行业普遍低效”“可以替代服役安全验收” |
| I2 | 区分获取完整图后的预测与获取哪些信息的决策 | 最近邻正文核实；用户研究目标 | “此前完全没有自主超声/图像回归” |
| I3 | 问题张力：表面先验可先得、内部内容逐步得、预算有限 | 信息权限与真实成本协议 | 推测未知材料物理规律或夸大VLM能力 |
| I4 | 方案路线：任务目标＋当前证据共同决定下一观测 | Methods和流程图 | 模型品牌清单替代工程思路 |
| I5 | 三项贡献，概述怎么评估而非结果最大值列表 | C1–C3；冻结比较 | first-ever、SOTA、guaranteed等未核实词 |

引言研究问题与第5章小节逐项对应，不能引入没有结果支撑的原位硬件验证、真实损伤定位或STOP目标。

## 2. Related work

### 2.1 Image-based residual-strength assessment
讲清全图/描述符→强度研究能做什么。重点读AEI图像CAI论文，不拼不同数据划分的R²排行。识别本文关注取得观测的过程，不声称首次提出CAI图像预测。

### 2.2 Adaptive acquisition for nondestructive inspection
核实Fuentes等自主超声工作，区分损伤指示/概率场目标与本文CAI回归代价。路径规划缩短位移的研究与选择信息的研究分别解释，不能混为同一个工业baseline。

### 2.3 Task-driven sequential information acquisition
以动态特征选择、价值导向获取为概念基础；引用实际算法来源。原文平方损失的条件方差结论不能直接充当本文绝对误差和有限数据网络的保证。本文不是GDFS精确复现。

本章末用4–6行矩阵总结：任务终点、决策前可见信息、是否依据新增反馈、成本定义、评价范围。每格均绑定原文位置。没有读到的项明确未核实。

## 3. Task-driven multimodal inspection framework

### 3.1 Problem formulation
定义：表面图S；隐藏C-scan图X；64格I；已测集合Omega_t及mask M_t；实际累计cost c_t；预测yhat_t；动作a_t；预算B。动作是原生图格，不是声学采样点。

目标写J=A+.25e_T，说明全部运行按预算终止，不把真实y送Actor。适当给优化期望，不虚构最优解保证。

### 3.2 Multimodal inspection-state representation
表面ResNet和VLM分别处理。VLM输出cue/alternative留在缓存，Actor实际只接区域、置信、可用性等数值；表面特征仍在无VLM对照中。VLM作用是检查先验，不是损伤真值。

用状态权限表说明S始终可见、X只在M_t为真处可见、y训练/评分侧可见。图格先裁再编码，环境可缓存完整特征但先mask后跨格运算。

### 3.3 Partial-observation CAI assessment
MEAN_SC全格融合pool＋已测pool＋成本回归；全格pool不是纯surface。无策略名/历史顺序，保证相同输入状态同预测的代码性质。预测器先训练冻结，OOF回报模型排除当前capture组。

### 3.4 State-dependent acquisition policy
SpatialCAIActor的65 token交互及动作分数；首步C0、后续释放；合法未测与可负担约束；训练采样、评估argmax；critic与CAI预测器分工。给一份10–18行Algorithm 1，不抄全部Python。

### 3.5 Acquisition timing and task cost
展示A恒等展开和g_t定义，解释时间权重；正文推导2–4行，SI给完整望远镜求和。明确它是损失函数的会计恒等式，不是因果贡献或新理论保证。实测阶段分解放5.2/5.3，不在Method提前写未经计算的“早期主要贡献”。

## 4. Experimental design

### 4.1 Data and specimen correspondence
一张表列六域实际来源/配置（核实才填）、物理试样数、来源组、split。CAI单位MPa及来源工作簿字段依实际清单。研究使用既有真实图像回放，不虚构新设备采集或健康样本。

### 4.2 Compared acquisition strategies
九方法全部定义清楚：四非学习固定方法、真实静态、主法、无VLM、开环、均值。FULL_SCAN同预测器＋完整C-scan＋表面，单列100%成本，没有A。规则名不冒充企业标准。

### 4.3 Learning and evaluation protocol
已有TRAIN/VALID、选模、OOF、种子和完整归档事实；明确本稿报告的VALID参与过checkpoint选择，TEST未评价。以表承载超参，正文不写研发审计史或SHA长串。

### 4.4 Cost, quality and statistical analysis
A域等权；MAE/MSE/R²原聚合；same budget cap不是exact same pixel count。每个预算的actual range放表或注。

等质量先群体MAE后最小b，两网格分开，null/负saving保留；first passage不是部署STOP，full只有点。5000组权重区间是事后逐点条件探索，不称正式显著。必要限定集中此处说明。

## 5. Results and discussion

### 5.1 Prediction quality under matched acquisition caps
主表25%摘要含全部九方法＋100%full；完整五预算在SI。先观察主方法vsgeometry/static，主A和端点各一结论，关键区间同表。noVLM最优不能隐藏。

### 5.2 From a shared order to evidence-dependent acquisition
static47.977→open46.501→main45.110作分层结果描述，解释信息权限；不能将相邻差除总差写成VLM/反馈的因果占比。空间/均值、VLM消融在同一表/图直接给出事实，不反复审计式否定。

### 5.3 Where in the trajectory are gains accumulated?
从本轮唯一派生数据写时机贡献；四阶段为观测完成阶段，正负均显示。方法差按共同e0和域等权核对；不是说每步都更好。结合既定三例展示实际选择和预测变化，不给模型代写语言推理。

### 5.4 Acquisition requirements at matched empirical quality
使用已有全部q与锚点数据。正文展示主网格和一组有出处目标，明确对geometry50%与static0%，细event-grid放SI且注明反弹，不用其中较大saving替换主结果。是否节省取决于目标和对照。

### 5.5 Full-information trade-off and engineering interpretation
full41.690 vs main44.286；从质量差到工程意义。不把没有达到full写成“无损替代”，不把25% raster换算75%时间。逐域结果3/6或5/6按各对照准确陈述。

### 5.6 Scope and outlook（短段，可并入5.5末尾）
只集中一段：单种子/selected VALID、图像回放与原生像素成本、有限可泛化证据。用具体结果界定用途和未来验证，不写一长串“本研究不足”。保留这些事实但不启动新实验。

## 6. Conclusions

三段以内：提出的任务/方法；本数据与协议下观察到的同成本和过程差异；评估质量与获取信息组织相连接的意义。数值引用与主表相同。结论不比摘要/Results更强，不在结论首次引入新结果或其他期刊工作。

## 主图/主表建议[D]（不是AEI版面限额）

| 编号 | 内容/角色 | 来源/工作 |
|---|---|---|
| Fig.1 | 多模态状态→Actor→获取→预测更新的闭环 | 新确定性示意；训练监督与执行流分开，VLM不在每步伪造新调用 |
| Fig.2 | 0–.25的MAE曲线，体现同投入比较 | 复用已有F1，完整参考线注明cost1；完整RMSE/R²曲线放SI |
| Fig.3 | 四项机制差及已有区间 | 复用F5；既有小/负方向完整保留 |
| Fig.4 | 一次新增时机分解的签名贡献 | 新派生图，不生成新CI；符号和A恒等式对应 |
| Fig.5 | 既定案例的过程 | 引用旧三例必要单图，保持分开文件；失败例保留SI或正文简述 |
| Fig.6 | 主网格等质量图或完整输入差图 | 复用F4，full质量gap表/旧F6按篇幅放SI；不能隐藏差距 |
| Table1 | 数据/来源组/配准与分割 | 从真实元信息整理 |
| Table2 | 方法与可见性/结构 | 代码事实 |
| Table3 | 25%九方法与100%full | 已有主表，关键配对区间可同表分栏 |
| Table4 | 关键配对与等质量示例 | 使用已有值，避免大表和曲线重复同一信息 |

选择更少主图可以，但删除原则是无新增论证功能；结论改变的不利信息不能只放SI。不要把当前六类图全部原样塞进正文后又全文重复每个数字。

## 作者需要确认、不阻塞完整草稿的事项

作者/通讯顺序、机构英文、基金号、利益冲突、CRediT、数据/代码公开范围与许可、最终投稿声明、AI使用审阅状态。科学事实确实缺失时用源ID占位并准确列缺口；不能因为作者信息不齐就只写引言。
