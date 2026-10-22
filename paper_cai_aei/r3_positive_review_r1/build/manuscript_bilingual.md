# Learning region selection and acquisition order in C-scan images for efficient compression-after-impact strength assessment of composite laminates / 通过学习 C-scan 区域选择与采集次序高效评估复合材料层合板冲击后压缩强度

# Abstract / 摘要

Compression-after-impact (CAI) strength assessment depends on which internal-image observations are supplied to the predictor and when they become available. We propose strength-assessment-driven multimodal sequential observation to learn C-scan region selection and acquisition order from specimen-level strength supervision. Initially available surface descriptors, acquired internal regions and the current CAI estimate form an observable decision state. A regional policy requests the next feasible region, and a common predictor updates strength after each observation. A trajectory objective combines process-average and terminal absolute errors to value useful estimates throughout acquisition. In offline regional replay on 50 development-validation specimens, also used for checkpoint selection, endpoint mean absolute error (MAE) was 42.385 MPa at an acquisition-area cap of 0.25, 9.65% below geometry-spread. The 0.0625 prefix of this same total-cap path reached MAE 43.324 MPa. On the five-cap reporting grid, first attainment of cohort MAE at most 47 MPa required a cap of 0.0625 versus 0.125 for geometry-spread. Recorded events showed larger early net error reductions. Fixed-final-set replay isolated arrival within selected content: the primary permutation-minus-original process contrast was +1.486 MPa, with a 95% interval of [−0.658, 3.384] MPa, leaving its average advantage uncertain. These results support task-driven observation organization for earlier useful CAI estimates within the selected development configuration.

冲击后压缩（CAI）强度评估取决于向预测器提供哪些内部图像观测，以及这些信息何时可用。本文提出强度评估驱动的多模态序贯观测方法，利用试样级强度监督学习 C-scan 区域选择与采集次序。初始表面描述符、已获取内部区域及当前 CAI 估计构成可观测状态；区域策略请求下一可行区域，共同预测器在每次观测后更新强度。轨迹目标联合过程平均绝对误差与终点误差，评价整个采集过程中的有用估计。在 50 件同时用于 checkpoint 选择的开发验证试样上进行离线区域回放时，方法于 0.25 采集面积上限达到 42.385 MPa 终点平均绝对误差（MAE），比几何分散低 9.65%；同一总上限路径的 0.0625 前缀达到 43.324 MPa。在五点报告网格上，首次达到不高于 47 MPa 的队列 MAE，本文所需上限为 0.0625，几何分散为 0.125。归档事件显示更大的早期净误差下降。固定最终集合回放检验选定内容内部的到达顺序：主要“置换减原序”的过程差为 +1.486 MPa，95% 区间 [−0.658, 3.384] MPa，平均优势仍不确定。结果支持在已选开发配置中，以任务驱动的观测组织较早形成有用 CAI 估计。

**Keywords:** compression-after-impact strength; composite laminates; C-scan images; sequential acquisition; region selection; task-driven assessment.

**关键词：** 冲击后压缩强度；复合材料层合板；C-scan 图像；序贯采集；区域选择；任务驱动评估。

# 1. Introduction / 引言

Compression-after-impact (CAI) strength connects the observed damage of a composite laminate to its residual compressive capacity. For carbon-fiber-reinforced polymer laminates, surface images and internal C-scan images provide complementary descriptions of the impacted specimen. Public paired-image datasets and measured CAI labels make this connection accessible to data-driven assessment \[1, 2\]. Image-based regression has further shown how C-scan spatial information can support strength prediction \[3\]. These developments create an opportunity to improve not only the predictor, but also the observations supplied to it.

冲击后压缩（compression-after-impact，CAI）强度将复合材料层合板的损伤状态与其剩余压缩承载能力联系起来。对于碳纤维增强聚合物层合板，表面图像与内部 C-scan 图像提供了受冲击试样的互补描述。公开配对图像及实测 CAI 标签为数据驱动的强度评估提供了基础 \[1, 2\]，图像回归研究进一步展示了 C-scan 空间信息对强度预测的支持作用 \[3\]。这些进展使强度评估的改进从预测器本身，延伸到向预测器提供哪些观测以及如何提供这些观测。

In sequential assessment, internal C-scan information enters prediction region by region, so the current strength estimate depends on the content acquired so far. Region selection determines what is observed; acquisition order determines when it becomes available. A useful observation obtained early can inform more subsequent estimates, making information arrival relevant to prediction quality throughout acquisition. CAI assessment can therefore be formulated as a coordinated task of region selection, observation update and strength prediction.

在序贯评估中，内部 C-scan 信息通过区域观测逐步进入预测，因而当前强度估计取决于已经获得的内容。区域选择决定"观察什么"，获取顺序决定"何时获得"。较早取得的有效观测能够参与更多后续估计，使信息到达时机直接影响整个采集过程的预测质量。由此，CAI 评估可以从对既有图像的回归，进一步拓展为区域选择、观测更新与强度预测相协同的任务。

The central design challenge is to assign regional observations value through the mechanical assessment they support. Specimen-level strength provides a direct task signal, whereas the usefulness of an individual region depends on its interaction with the information already available. The same final observed set can also support different intermediate estimates when its regions arrive in different orders. An assessment-oriented acquisition method thus needs to learn both which content to request and how to bring useful content into the prediction process early. A prescribed acquisition-area cap provides a common scale for comparing prediction quality, while a fixed prediction-quality target reveals how much internal information is needed to reach that quality.

这一任务的核心设计问题，是依据区域观测对力学评估的作用赋予其价值。试样级实测强度提供直接的任务监督，而单一区域的有效性取决于它与已有信息的相互作用。同一最终观测集合也会因区域到达顺序不同，形成不同的中间估计。因此，面向评估的采集方法需要共同学习请求哪些内容，以及如何让有用内容更早进入预测过程。预设采集面积上限为预测质量比较提供统一规模；固定预测质量目标则进一步刻画达到该质量所需的内部信息量。

Sequential inspection and active feature acquisition provide foundations for decisions based on an evolving information state \[4--6\]. Autonomous ultrasonic inspection links measurement selection to damage indication \[4\], while active feature acquisition learns to request information for downstream prediction \[5, 6\]. For CAI assessment, this decision structure must connect spatial image requests to a continuous mechanical output at every observed prefix. The unresolved question is how strength-supervised regional selection and information arrival can work together to improve the sequence of estimates, rather than treating acquisition and regression as separate stages.

序贯检查与主动特征获取为依据动态信息状态进行决策提供了研究基础 \[4--6\]。自主超声检查将测量选择与损伤指示联系起来 \[4\]，主动特征获取则学习为下游预测请求信息 \[5, 6\]。面向 CAI 评估，需要进一步将空间图像区域请求连接到每个观测前缀上的连续力学输出。本文关注的问题是：如何利用强度监督协同组织区域选择与信息到达，使采集和回归共同服务于整条强度估计序列。

We introduce strength-assessment-driven multimodal sequential observation. Initially available surface descriptors establish specimen-specific context. A regional policy combines this context with acquired internal descriptors, spatial position, measurement status, acquisition history, progress and the current CAI estimate to select the next feasible region. Each observation updates a common strength predictor and the state used for the next decision (Fig. 1). Training combines process-average absolute error with terminal error, assigning value to both early useful estimates and final prediction quality.

本文提出强度评估驱动的多模态序贯观测方法。初始可用的表面描述符建立试样相关上下文，区域策略结合已获取内部描述符、空间位置、测量状态、采集历史、进度及当前 CAI 估计，选择下一个可采集区域。每次观测更新共同强度预测器，并形成下一次决策的状态（图 1）。训练目标联合过程平均绝对误差与终点误差，同时优化有用估计的早期形成和最终预测质量。

The study makes three connected contributions. First, it formulates regional C-scan acquisition as a strength-supervised decision task in which every observed prefix produces a mechanical estimate. Second, the multimodal observable state and process-plus-terminal objective jointly organize which regions are requested and when their information enters a common predictor. Third, a frozen common-predictor evaluation links matched-cap accuracy, matched-quality information requirements and fixed-final-set order interventions, while retaining conditional uncertainty and adverse cases. These contributions concern task formulation, observation organization and supporting evidence.

本文形成三项相互衔接的贡献。首先，将区域 C-scan 采集建模为强度监督决策任务，使每个观测前缀均产生力学估计。其次，通过多模态可观测状态及过程与终点联合目标，共同组织请求哪些区域以及其信息何时进入共同预测器。最后，以冻结的共同预测器评价连接相同上限准确性、等质量信息需求与固定最终集合的次序干预，同时保留条件不确定性及不利案例。三项贡献分别对应任务建模、观测组织与支持证据。

The study evaluates this formulation through regional image replay on a paired composite-laminate dataset. Section 2 positions the work in image-informed CAI prediction and sequential acquisition. Section 3 presents the representation, predictor and policy-learning objective; Section 4 specifies the experimental protocol. Section 5 examines accuracy, acquisition efficiency and the behavior underlying earlier useful estimates, followed by the conclusions in Section 6.

本文通过配对复合材料层合板数据上的区域图像回放评价上述方法。第 2 节介绍图像支持的 CAI 预测与序贯采集研究；第 3 节给出状态表示、强度预测器和策略学习目标，第 4 节说明实验协议；第 5 节分析预测准确性、采集效率及较早形成有用估计的行为机制，第 6 节总结研究结论。

# 2. Related work / 相关研究

## 2.1. Image-informed compression-after-impact strength assessment / 图像支持的冲击后压缩强度评估

Image-informed strength assessment learns the relationship between an observed damage state and the subsequent mechanical response. Hasebe and colleagues released paired surface and internal-damage images after impact, together with subsequent CAI measurements \[1, 2\]. These resources support specimen-level investigation of complementary image modalities: the surface describes externally visible morphology, while C-scan images provide a spatial description of internal damage. Their pairing with mechanical labels makes it possible to evaluate how the supplied image information contributes to quantitative strength estimation.

图像支持的强度评估学习观测损伤状态与后续力学响应之间的关系。Hasebe 及同事公开了冲击后的配对表面与内部损伤图像，以及后续 CAI 测量结果 \[1, 2\]。这些资源支持在试样层面研究互补图像模态：表面图像描述外部可见形貌，C-scan 图像提供内部损伤的空间描述。图像与力学标签的配对，使研究者能够评价所提供的图像信息对定量强度估计的作用。

Mack and colleagues used a ResNet18-based model to predict impact energy and CAI strength from C-scan images \[3\]. This work establishes a close engineering precedent for learning image-to-strength relationships. We extend the assessment question from reading available image content to selecting the regional content that becomes available during prediction. A common strength predictor provides the readout for this comparison, so the evaluated change is the organization of its observations. Final accuracy describes the selected content at completion, and the prediction trajectory describes its usefulness as acquisition progresses.

Mack 及同事利用基于 ResNet18 的模型，从 C-scan 图像预测冲击能量与 CAI 强度 \[3\]，为学习图像与强度之间的关系提供了接近的工程研究基础。本文将评估问题从读取已有图像内容，拓展到选择在预测过程中逐步可用的区域内容。共同强度预测器为这一比较提供统一读出，因此被评价的变化是向其提供观测的组织方式。终点准确性描述最终选定内容的作用，预测轨迹则描述这些内容随采集推进而发挥的作用。

## 2.2. Sequential inspection and task-driven acquisition / 序贯检查与任务驱动采集

Sequential inspection treats the next measurement as a decision informed by previous observations. Fuentes and colleagues combined Bayesian optimization with robust outlier analysis to select ultrasonic observations from an evolving novelty field \[4\]. Their observation--update--decision loop connects spatial sampling to damage indication and probabilistic inspection. The present method adopts this regional-request perspective and links the decision criterion to measured CAI strength, making the sequence of continuous strength estimates the object of optimization.

序贯检查将下一次测量视为由已有观测支持的决策。Fuentes 及同事结合贝叶斯优化与稳健异常分析，根据不断更新的新颖性场选择超声观测 \[4\]。其观测---更新---决策循环将空间采样与损伤指示及概率检查连接起来。本文沿用区域请求这一决策视角，将选择准则进一步连接到实测 CAI 强度，并将连续强度估计序列作为优化对象。

Sequential inspection couples each request to an evolving information state. For strength assessment, the quality of that state can be evaluated through a continuous mechanical output after every completed regional observation. This task criterion motivates the present comparison between spatial allocation rules and a specimen-conditioned acquisition policy under a common predictor.

序贯检查将每次请求连接到不断演化的信息状态。对于强度评估，可以在每个区域观测完成后，通过连续力学输出评价该状态的质量。这一任务准则引出了本文在共同预测器下，对空间分配规则与试样条件化采集策略的比较。

## 2.3. Active feature acquisition and prediction with partial observations / 2.3. 主动特征获取与部分观测预测

Active feature acquisition learns which information to request for a downstream prediction task. Janisch and colleagues formulated costly-feature classification through sequential requests [5]. Covert and colleagues studied dynamic feature selection through conditional mutual information and amortized learning [6]. These approaches provide a decision perspective for assessing the value of information before all inputs are observed. Here, the requested units are C-scan regions and the output is continuous CAI strength throughout acquisition.

主动特征获取学习为下游预测任务请求哪些信息。Janisch 等将高成本特征分类建模为序贯请求 [5]；Covert 等通过条件互信息与摊销学习研究动态特征选择 [6]。这些研究提供了在全部输入尚未观测时评价信息价值的决策视角。本文请求单元为 C-scan 区域，输出为贯穿采集过程的连续 CAI 强度。

Partial observation also requires distinguishing available evidence from missing input. Residual-network image representations [7] and attention-based contextualization [8] supply building blocks for representing visible content and comparing candidate requests. The present method uses surface context and progressively revealed internal information for this purpose. Section 3 defines its specific masks, tokens and prediction interface; the experiments assess the complete observation-selection mechanism.

部分观测还要求区分已有证据与缺失输入。残差网络图像表示 [7] 与基于注意力的上下文化 [8]，为表示可见内容及比较候选请求提供基础构件。本文据此利用表面上下文与逐步揭示的内部信息，第 3 节定义具体掩膜、token 及预测接口，实验评价完整观测选择机制。

# 3. Strength-assessment-driven multimodal sequential observation / 强度评估驱动的多模态序贯观测

## 3.1. Framework overview: selecting regions and organizing their arrival / 方法总览：区域选择与信息到达组织

The proposed method couples a regional acquisition policy with a strength predictor to produce an ordered observation set and a CAI prediction trajectory. Before internal acquisition, surface descriptors supply specimen-specific context and support an initial strength estimate. The policy scores feasible unmeasured regions, requests one region and receives its internal content. The predictor updates the CAI estimate, while the measurement mask, acquisition history and acquired-area fraction update the state for the next decision. Figure 1 organizes this loop into surface representation, regional policy, observation/state update and strength assessment.

本文方法将区域采集策略与强度预测器相结合，输出带次序的观测集合及 CAI 预测轨迹。在内部采集开始前，表面描述符提供试样相关上下文并支持初始强度估计。策略为可采集的未测区域评分，请求一个区域并接收其内部内容；预测器随后更新 CAI，测量掩膜、采集历史与已获取面积比例同步构成下一次决策的状态。图 1 将该循环组织为表面表示、区域策略、观测及状态更新、强度评估四个模块。

Specimen-level strength supervises this sequence through its assessment cost. Each region gains training value from its effect on the estimates produced after acquisition, allowing regional preferences to be learned from specimen-level labels. The policy outputs a regional index, and the predictor outputs strength in MPa. Both models have fixed weights during evaluation; adaptation occurs through newly available observations and updated regional scores. Region selection and order emerge from the same decisions: requesting a region determines both its membership in the observed set and the prefixes in which it will be available.

试样级实测强度通过评估代价监督这一序列。区域的训练价值来自其被采集后对强度估计的作用，因此试样级标签即可支持区域偏好的学习。策略输出区域编号，预测器输出 MPa 单位的强度。评价时两个模型的权重保持固定，适应性体现为新增观测及区域评分的更新。区域选择与获取次序由同一系列决策产生：请求一个区域，同时确定它是否进入观测集合，以及从哪些预测前缀开始可用。

![Figure 1](figures/F1.png){width=6.5in}

**Figure 1.** Assessment-driven sequential observation. With fixed model weights, a regional request reveals internal content, updates the observable state and produces a new CAI estimate. During learning, measured specimen strength evaluates process-average and terminal errors. This task cost updates the policy and value head while the image encoder and strength readouts remain frozen. Execution uses the observable state; strength labels are used for training and retrospective evaluation.

**图 1。** 评估驱动的序贯观测。在模型权重固定时，区域请求揭示内部内容、更新可观测状态并产生新的 CAI 估计。学习时，试样实测强度评价过程平均误差与终点误差，任务代价更新策略及价值头，图像编码器与强度读出保持冻结。执行使用可观测状态，强度标签用于训练及事后评价。

## 3.2. Multimodal observable-state representation / 多模态可观测状态表示

A fixed regional representation connects requests to the same specimen locations throughout acquisition. Each specimen has a registered surface image and a paired two-dimensional C-scan raster. Orientation and outer cropping establish correspondence in normalized specimen coordinates. Both modalities are partitioned into an $8 \times 8$ grid with row-major indices $\mathcal{I} = \{ 0,\ldots,63\}$. These indices retain their spatial meaning throughout the episode, including when the native raster sizes differ. Each acquisition reveals one complete internal cell.

固定区域表示使采集请求在整个过程中始终对应同一试样位置。每件试样具有已登记的表面图像与配对二维 C-scan 图像，方向及外部裁切在归一化试样坐标下建立对应。两种模态均划分为 $8 \times 8$ 网格，采用逐行编号 $\mathcal{I} = \{ 0,\ldots,63\}$。即使两种原生图像尺寸不同，这些编号也在整条轨迹中保持相同空间含义。一次采集揭示一个完整内部格。

A frozen ResNet18 encoder maps each independently cropped region to a 512-dimensional descriptor \[7\]. Regional cropping precedes encoding, preserving the local support of each requested descriptor. Surface preprocessing fixes orientation and limits the longest edge to 1024 pixels. All 64 surface descriptors $S = \left( s_{i} \right)_{i \in \mathcal{I}}$ are available initially; internal descriptors $x_{i}$ become available through regional requests. The replay environment stores the descriptor bank, and a measurement mask controls which internal descriptors enter the models.

冻结的 ResNet18 编码器将每个独立裁切区域映射为 512 维描述符 \[7\]。区域裁切先于编码，以保持请求描述符的局部支持范围。表面预处理固定方向，并将最长边限制为 1024 像素。全部 64 个表面描述符 $S = \left( s_{i} \right)_{i \in \mathcal{I}}$ 最初可用，内部描述符 $x_{i}$ 则通过区域请求逐步可用。回放环境保存描述符库，测量掩膜控制哪些内部描述符能够进入模型。

Native pixel area defines the acquisition scale. For an internal crop of $H_{img} \times W_{img}$ pixels, region $i$ has support $\Omega_{i}$ and cost $w_{i} = \left| \Omega_{i} \right|/\left( H_{img}W_{img} \right)$, with $\sum_{i}^{}w_{i} = 1$. Let $\mathcal{M}_{t}$ be the observed set after $t$ acquisitions, $M_{t,i}$ its binary indicator and $c_{t} = \sum_{i \in \mathcal{M}_{t}}^{}w_{i}$ the acquired fraction. Under a prescribed cap $B$, feasible actions are

采集规模按原生像素面积定义。对于 $H_{img} \times W_{img}$ 像素的内部裁切图，区域 $i$ 的支持为 $\Omega_{i}$，成本为 $w_{i} = \left| \Omega_{i} \right|/\left( H_{img}W_{img} \right)$，满足 $\sum_{i}^{}w_{i} = 1$。令 $\mathcal{M}_{t}$ 表示获取 $t$ 个区域后的已测集合，$M_{t,i}$ 为其二值指示，$c_{t} = \sum_{i \in \mathcal{M}_{t}}^{}w_{i}$ 为已获取比例。在预设上限 $B$ 下，可行动作集合为

$$\mathcal{L}_{t} = \{ i \in \mathcal{I}:M_{t,i} = 0,\ c_{t} + w_{i} \leq B\}.\quad\quad(1)$$

Integer raster boundaries make cell areas slightly unequal, so the cap $B = 0.25$ admits 15 or 16 whole cells. Selection uses the known cell geometry and the remaining allowance; acquisition ends when no whole unmeasured cell fits. The history value $H_{t,i}$ is zero for an unmeasured region and otherwise equals its acquisition ordinal divided by 64. The observable policy state is

整数栅格边界使各格面积略有差异，因此 $B = 0.25$ 可容纳 15 或 16 个完整格。选择依据已知格几何和剩余面积进行，当剩余上限无法容纳任何完整未测格时结束。历史值 $H_{t,i}$ 在未测区域为零，在已测区域等于其获取序号除以 64。策略的可观测状态为

$$z_{t} = \left( S,X_{obs,t},M_{t},H_{t},{\widehat{y}}_{t},c_{t},B - c_{t} \right).\quad\quad(2)$$

The current estimate ${\widehat{y}}_{t}$ is computed from the available inputs. For regional scoring, a learned unknown descriptor $u$ represents unavailable internal content through ${\widetilde{x}}_{t,i} = M_{t,i}x_{i} + \left( 1 - M_{t,i} \right)u$, alongside the explicit measurement indicator. Surface descriptors establish initial context, acquired internal descriptors add evidence, position identifies requests, and history and progress describe the sequence. Together with the current strength estimate, these inputs allow priorities to depend on the specimen and its observation state (Fig. 2).

当前估计 ${\widehat{y}}_{t}$ 由可用输入计算。区域评分通过学习到的未知描述符 $u$ 表示尚不可用的内部内容，即 ${\widetilde{x}}_{t,i} = M_{t,i}x_{i} + \left( 1 - M_{t,i} \right)u$，并同时保留显式测量指示。表面描述符建立初始上下文，已获取内部描述符补充证据，位置标识请求，历史及进度描述序列。这些输入与当前强度估计共同支持依赖试样及其观测状态的区域优先级（图 2）。

![Figure 2](figures/F2.png){width=6.5in}

**Figure 2.** Observable-state representation and regional scoring. The actor combines known surface features, measured internal content or a learned unknown descriptor, mask, position and ordinal. Sixty-four regional tokens and one global token are contextualized by a two-layer, four-head Transformer. The 64 scores are restricted by a mask of unmeasured cells affordable within the remaining native area. The independent set-based predictor supplies the current strength estimate. Area and measured-cell fractions remain distinct; the two constant-zero channels per input block are detailed in S2. Diagrammatic connections specify information flow, not new data.

**图 2。** 可观测状态表示与区域评分。策略结合已知表面特征、已测内部内容或学习到的未知描述符、掩膜、位置及序号。64 个区域 token 与 1 个全局 token 经两层四头 Transformer 上下文化，再按剩余原生面积可容纳的未测格掩膜限制 64 个评分。独立的集合式预测器提供当前强度估计，面积比例与已测格比例保持区分；每个输入块的两个恒零通道详见 S2。示意连线规定信息流，不表示新数据。

## 3.3. Strength estimation throughout the observation process / 贯穿观测过程的强度估计

A common predictor converts each observed prefix into a CAI estimate, including the initial surface-only state. Its input is ${\widehat{y}}_{t} = f_{\phi}\left( S,X_{obs,t},M_{t},c_{t} \right)$. Separate affine projections $P_{s}\left( s_{i} \right) = W_{s}s_{i} + b_{s}$ and $P_{x}\left( M_{t,i}x_{i} \right) = W_{x}\left( M_{t,i}x_{i} \right) + b_{x}$ map the two modalities to width 64. Internal descriptors are masked before projection, retaining the affine bias. With normalized coordinates $p_{i} = \left( r_{i}/7,k_{i}/7 \right)$, a two-layer GELU multilayer perceptron produces a fused regional representation:

共同预测器将每个观测前缀转换为 CAI 估计，包括初始仅表面状态，其输入关系为 ${\widehat{y}}_{t} = f_{\phi}\left( S,X_{obs,t},M_{t},c_{t} \right)$。独立仿射投影 $P_{s}\left( s_{i} \right) = W_{s}s_{i} + b_{s}$ 和 $P_{x}\left( M_{t,i}x_{i} \right) = W_{x}\left( M_{t,i}x_{i} \right) + b_{x}$ 将两种模态映射至 64 维。内部描述符在投影前施加掩膜，保留仿射偏置。结合归一化坐标 $p_{i} = \left( r_{i}/7,k_{i}/7 \right)$，两层 GELU 多层感知机生成融合区域表示：

$$h_{t,i} = \psi_{\phi}\left( \left\lbrack P_{s}\left( s_{i} \right),P_{x}\left( M_{t,i}x_{i} \right),M_{t,i},p_{i} \right\rbrack \right).\quad\quad(3)$$

Two complementary summaries support regression. The all-cell mean preserves surface context and the represented observation state over the grid. The observed-cell mean focuses on acquired regions and is zero before internal acquisition. A regression head combines these summaries with the acquired fraction and converts its normalized output to MPa using training-only target statistics $\mu_{y}$ and $\sigma_{y}$:

两个互补汇总共同支持回归。全格均值保留整个网格上的表面背景及观测状态，已测格均值集中于已获取区域，并在内部采集开始前为零。回归头结合两个汇总及已获取比例，利用仅由训练数据计算的目标统计量 $\mu_{y}$ 与 $\sigma_{y}$，将归一化输出转换为 MPa：

$${\overline{h}}_{t} = \frac{1}{64}\sum_{i}^{}h_{t,i},\quad\quad{\overline{h}}_{t}^{obs} = \frac{\sum_{i}^{}M_{t,i}h_{t,i}}{\max\left( 1,\sum_{i}^{}M_{t,i} \right)}.\quad\quad(4)$$

$${\widehat{y}}_{t} = \mu_{y} + \sigma_{y}\rho_{\phi}\left( \left\lbrack {\overline{h}}_{t},{\overline{h}}_{t}^{obs},c_{t} \right\rbrack \right).\quad\quad(5)$$

Predictor training covers the information levels encountered during acquisition. The mask distribution allocates 10% to empty internal inputs, 60% to 1--16 cells from random or fixed geometric orders, 20% to 17--48 randomly selected cells and 10% to complete internal input. A Huber loss on the standardized residual trains the regressor. Three capture-group cross-fitted predictors provide training-side trajectory costs, with each training specimen evaluated by the predictor that excludes its fold. Evaluation uses one common frozen predictor fitted on the training partition.

预测器训练覆盖采集过程中出现的信息规模。掩膜分布包括 10% 空内部输入、60% 来自随机或固定几何次序的 1--16 格、20% 随机选取的 17--48 格，以及 10% 完整内部输入。回归使用标准化残差的 Huber 损失训练。三份按捕获组交叉拟合的预测器提供训练侧轨迹代价，每件训练试样由排除其所属折的预测器评价。正式比较使用在训练分区拟合的同一个冻结预测器。

Both means in Eq. (4) summarize a set: the all-cell mean includes the masked-input affine bias, and the observed-cell mean uses only measured cells. Acquisition history enters the policy, not this predictor. Identical final sets and acquired fractions therefore produce identical terminal inputs. Endpoint equality in the fixed-set intervention is a structural invariant; order can affect accuracy only at intermediate prefixes.

式（4）的两个均值均汇总集合：全格均值包含掩膜输入的仿射偏置，已测格均值只使用已测格。采集历史进入策略而不进入该预测器。因此，相同最终集合与获取比例对应相同终点输入。固定集合干预中的终点一致性是结构不变量，次序只能影响中间前缀的准确性。

## 3.4. State-conditioned regional scoring / 状态条件化区域评分

The regional policy compares candidates in the context of evidence already obtained elsewhere. It constructs 64 regional tokens and one global token, then contextualizes them jointly with a spatial Transformer \[8\]. Each regional token combines a surface projection, the visible or unknown internal projection, normalized coordinates, measurement status and acquisition history. Fixed position indices maintain each candidate's relation to the observed grid. The global token supplies task context through the current strength estimate and acquisition progress.

区域策略在其他位置已获得证据的上下文中比较候选区域。它构造 64 个区域 token 与 1 个全局 token，并利用空间 Transformer 进行联合上下文化 \[8\]。每个区域 token 结合表面投影、可见或未知内部投影、归一化坐标、测量状态及采集历史。固定位置编号维持候选区域与已测网格之间的关系，全局 token 则通过当前强度估计和采集进度提供任务背景。

Surface and internal projections each have width 48. Regional inputs are mapped to 128-dimensional tokens, and global inputs are mapped separately to the same width. The global values comprise acquired-area fraction, remaining allowance, standardized strength prediction and measured-cell fraction. Retaining both area and cell fractions represents the slightly unequal native cell costs. Two encoder layers, four attention heads and a feed-forward width of 256 contextualize the 65 tokens. The actor uses the learned unknown descriptor for missing internal content; the predictor uses the pre-projection zero mask described in Section 3.3.

表面与内部投影各为 48 维。区域输入映射为 128 维 token，全局输入独立映射至相同宽度。全局值包括已获取面积比例、剩余比例、标准化强度预测和已测格比例，同时保留面积比例与格比例以表示略有差异的原生格成本。两层编码器、四个注意力头及 256 维前馈层对 65 个 token 进行上下文化。策略使用学习到的未知描述符表示缺失内部内容，预测器采用第 3.3 节所述的投影前零掩膜。

A shared scoring head receives the concatenated contextual regional and global tokens and returns $s_{\theta}\left( z_{t},i \right)$. A separate value head reads the global token during training. At evaluation, the next requested region is the highest-scoring feasible candidate:

共享评分头读取上下文化区域 token 与全局 token 的拼接，输出 $s_{\theta}\left( z_{t},i \right)$；训练时另有价值头读取全局 token。评价中的下一请求为评分最高的可行候选区域：

$$a_{t} = {argmax}_{i \in \mathcal{L}_{t}}s_{\theta}\left( z_{t},i \right).\quad\quad(6)$$

The feasibility mask excludes measured cells and cells that exceed the remaining allowance. Training samples actions from a categorical softmax over feasible scores, allowing regional preferences to be explored. These probabilities describe selection preferences, and the value head estimates remaining policy cost. New observations, acquisition history and the current estimate can revise candidate priorities; the recorded-action analysis in Section 5.4 characterizes the revisions realized during execution.

可行性掩膜排除已测格及超过剩余面积的格。训练根据可行评分上的分类 softmax 分布采样动作，以探索区域偏好。其中，动作概率表示选择偏好，价值头估计策略的剩余代价。新增观测、采集历史与当前估计能够更新候选优先级，第 5.4 节利用实际动作记录分析执行中出现的调整。

## 3.5. Learning assessment relevance and information timing / 学习评估相关性与信息时机

### 3.5.1. Task supervision and trajectory objective / 任务监督与轨迹目标

A training episode samples a specimen, acquires feasible regions and updates strength estimates through its cross-fitted predictor. The measured training strength $y$ evaluates the resulting trajectory through $e_{t} = \left| {\widehat{y}}_{t} - y \right|$. Predictions update when a region is fully observed and remain constant between observation completions. If acquisition finishes at $c_{T} \leq B$, the last estimate is carried through the remaining interval to $B$. The acquisition-process average absolute error is

一条训练轨迹采样一件试样，获取可行区域，并通过对应交叉拟合预测器更新强度估计。训练试样的实测强度 $y$ 通过 $e_{t} = \left| {\widehat{y}}_{t} - y \right|$ 评价所形成的轨迹。预测在区域观测完成时更新，并在两次完成之间保持不变。当采集在 $c_{T} \leq B$ 结束时，最后估计延续至 $B$。采集过程平均绝对误差定义为

$$A(B) = \frac{1}{B}\int_{0}^{B}e(c)\, dc = \frac{\sum_{t = 0}^{T - 1}\left( c_{t + 1} - c_{t} \right)e_{t} + \left( B - c_{T} \right)e_{T}}{B}.\quad\quad(7)$$

The learning objective combines this process criterion with terminal accuracy:

学习目标将过程准则与终点准确性结合：

$$J = A(B) + \lambda_{T}e_{T},\quad\quad\lambda_{T} = 0.25.\quad\quad(8)$$

Both terms are expressed in MPa before training-scale normalization. The carry interval gives every trajectory the same integration range, including when the remaining allowance accommodates no whole cell. The terminal term preserves an incentive for final accuracy, including the effect of regions completed near the cap. This objective learns assessment relevance directly from the ability of the acquired observations to support prediction of measured specimen strength.

两项在训练尺度归一化之前均以 MPa 表示。最后预测的延续区间使全部轨迹具有相同积分范围，包括剩余面积无法容纳完整格的情况。终点项保留对最终准确性的优化作用，覆盖接近上限才完成的区域观测。该目标直接依据已获取观测对试样实测强度预测的支持作用，学习评估相关性。

### 3.5.2. Process value of earlier useful observations / 早期有效观测的过程价值

The trajectory objective makes the value of information arrival explicit. Let $\Delta e_{t} = e_{t - 1} - e_{t}$ denote the signed error decrease after the observation completed at $c_{t}$. Expanding the interval sum gives

轨迹目标明确刻画了信息到达的价值。令 $\Delta e_{t} = e_{t - 1} - e_{t}$ 表示在 $c_{t}$ 完成观测后的带符号误差下降。展开区间求和可得

$$A(B) = e_{0} - \sum_{t = 1}^{T}\left( 1 - \frac{c_{t}}{B} \right)\Delta e_{t},\quad\quad(9)$$

$$J = \left( 1 + \lambda_{T} \right)e_{0} - \sum_{t = 1}^{T}\left( 1 - \frac{c_{t}}{B} + \lambda_{T} \right)\Delta e_{t}.\quad\quad(10)$$

For a fixed specimen and predictor, the initial error is constant. An error decrease achieved at a smaller acquired fraction receives a larger process weight because it remains available over a longer part of the acquisition interval. Signed increases in error enter with the opposite effect. At $c_{t} = B$, the process weight is zero and the terminal weight remains $\lambda_{T}$. Thus, the objective favors useful information early while continuing to optimize endpoint quality.

对于固定试样和预测器，初始误差保持不变。在较小获取比例形成的误差下降具有更大过程权重，因为其作用覆盖更长的剩余采集区间；误差增加以相反方向进入目标。在 $c_{t} = B$ 处，过程权重为零，终点权重仍为 $\lambda_{T}$。因此，该目标在持续优化终点质量的同时，赋予较早获得的有效信息更高过程价值。

Supplementary S2 derives Eqs. (9–10) from Eq. (7), including the final carry interval. Their signed event contributions accommodate unequal native cell areas and prediction changes in either direction. Section 5.3 tests realized changes using archived events and fixed-set replay.

补充 S2 从式（7）推导式（9–10），并保留最后延续区间。带符号事件贡献适用于不等原生格面积及双向预测变化。第 5.3 节利用归档事件与固定集合回放检验实际变化。

### 3.5.3. Updating regional preferences from task cost / 以任务代价更新区域偏好

For a sampled decision at state $t$, the normalized remaining cost includes subsequent interval errors and terminal error:

对于状态 $t$ 处的采样决策，归一化剩余代价包括后续区间误差及终点误差：

$$G_{t} = \frac{1}{\sigma_{y}}\left\lbrack \sum_{k = t}^{T - 1}\frac{c_{k + 1} - c_{k}}{B}e_{k} + \left( \frac{B - c_{T}}{B} + \lambda_{T} \right)e_{T} \right\rbrack.\quad\quad(11)$$

With value estimate $v_{\theta}\left( z_{t} \right)$, the actor minimizes the score-function surrogate

给定价值估计 $v_{\theta}\left( z_{t} \right)$，策略最小化评分函数代理损失

$$\mathcal{L}_{actor} = \mathbb{E}\left\lbrack \sum_{t = 0}^{T - 1}\log\pi_{\theta}\left( a_{t} \mid z_{t} \right)\, stopgrad\left( G_{t} - v_{\theta}\left( z_{t} \right) \right) \right\rbrack.\quad\quad(12)$$

The positive sign implements cost minimization: a sampled action whose cost exceeds the value baseline receives a probability-reducing update. Actor terms are summed over decisions within an episode and averaged across the batch. Value squared errors and categorical entropy are averaged within episodes before batch aggregation. The total loss is $\mathcal{L}_{actor} + 0.5\mathcal{L}_{value} - \beta\mathcal{H}$, with $\beta$ decreasing linearly from 0.01 to zero. The encoder and strength predictors remain frozen, so policy updates learn which observations to supply to a fixed assessment model.

正号对应成本最小化：采样动作的代价高于价值基线时，更新会降低其概率。策略项在轨迹内按决策求和，再跨批次平均；价值平方误差和分类熵先在轨迹内平均，再进行批次聚合。总损失为 $\mathcal{L}_{actor} + 0.5\mathcal{L}_{value} - \beta\mathcal{H}$，其中 $\beta$ 从 0.01 线性降至零。编码器和强度预测器保持冻结，策略更新由此学习向固定评估模型提供哪些观测。

## 3.6. Sequential execution and prediction updates / 序贯执行与预测更新

Execution begins with an empty internal mask, zero history and zero acquired fraction. The surface-only estimate provides the initial task state. At each iteration, the policy constructs the feasible set, scores candidates, requests the highest-scoring region and receives its descriptor. The predictor then updates CAI, and the new observation, acquisition ordinal, completion fraction and prediction are recorded. Acquisition ends when the feasible set is empty. Since each cell is requested at most once, an episode has at most 64 actions.

执行从空内部掩膜、零历史及零已获取比例开始，仅表面估计提供初始任务状态。每轮构造可行集合、为候选评分、请求最高评分区域并接收其描述符，随后更新 CAI，记录新增观测、获取序号、完成比例及预测。可行集合为空时结束采集。每个格最多请求一次，因此一条轨迹最多包含 64 个动作。

**Algorithm 1. Learning and sequential execution**

**算法 1. 学习与序贯执行**

**1.** Fit the common predictor and three capture-group cross-fitted training predictors on the designated training data; freeze the image encoder and readouts.

**1.** 利用指定训练数据拟合共同预测器与三份按捕获组交叉拟合的训练预测器，冻结图像编码器及强度读出。

**2.** Train the regional policy on sampled feasible trajectories using $G_{t}$, the value baseline and entropy regularization. Select its checkpoint by development-validation $A$.

**2.** 利用 $G_{t}$、价值基线及熵正则，在采样的可行轨迹上训练区域策略，按开发验证 $A$ 选择 checkpoint。

**3.** For execution, fix all weights. Receive $S$, cell costs and $B$; initialize $M_{0} = H_{0} = 0$, $c_{0} = 0$ and ${\widehat{y}}_{0} = f_{\phi}(S,0,0,0)$.

**3.** 执行时固定全部权重，接收 $S$、格成本与 $B$，初始化 $M_{0} = H_{0} = 0$、$c_{0} = 0$ 及 ${\widehat{y}}_{0} = f_{\phi}(S,0,0,0)$。

**4.** While $\mathcal{L}_{t} \neq \varnothing$, select $a_{t}$ by Eq. (6), obtain $x_{a_{t}}$ and update $M_{t + 1,a_{t}} = 1$, $H_{t + 1,a_{t}} = (t + 1)/64$, $c_{t + 1} = c_{t} + w_{a_{t}}$.

**4.** 当 $\mathcal{L}_{t} \neq \varnothing$ 时，按式（6）选择 $a_{t}$，获得 $x_{a_{t}}$，更新 $M_{t + 1,a_{t}} = 1$、$H_{t + 1,a_{t}} = (t + 1)/64$、$c_{t + 1} = c_{t} + w_{a_{t}}$。

**5.** Update ${\widehat{y}}_{t + 1}$ and $z_{t + 1}$; record the requested region, completion fraction and prediction. Return the ordered observed set and complete prediction trajectory.

**5.** 更新 ${\widehat{y}}_{t + 1}$ 与 $z_{t + 1}$，记录请求区域、完成比例和预测，返回有序观测集合及完整预测轨迹。

Policy learning uses subsequent task cost to update sampled choices through the stopped-gradient advantage in Eq. (12); the value baseline reduces variation in this feedback. Evaluation uses deterministic feasible-score selection with fixed parameters, giving a reproducible sequence for a fixed specimen, checkpoint, preprocessing and cap. Measured strength enters training or retrospective scoring, while execution reads only the observable state. In the present implementation, each request reveals a stored two-dimensional C-scan regional descriptor.

策略学习通过式（12）中的停止梯度优势，利用后续任务代价更新采样选择，价值基线用于降低反馈波动。评价以固定参数执行确定性的可行最高评分选择，因此在试样、checkpoint、预处理及上限固定时，能够产生可复现序列。实测强度用于训练或事后评分，执行仅访问可观测状态。在当前实现中，每次请求揭示一个已保存的二维 C-scan 区域描述符。

# 4. Experimental setup / 实验设置

## 4.1. Paired images, mechanical labels and evaluation units / 配对图像、力学标签与评价单位

The study uses publicly released surface and internal-damage images of impacted carbon-fiber-reinforced polymer laminates, paired with measured CAI strength \[1, 2, 9\]. The assembled cohort contains 276 physical specimens from six source groups covering cross-ply and quasi-isotropic layups with 8, 16 and 24 layers. Each record connects a specimen identifier, a registered image pair and a mechanical label in MPa. Regional acquisition is evaluated through offline revelation of the two-dimensional C-scan raster; the strength labels are the source authors' mechanical measurements.

本文使用公开的碳纤维增强聚合物层合板冲击后表面与内部损伤图像，并配对实测 CAI 强度 \[1, 2, 9\]。汇总队列包含六个来源组的 276 件物理试样，覆盖 8、16 和 24 层的交叉铺层及准各向同性铺层。每条记录连接试样编号、登记图像对及 MPa 单位的力学标签。区域采集通过二维 C-scan 图像的离线逐区域揭示进行评价，强度标签来自源作者的力学测量。

Capture groups preserve dependencies associated with shared source images, paths or crop identities. Groups are formed before partitioning, and all members remain in one partition. Deterministic ordering within each source group assigns approximately 60%, 20% and 20% of capture groups to training, development validation and a reserved partition. This produces 161, 50 and 65 physical specimens, respectively, comprising 152, 48 and 59 capture groups. Table 1 reports the specimen counts for each configuration.

捕获组用于保留共享源图像、路径或裁切身份所形成的依赖关系。分组在数据划分之前建立，同组成员保持在同一分区。各来源组内采用确定性排序，将约 60%、20% 和 20% 的捕获组分配至训练、开发验证与预留分区，分别得到 161、50 和 65 件物理试样，对应 152、48 和 59 个捕获组。表 1 给出各配置的试样数量。

**Table 1.** Data configurations and physical-specimen split. Capture groups remain within partitions; all reported method comparisons use the same 50 development-validation specimens.

**表 1。** 数据配置及物理试样划分。捕获组保持在分区内；全部方法比较使用同一 50 件开发验证试样。

| Laminate | Train | Dev. | Reserved | Total |
| --- | --- | --- | --- | --- |
| 8-layer cross-ply | 25 | 9 | 11 | 45 |
| 24-layer quasi-isotropic | 29 | 9 | 11 | 49 |
| 16-layer quasi-isotropic | 25 | 8 | 10 | 43 |
| 24-layer cross-ply | 35 | 9 | 15 | 59 |
| 16-layer cross-ply | 25 | 8 | 9 | 42 |
| 8-layer quasi-isotropic | 22 | 7 | 9 | 38 |
| Total | 161 | 50 | 65 | 276 |

All reported effects use the 50 development-validation specimens, which also support checkpoint selection. The estimates therefore characterize the selected development configuration. The 65 reserved specimens are outside the reported evaluation. The specimen cohort and six diagnostic cases are retained in full. Random-order repetitions, image cells and prediction prefixes remain repeated observations of these specimens. Supplementary S1 records specimen identities, capture groups and partition membership.

全部效果使用上述 50 件开发验证试样，该分区也用于 checkpoint 选择，因此结果刻画的是已选开发配置。65 件预留试样未纳入所报告评价。试样队列及六个诊断案例全部保留；随机次序重复、图格与预测前缀均作为这些试样上的重复观测处理。补充 S1 记录试样身份、捕获组及分区归属。

## 4.2. Acquisition strategies and shared prediction conditions / 采集策略与共同预测条件

Five controls provide distinct ways of allocating the same allowed internal-image area (Table 2). Center-first sorts regions by distance to the grid center. Geometry-spread starts from a corner and repeatedly maximizes the minimum squared distance to previously selected cells, using the region index to break ties. Serpentine alternates horizontal direction between neighboring rows. Random uses five stored seeded permutations of all 64 cells. Learned-static optimizes 64 shared positional logits, giving all specimens the same initial order. The proposed policy uses the multimodal observable state for specimen-conditioned scoring.

五类对照提供分配相同允许内部图像面积的不同方式（表 2）。中心优先按区域到网格中心的距离排序；几何分散从角点开始，反复最大化到已选格的最小平方距离，并以区域编号破除并列；蛇形在相邻行之间交替水平方向；随机采用全部 64 格的五条已保存种子排列；学习固定排序优化 64 个共享位置分数，使全部试样具有相同初始次序。本文策略则利用多模态可观测状态进行试样条件化评分。

**Table 2.** Selection information and execution rules. All sequential strategies share the regional encoder, strength predictor, native-area costs and cap.

**表 2。** 选点信息与执行规则。全部序贯策略共享区域编码器、强度预测器、原生面积成本及比例上限。

| Strategy | Selection input | Rule |
| --- | --- | --- |
| Center-first | Grid position | Center-distance order |
| Geometry-spread | Position and observed locations | Farthest-point expansion |
| Serpentine | Grid position | Alternating row order |
| Random | Saved seeded permutations | Five repeats over 64 cells |
| Learned-static | 64 shared learned logits | Shared position order |
| Proposed | Surface, observed internal content, mask, history, prediction and progress | State-conditioned feasible argmax |
| Full-input reference | All 64 internal cells | Common predictor at fraction 1.0 |

All sequential methods share the frozen regional encoder, common strength predictor, grid geometry, native-area costs and total cap $B = 0.25$. Their strength estimates use the same complete surface information and the internal regions actually acquired. This fixes the readout while varying observation selection. A separate full-input reference supplies all 64 internal cells to the same predictor at area fraction 1.0. It provides an endpoint information-scale reference. Whole-cell acquisition retains each strategy's actual acquired fraction under the common cap.

所有序贯方法共享冻结区域编码器、共同强度预测器、网格几何、原生面积成本和总上限 $B = 0.25$。强度估计使用相同的完整表面信息，以及各策略实际获取的内部区域，由此固定读出并改变观测选择。独立的完整输入参照在面积比例 1.0 处向同一预测器提供全部 64 个内部格，用作终点信息规模参照。完整格采集保留各策略在共同上限下的实际获取比例。

## 4.3. Accuracy, efficiency and mechanism evaluation / 准确性、效率与机制评价

Accuracy is reported at caps $\mathcal{B}_{5} = \{ 0,0.0625,0.125,0.1875,0.25\}$. Each cap reads the last completed prediction at or below that value from the path planned with total cap 0.25. Thus, smaller-cap results query prefixes of the same execution, with the policy's original total-cap input retained. Endpoint MAE and RMSE pool physical specimens, and $R^{2}$ uses the same measured-target variation. For Random, absolute and squared losses are averaged within specimen before pooling; repeat-specific $R^{2}$ values are averaged.

准确性在比例上限 $\mathcal{B}_{5} = \{ 0,0.0625,0.125,0.1875,0.25\}$ 报告。每个上限读取总上限为 0.25 的路径中，不超过该值的最近一次已完成预测。因此，较小上限结果查询的是同一次执行的前缀，策略保持原始总上限输入。终点 MAE 与 RMSE 按物理试样池化，$R^{2}$ 使用相同实测目标变异。随机方法先在试样内平均绝对损失及平方损失再池化，$R^{2}$ 则先逐重复计算再平均。

Process error $A$ and early process error $A(0.0625)$ use a six-source-group-equal estimate: repeats are averaged within specimen, specimens within source group, and the six group means equally. Event-stage quantities use the same weighting. This differs from specimen-pooled endpoint MAE because group sizes differ. Prediction curves retain all saved completion breakpoints and hold the current estimate between events, including increases as well as decreases in error. Actual acquired fractions and repeat-level losses are reported in Supplementary S3.

过程误差 $A$ 与早期过程误差 $A(0.0625)$ 采用六来源组等权估计：先在试样内平均重复，再在来源组内平均试样，最后对六组均值等权平均。事件阶段量采用相同权重。由于各组样本量不同，该聚合与试样池化的终点 MAE 有所区别。预测曲线保留全部已保存完成断点，并在事件之间维持当前估计，包含误差增加和下降。实际获取比例及重复级损失列于补充 S3。

Matched-quality efficiency is the earliest supported common cap at which cohort MAE reaches a target $q$. We evaluate the five-cap grid and the union of recorded event breakpoints separately, denoting either by $\mathcal{B}$. The first-attainment cap and relative cap reduction are

等质量效率定义为队列 MAE 首次达到目标 $q$ 时，所支持的最小共同上限。本文分别使用五点网格及已记录事件断点的并集，统称为 $\mathcal{B}$。首次达标上限与相对上限减少量为

$$b^{\text{*}}(q) = \min\{ b \in \mathcal{B}:MAE(b) \leq q\},\quad\quad R(q) = 1 - \frac{b_{proposed}^{\text{*}}(q)}{b_{control}^{\text{*}}(q)}.\quad\quad(13)$$

The target set contains the inherited integer range 41--61 MPa and the full-input MAE anchor. Relative reduction is defined when both methods reach the target and the control cap is positive. We record first attainment and later recrossings separately. These targets describe cohort prediction quality at a common acquisition scale; they are analysis criteria rather than certified engineering acceptance thresholds or specimen-level stopping decisions.

目标集包含既定 41--61 MPa 整数范围及完整输入 MAE 锚点。当两种方法均达到目标且对照上限为正时，计算相对减少量。首次达标与后续回升分别记录。这些目标描述共同采集规模下的队列预测质量，是分析准则，而非经认证的工程验收阈值或试样级停止决策。

Event analysis uses 500 archived trajectories: 250 Random episodes and 50 from each of the other five strategies, yielding 7,849 acquisition events. Each event contributes a signed raw decrease $\Delta e_{t}$ and a process-weighted contribution $\left( 1 - c_{t}/B \right)\Delta e_{t}$. Completion fractions assign events to four right-closed stages: (0, 0.0625\], (0.0625, 0.125\], (0.125, 0.1875\] and (0.1875, 0.25\]. Events are summed within stage, then aggregated using the repeat, specimen and source-group scheme above. Paired stage contrasts compare the proposed policy with each control.

事件分析使用 500 条归档轨迹，包括 250 条随机轨迹和其余五种策略各 50 条轨迹，共形成 7,849 个采集事件。每个事件给出带符号原始下降 $\Delta e_{t}$ 及过程加权贡献 $\left( 1 - c_{t}/B \right)\Delta e_{t}$。依据完成比例将事件划入四个右闭阶段：(0, 0.0625\]、(0.0625, 0.125\]、(0.125, 0.1875\] 和 (0.1875, 0.25\]。阶段内先对事件求和，再按上述重复、试样与来源组方案聚合，通过配对阶段差比较本文策略及各对照。

The frozen fixed-final-set intervention retains the 15 or 16 cells selected by the proposed policy for each of the same 50 specimens. Original order, complete reversal and five preset hash-determined permutations provide 350 replay conditions. The archived intervention evaluated each intermediate mask with the common frozen predictor. Cumulative area uses integer native pixel counts divided by total crop pixels. The primary contrast is the within-specimen five-permutation mean minus original order; reversal minus original order is the prespecified secondary contrast. These repeated conditions do not increase the independent specimen count. Supplementary S4 provides the saved interventions, direction definitions and endpoint checks.

冻结的固定最终集合干预保留同一 50 件试样中本文策略选定的 15 或 16 格。原序、完全逆序及五次预设哈希置换提供 350 个回放条件。归档干预曾使用共同冻结预测器评价各中间掩膜；本次修订直接复用这些结果。累计面积采用整数原生像素数除以总裁切像素数。主要对比为试样内五次置换均值减原序，预设辅助对比为逆序减原序。这些重复条件不增加独立试样数。补充 S4 给出已保存的干预、方向定义及终点核验。

Paired 95% percentile intervals use 5,000 capture-group bootstrap draws sampled within source groups. Each sampled capture group carries all its specimens with the same multiplicity. MAE retains specimen pooling, while process and stage metrics retain equal source-group weighting. The five fixed-set permutations are averaged within specimen before resampling. These intervals are conditional on the selected checkpoints and are unadjusted for checkpoint selection or multiple comparisons. The independent sampling unit is the capture group, with repeated trajectories retaining their specimen identities.

配对 95% 百分位区间使用来源组内采样的 5,000 次捕获组 bootstrap。每个抽中的捕获组携带其全部试样，并保持相同重复倍数。MAE 保留试样池化，过程与阶段量保留来源组等权；固定集合的五次置换在重采样之前先于试样内平均。区间条件于已选 checkpoint，未校正 checkpoint 选择或多重比较。独立抽样单位为捕获组，重复轨迹保留其试样身份。

## 4.4. Training and checkpoint selection / 训练与 checkpoint 选择

The common mean-aggregation predictor completes 2,000 updates, with update 1,750 selected for evaluation. Three cross-fitted predictors supply the policy's training costs. The actor contains 378,978 parameters, including its training value head. It is optimized with AdamW, learning rate $3 \times 10^{- 4}$, weight decay $10^{- 4}$, batch size 16 and gradient-norm limit 1. The stored architecture retains two zero-valued placeholder channels in each of its two input blocks. Parameterization and model identities are documented in Supplementary S2.

共同均值汇总预测器完成 2,000 次更新，评价采用第 1,750 次更新的模型，三份交叉拟合预测器提供策略训练代价。策略包含训练价值头在内共 378,978 个参数，使用 AdamW 优化，学习率为 $3 \times 10^{- 4}$、权重衰减为 $10^{- 4}$、批次为 16、梯度范数上限为 1。存储架构在两个输入块中各保留两个恒零占位通道。参数化及模型身份记录于补充 S2。

The main policy initialization uses seed 2026091302 and completes 1,250 updates. Candidate checkpoints are compared every 250 updates using development-validation $A$. An earlier candidate is retained unless improvement exceeds $10^{- 12}$, selecting update 1,000. All reported policy results use this single selected initialization. Event and order analyses reuse the frozen predictor and policy choices, with the corresponding candidate scores and settings recorded in Supplementary S2.

主策略初始化种子为 2026091302，共完成 1,250 次更新。每 250 次依据开发验证 $A$ 比较候选，仅当改善超过 $10^{- 12}$ 时替换较早候选，最终选择第 1,000 次更新。所报告策略结果使用这一单一已选初始化。事件和次序分析复用冻结预测器及既有策略选择，相应候选成绩与设置记录于补充 S2。

# 5. Results and discussion / 结果与讨论

## 5.1. Task-driven region selection improves prediction at matched acquisition scales / 任务驱动区域选择改善相同采集规模下的预测

Task-driven observation selection improved CAI prediction under the common readout. At cap 0.25, the proposed method obtained the lowest point estimates of process error, early process error, endpoint MAE and RMSE, and the highest endpoint $R^{2}$, among the six sequential strategies (Table 3). Endpoint MAE decreased from 46.910 MPa for geometry-spread to 42.385 MPa, a reduction of 4.525 MPa or 9.65%. The paired 95% interval for this reduction was \[0.670, 8.276\] MPa. Against learned-static, endpoint MAE improved by 4.608 MPa \[0.656, 8.628\]. These comparisons demonstrate the value of selecting observations for the strength-assessment task within the evaluated configuration.

任务驱动的观测选择改善了共同读出器下的 CAI 预测。在 0.25 上限处，本文方法在六种序贯策略中取得最低的过程误差、早期过程误差、终点 MAE 与 RMSE 点估计，以及最高的终点 $R^{2}$（表 3）。终点 MAE 从几何分散的 46.910 MPa 降至 42.385 MPa，减少 4.525 MPa，即 9.65%，对应减少量的配对 95% 区间为 \[0.670, 8.276\] MPa。相对学习固定排序，终点 MAE 改善 4.608 MPa，区间为 \[0.656, 8.628\] MPa。这些比较展示了在已评价配置下，围绕强度评估任务选择观测的价值。

**Table 3.** Prediction performance. Errors are in MPa. Process $A$ and early $A$ use equal weighting of the six source groups; endpoint metrics pool specimens. Bold values identify the best sequential point estimates. The full-input reference uses area fraction 1.0 and has no acquisition-path metric.

**表 3。** 预测性能。误差单位为 MPa；过程 $A$ 与早期 $A$ 按六来源组等权，终点指标按试样池化。粗体表示序贯方法中的最佳点估计。完整输入参照使用面积比例 1.0，不定义采集路径指标。

| Strategy | A | Early A | MAE | RMSE | R² |
| --- | --- | --- | --- | --- | --- |
| Center-first | 48.475 | 51.821 | 48.027 | 65.223 | 0.577 |
| Geometry-spread | 47.311 | 51.880 | 46.910 | 62.278 | 0.614 |
| Serpentine | 48.911 | 52.170 | 48.013 | 63.041 | 0.605 |
| Random | 49.594 | 54.117 | 46.886 | 61.941 | 0.618 |
| Learned-static | 47.977 | 52.946 | 46.993 | 62.168 | 0.616 |
| **Proposed** | **43.597** | **48.771** | **42.385** | **56.534** | **0.682** |
| Full-input reference (area 1.0) | — | — | 41.690 | 53.944 | 0.711 |

Useful estimates emerged at the first nonzero reporting cap. All strategies started from the same surface-only cohort MAE of 58.551 MPa. At cap 0.0625, the proposed method reached 43.324 MPa with a mean actual acquired fraction of 0.057700, already below all five controls' endpoint MAEs at cap 0.25 (46.886--48.027 MPa). Its MAEs at caps 0.125, 0.1875 and 0.25 were 43.164, 42.928 and 42.385 MPa. This establishes an early prediction advantage along the common-cap evaluation path.

有用估计在首个非零报告上限即已形成。全部策略从相同的仅表面队列 MAE 58.551 MPa 开始。在 0.0625 上限处，本文方法达到 43.324 MPa，平均实际获取比例为 0.057700，已经低于五类对照在 0.25 上限处的终点 MAE（46.886--48.027 MPa）。随后在 0.125、0.1875 和 0.25 上限处的 MAE 分别为 43.164、42.928 和 42.385 MPa，显示出沿共同上限评价路径较早形成的预测优势。

The trajectory-level point estimates also favored the proposed policy. Process error $A$ was 43.597 MPa versus 47.311 MPa for geometry-spread, a reduction of 3.714 MPa or 7.85%. Its paired 95% interval, [−0.337, 7.283] MPa, crosses zero, so this process contrast remains uncertain despite the favorable mean. Relative to learned-static, the reduction was 4.380 MPa [0.199, 8.405]. Table 3 distinguishes this six-source-group-equal process metric from specimen-pooled endpoint MAE.

轨迹级点估计也有利于本文策略。过程误差 $A$ 为 43.597 MPa，几何分散为 47.311 MPa，减少 3.714 MPa，即 7.85%。配对 95% 区间为 [−0.337, 7.283] MPa，跨越零，因此该过程对比虽有利于本文，仍存在不确定性。相对学习固定排序，减少量为 4.380 MPa [0.199, 8.405]。表 3 区分了六来源组等权的过程指标与试样池化的终点 MAE。

Figure 3 shows how this advantage develops across the recorded observation events. Individual updates can raise as well as lower error: geometry-spread reached 46.189 MPa at cap 0.125 and ended at 46.910 MPa, while learned-static rose from 46.465 MPa at 0.0625 to 48.064 MPa at 0.125 before improving. The proposed policy improved at its four nonzero reporting points, with local reversals also visible on the finer event curve. Process and endpoint metrics capture these complementary aspects of assessment. The proposed mean final acquired fraction was 0.247174, compared with approximately 0.250 for geometry-spread.

图 3 展示这一优势随观测事件形成的过程。单次更新既可能降低误差，也可能使误差增加：几何分散在 0.125 上限达到 46.189 MPa，终点为 46.910 MPa；学习固定排序从 0.0625 处的 46.465 MPa 升至 0.125 处的 48.064 MPa，随后再改善。本文策略在四个非零报告点依次改善，更细事件曲线上也可见局部回升。过程与终点指标描述了评估的互补方面。本文最终平均实际获取比例为 0.247174，几何分散约为 0.250。

![Figure 3](figures/F3.png){width=6.5in}

**Figure 3.** Prediction quality throughout acquisition on 50 development-validation specimens. (a) Six-method full event curves. (b) A local view of proposed recrossings, with the 42 MPa target dashed. Step curves preserve all recorded observation-completion breakpoints; dots mark the five reporting caps. Cohort MAE pools physical specimens, with Random losses averaged within specimen over five saved orders. All methods share the predictor and surface inputs. The horizontal reference is full-input MAE at area fraction 1.0. Process $A$ in Table 3 uses equal source-group weighting.

**图 3。** 50 件开发验证试样在采集过程中的预测质量。(a) 六方法完整事件曲线；(b) 本文局部回升，虚线表示 42 MPa 目标。阶梯曲线保留全部观测完成断点，圆点表示五个报告上限。队列 MAE 按物理试样池化，随机损失先在试样内对五条保存次序平均。全部方法共享预测器及表面输入，水平参照为面积比例 1.0 下的完整输入 MAE。表 3 中的过程 $A$ 采用来源组等权聚合。

## 5.2. Selected observations reach matched prediction quality earlier / 选定观测更早达到相同预测质量

The accuracy trajectories translate into lower acquisition requirements at defined quality targets. For a cohort MAE target of 47 MPa, the first supported five-grid cap was 0.0625 for the proposed method, 0.125 for geometry-spread and 0.25 for Random (Table 4). The required cap was therefore 50% smaller than geometry-spread and 75% smaller than Random. Learned-static also reached this target at 0.0625 on the five-cap grid. At the more demanding target of 44 MPa, the proposed method reached the first nonzero reporting cap, while all five controls remained above the target over the reported range.

准确性轨迹进一步转化为达到指定质量所需的采集规模。对于 47 MPa 队列 MAE 目标，五点网格首次支持的上限分别为本文方法 0.0625、几何分散 0.125 和随机 0.25（表 4）。所需上限相对几何分散减少 50%，相对随机减少 75%。学习固定排序在该五点网格上同样于 0.0625 达到此目标。对于要求更高的 44 MPa 目标，本文在首个非零报告上限达标，而五类对照在报告范围内均高于该目标。

**Table 4.** First attainment of cohort-MAE targets. Five-cap and event grids are reported separately. NR indicates that the target was not reached within cap 0.25. Entries describe first crossings under a common cap.

**表 4。** 队列 MAE 目标的首次达标。五点网格与事件网格分别报告；NR 表示在 0.25 上限内未达标，数值表示共同上限条件下的首次交叉。

| Strategy | 47 MPa, five caps | 44 MPa, five caps | 47 MPa, events |
| --- | --- | --- | --- |
| Proposed | 6.25% | 6.25% | 3.1203% |
| Center-first | NR | NR | NR |
| Geometry-spread | 12.5% | NR | 9.3426% |
| Serpentine | NR | NR | NR |
| Random | 25% | NR | 24.9630% |
| Learned-static | 6.25% | NR | 6.2407% |

The event grid resolves information arrival more finely. At the same 47 MPa target, the first cap was 0.031203 for the proposed method, 0.093426 for geometry-spread, 0.062407 for learned-static and 0.249630 for Random. The relative reduction against geometry-spread was approximately 66.60% at this resolution. Each value applies one common cap to the cohort before calculating MAE. The coarse and event grids therefore provide different resolutions of the same prediction trajectories.

事件网格以更细分辨率刻画信息到达。对同一 47 MPa 目标，首次上限分别为本文方法 0.031203、几何分散 0.093426、学习固定排序 0.062407 和随机 0.249630；在这一分辨率下，相对几何分散减少约 66.60%。每个数值均先对队列施加统一上限，再计算 MAE。粗网格与事件网格由此提供同一预测轨迹在不同分辨率下的描述。

The full target range reveals where this advantage matters (Fig. 4). At 55 MPa, all six strategies first met the five-grid criterion at 0.0625. Tighter targets differentiated their information requirements. The proposed event curve first reached 43 MPa at 0.062591, compared with 0.1875 on the five-cap grid. It also crossed 42 MPa at 0.078844 before rising again and ending at 42.385 MPa; the five-cap grid recorded no 42 MPa crossing. These observations distinguish first useful attainment from sustained attainment and retain the target dependence of information efficiency. The full-input anchor of 41.690 MPa was not reached within cap 0.25.

完整目标范围说明了这种优势出现的条件（图 4）。在 55 MPa 目标下，六种策略都于 0.0625 首次满足五点条件；更严格的目标则区分了各策略的信息需求。本文事件曲线首次达到 43 MPa 的上限为 0.062591，五点网格为 0.1875；事件曲线也在 0.078844 处首次跨越 42 MPa，随后回升并以 42.385 MPa 结束，五点网格未记录到 42 MPa 交叉。这些结果区分首次有用达标与持续达标，并保留信息效率对目标的依赖。完整输入锚点 41.690 MPa 在 0.25 上限内未达到。

![Figure 4](figures/F4.png){width=6.5in}

**Figure 4.** Acquisition scale required for matched cohort prediction quality. (a) First supported cap on the five-point grid. (b) First supported cap on the event grid. Both panels retain all preset integer targets from 41 to 61 MPa. The NR band explicitly marks nonattainment; initially satisfied targets are shown at zero cost. Lines guide comparisons between evaluated targets. The full-input anchor is reported separately in the text.

**图 4。** 达到相同队列预测质量所需的采集规模。(a) 五点网格中的首次支持上限；(b) 事件网格中的首次支持上限。两图均保留全部预设 41--61 MPa 整数目标。NR 区带显式表示未达标，初始已满足的目标成本为零。连线用于辅助比较已评价目标，完整输入锚点在正文中另行报告。

## 5.3. Earlier useful estimates arise from early error reduction and information arrival / 早期误差下降与信息到达支持更早的有用估计

### 5.3.1. Early observations produce larger realized error reductions / 早期观测形成更大的实际误差下降

Early acquisition events produced larger realized improvements under the proposed policy. For observations completed in (0, 0.0625\], the unweighted net error decrease exceeded geometry-spread by 6.865 MPa, with a paired 95% interval of \[0.362, 12.502\] MPa, and exceeded Random by 9.707 MPa \[4.862, 14.131\] (Fig. 5a). These stage sums retain both favorable and adverse events. They show that useful early observations generated larger actual reductions, providing empirical support for the early-process value encoded in the learning objective.

本文策略在早期采集事件中产生了更大的实际改善。对于 (0, 0.0625\] 内完成的观测，未加权净误差下降比几何分散多 6.865 MPa，配对 95% 区间为 \[0.362, 12.502\] MPa；比随机多 9.707 MPa，区间为 \[4.862, 14.131\] MPa（图 5a）。阶段求和同时保留有利和不利事件，表明有效早期观测确实形成了更大的实际下降，为学习目标中的早期过程价值提供了经验证据。

Process weighting connects these realized changes to the integrated assessment gain. The first-stage weighted contribution improved by 5.926 MPa \[−0.117, 11.095\] against geometry-spread and 8.617 MPa \[4.194, 12.678\] against Random. Against geometry-spread, the four stage gains were +5.926, −2.475, +0.199 and +0.063 MPa (Fig. 5b); their full-precision sum recovered the 3.714 MPa reduction in process $A$. Geometry-spread recovered part of the initial gap in the second stage, while the proposed sequence retained a lower average process error.

过程加权将这些实际变化连接到积分评估改善。首阶段加权贡献相对几何分散改善 5.926 MPa，区间为 \[−0.117, 11.095\] MPa；相对随机改善 8.617 MPa，区间为 \[4.194, 12.678\] MPa。相对几何分散，四阶段改善依次为 +5.926、−2.475、+0.199 和 +0.063 MPa（图 5b），全精度求和复现了过程 $A$ 的 3.714 MPa 减少量。几何分散在第二阶段追回部分初始差距，本文序列仍保持较低的平均过程误差。

For the proposed method, the four weighted contributions were 13.069, 0.196, 0.075 and 0.111 MPa. Most of the integrated gain relative to its source-group-equal initial error was established in the first stage. The raw event changes establish the realized reduction, while the weights account for how long each change remains useful. Together, they explain how early information contributes to the observed trajectory advantage.

本文方法的四阶段加权贡献为 13.069、0.196、0.075 和 0.111 MPa，相对来源组等权初始误差的大部分积分改善在首阶段建立。原始事件变化描述实际下降，加权项刻画每次变化持续发挥作用的区间，两者共同解释了早期信息如何形成观测到的轨迹优势。

### 5.3.2. Arrival order changes process quality with final content fixed / 最终内容固定时，到达顺序改变过程质量

Holding the final selected content fixed exposed the effect of its arrival order on intermediate assessment. Original order gave process $A = 43.597$ MPa and early $A = 48.771$ MPa. The five-permutation mean was 45.083 and 50.399 MPa, and complete reversal gave 47.526 and 52.099 MPa, respectively. All three conditions had the same endpoint MAE of 42.385 MPa. Their intermediate trajectories differed and converged at the identical terminal input (Fig. 5c).

固定最终选定内容后，其到达顺序对中间评估的作用得以显现。原序的过程 $A = 43.597$ MPa、早期 $A = 48.771$ MPa；五次置换均值分别为 45.083 和 50.399 MPa，完全逆序分别为 47.526 和 52.099 MPa。三种条件具有相同的终点 MAE 42.385 MPa，中间轨迹存在差异，并在相同终点输入处收敛（图 5c）。

The primary permutation-mean-minus-original process contrast was +1.486 MPa, with a paired 95% interval of [−0.658, 3.384] MPa (Fig. 5d). Thus, the lower mean under original order does not establish a resolved average advantage against the preset permutations. The prespecified secondary reversal contrast was +3.929 MPa [0.145, 7.405], supporting an order effect for that intervention. Early-process contrasts were +1.628 MPa [−2.190, 5.180] for permutation and +3.328 MPa [−1.527, 7.979] for reversal; both intervals cross zero. These contrasts describe arrival within the selected sets and do not partition the total policy gain into selection and order percentages.

主要“置换均值减原序”的过程对比为 +1.486 MPa，配对 95% 区间为 [−0.658, 3.384] MPa（图 5d）。因此，原序均值更低尚不能确立其相对预设置换的平均优势。预设辅助逆序对比为 +3.929 MPa [0.145, 7.405]，支持该干预下的次序作用。早期过程对比分别为置换 +1.628 MPa [−2.190, 5.180] 和逆序 +3.328 MPa [−1.527, 7.979]，两者区间均跨零。这些对比描述选定集合内部的到达顺序，不将总体策略改善分解为选择与排序的百分比。

Original-order replay reproduced the archived predictions with a maximum absolute deviation of $6.104 \times 10^{- 5}$ MPa, and all 350 terminal-set checks preserved the terminal input and prediction. The observed process differences thus came from predictions recomputed on the corresponding intermediate prefixes. This intervention shows how the same final regional content can support different assessment trajectories through its arrival sequence.

原序回放复现归档预测，最大绝对偏差为 $6.104 \times 10^{- 5}$ MPa，350 项终点集合核验均保持终点输入和预测一致。因此，观测到的过程差异来自对应中间前缀上的实际重算预测。该干预展示了同一最终区域内容如何通过不同到达序列，支持不同的评估轨迹。

![Figure 5](figures/F5.png){width=6.5in}

**Figure 5.** Evidence linking information arrival to earlier useful assessment. (a) Proposed-minus-control gains in unweighted net error decrease for completions in (0, 0.0625\]. (b) Proposed-minus-geometry-spread weighted gains in stages 1--4. (c) Recomputed cohort-MAE trajectories for original, permutation-mean and reversed arrivals of each identical final selected set. (d) Primary permutation-mean-minus-original and secondary reverse-minus-original contrasts in process $A$. Positive values in (a,b) favor the proposed method; positive values in (d) favor original order. Error bars are paired 95% capture-group percentile intervals from 5,000 draws. Panels (a,b,d) use equal source-group weighting; (c) uses specimen-pooled MAE.

**图 5。** 信息到达与较早形成有用评估之间的证据。(a) (0, 0.0625\] 完成事件的未加权净误差下降改善，方向为本文减对照；(b) 四阶段加权改善，方向为本文减几何分散；(c) 每件试样固定最终集合后，重算原序、置换均值及逆序的队列 MAE 轨迹；(d) 过程 $A$ 的主要置换均值减原序和辅助逆序减原序对比。(a,b) 正值有利于本文方法，(d) 正值有利于原序。误差条为 5,000 次捕获组抽样的配对 95% 百分位区间。(a,b,d) 采用来源组等权，(c) 采用试样池化 MAE。

Together, the event and fixed-set results connect earlier useful estimates to realized early reductions and the timing of selected content, with the primary permutation contrast remaining uncertain.

事件与固定集合结果共同将较早的有用估计联系到实际早期下降及选定内容的到达时机，同时主要置换对比仍具有不确定性。

## 5.4. Specimen-specific initialization and selective later reordering / 试样相关初始化与选择性后续重排序

The fixed diagnostic cases show how specimen information enters regional decisions. The first selected cells were 62, 31, 14, 27, 63 and 63 for c8-16, q24-48, q16-29, c24-9t, c16-24 and q8-22, respectively. These choices differed before any internal region was visible, demonstrating specimen-specific initialization from the initially available state. Figure 6 follows c8-16 and q24-48 through the registered surface image, internal-image reference with acquired-cell order, and strength updates. All six fixed cases are documented in Supplementary S5.

固定诊断案例展示了试样信息如何进入区域决策。c8-16、q24-48、q16-29、c24-9t、c16-24 和 q8-22 的首个选择编号分别为 62、31、14、27、63 和 63。这些差异出现在任何内部区域可见之前，展示了基于初始可用状态的试样相关初始化。图 6 串联 c8-16 与 q24-48 的登记表面图、叠加已获取格次序的内部图参照及强度更新，全部六个固定案例记录于补充 S5。

The subsequent decisions showed selective reordering of the initial preferences. Each recorded action was compared with the highest initially scored cell still feasible on the actual path, accounting for removal of observed and unaffordable cells. Nine of 94 decisions differed from this reference: two for c8-16, two for q16-29, four for c16-24 and one for q8-22. The q24-48 and c24-9t cases retained their initial feasible ranking throughout. The observed behavior therefore combined specimen-specific starting priorities with selective adjustment during acquisition. These nine departures concern only the six fixed cases. Because the changing state jointly includes measured content, history, progress and prediction, they cannot be attributed causally to internal-image feedback alone.

后续决策表现为对初始偏好的选择性重排序。每个实际动作与真实路径上仍可行的初始最高评分格比较，以计入已测格及超过剩余面积格被移除的影响。94 个决策中有九次不同于这一参照：c8-16 两次、q16-29 两次、c16-24 四次、q8-22 一次；q24-48 与 c24-9t 全程保持初始可行排序。因此，实际行为结合了试样相关的起始优先级与采集中的选择性调整。 这九次偏离仅涉及六个固定案例。由于变化状态同时包含已测内容、历史、进度和预测，不能将这些偏离因果归因于内部图像反馈这一单独因素。

The prediction traces expose both the benefit of acquisition and the residual error of the strength model. For c8-16, measured CAI was 264.163 MPa; the estimate changed from 301.366 to 232.095 MPa, reducing absolute error from 37.203 to 32.068 MPa through updates in both directions. For q24-48, the estimate moved from 302.099 to 258.932 MPa against measured strength of 136.680 MPa. Acquisition reduced its initial error, but the final residual remained 122.252 MPa. These fixed cases make the complete request--observation--prediction process visible, including an instance where improved information supply left substantial prediction error.

预测轨迹同时呈现采集作用与强度模型的剩余误差。c8-16 的实测 CAI 为 264.163 MPa，估计从 301.366 MPa 变为 232.095 MPa，经历双向更新后，绝对误差从 37.203 MPa 降至 32.068 MPa。q24-48 的估计从 302.099 MPa 变为 258.932 MPa，实测强度为 136.680 MPa；采集降低了初始误差，但终点仍保留 122.252 MPa 残差。这些固定案例展示了完整请求---观测---预测过程，其中包括信息供给改善后仍存在较大预测误差的情形。

![Figure 6](figures/F6.png){width=6.5in}

**Figure 6.** Real-image evidence and CAI trajectories for c8-16 (a–c) and q24-48 (d–f). Each row shows the registered surface view, retrospective full-image C-scan reference with observed cells and acquisition ordinals overlaid, and saved prediction updates. Unnumbered cells were unavailable to the policy; the displayed full C-scan was never an execution input. Ordinals follow the native row-major grid. The final prediction is carried to cap 0.25; the q24-48 final residual of 122.252 MPa is retained. Images: S. Hasebe, R. Higuchi, T. Yokozeki and S. Takeda, Mendeley Data v1 (2022), CC BY 4.0 [10,11]. Surface images follow the frozen clockwise-90° registration and display resize; no C-scan contrast or rotation is added.

**图 6。** c8-16（a–c）与 q24-48（d–f）的真实图像证据及 CAI 轨迹。每行依次为登记表面图、叠加已测格及获取序号的事后完整 C-scan 参照、已保存的预测更新。未编号格对策略不可用，图示完整 C-scan 从未作为执行输入。序号遵循原生行优先网格，最后预测延续至 0.25 上限，保留 q24-48 的 122.252 MPa 终点残差。图像来源：Hasebe 等，Mendeley Data v1（2022），CC BY 4.0 [10,11]。表面图遵循冻结的顺时针 90° 登记与显示缩放，C-scan 不额外调整对比度或旋转。

Information-use diagnostics complement the executed trajectories. Action probabilities quantify selection preferences. Surface attribution measures local sensitivity of the selected action's log probability, including the route through the current strength prediction. Attention summarizes global-to-regional interactions among fused tokens. Supplementary S5 defines these quantities separately. The maps describe how the policy uses its represented inputs, while the strength traces measure the resulting assessment outcome; the diagnostic values are not regional mechanical labels or calibrated damage probabilities.

信息使用诊断为执行轨迹提供补充。动作概率量化选择偏好；表面归因衡量所选动作对数概率的局部敏感性，包括经过当前强度预测的路径；注意力汇总融合 token 中的全局---区域交互。补充 S5 分别定义这些量。诊断图描述策略如何使用表示输入，强度轨迹衡量其实际评估结果；这些诊断值不是区域力学标签或校准的损伤概率。

## 5.5. Groupwise performance and the partial-to-full information trade-off / 分组表现与部分至完整信息的性能权衡

The groupwise results locate the configurations contributing to the overall improvement (Table 5). Relative to geometry-spread, the 24-layer quasi-isotropic group improved process $A$ by 13.061 MPa and endpoint MAE by 16.440 MPa. The 16-layer cross-ply group improved these metrics by 8.157 and 5.875 MPa. The 8-layer cross-ply group also improved both metrics. Other configurations showed smaller or mixed changes: 24-layer cross-ply had a lower process error but 2.097 MPa higher endpoint MAE, and the 8-layer quasi-isotropic group was worse on both metrics. The 16-layer quasi-isotropic group had nearly unchanged process error and a modest endpoint improvement.

分组结果定位了总体改善所对应的配置（表 5）。相对几何分散，24 层准各向同性组的过程 $A$ 改善 13.061 MPa，终点 MAE 改善 16.440 MPa；16 层交叉铺层组分别改善 8.157 和 5.875 MPa，8 层交叉铺层组也在两项指标上改善。其他配置变化较小或方向不同：24 层交叉铺层组过程误差更低，终点 MAE 却高 2.097 MPa；8 层准各向同性组两项指标均较差；16 层准各向同性组的过程误差基本不变，终点略有改善。

**Table 5.** Groupwise performance in MPa. Gains are geometry-spread minus the proposed method, so positive values favor the proposed method. These are source-group strata within the evaluated partition.

**表 5。** 分组性能，单位为 MPa。改善定义为几何分散减本文方法，正值有利于本文方法。这些结果为评价分区内的来源组分层。

| Laminate | n | Proposed A | Proposed MAE | Gain in A | Gain in MAE |
| --- | --- | --- | --- | --- | --- |
| 8-layer cross-ply | 9 | 31.330 | 28.655 | +0.893 | +5.634 |
| 24-layer quasi-isotropic | 9 | 68.749 | 64.067 | +13.061 | +16.440 |
| 16-layer quasi-isotropic | 8 | 47.030 | 44.088 | −0.105 | +1.615 |
| 24-layer cross-ply | 9 | 48.163 | 46.433 | +0.760 | −2.097 |
| 16-layer cross-ply | 8 | 48.059 | 48.408 | +8.157 | +5.875 |
| 8-layer quasi-isotropic | 7 | 18.250 | 18.124 | −0.485 | −1.922 |

At specimen level, the proposed method improved process $A$ for 33 of 50 specimens and endpoint absolute error for 29 of 50 relative to geometry-spread. These results characterize a useful but heterogeneous selection policy. The source-group strata combine differences in imagery, labels and acquisition source with laminate configuration; they describe within-partition performance rather than leave-one-configuration-out transfer. Their variation identifies concrete settings in which observation selection helps and settings requiring further investigation.

在试样层面，相对几何分散，本文方法在 50 件中的 33 件改善过程 $A$，在 29 件改善终点绝对误差，呈现出有效但存在异质性的选择策略。来源组分层同时包含图像、标签、采集来源及层合板配置差异，描述的是分区内表现，而非留一配置迁移。分组变化由此指出了观测选择发挥作用的具体条件，以及需要进一步研究的配置。

The full-input reference places the result on an information-scale axis. The proposed method acquired a mean 24.7174% of internal pixels, with specimen fractions ranging from 0.234492 to 0.250000, and reached MAE 42.385 MPa. Full input gave 41.690 MPa, a partial-minus-full difference of 0.695 MPa with paired interval \[−4.009, 5.327\] MPa. RMSE was 56.534 versus 53.944 MPa, and $R^{2}$ was 0.682 versus 0.711. These results quantify the prediction--information trade-off under one readout: selected partial input approached the full-input mean error while retaining differences in endpoint metrics. This is a difference estimate; equivalence or noninferiority was not tested with a prespecified margin.

完整输入参照将结果置于信息规模维度。本文平均获取 24.7174% 的内部像素，各试样比例为 0.234492--0.250000，MAE 达到 42.385 MPa；完整输入为 41.690 MPa，部分减完整输入差为 0.695 MPa，配对区间为 \[−4.009, 5.327\] MPa。RMSE 分别为 56.534 和 53.944 MPa，$R^{2}$ 分别为 0.682 和 0.711。结果量化了共同读出器下的预测---信息权衡：选定部分输入的平均误差接近完整输入，同时保留终点指标差异。这里报告的是差异估计，未按预设界限开展等效性或非劣效检验。

## 5.6. Implications for inspection-and-assessment workflows / 对检查与评估流程的意义

The central engineering contribution is to make information provision an explicit, learnable part of strength assessment. Regional requests determine the observations available to a common predictor, and the trajectory objective connects these requests to both immediate and final assessment quality. Accuracy comparisons establish the value of observation organization; matched-quality analysis expresses that value as an information requirement; and event and fixed-set analyses explain how information arrival contributes to the process. Together, they describe an assessment workflow in which acquisition is guided by the mechanical quantity being estimated.

本文的核心工程贡献，是将信息供给变为强度评估中明确、可学习的组成部分。区域请求决定共同预测器可获得的观测，轨迹目标将请求连接到过程与最终评估质量。准确性比较说明观测组织的价值，等质量分析将其表达为信息需求，事件与固定集合分析则解释信息到达对过程的作用。三者共同形成了由待估计力学量引导采集的评估流程。

This formulation extends image-informed CAI regression \[3\] by optimizing the input sequence as well as evaluating its readout. It also connects the observation--update--decision structure of autonomous inspection \[4\] and active feature acquisition \[5, 6\] to a continuous strength trajectory. The common-predictor design attributes the evaluated difference to the complete selection mechanism. Surface conditioning, observed internal content, history and global task state operate together in that mechanism; their separate contributions would require matched component interventions. Similarly, the fixed-set experiment evaluates arrival within the selected content, rather than assigning additive percentages of the total gain to selection and order.

这一建模通过优化输入序列及评价其读出，扩展了图像支持的 CAI 回归 \[3\]，并将自主检查 \[4\] 和主动特征获取 \[5, 6\] 的观测---更新---决策结构连接到连续强度轨迹。共同预测器设计将被评价的差异归于完整选择机制。在该机制中，表面条件、已测内部内容、历史与全局任务状态共同作用；各自的独立贡献需要匹配的组件干预检验。同样，固定集合实验评价的是选定内容内部的到达次序，而非将总体改善按可加百分比分配给选择与排序。

The present evidence concerns regional revelation of registered two-dimensional images. Native-area fraction measures the quantity of internal-image information supplied to assessment. Extending the request interface to physical inspection would require accounting for scanner movement, coupling, setup and acquisition latency, and validating an execution-time stopping criterion. The method supplies a concrete interface for that extension: regional indices specify requests, completed observations update the state, and the predictor supplies a strength estimate after each update.

当前证据对应已登记二维图像的逐区域揭示，原生面积比例衡量提供给评估的内部图像信息量。将请求接口扩展至物理检查，需要计入扫描运动、耦合、准备及采集延迟，并验证执行时的停止准则。本文方法为这一扩展提供了明确接口：区域编号规定请求，完成观测更新状态，预测器在每次更新后输出强度估计。

Independent evaluation would establish how this capability transfers beyond the selected development configuration. The present cohort also selected checkpoints, and the policy results use one initialization; the groupwise reversals and residual cases further identify variability that broader evaluation should resolve. Independent specimens and repeated initializations can test the stability of the accuracy and information-efficiency gains, while physical regional acquisition can test their translation into inspection performance. The demonstrated result is a task-driven way to obtain useful CAI estimates earlier along regional image acquisition and to use the supplied internal information more effectively.

独立评价将进一步检验该能力能否迁移至已选开发配置之外。当前队列同时用于 checkpoint 选择，策略结果使用一次初始化；分组反向变化与残差案例进一步指出了需要在更广评价中检验的变异。独立试样及重复初始化可检验准确性与信息效率改善的稳定性，物理区域采集可检验其向检查性能的转化。本文已经展示了以任务驱动方式，在区域图像采集进程中较早获得有用 CAI 估计，并更有效地利用所提供内部信息的途径。

# 6. Conclusions / 结论

We presented strength-assessment-driven multimodal sequential observation for efficient CAI estimation. The method uses initially available surface descriptors and progressively acquired internal C-scan regions to learn specimen-conditioned requests. A common predictor updates strength after each observation, and a trajectory objective combines process-average and terminal errors to organize regional content and its arrival around the assessment task. The event identity explains why useful error reductions obtained earlier contribute more to process quality.

本文提出用于高效 CAI 估计的强度评估驱动多模态序贯观测方法。该方法利用初始可用的表面描述符与逐步获取的内部 C-scan 区域，学习试样条件化区域请求；共同预测器在每次观测后更新强度，轨迹目标联合过程平均误差与终点误差，围绕评估任务组织区域内容及其到达时机。事件恒等式解释了较早形成的有效误差下降为何具有更大过程贡献。

On the 50 development-validation specimens used for selection and effect estimation, endpoint MAE reached 42.385 MPa at cap 0.25, 9.65% below geometry-spread; the 0.0625 prefix reached 43.324 MPa. The five-grid cap first meeting a 47 MPa cohort-MAE target was 0.0625 versus 0.125 for geometry-spread. Larger early net reductions and fixed-set trajectories support the value of information arrival, although the primary permutation contrast crosses zero and endpoint equality follows from the set-based predictor. These conditional results establish observation organization as an optimization target for image-based assessment. Independent specimens, repeated initializations and physical acquisition remain necessary to establish broader reliability and inspection-time benefits.

在同时用于选择与效果估计的 50 件开发验证试样上，方法于 0.25 上限达到 42.385 MPa 终点 MAE，比几何分散低 9.65%；0.0625 前缀达到 43.324 MPa。首次达到 47 MPa 队列 MAE 的五点网格上限为 0.0625，几何分散为 0.125。更大的早期净下降与固定集合轨迹支持信息到达的价值，但主要置换对比区间跨零，终点一致性来自集合式预测器结构。这些条件性结果使观测组织成为图像评估的优化对象。更广可靠性与检查时间收益仍需独立试样、重复初始化及物理采集检验。

# Data availability / 数据可用性

The original surface and internal images are available in the six versioned Mendeley Data collections \[10--15\], and the CAI measurements are available in the corresponding strength-testing collection \[9\]. Supplementary S1--S5 describe specimen and capture-group manifests, model identities, frozen prediction trajectories, full-precision results, fixed-case diagnostic records and the scripts used to regenerate the manuscript displays. Source-file hashes identify the frozen evidence associated with the reported analyses.

原始表面及内部图像可从六份带版本的 Mendeley Data 集合获取 \[10--15\]，CAI 测量值来自对应强度试验集合 \[9\]。补充 S1--S5 说明试样与捕获组清单、模型身份、冻结预测轨迹、全精度结果、固定案例诊断记录及用于重建论文图表的脚本，源文件哈希标识所报告分析对应的冻结证据。

# References / 参考文献

\[1\] Saki Hasebe, Ryo Higuchi, Tomohiro Yokozeki, and Shin-ichi Takeda. Dataset for surface and internal damage after impact on CFRP laminates. Data in Brief, 43:108462, 2022. doi: 10.1016/j.dib.2022.108462.

\[2\] Saki Hasebe, Ryo Higuchi, Tomohiro Yokozeki, and Shin-ichi Takeda. Dataset of compression after impact testing on carbon fiber reinforced plastic laminates. Data in Brief, 60:111509, 2025. doi: 10.1016/j.dib.2025.111509.

\[3\] Jason P. Mack, Faizan Mirza, Zhong-Hui Duan, Saki Hasebe, Ryo Higuchi, Tomohiro Yokozeki, and K. T. Tan. Deep learning for predicting impact energy and compression after impact strength of composite materials using C-scan images. Advanced Engineering Informatics, 72:104518, 2026. doi: 10.1016/j.aei.2026.104518.

\[4\] R. Fuentes, P. Gardner, C. Mineo, T. J. Rogers, S. G. Pierce, K. Worden, N. Dervilis, and E. J. Cross. Autonomous ultrasonic inspection using Bayesian optimisation and robust outlier analysis. Mechanical Systems and Signal Processing, 145:106897, 2020. doi: 10.1016/j.ymssp.2020.106897.

\[5\] Jaromír Janisch, Tomáš Pevný, and Viliam Lisý. Classification with Costly Features Using Deep Reinforcement Learning. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01):3959--3966, 2019. doi: 10.1609/aaai.v33i01.33013959.

\[6\] Ian Connick Covert, Wei Qiu, Mingyu Lu, Na Yoon Kim, Nathan J. White, and Su-In Lee. Learning to Maximize Mutual Information for Dynamic Feature Selection. Proceedings of the 40th International Conference on Machine Learning, 202:6424--6447, 2023. https://proceedings.mlr.press/v202/covert23a.html.

\[7\] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. arXiv, 2015. doi: 10.48550/arXiv.1512.03385.

\[8\] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention Is All You Need. arXiv, 2017. doi: 10.48550/arXiv.1706.03762.

\[9\] Saki Hasebe, Ryo Higuchi, Tomohiro Yokozeki, and Shin-ichi Takeda. Datasets on CFRP specimens subjected to compression after impact tests. Mendeley Data, 2025, version 3. https://data.mendeley.com/datasets/8scdmfdcfb/3.

\[10\] Saki Hasebe, Ryo Higuchi, Tomohiro Yokozeki, and Shin-ichi Takeda. Datasets on CFRP specimens after low-velocity impact tests: 8-layer cross-ply laminates. Mendeley Data, 2022, version 1. [8-layer cross-ply dataset, version 1](https://data.mendeley.com/datasets/74t7kcdgkr/1).

\[11\] Saki Hasebe, Ryo Higuchi, Tomohiro Yokozeki, and Shin-ichi Takeda. Datasets on CFRP specimens after low-velocity impact tests: 24-layer quasi-isotropic laminates. Mendeley Data, 2022, version 1. [24-layer quasi-isotropic dataset, version 1](https://data.mendeley.com/datasets/cgtnjyggtm/1).

\[12\] Saki Hasebe, Ryo Higuchi, Tomohiro Yokozeki, and Shin-ichi Takeda. Datasets on CFRP specimens after low-velocity impact tests: 16-layer quasi-isotropic laminates. Mendeley Data, 2022, version 1. [16-layer quasi-isotropic dataset, version 1](https://data.mendeley.com/datasets/w68dtmpfyf/1).

\[13\] Saki Hasebe, Ryo Higuchi, Tomohiro Yokozeki, and Shin-ichi Takeda. Datasets on CFRP specimens after low-velocity impact tests: 24-layer cross-ply laminates. Mendeley Data, 2022, version 1. [24-layer cross-ply dataset, version 1](https://data.mendeley.com/datasets/xcmzfsbd9t/1).

\[14\] Saki Hasebe, Ryo Higuchi, Tomohiro Yokozeki, and Shin-ichi Takeda. Datasets on CFRP specimens after low-velocity impact tests: 16-layer cross-ply laminates. Mendeley Data, 2022, version 1. [16-layer cross-ply dataset, version 1](https://data.mendeley.com/datasets/yfxyg8jm46/1).

\[15\] Saki Hasebe, Ryo Higuchi, Tomohiro Yokozeki, and Shin-ichi Takeda. Datasets on CFRP specimens after low-velocity impact tests: 8-layer quasi-isotropic laminates. Mendeley Data, 2022, version 1. [8-layer quasi-isotropic dataset, version 1](https://data.mendeley.com/datasets/ykhs7s2dck/1).
