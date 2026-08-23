# Codex Prompt — Mechanical-Value Acquisition for Task-Driven Ultrasonic C-scan Assessment

## 0. Role and repository

你现在继续接管 CFRP ultrasonic C-scan → CAI 项目。

Repository:

```bash
git@github.com:Orangekostar/diff.git
```

当前 main 已包含并冻结：

- G1/G2
- P1/P3/P5/P6/P7
- MASI A0–A5
- Multi-view E1–E3
- MSSS S1
- MGMR M0

最近 MGMR 提交：

```text
6b1ce961129317354f7907f2d1bcf6a078206384
```

正式状态：

```text
MGMR_NO_GO
```

禁止修改历史 gate 以解锁 MGMR。

---

# 1. 本阶段不要再做 representation guessing

停止以下方向：

```text
MASI / feature invariance
multi-view fusion
MSSS rescue
MGMR rescue
wavelet + GNN stacking
new attention block
new backbone search
full-image diffusion reconstruction
```

过去实验已经说明：

> 从现有 276 个 specimen 中不断猜测 latent representation，再开发新模块，容易产生 representation mismatch。

新的科学问题必须建立在一个**直接可观测的工程事实**上：

\[
\boxed{
\text{dense ultrasonic observations contain substantial redundancy for CAI prediction}
}
\]

P5/P7/MSSS sampling 已经支持：

```text
FULL internal-only ≈ 0.0896358
25% sparse-only   ≈ 0.090116
```

且更低 sampling density 仍保留大量 CAI value。

因此新的问题不是：

> What feature should represent damage?

而是：

> **Given a limited ultrasonic measurement budget, which additional measurements provide the highest value for CAI assessment?**

---

# 2. New research direction

暂命名：

# MVA — Mechanical-Value Acquisition

完整概念：

> **Mechanics-Aware Task-Driven Ultrasonic Acquisition for CAI Assessment**

核心思想：

```text
Coarse global survey
        ↓
Current sparse observation
        ↓
Estimate mechanical value of candidate measurements
        ↓
Refine the highest-value region
        ↓
Update CAI prediction
        ↓
Repeat until budget exhausted
```

核心对象不是新 feature，而是：

# Mechanical Value of Measurement — MVoM

---

# 3. 核心定义

对于 specimen \(i\)，当前已测 measurement set：

\[
S_t
\]

当前 CAI predictor：

\[
\hat y_i(S_t)
\]

候选新增测量区域：

\[
p
\]

新增后：

\[
S_t^p = S_t\cup p
\]

Primary loss：

\[
\ell(y,\hat y)=|y-\hat y|
\]

定义：

\[
\boxed{
V_i^{CAI}(p|S_t)
=
\ell(y_i,\hat y_i(S_t))
-
\ell(y_i,\hat y_i(S_t^p))
}
\]

解释：

```text
V > 0:
measurement improves CAI prediction

V ≈ 0:
measurement provides little mechanical value

V < 0:
measurement harms the current predictor
```

第一阶段只把该 quantity 当：

> retrospective oracle / training supervision

绝不能在 deployment 时使用 ground-truth CAI。

---

# 4. 为什么这个问题值得研究

当前 evidence chain：

```text
A/W/H
    ↓
insufficient

Full C-scan
    ↓
strong CAI information

P3 spatial shuffle
    ↓
spatial organization is important

P5/P7 sparse sensing
    ↓
many measurement locations are redundant

P6 reconstruction
    ↓
better image reconstruction does not imply better CAI

MASI
    ↓
forcing invariance loses CAI information

Multi-view
    ↓
multiple fixed sparse/full views are predictively redundant

MSSS
    ↓
no universal single spatial scale established

MGMR
    ↓
predefined morphology decomposition not validated
```

因此：

\[
\boxed{
\text{Do not guess what information is useful.}
}
\]

直接研究：

\[
\boxed{
\text{Which measurement changes the mechanical decision?}
}
\]

---

# 5. 最重要的研究假设

## H1 — Task value differs from reconstruction value

\[
V^{CAI}(p)
\neq
V^{reconstruction}(p)
\]

即：

> 最值得恢复图像的位置，不一定是最值得预测 CAI 的位置。

---

## H2 — Task value differs from visual damage value

\[
V^{CAI}(p)
\neq
V^{damage}(p)
\]

即：

> visually strongest damage region 不一定是对剩余承载能力最有信息的区域。

---

## H3 — Adaptive allocation has headroom

在相同 measurement budget 下：

\[
MAE_{\rm CAI-oracle}
<
MAE_{\rm uniform}
\]

且最好：

\[
MAE_{\rm CAI-oracle}
<
MAE_{\rm recon-oracle}
\]

---

## H4 — Deployable policy can imitate the oracle

仅依赖当前观测：

\[
S_t
\]

而不依赖 unseen full C-scan / true CAI，

学习：

\[
\pi(a_{t+1}|S_t)
\]

能明显优于 uniform/random sampling。

---

## H5 — Laminate structure may condition measurement value

后续测试：

\[
V(p|S_t,L)
\]

是否随 laminate architecture \(L\) 变化。

只有 H1–H4 成立后再测试 H5。

---

# 6. Mandatory reference audit

首先创建：

```text
docs/MVA_REFERENCE_METHOD_AUDIT.md
```

深入阅读并记录以下工作。

---

## R1 — AdaSTMAE / Adaptive ultrasonic sampling

重点理解：

```text
coarse exploration
→ identify informative/damage-prone regions
→ locally increase sampling density
→ reconstruct sparse wavefield
```

借：

- coarse-to-fine spatial acquisition
- adaptive allocation of measurement budget
- comparison with uniform/random strategies

不要借：

- reconstruction as final objective
- STMAE architecture as current main model

本项目的区别：

```text
AdaSTMAE:
sampling → reconstruction fidelity

MVA:
sampling → CAI mechanical utility
```

---

## R2 — TACKLE

重点理解：

```text
sampler
→ retriever
→ downstream predictor
```

以及：

> acquisition pattern should be optimized for the downstream task rather than image reconstruction alone.

借：

- task-driven acquisition formulation
- budget-constrained learnable mask
- two-stage training philosophy
- deterministic top-k inference mask

不要搬：

- MRI forward operator
- VarNet-specific architecture

---

## R3 — LOUPE / differentiable acquisition mask

重点理解：

\[
p_j=\sigma(q_j)
\]

budget normalization：

\[
\sum_jp_j=B
\]

以及 Bernoulli / STE / top-k acquisition。

作为后续：

> Global Task-Aware Static Mask

baseline。

---

## R4 — EDDI / value-of-information acquisition

重点理解：

> acquisition value should be defined relative to uncertainty/information about the target variables.

借：

\[
\text{target-oriented value of acquisition}
\]

不要搬：

- Partial VAE architecture

因为当前已有 strong CAI predictor。

---

## R5 — Active Feature Acquisition / oracle imitation

重点理解：

```text
full training information
→ derive oracle acquisition order
→ train deployable policy from partial observations
```

这是本项目 sequential policy 的主要方法论模板。

不要直接上 RL。

---

## R6 — Active MRI Acquisition

仅作为：

- sequential acquisition
- POMDP
- budgeted actions

背景参考。

当前数据量禁止第一版使用：

```text
PPO
DQN
Decision Transformer
```

---

## R7 — Task-based sensing / task-based quantization

理解理论：

如果 downstream variable 是：

\[
y
\]

则 sensing system 应优化：

\[
\mathcal L(y,\hat y)
\]

而不是：

\[
\mathcal L(x,\hat x)
\]

该理论用于论文中支撑：

> mechanical task-driven acquisition。

---

## R8 — Lamination parameters literature

只为后期 H5 使用。

理解：

> 如何把 variable-length stacking sequence 映射成 fixed-dimensional continuous mechanical descriptor。

不要立即实现。

---

# 7. Licensing / provenance rule

外部代码只能：

- 学习方法；
- 参考公开算法；
- 按 license 合法复用。

创建：

```text
docs/MVA_EXTERNAL_CODE_PROVENANCE.md
```

记录：

```text
paper
repository
license
borrowed idea
borrowed code if any
modified files
```

禁止无归属复制代码。

---

# 8. Critical physical limitation

当前公开 Hasebe C-scan 数据需要先确认：

> 是 scanner screenshot / rasterized C-scan image，而不是逐 measurement point 的原始 A-scan 数值数据库。

因此当前实验严格属于：

\[
\boxed{
\text{retrospective spatial acquisition simulation}
}
\]

不得在文档/论文中直接声称：

```text
75% actual inspection time reduction
75% physical scanning time saving
real online adaptive scanner
```

除非以后获得真实 scanner interface / raw measurement validation。

当前只能说：

```text
simulated measurement budget
retrospective sparse acquisition
spatial observation reduction
```

---

# 9. Phase structure

严格顺序：

# A0 — Acquisition Semantics Audit

# A1 — Retrospective Acquisition Simulator

# A2 — Mechanical-Value Oracle Audit

# A3 — Oracle Headroom Gate

只有 A3 GO：

# A4 — Global Task-Aware Static Acquisition

只有 A4 / A3 支持 adaptive：

# A5 — Deployable Oracle-Imitation Policy

只有 A5 GO：

# A6 — Laminate-Conditioned Acquisition

最后：

# A7 — Structured Transfer Evaluation

---

# 10. A0 — Acquisition Semantics Audit

先创建：

```text
docs/MVA_ACQUISITION_SEMANTICS_AUDIT.md
```

必须回答：

1. 原始 C-scan 文件尺寸；
2. published scan physical area；
3. nominal scan pitch；
4. screenshot pixel grid 与 physical scan point 是否一一对应；
5. 是否存在 resize/crop；
6. 当前 P5 sampling 操作发生在哪个 grid；
7. 一个 sampled image pixel 是否能合理解释为一个 physical measurement；
8. 哪些 physical claims 可以做；
9. 哪些只能称 retrospective simulation；
10. specimen 是否具有统一 orientation；
11. scan ROI 是否一致；
12. impact center 是否一致；
13. missing / border / annotation pixels；
14. pseudocolor semantics；
15. measured-point restoration 如何实现。

如果 pixel ↔ physical point 无法严格证明：

> 所有 primary experiment 使用 normalized spatial observation grid，不使用虚假的 mm / pitch claim。

---

# 11. Reuse existing P5 sampling semantics

禁止重新发明另一套 sparse generation。

优先复用当前已经验证的：

```text
sampling coordinates
bilinear interpolation
measured point restoration
rounding
border handling
```

建立：

```text
src/cmc_bbdm/mva/acquisition_grid.py
```

并引用 existing P5 implementation。

---

# 12. A1 — Acquisition simulator

新的 simulator 不只是 uniform density。

状态：

\[
S_t=(M_t,X_t)
\]

其中：

\[
M_t
\]

是 measured location mask，

\[
X_t
\]

是已经观测到的真实 C-scan values。

未观测部分通过冻结 deterministic interpolation 得到：

\[
\tilde X_t
\]

因此 CAI predictor 输入：

\[
\tilde X_t
\]

但 measurement mask：

\[
M_t
\]

必须单独保留供 acquisition policy 使用。

---

# 13. Coarse-to-fine design

不要让 action 是“任意单 pixel”。

Primary acquisition unit：

\[
\boxed{\text{spatial cell refinement}}
\]

将有效 ROI 划成：

```text
8×8 cells
```

Primary：

\[
K=64
\]

Sensitivity：

```text
6×6
10×10
```

但是只有 primary GO 后才跑 sensitivity。

---

# 14. Nested measurement grids

每个 cell 有多个 sampling level：

```text
level 0 = coarse survey
level 1 = intermediate refinement
level 2 = fine/full local observation
```

实际 stride 必须根据 native grid 可整除性确定。

保证：

\[
S^{(0)}
\subset
S^{(1)}
\subset
S^{(2)}
\]

即新增 refinement 只能：

> add measurements

不能改变已有测量点。

---

# 15. Initial survey

不要未经验证固定 6.25%。

先 audit candidate coarse budgets：

```text
1.5625%
3.125%
6.25%
```

如果 native grid 无法精确实现，使用合法 nested lattice 并报告 effective budget。

原则：

- 6.25% 可能已经过强，adaptive headroom 太小；
- 过低又可能 completely destroy initial information。

因此在 source-only pilot 中选择一个：

> 既有非平凡 error，又仍能提供全局 damage context 的 initial survey。

Outer test 不得参与该选择。

---

# 16. Budget definition

所有 acquisition results 必须基于：

\[
B_t=
\frac{
\#\text{unique measured locations}
}{
\#\text{full-grid locations}
}
\]

禁止用：

```text
number of refinement actions
```

代替 budget。

不同 cell 边界可能对应不同新增 measurement 数量，因此必须按真实 unique locations 计算。

---

# 17. Budget checkpoints

Primary checkpoints：

```text
3.125%
6.25%
9.375%
12.5%
18.75%
25%
50%
100%
```

若 initial survey 已高于某 checkpoint，则跳过。

报告 effective budget，而不是伪造精确比例。

---

# 18. Strong CAI evaluator

A0–A3 禁止更换 CAI backbone。

使用当前 strongest registered pipeline：

```text
Frozen ImageNet ResNet18
→ final 512D embedding
→ fold-local preprocessing/PCA
→ registered regressor
```

目标：

> isolate acquisition effect。

Frozen full baseline 必须精确复算：

\[
MAE=0.0896358
\]

误差超过既有 tolerance：

> STOP。

---

# 19. CAI predictor under sparse observations

对每一个 state：

\[
S_t
\]

生成：

\[
\tilde X_t
\]

使用同一 frozen encoder。

允许 source-only 对 sparse-state regression head 重新训练，

但必须比较两种协议：

### P-A

Full-trained frozen CAI evaluator applied to sparse reconstruction.

### P-B

Budget-specific source-trained sparse predictor.

Primary oracle audit 优先 P-B：

因为真正系统会知道其 acquisition regime。

但不得利用 outer target。

---

# 20. Strict cross-fitting requirement

机械价值 oracle 会用 ground-truth CAI。

因此训练 source specimen 的 oracle 轨迹绝不能使用：

> 对该 specimen 训练过的 CAI predictor。

必须：

\[
\boxed{
\text{strict OOF predictor for every oracle-labelled source specimen}
}
\]

例如 outer fold 内：

```text
source domains
    ↓
inner grouped / domain CV
    ↓
OOF CAI predictions
    ↓
generate oracle values
```

新增：

```text
test_mva_oracle_uses_oof_predictor.py
```

---

# 21. A2 — Three fundamentally different value definitions

必须至少实现三类 oracle。

---

## O1 — Reconstruction-Value Oracle

当前 reconstruction：

\[
\tilde X_t
\]

full C-scan：

\[
X
\]

candidate \(p\) 加入前后：

\[
E_t^{rec}
=
L_{rec}(X,\tilde X_t)
\]

\[
E_{t,p}^{rec}
=
L_{rec}(X,\tilde X_t^p)
\]

定义：

\[
V^{rec}(p|S_t)
=
E_t^{rec}-E_{t,p}^{rec}
\]

Primary：

```text
MSE / normalized MSE
```

Secondary：

```text
SSIM
```

不要让 SSIM 阻塞主实验。

---

# 22. O2 — Appearance / Damage-Focused Oracle

只有存在可信 damage mask / amplitude semantics 时才叫：

```text
damage oracle
```

否则必须称：

```text
appearance-intensity oracle
```

可定义：

- absolute local deviation from estimated background；
- local C-scan contrast；
- measured damage-mask occupancy if authoritative mask exists。

绝不能把 pseudocolor intensity 自动解释成物理 damage severity。

---

# 23. O3 — CAI Mechanical-Value Oracle

Primary：

\[
V^{CAI}(p|S_t)
=
|y-\hat y_t|
-
|y-\hat y_t^p|
\]

同时 secondary：

\[
V^{MSE}
=
(y-\hat y_t)^2
-
(y-\hat y_t^p)^2
\]

但 primary 排序使用 absolute-error reduction，

因为最终主要指标是 MAE。

---

# 24. Greedy CAI oracle trajectory

对于 source specimen：

```text
S0 = coarse survey
```

每一步：

```text
for every unrefined candidate cell p:
    simulate refining p
    update sparse reconstruction
    run frozen/source-only CAI predictor
    compute V_CAI(p | S_t)

choose:
    p* = argmax V_CAI
```

然后：

\[
S_{t+1}=S_t\cup p^*
\]

直到：

- maximum budget；
- 或所有 cells refined。

保存：

```text
specimen_id
step
budget_before
candidate
value
budget_after
current_prediction
new_prediction
current_error
new_error
```

---

# 25. Oracle is diagnostic, not deployable

必须在所有 artifacts 明确：

```text
CAI oracle uses true CAI and unobserved candidate values.
It is an upper-bound diagnostic, not a deployable policy.
```

禁止在摘要中把 oracle performance 当方法性能。

---

# 26. Outer-test oracle rule

最终 outer held-out domain 可以在所有 protocol 冻结后计算：

```text
oracle upper-bound curve
```

但：

- 不能用于 method selection；
- 不能用于 hyperparameter selection；
- 不能决定是否修改 policy；
- 必须标注 diagnostic upper bound。

GO/NO-GO 最好优先根据 cross-fitted source / preregistered aggregate evidence。

---

# 27. A2 Baselines

同 budget 下至少比较：

## B0 — Uniform

existing registered uniform grid。

## B1 — Random

至少：

```text
100 deterministic seeds
```

报告：

```text
mean
median
5–95 percentile
```

---

## B2 — Center-first

从 impact center 向外 refine。

仅在 impact-center authority 成立时使用。

---

## B3 — Appearance-first

根据 observed/full appearance heuristic 排序。

分清：

- deployable observed heuristic；
- oracle full-image heuristic。

---

## B4 — Reconstruction oracle

最大化：

\[
V^{rec}
\]

---

## B5 — Mechanical CAI oracle

最大化：

\[
V^{CAI}
\]

这是 diagnostic upper bound。

---

# 28. Main oracle plots

必须生成同一 specimen 示例：

# Figure O1

Current sparse scan + mask

# Figure O2

Reconstruction-value map

# Figure O3

Appearance/damage-value map

# Figure O4

CAI mechanical-value map

# Figure O5

Acquisition trajectories

颜色/图例必须统一。

---

# 29. 最重要的 quantitative plot

# Error–Budget Curve

横轴：

\[
B
\]

measurement fraction。

纵轴：

\[
MAE(B)
\]

方法：

```text
Uniform
Random
Appearance-first
Reconstruction-oracle
CAI-oracle
Full
```

---

# 30. Area under error–budget curve

定义：

\[
AUEBC
=
\int MAE(B)dB
\]

使用统一离散 trapezoidal integration。

预算范围固定：

例如：

\[
B\in[B_0,0.25]
\]

避免不同方法积分区间不一致。

越低越好。

---

# 31. Mechanical Sufficiency Budget

定义：

\[
B_{5\%}
=
\min B:
MAE(B)
\le
1.05MAE_{FULL}
\]

其中：

\[
MAE_{FULL}=0.0896358
\]

因此 threshold：

\[
\approx0.09412
\]

同时 secondary：

\[
B_{2.5\%}
\]

和：

\[
B_{7.5\%}
\]

但 primary 是 5%。

---

# 32. Simulated Measurement Saving

相对 uniform：

\[
Saving_{MVA}
=
1-
\frac{
B^{MVA}_{5\%}
}{
B^{uniform}_{5\%}
}
\]

只能称：

```text
simulated measurement reduction
```

禁止：

```text
inspection-time reduction
scanner-time saving
```

---

# 33. A3 — Oracle Headroom Gate

这是整个项目最重要的 GO/NO-GO。

只有通过才开发 policy。

---

## Gate H1 — CAI oracle beats uniform

在关键低预算区间：

\[
6.25\%-25\%
\]

CAI oracle 必须明显优于 uniform。

建议至少满足：

\[
\ge5\%
\]

relative MAE improvement at one preregistered low budget，

且：

```text
>=4/6 domains same direction
```

---

## Gate H2 — CAI oracle beats reconstruction oracle

必须证明：

\[
AUEBC_{CAI-oracle}
<
AUEBC_{reconstruction-oracle}
\]

最好同时：

\[
B_{5\%,CAI}
<
B_{5\%,reconstruction}
\]

否则：

> 机械价值与 reconstruction value 并未产生足够区别。

---

## Gate H3 — CAI oracle beats appearance heuristic

若可信 appearance/damage baseline 存在：

\[
AUEBC_{CAI}
<
AUEBC_{appearance}
\]

否则不要把“damage value 与 mechanical value 不同”写成正式 claim。

---

## Gate H4 — Meaningful adaptive headroom

相对最强 deployable fixed/uniform strategy，

oracle 至少有：

```text
>= 10% relative AUEBC headroom
```

或者：

```text
>= 25% reduction in B_5%
```

两者至少一个达到。

如果 oracle 只提高 1–2%：

\[
\boxed{
MVA\_NO\_GO
}
\]

不要开发复杂 policy。

---

# 34. A3 NO-GO meaning

如果：

```text
CAI oracle ≈ uniform
```

结论：

> 当前数据中的 spatial measurement value 很均匀，adaptive acquisition headroom 不足。

如果：

```text
CAI oracle ≈ reconstruction oracle
```

结论：

> downstream mechanical value 没有明显区别于 image recovery value。

如果：

```text
oracle headroom large only after using target information
```

结论：

> 不可推广。

全部停止。

---

# 35. A4 — Global Task-Aware Static Acquisition

只有 A3 GO 后执行。

先不要马上做 sequential policy。

测试：

> 是否一套全局固定 task-aware pattern 已经足够？

---

# 36. Simple global MVoM mask

第一版不要 STE 神经 mask。

直接利用 source OOF oracle values：

对 region \(k\)：

\[
\bar V_k
=
\frac1N\sum_iV_i^{CAI}(k|S_0)
\]

或者 aggregate rank。

得到：

\[
\boxed{
\text{Global MVoM ranking}
}
\]

测试时：

> 所有 specimen 使用相同 top-ranked refinement pattern。

这非常适合当前小样本。

---

# 37. TACKLE/LOUPE-style differentiable mask

只有 simple global MVoM mask 已显示价值，

再实现：

\[
p_k=\sigma(q_k)
\]

budget-constrained differentiable global mask。

作为 secondary algorithm。

不得让复杂 differentiable mask 阻塞 A4。

---

# 38. A4 comparison

```text
uniform
random
global reconstruction mask
global appearance mask
global MVoM mask
optional differentiable task mask
```

如果 global MVoM 已经接近 oracle：

> specimen-specific policy 不一定需要。

---

# 39. Static-vs-adaptive headroom gate

定义：

\[
H_{adaptive}
=
AUEBC_{global-MVoM}
-
AUEBC_{CAI-oracle}
\]

如果很小：

```text
<3% relative
```

不要开发 adaptive policy。

直接把：

> task-aware fixed acquisition design

作为主方法候选。

---

# 40. A5 — Oracle-Imitation Adaptive Policy

只有：

```text
CAI oracle strongly > global MVoM
```

才执行。

禁止第一版使用 RL。

采用：

\[
\boxed{
\text{supervised oracle imitation}
}
\]

---

# 41. Policy training labels

训练 source specimen 的 oracle trajectories 给出：

\[
a_t^*
\]

或完整：

\[
V^{CAI}_{t,1},...,V^{CAI}_{t,K}
\]

policy 输入只允许当前已观察状态。

不能访问：

```text
unmeasured true pixels
true CAI
full-image features
oracle values
```

---

# 42. Policy state

第一版：

\[
s_t=
[
z_t,
m_t,
\hat y_t,
B_t
]
\]

其中：

### \(z_t\)

当前 sparse/interpolated C-scan 的 frozen ResNet embedding。

### \(m_t\)

measurement mask encoding。

### \(\hat y_t\)

current CAI prediction。

### \(B_t\)

remaining/used budget。

不要一开始加入 laminate context。

---

# 43. Mask encoding

先用简单：

```text
candidate cell measured/refined flags
```

即：

\[
m_t\in\{0,1,2\}^K
\]

必要时加入：

```text
fraction of measured points per cell
```

不要先做大型 CNN mask encoder。

---

# 44. Candidate features

对 candidate region \(k\)：

```text
normalized x
normalized y
distance to impact center if authoritative
current refinement level
local interpolation gradient
local interpolation variance proxy
distance to nearest measured point
```

这些都必须由当前 observed state 计算。

不能使用 full unobserved image。

---

# 45. Policy architecture

第一版：

```text
global state embedding
        ↓
small MLP

candidate features
        ↓
small MLP

concat
        ↓
candidate score
```

共享 scorer：

\[
s_k=f_\theta(s_t,c_k)
\]

trainable parameters：

\[
<200k
\]

最好：

\[
<100k.
\]

---

# 46. Policy objective

Primary 不建议回归 exact value。

采用 ranking。

若：

\[
V_a^{CAI}>V_b^{CAI}
\]

则要求：

\[
s_a>s_b
\]

可使用：

```text
pairwise logistic ranking loss
```

加一个可选：

```text
top-1 cross entropy
```

不要一开始同时加 5 个 loss。

---

# 47. Policy evaluation

在 strict held-out domain：

每一步：

```text
policy sees only current partial state
→ chooses cell
→ simulator reveals corresponding true measurements
→ predictor updates
```

直到各预算 checkpoint。

报告：

\[
MAE(B)
\]

和：

\[
AUEBC
\]

---

# 48. A5 comparators

```text
uniform
random
center-first
appearance-driven deployable heuristic
reconstruction-error heuristic
global MVoM mask
imitation policy
oracle upper bound
```

其中 oracle 单独虚线显示，不能和 deployable method 混为一类。

---

# 49. A5 GO

至少：

\[
AUEBC_{policy}
<
AUEBC_{global-MVoM}
\]

并：

```text
>=4/6 domains improve
```

最好同时达到：

\[
B^{policy}_{5\%}
<
B^{global}_{5\%}.
\]

---

# 50. Oracle gap closure

定义：

\[
GapClosure=
\frac{
AUEBC_{baseline}-AUEBC_{policy}
}{
AUEBC_{baseline}-AUEBC_{oracle}
}
\]

报告 policy 关闭了多少 oracle headroom。

如果：

```text
GapClosure < 20%
```

说明当前数据不足以学 adaptive behavior。

不要增加 Transformer/RL。

---

# 51. A6 — Laminate-Conditioned Acquisition

只有 A5 GO 后执行。

首先建立：

```text
docs/MVA_LAMINATE_AUTHORITY_AUDIT.md
```

确认 exact stacking sequences。

---

# 52. Laminate representation hierarchy

按复杂度严格递进：

## L0

无 laminate information。

## L1

```text
ply_count
layup_family
```

仅 baseline。

## L2

orientation histogram：

\[
[
p_0,p_{90},p_{45},p_{-45}
]
\]

## L3

标准 lamination parameters。

只有 exact stacking sequence + thickness authority 完整时允许。

---

# 53. Do not fabricate lamination parameters

如果 specimen-level sequence 不完整：

禁止根据 domain label 猜 sequence。

只使用 L1/L2。

---

# 54. Conditioning mechanism

最简单：

\[
z_L=g(L)
\]

然后：

\[
s_k=f(s_t,c_k,z_L)
\]

不要一开始设计：

```text
laminate Transformer
physics GNN
```

先验证：

> laminate context 是否改变 measurement value prediction。

---

# 55. Critical laminate test

比较：

```text
policy without laminate context
policy with laminate context
```

尤其：

```text
leave-one-ply-count-out
leave-one-layup-family-out
```

如果 conditioned policy 只在 seen configurations 上提高：

> 不支持 transfer claim。

---

# 56. A7 — Structured transfer

Primary：

# Six-domain LODO

然后：

# Leave-one-ply-count-out

```text
16 + 24 → 8
8 + 24 → 16
8 + 16 → 24
```

# Leave-one-layup-family-out

```text
cross-ply → quasi-isotropic
quasi-isotropic → cross-ply
```

所有：

- acquisition design；
- oracle imitation；
- policy model；
- hyperparameters；

只能使用 source。

---

# 57. Strongest scientific comparison

最终必须对比：

\[
\boxed{
\text{Reconstruction-driven sampling}
}
\]

vs

\[
\boxed{
\text{Mechanical-value-driven sampling}
}
\]

在同 budget 下同时报告：

```text
reconstruction MSE
SSIM
CAI MAE
```

最理想现象：

```text
Reconstruction method:
better PSNR/SSIM
worse CAI

MVA:
worse or equal image fidelity
better CAI
```

这将直接支持：

\[
\boxed{
\text{image fidelity is not mechanical utility}
}
\]

---

# 58. Do not optimize policy against historical outer results

当前 6 domains 已经经过大量 researcher exposure。

因此：

- 代码层面继续严格 outer isolation；
- 不允许手工根据某一 domain 调策略；
- 所有 candidate sets 在 protocol 冻结前定义；
- 最终论文必须诚实说明 current cohort repeatedly informed method development。

若以后得到独立数据：

> reserve it as final external confirmation。

---

# 59. Code structure

新建：

```text
src/cmc_bbdm/mva/
```

建议：

```text
authority.py
acquisition_grid.py
measurement_state.py
refinement_simulator.py
interpolation.py

cai_evaluator.py
crossfit.py

reconstruction_value.py
appearance_value.py
mechanical_value.py

oracle.py
oracle_trajectory.py
budget_metrics.py

global_mask.py
differentiable_mask.py

policy_state.py
candidate_features.py
ranking_policy.py
imitation.py

laminate_context.py

evaluation.py
statistics.py
artifacts.py
replay.py
```

---

# 60. Tests

至少：

```text
test_mva_baseline_reproduction.py

test_acquisition_grid_nested.py
test_measurements_only_added.py
test_budget_counts_unique_measurements.py
test_measured_values_restored_exactly.py

test_mva_oracle_uses_oof_predictor.py
test_oracle_candidate_does_not_access_future_state.py

test_reconstruction_value_definition.py
test_mechanical_value_definition.py

test_outer_domain_not_used_for_oracle_training.py
test_outer_domain_not_used_for_policy_selection.py

test_policy_never_reads_unobserved_pixels.py
test_policy_never_reads_true_cai.py

test_budget_curve_monotonic_measurement_count.py
test_auebc.py
test_b5_metric.py

test_mva_replay.py
```

---

# 61. Result layout

```text
results/mva/
```

---

## A0

```text
a0_acquisition_audit/
```

---

## A1

```text
a1_simulator/
```

---

## A2

```text
a2_oracle_value/
```

Required:

```text
oracle_values.parquet
oracle_trajectories.parquet

uniform_curve.csv
random_curve.csv
appearance_curve.csv
reconstruction_oracle_curve.csv
mechanical_oracle_curve.csv

budget_metrics.csv
domain_metrics.csv

summary.json
REPORT.md
```

---

## A4

```text
a4_global_task_mask/
```

---

## A5

```text
a5_imitation_policy/
```

---

## A6

```text
a6_laminate_policy/
```

---

# 62. Required A2 report questions

`REPORT.md` 必须直接回答：

1. CAI oracle 是否明显优于 uniform？
2. CAI oracle 是否优于 reconstruction oracle？
3. CAI oracle 是否优于 appearance/damage heuristic？
4. advantage 出现在哪些 budgets？
5. 是否所有 domain 都存在 headroom？
6. 哪些 specimens 得益最多？
7. mechanical-value map 与 reconstruction map 有多大差异？
8. mechanical-value map 是否只是 damage map 的复制？
9. Oracle 能把 \(B_{5\%}\) 降低多少？
10. headroom 是否足以支持学习 policy？

---

# 63. Additional diagnostic — value-map similarity

对每 specimen：

比较：

\[
V^{CAI}
\]

与：

\[
V^{rec}
\]

的：

```text
Pearson
Spearman
top-k overlap
rank-biased overlap
```

同样比较 appearance value。

如果：

\[
\rho(V^{CAI},V^{rec})\approx1
\]

则：

> mechanical-value concept 没有形成新的 acquisition principle。

---

# 64. Top-k acquisition overlap

例如 top 10% candidate cells：

\[
Overlap_{CAI,recon}
=
\frac{
|Top_{CAI}\cap Top_{recon}|
}{
|Top_{CAI}|
}
\]

如果 overlap 很低但 CAI oracle 明显更优：

这是非常强的结果。

---

# 65. Oracle stability

检查同一 specimen：

\[
a_t^*
\]

对：

- bootstrap predictor；
- regression family；
- sparse interpolation；

是否极端不稳定。

如果 oracle value ranking 本身高度不稳定：

> policy supervision 不可靠。

不要直接训练 policy。

---

# 66. Mechanical-value uncertainty

可用不同 source-bootstrap CAI predictors 得：

\[
V_1,\ldots,V_M
\]

计算：

\[
\mu_V,\sigma_V.
\]

可选 robust oracle：

\[
score(p)
=
\mu_V(p)-\lambda\sigma_V(p)
\]

但仅作为 secondary。

第一版 primary 仍使用 mean oracle。

---

# 67. Stop rules

## Stop 1

Full baseline 不能精确复算：

```text
STOP
```

## Stop 2

Simulator 改变 measured values：

```text
STOP
```

## Stop 3

CAI oracle 没有 meaningful headroom：

```text
MVA_NO_GO
```

## Stop 4

CAI oracle ≈ reconstruction oracle：

```text
task-driven novelty unsupported
```

## Stop 5

Global task-aware mask 已关闭绝大多数 oracle gap：

```text
do not build adaptive policy
```

## Stop 6

Imitation policy 无法关闭至少 20% oracle gap：

```text
do not add RL/Transformer
```

## Stop 7

Laminate conditioning 无 structured-transfer gain：

```text
remove laminate module
```

---

# 68. Explicitly forbidden rescue behavior

如果结果失败，不允许：

```text
try dozens of backbones
open outer test repeatedly
choose budget after seeing target
change oracle definition post hoc
add GNN
add Transformer
add diffusion
add RL
add attention modules
```

除非形成一个新的冻结 hypothesis。

---

# 69. Expected paper narrative if successful

不要写：

> Sparse scanning can reduce image pixels.

写：

> **Dense ultrasonic inspection is conventionally designed to recover the damage field, whereas structural integrity assessment only requires measurements that are informative about residual mechanical performance. We therefore formulate ultrasonic acquisition as a downstream mechanical-value optimization problem. Starting from a coarse survey, the proposed method estimates the value of candidate local measurements according to their expected contribution to CAI prediction and selectively allocates additional sensing budget to mechanically informative regions.**

---

# 70. Core scientific distinction

Existing paradigm:

\[
\boxed{
\text{Where is image information missing?}
}
\]

or:

\[
\boxed{
\text{Where does damage look strong?}
}
\]

Our paradigm:

\[
\boxed{
\text{Where will an additional measurement change the structural-integrity estimate?}
}
\]

---

# 71. Potential contributions if supported

## Contribution 1

Formalize:

\[
\boxed{
\text{Mechanical Value of Measurement}
}
\]

for ultrasonic CAI assessment.

---

## Contribution 2

Demonstrate experimentally that:

\[
\text{reconstruction value}
\neq
\text{mechanical value}
\]

under controlled acquisition simulation.

---

## Contribution 3

Develop task-driven coarse-to-fine ultrasonic acquisition that allocates measurements according to CAI value rather than full-field reconstruction fidelity.

---

## Contribution 4

If A6 PASS:

show that laminate structural context helps predict where measurements are mechanically informative under unseen configurations.

---

# 72. Claim boundary

Until real scanner validation exists:

use:

```text
retrospective acquisition simulation
simulated measurement budget
measurement-location reduction
task-driven sampling design
```

Do not use:

```text
real-time adaptive scanner
inspection time reduced by X%
physical acquisition speedup
online industrial deployment
```

---

# 73. First execution package

现在只执行：

```text
A0
A1
A2
A3
```

也就是说：

1. repository/acquisition audit；
2. nested coarse-to-fine simulator；
3. uniform/random/reconstruction/appearance/CAI oracle；
4. error–budget curves；
5. AUEBC；
6. \(B_{5\%}\)；
7. oracle map similarity；
8. domain-level headroom；
9. GO/NO-GO。

**暂时不要实现 A4–A7。**

---

# 74. First required documents

提交正式实验前先生成：

```text
docs/MVA_REFERENCE_METHOD_AUDIT.md
docs/MVA_ACQUISITION_SEMANTICS_AUDIT.md
docs/MVA_A0_A3_PROTOCOL.md
docs/MVA_CLAIM_EVIDENCE_MATRIX.md
```

---

# 75. Claim-Evidence Matrix

初始：

| Claim | Status |
|---|---|
| Dense C-scan predicts CAI | PROVEN |
| Spatial organization matters | PROVEN |
| Uniform sparse sensing retains much CAI value | PROVEN |
| Reconstruction fidelity is sufficient objective | NOT SUPPORTED |
| Mechanical measurement value differs from reconstruction value | TO TEST |
| Mechanical oracle outperforms uniform sampling | TO TEST |
| Mechanical oracle outperforms reconstruction-driven sampling | TO TEST |
| Adaptive headroom is large enough to learn a policy | TO TEST |
| Global task-aware mask improves fixed acquisition | LOCKED |
| Deployable policy can imitate mechanical oracle | LOCKED |
| Laminate context improves acquisition decisions | LOCKED |

---

# 76. Exact execution order

```text
STEP 0
Checkout and audit current main.

STEP 1
Reproduce FULL baseline 0.0896358.

STEP 2
Audit actual C-scan/image acquisition semantics.

STEP 3
Freeze normalized acquisition-grid definition.

STEP 4
Implement nested coarse-to-fine simulator.

STEP 5
Validate measurement budget accounting.

STEP 6
Validate P5-equivalent uniform sampling reproduction.

STEP 7
Freeze source-only cross-fitting protocol.

STEP 8
Generate reconstruction-value oracle.

STEP 9
Generate appearance/damage baseline if scientifically valid.

STEP 10
Generate CAI mechanical-value oracle.

STEP 11
Generate greedy oracle trajectories.

STEP 12
Run uniform + random controls.

STEP 13
Compute MAE-budget curves.

STEP 14
Compute AUEBC.

STEP 15
Compute B_5%.

STEP 16
Compare value-map correlations/top-k overlap.

STEP 17
Compute per-domain oracle headroom.

STEP 18
Run synchronized bootstrap.

STEP 19
Issue:

MVA_ORACLE_GO
or
MVA_ORACLE_NO_GO

STEP 20
STOP.

Do not implement policy in this execution.
```

---

# 77. Final principle

整个阶段始终遵守：

\[
\boxed{
\text{Do not design the policy before proving that an oracle policy has value.}
}
\]

\[
\boxed{
\text{Do not optimize image recovery when the engineering task is mechanical assessment.}
}
\]

\[
\boxed{
\text{Do not claim physical scan-time savings from retrospective screenshot subsampling.}
}
\]

以及最重要的一句：

\[
\boxed{
\text{Measure what changes the mechanical decision.}
}
\]

当前唯一目标是回答：

> **If the full C-scan is available only during training, does knowledge of the true mechanical target reveal a spatial acquisition strategy that is substantially more efficient than uniform, appearance-driven, or reconstruction-driven sampling?**

只有答案是 YES，

才值得继续开发真正的 MVA acquisition policy。