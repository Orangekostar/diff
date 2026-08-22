# Codex Prompt — Discovering and Validating a Transferable Mechanically Sufficient Spatial Scale for C-scan CAI Assessment

## 0. Role

你现在继续接管当前 CFRP ultrasonic C-scan → CAI 项目。

当前仓库：

```bash
git@github.com:Orangekostar/diff.git
```

当前主分支已经包含：

- G1 / G2
- P1 / P3 / P5 / P6 / P7
- A0–A5 MASI
- E1–E3 multi-view experiments

当前最近已完成多视图结果提交：

```text
1650a2d
```

目标期刊：

> **Advanced Engineering Informatics**

本阶段不要再继续：

- feature-level invariance；
- MASI factorization；
- Cooperative Learning；
- static multi-view fusion；
- GMvR；
- MoE；
- full-image diffusion reconstruction；
- pixel-frequency diffusion。

这些方向已有明确实验结果。

---

# 1. 当前新的科学主线

当前已有结果共同指向：

```text
A/W/H
    ↓
too coarse / mechanically insufficient

Full C-scan
    ↓
strong CAI information

Spatial destruction
    ↓
strong performance loss

Reduced sensing
    ↓
large fraction of CAI information retained

Feature invariance
    ↓
mechanical information loss

Multi-view fusion
    ↓
no gain in ordinary 6-domain LODO

Structured ply / layup extrapolation
    ↓
reduced-resolution representations show positive signals
```

因此下一阶段不再研究：

> 如何融合 FULL / 50% / 25%？

新的科学问题是：

> **At what spatial scale does internal impact damage form a mechanically sufficient and transferable representation for CAI assessment?**

中文：

> **内部冲击损伤在什么空间尺度上能够形成既具有 CAI 机械充分性、又具有跨铺层和层数迁移能力的表征？**

---

# 2. 新的 Positive Hypothesis

不要把主张写成：

> Fine details are useless.

新的 positive hypothesis 是：

> **Internal impact damage contains a compact mesoscale representation that preserves mechanically important spatial organization and provides transferable CAI information across laminate configurations.**

最终要寻找一个：

\[
\boxed{s^*}
\]

称为：

# Mechanically Sufficient Spatial Scale — MSSS

其定义不是：

\[
s^*=\arg\min MAE
\]

而是：

> **在保持 CAI mechanical performance 和 spatial specificity 的前提下，能够使用的最紧凑空间尺度。**

---

# 3. 本阶段只有两个主任务

## S1 — Mechanically Sufficient Scale Discovery

回答：

> 哪个空间尺度足以表达 CAI-relevant damage information？

## S2 — Structured Transfer Validation

回答：

> 这个尺度是否在 unseen ply count / layup family 下比 full-resolution representation 更 transferable？

所有新增代码、实验和论文修改都只能围绕这两个问题。

---

# 4. 必须首先阅读的参考工作

不要直接复制外部 architecture。

重点理解实验思想。

---

## Reference A — MFC-MIL, ICLR 2025

论文：

> A Multiscale Frequency Domain Causal Framework for Enhanced Pathological Analysis

官方代码：

```text
WissingChen/MFC-MIL
```

重点阅读：

- Multi-scale Spatial Representation Module
- Frequency Domain Structure Representation Module
- 多尺度和频率信息如何分开进行实验验证
- intervention / ablation 设计

本项目只借：

> **spatial scale 和 frequency structure 应分开验证**

不要搬 MIL architecture。

---

## Reference B — WaveRNet

官方仓库：

```text
Chanchan-Wang/WaveRNet
```

重点理解：

- DWT / wavelet decomposition
- multi-frequency representation
- domain-generalization evaluation
- frequency-aware test strategy

本项目只借：

> wavelet decomposition 的工程实现和 frequency-scale evaluation 思想。

---

## Reference C — FreqGRL

论文：

> Frequency-guided generalizable representation learning for cross-domain few-shot learning

重点理解：

> 不同 frequency bands 的 domain transferability 必须由任务验证。

不要直接采用：

```text
low frequency = nuisance
high frequency = transferable
```

这样的结论。

本项目必须通过 CAI 实验自行识别 scale。

---

# 5. 必须先冻结已有证据

创建：

```text
docs/MSSS_EXISTING_EVIDENCE.md
```

明确列出：

## E1

A/W/H → CAI：

```text
FAIL
```

## E2

Full C-scan：

```text
mechanically informative
```

## E3

P3 spatial destruction：

```text
spatial organization necessary
```

## E4

P5 reduced sensing：

```text
reduced observations retain substantial CAI value
```

## E5

P6：

```text
full-image generative reconstruction unnecessary
```

## E6

MASI A5：

```text
feature-level invariance insufficient
```

## E7

Multi-view E1–E3：

```text
FULL / 50 / 25 predictively redundant under ordinary LODO
```

## E8

Structured stress tests：

```text
reduced-resolution representations show exploratory positive transfer signals
```

最后明确：

> 当前还没有证明 MSSS 存在。

状态：

```text
MSSS = TO TEST
TRANSFERABILITY = TO TEST
```

---

# 6. S1 总体原则

S1 绝对不能只是：

```text
try many resize ratios
choose the best MAE
```

这会退化成超参数搜索。

必须通过三条彼此独立的 scale axis 交叉验证：

\[
\boxed{
Sampling\ scale
}
\]

\[
\boxed{
Spatial\ smoothing\ scale
}
\]

\[
\boxed{
Frequency/wavelet\ scale
}
\]

如果三条轴均指向相近的 performance plateau / transition region，才能支持：

> 存在真实 mechanically sufficient scale。

---

# 7. S1-A — Sampling Scale Axis

复用 P5 的 sampling implementation。

不要重新实现不同 sampling semantics。

首先审计并绑定：

```text
sampling coordinates
rounding
endpoint policy
measured-point restoration
interpolation
```

Primary interpolation：

```text
BILINEAR
```

避免 interpolation 变成额外变量。

---

# 8. Sampling Density Grid

新增更细的 density grid：

```text
100%
75%
62.5%
50%
37.5%
25%
18.75%
12.5%
6.25%
```

如果原始 grid 无法精确实现某个比例：

> 使用最接近且可重复的 integer stride / coordinate count，并记录 effective density。

不要强制虚假的百分比。

输出：

```text
requested_density
effective_density
sampling_stride
n_measured_points
```

---

# 9. S1-B — Gaussian Spatial Scale-Space

保持：

```text
image resolution
pixel grid
ResNet input resolution
```

全部不变。

仅控制 spatial bandwidth：

\[
D_\sigma=G_\sigma*D
\]

候选：

```text
sigma_px:
0
0.5
1
1.5
2
3
4
6
8
```

如果可以从数据 metadata 恢复 pixel→mm：

同时记录：

\[
\sigma_{\mathrm{mm}}
\]

论文优先使用物理尺度。

如果无法可靠恢复：

> 不得制造 mm 数字，保留 pixel scale。

---

# 10. Gaussian Scale 控制

必须避免 blur 改变：

- overall image size；
- intensity range semantics；
- normalization pipeline。

使用相同：

```text
Frozen ResNet18
same preprocessing
same fold-local PCA
same regression family
```

唯一变量：

\[
\sigma
\]

---

# 11. S1-C — Wavelet Scale Axis

优先使用 PyWavelets 或经过审计的轻量 DWT 实现。

第一轮 wavelet：

```text
haar
db2
db4
```

primary：

```text
db2
```

其余作为 sensitivity。

---

# 12. Multi-Level DWT

对于 full C-scan：

\[
D
\rightarrow
\{LL_j,LH_j,HL_j,HH_j\}
\]

建立 cumulative low-pass reconstruction：

```text
level 0 = full
level 1
level 2
level 3
...
```

并恢复到与原始 ResNet 输入完全相同尺寸。

同时建立：

```text
low-pass only
low + selected detail bands
```

用于区分：

> spatial scale 与 edge/detail information。

---

# 13. 不要提前假设哪一个频带最好

禁止在代码/文档中写：

```text
low frequency = mechanical
high frequency = nuisance
```

只能写：

```text
candidate scale bands
```

最终由 CAI 与 transfer evaluation 判断。

---

# 14. Fourier Axis 可作为 Secondary

如果实现成本低，可增加 radial FFT cutoff：

\[
D_{f_c}
\]

候选 normalized cutoff：

```text
1.00
0.75
0.50
0.35
0.25
0.15
0.10
```

但：

> Fourier 不得阻塞 S1。

Sampling + Gaussian + Wavelet 为 primary 三轴。

---

# 15. 每个 Scale 必须使用同一 Predictor Protocol

Primary：

```text
Frozen ResNet18
→ fold-local StandardScaler
→ fold-local PCA
→ registered regression model
→ CAI
```

不要在不同 scale 下：

- 换 backbone；
- fine-tune encoder；
- 加新的 CNN；
- 改 target。

目的：

> isolate scale effect。

---

# 16. Scale Model Selection

对于每一个 candidate scale：

允许 inner-domain 选择：

- PCA dim；
- regressor；
- regressor hyperparameters。

但是：

> 不允许 outer target domain 选择 scale。

Scale selection 必须 source-only。

---

# 17. Mechanically Sufficient Set

不要用：

\[
argmin_s MAE
\]

定义 MSSS。

定义：

\[
\mathcal S_{\rm MS}
=
\{
s:
MAE_s
\le
MAE_{\rm full}+\delta
\}
\]

其中：

\[
\delta
\]

必须在运行正式 outer scale test 前冻结。

---

# 18. Non-Inferiority Margin

Primary 推荐：

\[
\delta_{rel}=5\%
\]

相对于 FULL MAE。

即：

\[
MAE_s
\le
1.05\times MAE_{\rm full}
\]

视为 mechanically sufficient candidate。

同时报告：

```text
2.5%
5%
7.5%
```

sensitivity，

但 5% 是 primary。

---

# 19. MSSS Definition

在：

\[
\mathcal S_{\rm MS}
\]

中选择：

> **coarsest / most compact candidate scale**

作为：

\[
s^*
\]

注意：

不同 axis 的 “coarse” 定义不同。

Sampling：

```text
lowest measurement density
```

Gaussian：

```text
largest sigma
```

Wavelet：

```text
coarsest retained scale
```

---

# 20. 还必须加入 Spatial Specificity Gate

一个 scale：

\[
s
\]

只有 CAI MAE 不劣于 FULL 还不够。

必须证明：

> 该尺度仍然使用真正的 spatial organization。

所以对每一个重点 candidate：

\[
D_s
\]

生成：

\[
T^-(D_s)
\]

primary destructive control：

```text
registered P3 8×8 patch shuffle
```

必要时：

```text
pixel shuffle
```

作为 secondary。

---

# 21. Spatial Specificity Requirement

定义：

\[
SSG(s)
=
MAE[T^-(D_s)]-MAE[D_s]
\]

要求：

\[
SSG(s)>0
\]

并且在多数 held-out domains 上同向。

Primary promotion：

```text
>=4/6 domains positive
```

最好：

```text
simultaneous CI lower > 0
```

---

# 22. MSSS 必须同时满足两个条件

最终：

\[
s^*
\]

必须：

## Mechanical Sufficiency

\[
MAE(s^*)
\le
MAE(full)+\delta
\]

以及：

## Spatial Specificity

\[
MAE(T^-s^*)
>
MAE(s^*)
\]

否则不能称：

> mechanically sufficient spatial representation。

---

# 23. 三轴 Convergence Analysis

S1 完成后，需要检查：

```text
sampling MSSS
Gaussian MSSS
Wavelet MSSS
```

是否指向相近的信息尺度 regime。

不能强行把它们换算成同一个数字。

可以定义 normalized spatial retention index：

\[
r_s\in[0,1]
\]

仅用于可视化。

真正结果仍分别报告。

---

# 24. 最理想的 S1 结果

希望观察到：

```text
FULL
↓
large performance plateau
↓
MSSS boundary
↓
rapid performance degradation
↓
A/W/H / over-coarse regime
```

也就是：

\[
\boxed{
\text{plateau + knee}
}
\]

而不是一条随机震荡的曲线。

---

# 25. S1 GO / NO-GO

## GO

至少两条 independent scale axes：

- 存在 mechanically sufficient plateau；
- scale boundary 相对稳定；
- P3 spatial specificity 在 plateau 上保持。

## STRONG GO

三条 axis 均出现相似 scale regime。

## NO-GO

如果：

- scale performance 无规律；
- MSSS 对 outer fold 极不稳定；
- coarse scale 的 spatial specificity 消失；

则不得宣称 universal MSSS。

---

# 26. 如果 S1 NO-GO

不要强行停止整个项目。

转入：

> configuration-dependent mechanical scale

分析。

检查：

```text
ply count
layup family
domain
damage size
```

与 selected scale 的关系。

这可能形成：

# Scale–Laminate Coupling

但必须作为新假设重新验证。

---

# 27. S1 必须输出的主文件

```text
results/msss/s1_scale_discovery/
```

至少：

```text
sampling_curve.csv
gaussian_curve.csv
wavelet_curve.csv
fourier_curve.csv          # optional

domain_scale_metrics.csv
spatial_specificity.csv
msss_selection.csv
selection_stability.csv

summary.json
REPORT.md
```

---

# 28. S1 主图

必须自动生成：

## Figure A

Sampling density vs CAI MAE

## Figure B

Gaussian spatial scale vs CAI MAE

## Figure C

Wavelet scale vs CAI MAE

## Figure D

Mechanical sufficiency + spatial specificity combined plot

突出：

```text
FULL
MSSS
OVER-COARSE
```

---

# 29. S2 — Structured Transfer Validation

S2 只允许在 S1 protocol 冻结后执行。

核心原则：

\[
\boxed{
\text{Target domain cannot participate in scale selection}
}
\]

---

# 30. Source-Only MSSS Selection

每一个 outer transfer task：

1. 只使用 source domains；
2. 在 source 内 nested selection；
3. 得到：
   \[
   s^*_{\rm source}
   \]
4. 完全冻结；
5. 测试 unseen target。

禁止：

```text
see target performance
→ change scale
```

---

# 31. S2-A — Six-Domain LODO

继续现有：

```text
6 datasets
each one held out
```

作用：

> ordinary configuration transfer。

比较：

```text
FULL
fixed 25%
source-selected s*
over-coarse
```

---

# 32. S2-B — Leave-One-Ply-Count-Out

必须从 metadata / source authority 正式确认：

```text
8 ply
16 ply
24 ply
```

mapping。

三个 transfer tasks：

```text
16 + 24 → 8
8 + 24 → 16
8 + 16 → 24
```

每一个：

> source-only select \(s^*\)。

---

# 33. S2-C — Leave-One-Layup-Family-Out

正式确认：

```text
cross-ply
quasi-isotropic
```

两个任务：

```text
CP → QI
QI → CP
```

同样：

> target family 不能参与 scale selection。

---

# 34. Optional S2-D — Impact Condition Shift

如果 metadata 支持足够 sample：

可设计：

```text
leave-one-impact-energy-bin-out
leave-one-impactor-type-out
```

但必须先做样本量 audit。

不能为了增加实验强行切出极小 group。

---

# 35. S2 Primary Comparators

所有 structured transfer 统一比较：

## B0 — FULL

full C-scan。

## B1 — Fixed 25%

现有 P5 工程 baseline。

## B2 — Source-Selected MSSS

\[
s^*_{\rm source}
\]

主方法。

## B3 — Over-Coarse

使用已在 S1 中确认跨过 sufficiency boundary 的尺度。

作用：

> 证明不是越粗越好。

---

# 36. Transfer Gain

定义：

\[
TG(s)
=
MAE_{\rm FULL}
-
MAE_s
\]

因此：

\[
TG>0
\]

表示：

> scale representation 比 FULL 在 target shift 下更好。

必须逐 transfer group 报告。

---

# 37. Relative Transfer Gain

同时：

\[
RTG(s)
=
\frac{
MAE_{\rm FULL}-MAE_s
}{
MAE_{\rm FULL}
}
\]

报告百分比。

---

# 38. Transferability Positive Claim

至少需要：

\[
MAE_{\rm target}(s^*)
\le
MAE_{\rm target}(FULL)
\]

在多数 structured shifts 成立。

Strong claim：

\[
TG(s^*)>0
\]

在：

```text
>=2/3 ply-count shifts
and
>=1/2 layup shifts
```

成立。

更理想：

```text
all or nearly all structured shifts
```

---

# 39. Scale Selection Stability

记录每一个 source split 选出的：

\[
s^*
\]

分析：

```text
selected_scale
source domains
target structure
```

如果多数：

\[
s^*
\]

落在相邻 regime：

支持：

> transferable mesoscale band。

---

# 40. 如果 Scale 随 Laminate Systematic Change

不要判失败。

分析：

\[
s^*=f(
ply\ count,
layup
)
\]

若出现规律：

例如：

```text
increasing ply count
→ systematic shift in sufficient scale
```

进入新 hypothesis：

> laminate-dependent mechanically sufficient scale。

但是：

> 只能 exploratory，不能 post-hoc 直接作为主 conclusion。

---

# 41. S2 必须输出

```text
results/msss/s2_transfer/
```

包含：

```text
six_domain_lodo.csv
leave_ply.csv
leave_layup.csv
impact_shift.csv       # optional

scale_selection.csv
transfer_gain.csv
group_metrics.csv
bootstrap.csv

summary.json
REPORT.md
```

---

# 42. Statistical Protocol

保持当前项目已有：

- specimen-level unit；
- equal-domain evaluation；
- synchronized bootstrap；
- source-only model selection；
- immutable authorities；
- hashes/checksums。

---

# 43. 禁止 Image-Level Split

所有 derived images：

```text
FULL
50%
25%
Gaussian
Wavelet
```

属于同一个 specimen。

永远：

\[
\boxed{
split\ by\ specimen
}
\]

---

# 44. Scale Search 也不能看到 Outer Target

不仅 regressor 不能看到。

以下全部不能看到 target：

```text
scale
sigma
wavelet
level
frequency cutoff
PCA dim
regressor
non-inferiority selection
```

---

# 45. Reference Methods 的使用方式

新建：

```text
docs/MSSS_REFERENCE_METHOD_AUDIT.md
```

记录：

## MFC-MIL

借：

- multi-scale spatial analysis
- frequency-domain structural analysis
- intervention-separated ablation

不借：

- MIL architecture
- pathology task

## WaveRNet

借：

- DWT implementation ideas
- frequency-scale decomposition
- DG evaluation perspective

不借：

- retinal-specific segmentation network

## FreqGRL

借：

> frequency-domain transferability must be empirically diagnosed

不借：

> “low/high frequency 哪个一定 transferable”的具体任务结论。

---

# 46. 新代码结构

建议：

```text
src/cmc_bbdm/msss/
```

包含：

```text
authority.py

sampling_scale.py
gaussian_scale.py
wavelet_scale.py
fourier_scale.py

scale_features.py
scale_evaluator.py
noninferiority.py
msss_selector.py

spatial_specificity.py

transfer_tasks.py
source_only_selection.py
transfer_metrics.py

statistics.py
artifacts.py
replay.py
```

---

# 47. 必须复用

不要重写：

```text
Frozen ResNet18
PCA
regression registry
LODO protocol
bootstrap
sampling coordinate logic
P3 corruption implementation
artifact hashing
```

---

# 48. Tests

新增：

```text
test_msss_sampling_semantics.py
test_msss_gaussian_scale.py
test_msss_wavelet_reconstruction.py

test_msss_specimen_grouping.py
test_msss_target_not_seen_in_scale_selection.py

test_msss_noninferiority_selection.py
test_msss_spatial_specificity.py

test_leave_ply_authority.py
test_leave_layup_authority.py

test_transfer_gain.py
test_msss_replay.py
```

尤其必须：

```text
test_target_not_seen_in_scale_selection
```

---

# 49. 新 Claim-Evidence Matrix

创建：

```text
docs/MSSS_CLAIM_EVIDENCE_MATRIX.md
```

初始：

| Claim | Status |
|---|---|
| spatial organization matters | PROVEN |
| reduced sampling retains CAI value | PROVEN, protocol-specific |
| mechanically sufficient scale exists | TO TEST |
| same scale appears across independent scale axes | TO TEST |
| MSSS remains spatially specific | TO TEST |
| MSSS is transferable across ply count | TO TEST |
| MSSS is transferable across layup | TO TEST |
| MSSS improves severe structural extrapolation | TO TEST |
| scale depends systematically on laminate architecture | EXPLORATORY |

---

# 50. 最重要的执行顺序

严格：

```text
STEP 0
Audit current HEAD and reproduce frozen baselines.

STEP 1
Freeze S1 scientific protocol.

STEP 2
Implement sampling scale axis.

STEP 3
Implement Gaussian scale axis.

STEP 4
Implement Wavelet scale axis.

STEP 5
Run source-domain nested scale curves.

STEP 6
Apply non-inferiority definition.

STEP 7
Identify candidate MSSS.

STEP 8
Run P3-style spatial specificity at candidate scales.

STEP 9
Issue S1 GO / NO-GO.

IF S1 GO:
    continue.

STEP 10
Freeze S2 transfer protocol.

STEP 11
Run six-domain source-selected MSSS LODO.

STEP 12
Run leave-one-ply-count-out.

STEP 13
Run leave-one-layup-family-out.

STEP 14
Compare FULL / fixed25 / MSSS / over-coarse.

STEP 15
Compute TG / RTG and bootstrap.

STEP 16
Audit scale-selection stability.

STEP 17
Issue S2 GO / NO-GO.

STEP 18
Only after S1/S2 freeze:
decide whether scale-adaptive learning is justified.
```

---

# 51. 不要提前开发 Scale-Adaptive Network

当前阶段禁止直接做：

```text
scale attention
multi-scale transformer
MoE
wavelet CNN
frequency adapter
Diffusion
```

只有当 S2 证明：

```text
different structures prefer systematically different scales
```

才允许进入下一阶段：

# Scale-Adaptive Mechanical Representation

否则：

> simple MSSS is already the correct scientific conclusion。

---

# 52. 如果后续允许 Scale-Adaptive Extension

只有 S2 出现 clear architecture-dependent scale 后，

才设计：

\[
w_s(x)
\]

自动选择多个 scale experts。

但该阶段另开 protocol。

不要和 MSSS discovery 混在同一次实验中。

---

# 53. S1 成功后应得到什么结论

如果存在稳定 plateau：

> **CAI-relevant internal damage information can be represented at a compact spatial scale substantially coarser than the full-resolution C-scan while retaining its essential spatial organization.**

这就是：

> mechanically sufficient。

---

# 54. S2 成功后应得到什么结论

如果：

\[
MAE_{shift}(s^*)<MAE_{shift}(FULL)
\]

则：

> **The mechanically sufficient representation is not only compact but more transferable across laminate configurations.**

这是：

> transferable。

---

# 55. 最终 Positive Story

最终论文不是：

```text
high-resolution details are useless
```

而是：

```text
Internal damage
↓
contains spatial mechanical information
↓
that information is concentrated at a sufficient scale
↓
the sufficient scale preserves morphology
↓
the compact representation transfers across laminate structures
```

即：

\[
\boxed{
\text{Discover}
\rightarrow
\text{Quantify}
\rightarrow
\text{Transfer}
}
\]

---

# 56. 最终核心主张

目标主张：

> **Internal impact damage contains a compact mesoscale representation that is mechanically sufficient for CAI assessment and more transferable across laminate configurations than the full-resolution C-scan representation.**

中文：

> **冲击内部损伤中存在一种紧凑的中尺度空间表征，它能够充分表达 CAI 相关机械信息，并且相比完整分辨率 C-scan 在跨铺层结构中具有更好的迁移能力。**

注意：

> 只有 S1 + S2 均 PASS 后才能使用这句话。

---

# 57. Final Principle

本阶段必须始终遵守：

\[
\boxed{
\text{Do not assume which scale is transferable. Measure it.}
}
\]

\[
\boxed{
\text{Do not optimize scale on the target domain. Select it from sources.}
}
\]

\[
\boxed{
\text{Do not confuse lower resolution with mechanical sufficiency. Preserve spatial specificity.}
}
\]

\[
\boxed{
\text{The goal is to discover transferable mechanical information, not to remove image details.}
}
\]

---

# 58. First Deliverables

在运行正式 S1 前，先生成：

```text
docs/MSSS_S1_PROTOCOL.md
docs/MSSS_S2_TRANSFER_PROTOCOL.md
docs/MSSS_REFERENCE_METHOD_AUDIT.md
docs/MSSS_CLAIM_EVIDENCE_MATRIX.md
```

然后先只执行：

```text
sampling
Gaussian
wavelet
non-inferiority
spatial specificity
```

没有 S1 GO：

> 不执行 S2。

没有 S2 GO：

> 不增加新的 scale-adaptive AI 方法。

所有结论由冻结结果决定。