# Codex Prompt — MGMR: Mechanics-Guided Multiscale Morphology Representation for Cross-Configuration CAI Assessment

## 0. Role

你现在继续接管当前 CFRP ultrasonic C-scan → CAI 项目，并基于已有完整实验链设计下一阶段真正具有 **Advanced Engineering Informatics (AEI)** 方法学贡献的新模型。

当前仓库：

```bash
git@github.com:Orangekostar/diff.git
```

当前主线已经完成并冻结：

- G1 / G2
- P1 / P3 / P5 / P6 / P7
- MASI A0–A5
- Multi-view E1–E3
- MSSS S1

当前 MSSS 结果提交：

```text
db335cc
```

不要重新修改这些正式结果。

目标期刊：

> **Advanced Engineering Informatics**

本阶段新的核心故事：

> **我们已经通过连续干预实验识别出内部冲击损伤中两类潜在互补的机械信息：整体损伤组织（coarse morphology）以及方向性边界形貌（directional/boundary morphology）。下一阶段的目标是验证这两类信息是否真正具有增量互补性，并在此基础上将它们编码成铺层感知的多尺度空间表示，以提高跨铺层和跨层数条件下的 CAI 预测。**

---

# 1. 当前已经冻结的科学证据

以下结论视为正式 evidence。

不得为了新模型重新修改或重新解释。

---

## F1 — Global Damage Scalars Are Insufficient

G2 已证明：

```text
damage area
damage width
damage height
```

不能稳定改善 CAI。

因此：

\[
\boxed{
\text{global damage size}
\neq
\text{sufficient mechanical representation}
}
\]

新方法不能退化成简单 scalar feature engineering。

---

## F2 — Full C-scan Contains Strong Mechanical Information

P1：

```text
Surface baseline                  MAE ≈ 0.188121
Surface + full C-scan             MAE ≈ 0.128489
Internal-only frozen C-scan       MAE ≈ 0.0896358
```

当前所有新模型必须以：

\[
\boxed{
MAE_{\rm baseline}=0.0896358
}
\]

作为正式主基线。

---

## F3 — Spatial Organization Matters

P3 已证明：

```text
pixel shuffle
patch shuffle
canonical compact field
```

破坏真实空间组织后性能显著下降。

因此：

\[
\boxed{
\text{spatial morphology is mechanically relevant}
}
\]

任何新方法都必须显式保留：

- spatial location
- relative geometry
- directionality
- damage organization

而不是只做 global pooling。

---

## F4 — Reduced Sampling Retains Large CAI Value

P5 / MSSS sampling axis 已证明：

即使大幅减少 spatial sampling，CAI performance 仍可接近 FULL。

MSSS S1 中：

```text
6.25% sampling
```

仍保持很强性能和 spatial specificity。

因此已有证据支持：

> **coarse/global spatial organization is highly informative and sampling-redundant.**

但不要写：

> low resolution is sufficient in general.

正式 MSSS：

```text
S1 = NO_GO
```

仍保持冻结。

---

## F5 — Pure Isotropic Smoothing Is Not Sufficient

Gaussian scale-space 表明：

随着 smoothing 增强：

```text
MAE progressively worsens
```

因此不得使用：

\[
\boxed{
\text{low frequency = all useful mechanics}
}
\]

这样的假设。

---

## F6 — Wavelet Low-Only Loses Information

Wavelet low-only：

```text
level increases
→
CAI performance worsens
```

因此：

> coarse morphology alone is not guaranteed sufficient.

---

## F7 — Coarse + Directional Details Is a Positive Clue

MSSS wavelet sensitivity 中：

```text
low + selected boundary/directional details
```

明显优于：

```text
low-only
```

部分配置接近 FULL。

这不是正式已证明的“互补性”。

它只是当前最重要的：

\[
\boxed{
\text{architecture-design hypothesis}
}
\]

新的实验必须首先验证这一点。

---

## F8 — Feature Invariance Does Not Work

MASI A4/A5：

```text
Raw full MAE       ≈ 0.08964
Stable code MAE    ≈ 0.09784
```

虽然 stable code 更 observation-invariant，但 CAI 信息下降。

因此新方法禁止再次：

```text
force different views/features into a common invariant space
```

---

## F9 — FULL / 50% / 25% Are Predictively Redundant

Multi-view E1：

```text
prediction correlation > 0.99
residual correlation > 0.987
```

Cooperative / fusion / GMvR 均未超过 FULL。

因此：

> FULL / 50% / 25% 不应继续作为 complementary modalities。

不要继续多视图融合。

---

# 2. 新科学假设

新的研究假设是：

> **内部冲击损伤中的 CAI 相关信息不是由单一空间尺度决定，而可能由至少两类结构性信息共同组成：**
>
> 1. **Coarse morphology**：整体损伤位置、范围、传播范围、中心—外围组织；
> 2. **Directional/boundary morphology**：边界、方向性、各向异性、局部轮廓及尺度匹配的 directional detail。

进一步假设：

> **这两类信息并不是简单冗余关系，而可能具有增量机械互补性。**

只有首先证明：

\[
\boxed{
\text{coarse}
+
\text{directional}
>
\text{coarse alone}
}
\]

且 directional component 对 baseline residual 提供稳定增量信息，

才允许开发完整 MGMR 方法。

---

# 3. 新方法暂命名

# MGMR

> **Mechanics-Guided Multiscale Morphology Representation**

如果后续加入铺层感知图推理并且结果成立，可升级名称为：

# LA-MGMR

> **Laminate-Aware Mechanics-Guided Multiscale Morphology Representation**

不要在 M0 尚未通过前提前使用最终方法名写论文。

---

# 4. 本阶段严格分成两个阶段

## Phase M0 — Complementarity Feasibility

目标：

> 验证 coarse morphology 与 directional/boundary morphology 是否真正存在 CAI 增量互补性。

---

## Phase M1 — Structured Mechanics-Guided Representation

只有 M0 PASS 后：

> 构建 impact-centered spatial graph + directional wavelet features + laminate-aware relation modeling。

---

# 5. 参考方法与开源代码

正式开发前必须阅读并审计以下工作。

建立：

```text
docs/MGMR_REFERENCE_METHOD_AUDIT.md
```

---

## Reference A — MFC-MIL

论文：

> A Multiscale Frequency Domain Causal Framework for Enhanced Pathological Analysis, ICLR 2025

开源仓库：

```text
WissingChen/MFC-MIL
```

重点阅读：

```text
modules/multi_level.py
frequency-related modules
feature fusion
ablation implementation
```

只借：

- multi-scale information separated before fusion
- spatial-scale and frequency-scale independent validation
- lightweight feature-level fusion
- intervention-style ablation

不要借：

- MIL architecture
- pathology-specific modules
- causal-memory claims

---

## Reference B — WaveRNet

开源仓库：

```text
Chanchan-Wang/WaveRNet
```

重点阅读：

```text
models/waverNet.py
SimpleWaveletTransform
frequency branch
```

只借：

> **对 learned feature map 做 DWT，而不是只在原始 RGB image 上做 wavelet。**

这与当前项目特别重要：

当前公开 C-scan 为 pseudocolor screenshot。

feature-domain DWT 比 raw-RGB wavelet 更合理。

---

## Reference C — AEI Physics-Guided GCN / Graph Papers

重点理解：

> graph adjacency 应由 engineering relationships 定义，而不是任意 fully-connected graph。

只借：

- physics/spatial knowledge determines topology
- fully connected vs structured graph ablation
- engineering knowledge explicitly enters message passing

不借：

- Lamb-wave path graph
- bridge topology
- 原任务网络

---

# 6. 开发前必须完成 Repository Audit

生成：

```text
docs/MGMR_REPOSITORY_AUDIT.md
```

至少回答：

1. 当前 exact baseline 如何复现 0.0896358；
2. frozen ResNet18 权重 SHA；
3. 当前 C-scan crop 尺寸；
4. crop 是否保持冲击中心；
5. impact center 是否可可靠恢复；
6. layer2 / layer3 spatial feature map shape；
7. 当前 preprocessing 是否保持方向；
8. 是否存在 rotation / flip augmentation；
9. specimen → dataset → ply count → layup mapping；
10. stacking sequence 是否可以逐 specimen authority 化；
11. 当前 wavelet implementation 具体保留哪些 coefficients；
12. P3 destructive transformations 如何复用；
13. LODO / leave-ply / leave-layup authority；
14. 是否有任何 previous outer-test exposure 会影响新实验。

---

# 7. Laminate Metadata Authority

建立：

```python
LaminateAuthority
```

逐 specimen 提供：

```text
specimen_id
domain_id
ply_count
layup_family
stacking_sequence
ply_orientation_histogram
nominal_thickness
```

禁止仅根据：

```text
domain_id
```

在网络里 hardcode ply/layup。

所有 mapping 必须来自：

- public dataset metadata；
- workbook；
- manifest；
- 已验证 source documents。

并缓存 hash。

---

# 8. Laminate Context

第一版只构造低维、稳定的 laminate context：

\[
k_L=
[
n_{\rm ply}/24,
p_0,
p_{90},
p_{45},
p_{-45}
]
\]

其中：

\[
p_\alpha
=
\frac{\#\text{plies at orientation }\alpha}
{N_{\rm ply}}
\]

可选加入：

```text
layup_family
nominal thickness
```

但是：

> 不使用 sequence Transformer。

当前 276 specimens 不支持过度复杂 laminate encoder。

---

# 9. Impact Center Alignment Audit

在使用 polar graph 之前必须验证：

\[
(x_c,y_c)
\]

是否能够可靠定义。

检查：

- 原试样冲击中心；
- C-scan screenshot；
- crop；
- resize；
- registered masks / metadata。

输出：

```text
docs/MGMR_IMPACT_CENTER_AUDIT.md
```

若无法可靠对齐：

> Phase M1 第一版使用 Cartesian spatial grid。

禁止假装 center alignment 成立。

---

# 10. Frozen Spatial Encoder

当前 frozen ResNet18 已经支持：

```text
layer1
layer2
layer3
layer4
```

但现有接口进行了 global pooling。

新增：

```python
encode_spatial(
    images,
    layer="layer3",
)
```

返回：

```text
N × C × H × W
```

Primary：

```text
layer3
```

Secondary：

```text
layer2
```

不得改变：

- pretrained weights；
- normalization；
- image resize；
- encoder checkpoint。

目标：

> isolate representation design effect。

---

# 11. 新 Feature Bank

建立：

```text
results/mgmr/feature_bank/
```

预计算：

```text
FULL layer3 spatial map
coarse-input layer3 spatial map
```

以及必要 layer2 sensitivity。

所有 feature：

- immutable；
- specimen ordered；
- checksum bound；
- cached。

不要每个 trial 重跑 ResNet。

---

# 12. Phase M0 — Complementary Component Gate

这是整个 MGMR 项目的硬门禁。

在 M0 PASS 前：

> 禁止开发 GNN、laminate-aware graph、Diffusion 或大型新网络。

---

# 13. M0 的核心问题

回答：

> **Does directional/boundary morphology provide incremental CAI information beyond coarse damage organization?**

注意：

不是问：

> coarse 和 boundary 各自能不能预测 CAI？

必须问：

\[
\boxed{
\text{Does boundary information explain CAI residuals left by coarse morphology?}
}
\]

这是前面 multi-view 失败后必须吸取的教训。

---

# 14. M0 — Coarse Component Definition

不要再搜索 MSSS。

Primary coarse representation 固定从已有正证据中选一个。

建议第一版：

```text
25% or 18.75% bilinear sparse reconstruction
```

具体选哪一个：

只能在 source-domain protocol 内根据：

- P5/MSSS frozen results；
- stability；
- engineering simplicity

提前冻结。

不要使用新 outer results 重新选。

建立：

```text
docs/MGMR_M0_PROTOCOL.md
```

明确 coarse input。

---

# 15. Coarse Spatial Feature

\[
D_c
\rightarrow
E_{\rm spatial}
\rightarrow
F_c
\]

得到：

\[
F_c\in\mathbb R^{C\times H\times W}
\]

先不做 graph。

Primary M0 可：

\[
f_c=GAP(F_c)
\]

然后 fold-local PCA + existing regressor。

目的：

> 建立 coarse baseline。

---

# 16. Directional/Boundary Component

Primary 方法不再对原 RGB C-scan 做 wavelet。

改为：

\[
F=E_{\rm spatial}(D_{\rm FULL})
\]

然后：

\[
DWT(F)
\rightarrow
F_{LL},
F_{LH},
F_{HL},
F_{HH}
\]

第一版 wavelet：

```text
haar
db2
```

Primary：

```text
db2
```

---

# 17. Feature-Domain DWT

必须验证：

1. reconstruction identity；
2. orientation of H/V/D bands；
3. dtype；
4. border mode；
5. feature-map size；
6. deterministic output。

新增：

```text
tests/test_mgmr_feature_wavelet.py
```

---

# 18. Directional Feature

不要先平均三个 detail band。

分别：

\[
f_{LH}=GAP(F_{LH})
\]

\[
f_{HL}=GAP(F_{HL})
\]

\[
f_{HH}=GAP(F_{HH})
\]

directional representation：

\[
f_b=
[
f_{LH};
f_{HL};
f_{HH}
]
\]

先 PCA，再 regression。

---

# 19. M0 Baselines

严格比较：

## B0 — Full Frozen ResNet

\[
MAE=0.0896358
\]

---

## B1 — Coarse Only

\[
f_c
\]

---

## B2 — Boundary/Directional Only

\[
f_b
\]

---

## B3 — Coarse + Directional Concatenation

\[
[f_c;f_b]
\]

---

## B4 — Full Feature + Directional Correction

\[
[f_{\rm full};f_b]
\]

这是非常重要的 strong baseline。

---

# 20. M0 Residual Complementarity Audit

这是本阶段最重要的实验。

先使用 strict OOF coarse predictor：

\[
\hat y_c^{OOF}
\]

得到：

\[
r_c
=
y-\hat y_c^{OOF}
\]

然后：

\[
f_b\rightarrow r_c
\]

使用 strict nested source-domain validation。

同时：

\[
f_b\rightarrow r_{\rm full}
\]

其中：

\[
r_{\rm full}
=
y-\hat y_{\rm full}^{OOF}
\]

用于检验 directional branch 是否还能补充当前最强 baseline。

---

# 21. M0 必须报告

```text
MAE coarse
MAE boundary
MAE coarse+boundary
MAE full+boundary

residual MAE coarse → boundary
residual MAE full → boundary

per-domain effects
worst-domain effects
residual correlations
```

---

# 22. Spatial Specificity Control

对：

```text
FULL
```

应用已冻结 P3：

```text
8×8 patch shuffle
```

然后重新计算 directional representation。

如果 boundary branch 真利用 spatial morphology：

其增量价值应该在 shuffled input 中明显下降。

定义：

\[
\Delta_{\rm boundary}^{real}
\]

和：

\[
\Delta_{\rm boundary}^{shuffle}
\]

要求：

\[
\Delta_{\rm boundary}^{real}
>
\Delta_{\rm boundary}^{shuffle}
\]

---

# 23. M0 GO Criteria

进入完整 MGMR 至少满足：

### Gate A — Component Complementarity

\[
MAE_{\rm coarse+boundary}
<
\min(
MAE_{\rm coarse},
MAE_{\rm boundary}
)
\]

并：

```text
>= 4/6 domains improve relative to coarse
```

---

### Gate B — Incremental Residual Value

Directional component 对 coarse residual：

```text
stable positive incremental prediction
```

至少：

```text
>= 4/6 domains
```

方向一致。

---

### Gate C — Preferably Full Residual Value

如果：

\[
f_b
\]

还能改善：

\[
r_{\rm full}
\]

则强 GO。

这是最理想结果。

---

### Gate D — Spatial Specificity

真实 spatial layout 下的 directional benefit 必须明显大于 P3-shuffled control。

---

# 24. M0 NO-GO

如果：

```text
coarse + boundary
```

不优于 coarse，

或者：

```text
boundary -> residual
```

无稳定增量，

则：

\[
\boxed{
\text{MGMR_NO_GO}
}
\]

不要继续开发 GNN。

此时当前“两个互补机械成分”故事未被数据支持。

---

# 25. M0 Success Target

最好目标：

\[
MAE_{\rm full+boundary}
<
0.0896358
\]

正式 positive：

\[
MAE\le0.08515
\]

strong：

\[
MAE\le0.08247
\]

但 M0 的首要目标不是直接达到最终 SOTA，

而是：

> **证明 directional morphology 提供真实 incremental mechanical information。**

---

# 26. Phase M1 — MGMR Structured Representation

只有 M0 PASS 才允许执行。

---

# 27. M1 Architecture Overview

```text
                    C-scan
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
      Coarse Input           Full C-scan
          │                       │
          ▼                       ▼
 Frozen Spatial Encoder   Frozen Spatial Encoder
          │                       │
          │                       ▼
          │                    DWT
          │               ┌────┬────┬────┐
          │               LH   HL   HH
          │               │    │    │
          └──────────┬────┴────┴────┘
                     │
             spatial node pooling
                     │
                     ▼
              Impact-centered graph
                     │
              laminate-aware edges
                     │
                     ▼
              lightweight GNN
                     │
                     ▼
                 h_graph
                     │
          ┌──────────┴──────────┐
          │                     │
 global frozen feature      laminate context
          │                     │
          └──────────┬──────────┘
                     ▼
                  fusion
                     │
                     ▼
               CAI correction
```

---

# 28. Spatial Graph Type

第一版不要做 detected-component graph。

Primary：

# Impact-Centered Polar Graph

如果 impact center audit PASS：

```text
4 radial rings × 8 angular sectors
```

约：

```text
32 nodes
```

---

# 29. Fallback

如果 impact center alignment 不可靠：

使用：

```text
6×6
or
7×7 Cartesian grid
```

但在论文中必须明确原因。

---

# 30. Node Features

每个 node \(v\)：

\[
x_v=
[
P(F_c|_v),
P(F_{LH}|_v),
P(F_{HL}|_v),
P(F_{HH}|_v),
r_v,
\sin\theta_v,
\cos\theta_v
]
\]

所有 feature channel 必须先用轻量 linear projection：

```text
C → 32 / 64
```

避免维数过大。

---

# 31. Spatial Graph Adjacency

Primary Graph：

- same ring: angular neighbors；
- adjacent rings: same angle；
- adjacent rings: neighboring angle。

不要一开始 fully connect。

---

# 32. Graph Baselines

必须实现：

## G0 — No Graph

简单 concatenation。

## G1 — Fully Connected Graph

证明 GNN 本身。

## G2 — Spatial Graph

显式 spatial topology。

## G3 — Laminate-Aware Spatial Graph

最终方法。

---

# 33. Laminate Direction Alignment

对于 edge：

\[
i\rightarrow j
\]

计算 edge direction：

\[
\phi_{ij}
\]

定义 orientation alignment feature。

可从简单式开始：

\[
a_{ij}^{lam}
=
\sum_{\alpha}
p_\alpha
|\cos 2(\phi_{ij}-\alpha)|
\]

其中：

\[
\alpha\in
\{0,90,45,-45\}
\]

该式作为：

> engineering prior

不是严格 physical law。

必须在论文中明确。

---

# 34. Laminate-Aware Edge

Edge feature：

\[
e_{ij}
=
[
d_{ij},
\cos\phi_{ij},
\sin\phi_{ij},
a_{ij}^{lam}
]
\]

第一版使用：

```text
GATv2 edge bias
or
GraphSAGE + edge-conditioned scalar
```

选择一个实现。

不要同时堆多个 GNN。

---

# 35. GNN Capacity

严格控制：

```text
2 graph layers
hidden dim 32 or 64
dropout
```

trainable parameter target：

\[
<0.5M
\]

优先：

\[
<0.2M
\]

---

# 36. Global Skip Branch

保留当前最强：

\[
f_{\rm full}
\]

或者 coarse global feature。

不要让新 graph 完全替代 P1 baseline。

最终：

\[
h=
Fusion(
f_{\rm full},
h_G,
k_L
)
\]

---

# 37. Residual CAI Head

Primary 设计：

\[
\hat y
=
\hat y_{\rm frozen}
+
\beta\Delta\hat y_{\rm MGMR}
\]

其中：

\[
\hat y_{\rm frozen}
\]

来自当前冻结 baseline。

MGMR 学：

\[
\Delta y
=
y-\hat y_{\rm frozen}^{OOF}
\]

理由：

> 当前 baseline 已经很强，新模型只学习它遗漏的结构化机械信息。

---

# 38. Residual Training 必须 OOF

训练 MGMR residual target：

\[
r_i
=
y_i-\hat y_i^{baseline,OOF}
\]

禁止使用 baseline 对训练 specimen 的 in-sample prediction 构造 residual。

新增：

```text
test_mgmr_residual_target_is_strict_oof
```

---

# 39. M1 Baseline Table

至少比较：

```text
Frozen ResNet18
Coarse only
Boundary only
Coarse + boundary concat
Full + boundary concat

Spatial node concat
Fully connected GNN
Spatial GNN
Spatial GNN + laminate metadata concat
Laminate-aware spatial GNN
Full MGMR residual model
```

---

# 40. 必须证明的两条方法学关系

## Relation 1

\[
\boxed{
SpatialGraph > FullyConnectedGraph
}
\]

证明 spatial organization knowledge 有价值。

---

## Relation 2

\[
\boxed{
LaminateAwareGraph > SpatialGraph
}
\]

证明 laminate orientation knowledge 对 message passing 有增量价值。

---

# 41. Metadata Concat Control

必须有：

```text
Spatial GNN
+
plain laminate metadata concatenation
```

与：

```text
laminate-aware edge modeling
```

比较。

否则 reviewer 会说：

> 提升只是因为多输入了 ply count / layup。

---

# 42. Training Objective

第一版只使用：

\[
L_{\rm CAI}
\]

或者 residual MSE/Huber。

禁止同时加入：

- DANN
- contrastive loss
- causal loss
- diffusion loss
- invariance loss

先证明 representation 本身有效。

---

# 43. Optional Robust Objective

只有基础 MGMR GO 后，

测试：

```text
GroupDRO
VREx
```

作为 secondary DG extension。

---

# 44. Main Evaluation

Primary：

```text
six-domain LODO
```

必须报告：

```text
equal-domain MAE
per-domain MAE
worst-domain MAE
domain SD
```

---

# 45. Structured Transfer

必须运行：

## Leave-One-Ply-Count-Out

```text
16+24 → 8
8+24 → 16
8+16 → 24
```

---

## Leave-One-Layup-Family-Out

```text
cross-ply → quasi-isotropic
quasi-isotropic → cross-ply
```

---

# 46. 主方法的真正卖点

如果 ordinary LODO 仅小幅提升，

但：

```text
leave-ply
leave-layup
```

明显改善，

仍然属于 strong positive result。

因为方法目标就是：

> cross-configuration CAI transfer。

---

# 47. Spatial Knowledge Ablation

必须至少：

```text
remove radial position
remove angular encoding
fully connect graph
shuffle graph topology
```

验证：

> graph 不是普通 feature mixer。

---

# 48. Directional Knowledge Ablation

必须：

```text
LL only
LH only
HL only
HH only
all directional details
directional details shuffled
```

不要为了 paper 简洁而省略。

这些 ablation 是解释：

> “directional morphology”

的关键证据。

---

# 49. Laminate Knowledge Ablation

至少：

```text
no laminate context
ply count only
orientation histogram only
metadata concat
edge alignment
```

确定真正贡献来自哪里。

---

# 50. P3 Compatibility Check

MGMR 最终方法必须重新跑：

```text
registered P3 8×8 patch shuffle
```

如果模型在 spatially destroyed field 上表现几乎不变：

> 它并没有真正使用 claimed morphology information。

该方法不得作为主方法。

---

# 51. Success Criteria

## Minimum Positive

\[
MAE<0.0896358
\]

并：

```text
>=4/6 domains improve
```

---

## Formal Positive

\[
MAE\le0.08515
\]

约：

```text
>=5% improvement
```

---

## Strong Positive

\[
MAE\le0.08247
\]

约：

```text
>=8%
```

---

# 52. Structured Transfer Success

即使 ordinary LODO 只改善 3–5%，若：

```text
leave-ply improves consistently
leave-layup improves consistently
worst-domain decreases
```

可作为 strong AEI transfer result。

---

# 53. M1 NO-GO

如果：

```text
spatial graph <= concat
```

则图结构无价值。

如果：

```text
laminate-aware <= spatial graph
```

则铺层感知机制无价值。

如果：

```text
MGMR <= frozen ResNet
```

且 structured transfer 也无增益：

正式停止该方法。

不得继续堆 Transformer / Diffusion 救结果。

---

# 54. Diffusion 的角色

当前阶段：

\[
\boxed{
\text{Diffusion = NOT AUTHORIZED}
}
\]

只有 MGMR 已经证明：

\[
coarse + directional
\]

确实是有效互补结构，

且后续存在：

> boundary/directional component uncertainty modeling

的明确需求时，

才允许设计：

\[
p(
h_{\rm boundary}
|
h_{\rm coarse},k_L
)
\]

这样的 conditional diffusion extension。

不得重新做：

```text
image reconstruction
pixel residual diffusion
feature-invariance diffusion
```

---

# 55. 新代码结构

新建：

```text
src/cmc_bbdm/mgmr/
```

至少：

```text
authority.py
laminate_authority.py
impact_center.py

spatial_encoder.py
feature_wavelet.py

m0_components.py
m0_residual_audit.py

polar_graph.py
cartesian_graph.py

laminate_context.py
laminate_alignment.py

graph_models.py
fusion.py
residual_head.py

evaluation.py
statistics.py
formal_outer.py

artifacts.py
replay.py
```

---

# 56. Tests

新增：

```text
test_mgmr_baseline_reproduction.py
test_mgmr_laminate_authority.py
test_mgmr_impact_center.py

test_mgmr_spatial_encoder.py
test_mgmr_feature_wavelet.py

test_mgmr_m0_components.py
test_mgmr_residual_target_is_strict_oof.py

test_mgmr_graph_topology.py
test_mgmr_laminate_alignment.py

test_mgmr_no_outer_leakage.py
test_mgmr_specimen_grouping.py

test_mgmr_p3_compatibility.py
test_mgmr_replay.py
```

---

# 57. Result Directory

```text
results/mgmr/
```

子目录：

```text
m0_component_gate/
m1_spatial_graph/
m2_laminate_graph/
m3_structured_transfer/
m4_ablation/
```

每个必须：

```text
config.yaml
aggregate_metrics.csv
domain_metrics.csv
bootstrap.csv
summary.json
REPORT.md
artifact_manifest.json
CHECKSUMS.sha256
```

---

# 58. Claim-Evidence Matrix

创建：

```text
docs/MGMR_CLAIM_EVIDENCE_MATRIX.md
```

初始：

| Claim | Status |
|---|---|
| global A/W/H insufficient | PROVEN |
| spatial organization matters | PROVEN |
| reduced sampling retains CAI value | PROVEN |
| pure low-pass representation is insufficient | PROVEN |
| low + directional detail is promising | EXPLORATORY |
| coarse and directional components are complementary | TO TEST M0 |
| directional component predicts baseline residual | TO TEST M0 |
| explicit spatial graph adds value | TO TEST M1 |
| laminate orientation context adds value | TO TEST M2 |
| MGMR improves ordinary LODO | TO TEST |
| MGMR improves leave-ply transfer | TO TEST |
| MGMR improves leave-layup transfer | TO TEST |

---

# 59. First Deliverables

在实现完整 GNN 前，必须先生成：

```text
docs/MGMR_REPOSITORY_AUDIT.md
docs/MGMR_REFERENCE_METHOD_AUDIT.md
docs/MGMR_IMPACT_CENTER_AUDIT.md
docs/MGMR_M0_PROTOCOL.md
docs/MGMR_CLAIM_EVIDENCE_MATRIX.md
```

然后只运行 M0。

---

# 60. 严格执行顺序

```text
STEP 0
Audit current repository and reproduce 0.0896358.

STEP 1
Audit laminate metadata / stacking sequences.

STEP 2
Audit impact-center alignment.

STEP 3
Add frozen spatial feature-map API.

STEP 4
Implement feature-domain wavelet extraction.

STEP 5
Freeze M0 protocol.

STEP 6
Run coarse-only.

STEP 7
Run directional-only.

STEP 8
Run coarse + directional.

STEP 9
Run full + directional.

STEP 10
Run strict OOF residual complementarity audit.

STEP 11
Run P3 spatial-specificity control.

STEP 12
Issue M0 GO / NO-GO.

IF M0 NO-GO:
    stop MGMR.

IF M0 GO:
    continue.

STEP 13
Build fixed spatial graph.

STEP 14
Run concat vs FC graph vs spatial graph.

STEP 15
Build LaminateAuthority.

STEP 16
Add laminate-context concat baseline.

STEP 17
Add laminate-aware edge modulation.

STEP 18
Run ordinary six-domain LODO.

STEP 19
Run leave-ply transfer.

STEP 20
Run leave-layup transfer.

STEP 21
Run full ablation.

STEP 22
Freeze evidence matrix.

STEP 23
Only then modify manuscript.
```

---

# 61. AEI Story if Successful

最终论文不是：

> We combine wavelet and GNN.

真正故事：

> **Previous intervention studies reveal that CAI-relevant internal damage information cannot be represented by global dimensions or a single coarse spatial scale. The evidence instead suggests a structured multiscale representation composed of global damage organization and directional boundary morphology. MGMR explicitly encodes these components and their spatial relations, while laminate orientation information modulates message passing to improve cross-configuration CAI assessment.**

中文：

> **前期干预实验表明，CAI 相关内部损伤信息既不能由面积、宽度和高度等标量表示，也不能简化为单一粗尺度形貌。实验进一步提示，整体损伤组织与方向性边界结构可能构成互补的机械信息。本研究据此提出 MGMR，将这两类信息及其空间关系显式编码，并利用铺层方向知识调节区域间的信息传播，从而提高跨材料配置的 CAI 评估能力。**

---

# 62. Final Contributions if Fully Supported

## Contribution 1 — Evidence-Driven Mechanical Knowledge Discovery

通过已有：

- scalar failure；
- spatial intervention；
- sparse sensing；
- scale decomposition；

识别候选机械信息成分。

---

## Contribution 2 — Multiscale Morphology Representation

显式建模：

\[
\boxed{
\text{coarse damage organization}
+
\text{directional boundary morphology}
}
\]

而不是依赖单一 global CNN embedding。

---

## Contribution 3 — Spatial Reasoning

用 impact-centered graph 显式表示：

- location；
- radial relation；
- angular relation；
- neighborhood structure。

---

## Contribution 4 — Laminate-Aware Reasoning

利用真实 stacking orientation information 调节 graph message passing，

而不是将 laminate 仅作为普通 domain ID。

---

# 63. Final Principle

整个阶段必须始终遵守：

\[
\boxed{
\text{The architecture must follow evidence, not fashion.}
}
\]

以及：

\[
\boxed{
\text{Two useful feature sets are not automatically complementary.}
}
\]

因此：

> **先证明 coarse morphology 和 directional morphology 具有增量机械互补性，再开发 MGMR。**

不要重复此前 multi-view 实验的错误：

> 两个 representation 都能预测 CAI ≠ 融合后一定更好。

真正的 M0 门禁是：

\[
\boxed{
\text{directional morphology explains what coarse/full baseline misses}
}
\]

只有这件事成立，

MGMR 才具有科学基础。