# Codex Optimization Prompt — Result-Oriented Diffusion Marginalization for Cross-Domain CAI

## 0. Role

你现在作为当前项目的主研究工程师与实验负责人，直接在现有仓库基础上继续开发新的 diffusion-based positive experiment。

项目仓库：

```bash
git@github.com:Orangekostar/diff.git
```

当前参考提交：

```text
075ec73132984479d3b0eed54ae38107033c202a
```

目标期刊：

> **Composites Part B: Engineering**

本轮任务的核心目标非常明确：

> **在保持正式 outer-domain 测试不可泄漏的前提下，允许在训练域和内部验证域进行大规模调参、模型筛选、方法组合和自动搜索，尽最大可能找到一个能够稳定击败当前最佳 internal-only C-scan baseline 的 diffusion-based 方法。**

当前需要击败的核心 baseline：

\[
\boxed{
MAE_{\mathrm{baseline}}=0.089636
}
\]

对应：

```text
Measured full C-scan
→ frozen pretrained ResNet18
→ fold-local PCA
→ lightweight regression
→ CAI
```

不使用 surface features。

---

# 1. 当前已经冻结的科学事实

以下实验结果视为已完成证据，不重新修改结果，也不继续救失败路线。

## G1

```text
Surface → internal A/W/H
FAIL
```

三个标量改善均低于 10%，同步区间跨 0。

---

## G2

```text
Measured/predicted A/W/H → CAI
FAIL
```

说明：

> global damage dimensions 不能承载 full C-scan 中真正有效的 CAI 信息。

---

## P1

```text
Measured full-field C-scan → CAI
PASS
```

已有核心结果：

```text
Surface baseline     = 0.188121
Surface + full-field = 0.128489
Improvement          = 31.70%
```

更重要的是：

```text
Internal-only frozen C-scan = 0.089636
```

因此新的主 baseline 必须使用：

\[
0.089636
\]

而不是 0.128489。

---

## P3

```text
Within-C-scan spatial destruction
PASS
```

空间结构破坏导致明显性能下降，并表现出跨域一致性。

因此已经支持：

> mechanically relevant C-scan information contains specimen-specific spatial organization.

---

## P4

```text
DINOv2 / DDPM feature / fusion
NO-GO
```

当前复杂 frozen representation 没有稳定超过 frozen ResNet18。

不要继续单纯换 backbone。

---

## P5

```text
Reduced C-scan sampling
PASS
```

25% sampling 仍然保留大部分 full-field mechanical gain。

因此：

> fine spatial detail 并非全部必要。

---

## P6

```text
Diffusion reconstruction
NO_MECHANICAL_GAIN
```

已有结果：

```text
PCHIP      ≈ 0.129320
Bilinear   ≈ 0.129802
Diffusion  ≈ 0.142201
```

因此：

> diffusion 不应继续作为 full-image reconstruction 方法。

---

## P7

```text
Surface + sparse internal
NO_COMPLEMENTARITY
```

加入 surface features 出现稳定 negative transfer。

所以本轮禁止重新加入 surface branch。

---

# 2. 新实验核心假设

现有结果共同指向：

```text
A/W/H 太粗
↓
完整 spatial field 有用
↓
破坏 spatial organization 会失败
↓
但高分辨率细节具有明显冗余
↓
复杂 reconstruction 又没有额外机械收益
```

因此提出新的假设：

> **C-scan 中与 CAI 真正相关的是稳定的 low-/mid-frequency damage morphology；部分 fine-scale texture、成像风格和高频局部变化可能属于跨域 nuisance。**

新的 diffusion 角色不是：

```text
reconstruct missing mechanical information
```

而是：

```text
model and marginalize non-mechanical nuisance variability
```

即：

\[
\boxed{
\text{Morphology-Preserving Diffusion Marginalization}
}
\]

---

# 3. 新实验命名

暂命名：

> **D8 — Morphology-Preserving Diffusion Marginalization**

可以在代码中简称：

```text
D8-MPDM
```

---

# 4. 本轮实验的“成果导向”规则

本轮允许：

- 大规模超参数搜索；
- 自动 Bayesian / Optuna optimization；
- 多种 decomposition；
- 多种 residual 定义；
- 多种 diffusion sampling 参数；
- 多种 consistency loss；
- 多种 feature aggregation；
- 多种 test-time ensemble；
- 多种 regression model；
- 多模型 ensemble；
- staged model selection；
- top-k candidate reranking。

但正式规则是：

> **Outer held-out domains 不允许直接参与配置选择。**

即：

```text
Exploration / inner validation:
允许充分调参和挑模型

Final outer-domain evaluation:
只允许使用冻结后的配置
```

目标是：

> **最大化真实成功概率，而不是限制探索。**

---

# 5. Primary Goal

主指标：

\[
\text{equal-domain CAI ratio MAE}
\]

当前 baseline：

\[
0.089636
\]

最低 positive target：

\[
MAE<0.089636
\]

推荐正式 promotion target：

\[
MAE\le0.085154
\]

即：

\[
\ge5\%
\]

relative improvement。

Strong target：

\[
MAE\le0.082465
\]

即：

\[
\ge8\%
\]

relative improvement。

Stretch target：

\[
MAE<0.080
\]

---

# 6. 新方法总体结构

```text
Measured C-scan D
        |
        v
Multi-scale decomposition
        |
        +-------------------------+
        |                         |
        v                         v
Stable morphology S          residual R
low / mid frequency        fine-scale nuisance
        |                         |
        |                 diffusion modeling
        |                         |
        |               sample R1 ... RK
        |                         |
        +------------+------------+
                     |
                     v
      morphology-preserving variants
                     |
                     v
             frozen ResNet18
                     |
                     v
      consistency / marginalization
                     |
                     v
                   CAI
```

---

# 7. Multi-Scale Decomposition

不要只固定 Gaussian low-pass。

必须允许系统搜索以下 decomposition family。

## Family A — Gaussian

\[
S=G_\sigma*D
\]

\[
R=D-S
\]

搜索：

```text
sigma ∈ [0.5, 8.0]
```

可用 log-scale 或预设离散网格。

---

## Family B — Wavelet

使用：

- Haar；
- db2；
- db4；
- symlet；

等轻量 wavelet。

将：

```text
LL / coarse bands
```

作为 morphology，

将：

```text
LH / HL / HH
```

及部分中尺度 band 作为 residual。

搜索：

- wavelet family；
- decomposition level；
- retained bands。

---

## Family C — Fourier

按 normalized radial frequency：

\[
f_c
\]

切分：

```text
low
mid
high
```

搜索 cutoff。

允许：

```text
low-only
low+mid
mid+high residual
high-only residual
```

---

# 8. Morphology Preservation Principle

所有 diffusion augmentation 必须尽可能保持：

- damage footprint；
- centroid；
- large-scale anisotropy；
- radial distribution；
- major topology；
- low-/mid-frequency structure。

只随机化：

- local texture；
- high-frequency intensity pattern；
- minor boundary detail；
- acquisition-like residual；
- small local appearance variations。

---

# 9. Morphology Gate

允许搜索 gate 阈值，但只能通过 inner validation 确定。

至少评估：

## Area deviation

\[
\Delta A
\]

## Width deviation

\[
\Delta W
\]

## Height deviation

\[
\Delta H
\]

## Centroid shift

\[
\Delta c
\]

## Low-frequency correlation

\[
\rho_{LP}
\]

## Radial-profile consistency

\[
\rho_r
\]

## Optional Dice

若可靠 damage mask 存在。

探索阶段可搜索不同 gate，例如：

```text
A/W/H tolerance:
2.5%
5%
7.5%
10%

low-frequency corr:
0.95
0.97
0.98
0.99
```

最终配置由 inner-domain performance 选择。

---

# 10. Diffusion Residual Modeling

正式目标：

\[
p_\theta(R|S)
\]

而不是：

\[
p_\theta(D)
\]

也不是：

\[
p_\theta(D|CAI).
\]

禁止把 CAI label 输入 diffusion。

---

# 11. 第一阶段优先复用已有 P6 diffusion

在重新训练新 diffusion 前，先最大化利用：

```text
P6 trained checkpoints
P6 posterior draws
P6 reconstruction residuals
```

进行 D8-Pilot。

---

# 12. D8-Pilot Residual Construction

对于：

\[
D_i
\]

以及 P6 diffusion output：

\[
D^{diff}_{i,k}
\]

计算：

\[
\Delta_{i,k}
=
D^{diff}_{i,k}-D_i
\]

然后构造：

\[
R_{i,k}^{band}
=
BandPass(\Delta_{i,k})
\]

搜索：

- high-pass；
- mid-pass；
- high+mid；
- wavelet residual；
- Fourier residual。

---

# 13. Variant Construction

构造：

\[
\tilde D
=
D+\alpha R
\]

搜索：

```text
alpha ∈ [0.02, 1.0]
```

推荐使用 Optuna continuous search。

同时允许：

```text
alpha < 0
```

作为探索性对称扰动，例如：

```text
[-0.5, 1.0]
```

但必须满足 morphology gate。

---

# 14. Augmentation Count

允许搜索：

```text
K ∈ {1,2,4,8,16}
```

训练阶段和 test-time 可以使用不同 K：

```text
K_train
K_test
```

均由 inner validation 决定。

---

# 15. Non-Diffusion Controls

必须保留，用于判断 diffusion-specific value。

## B0 Raw

```text
Measured full C-scan
→ frozen ResNet18
```

baseline：

\[
0.089636
\]

---

## B1 Low-pass

只用 morphology component。

---

## B2 Gaussian noise residual

匹配 residual variance。

---

## B3 Spectrum-matched phase randomization

保留 amplitude spectrum，随机 phase。

---

## B4 Empirical residual bootstrap

从训练 specimen residual bank 随机交换 residual。

---

## B5 Diffusion residual augmentation

只做 diffusion augmentation。

---

## B6 Diffusion + prediction consistency

---

## B7 Diffusion + test-time marginalization

---

## B8 Full proposed

```text
diffusion augmentation
+
consistency
+
test-time marginalization
+
optional model ensemble
```

---

# 16. Frozen Encoder

主实验默认：

> **Frozen ImageNet ResNet18**

不得先 fine-tune。

但 exploration 阶段允许测试：

```text
ResNet18 raw global feature
ResNet18 intermediate block feature
multi-layer pooled ResNet features
```

如果 intermediate/multi-layer feature 在 inner validation 中明显更好，可进入最终候选。

这仍属于 frozen feature exploitation，不算重新训练 backbone。

---

# 17. Feature-Level Marginalization

除 prediction averaging 外，还必须探索：

## Mean feature

\[
\bar f
=
\frac1K
\sum_k f_k
\]

---

## Median feature

coordinate-wise median。

---

## Robust trimmed feature

去掉 feature-space extreme samples。

---

## Covariance-aware representation

计算：

\[
\mu_f,\Sigma_f
\]

然后输入：

```text
mean
mean + diagonal variance
mean + selected variance features
```

测试 diffusion variation 是否能提供额外 uncertainty/invariance signal。

---

# 18. Prediction-Level Marginalization

允许：

```text
mean
median
trimmed mean
weighted mean
```

权重可以由 inner validation 学习。

也允许：

\[
w_k
\propto
\exp(-\beta d_{\mathrm{morph}}(D,\tilde D_k))
\]

搜索：

```text
beta
```

使与原 morphology 更一致的 variant 权重大。

---

# 19. Consistency Strategies

不要只测试一个 loss。

允许：

## Prediction consistency

\[
L_{pred}
\]

---

## Feature consistency

约束 same-specimen variant feature：

\[
\|f_k-\bar f\|^2
\]

但不要直接让所有 feature 完全相同。

可只作用于 selected PCA subspace。

---

## Pairwise ranking consistency

同一 specimen 的 variant 应保持与其他 specimen 的 CAI ranking。

---

## Variance penalty

直接降低：

\[
Var_k(\hat y_{ik})
\]

---

# 20. Regression Model Search

允许在 inner validation 搜索：

- Ridge；
- ElasticNet；
- PLS；
- Huber regression；
- kernel ridge；
- SVR；
- shallow MLP；
- gradient boosted regressor。

但必须注意：

> 模型复杂度相对于 276 specimen 不能无限膨胀。

最终优先选择：

```text
performance
+
cross-domain stability
+
reasonable complexity
```

---

# 21. Hyperparameter Optimization

优先使用：

> **Optuna / Bayesian optimization**

不要只使用固定网格。

每个 outer fold 内：

```text
inner-domain objective
```

定义为：

\[
J
=
MAE_{\text{mean}}
+
\lambda_w MAE_{\text{worst}}
+
\lambda_s SD_{\text{domain}}
\]

建议搜索：

```text
lambda_w ∈ [0, 0.5]
lambda_s ∈ [0, 0.25]
```

或者先固定：

```text
J = mean_MAE + 0.25 * worst_MAE + 0.10 * domain_SD
```

核心目的是避免配置只对一个 domain 有效。

---

# 22. Search Budget

允许较大搜索量。

建议：

## Stage 1

```text
50–100 trials / outer fold
```

快速 surrogate search。

---

## Stage 2

取：

```text
top 10–20 configurations
```

进行更严格 repeated inner-domain evaluation。

---

## Stage 3

取：

```text
top 3–5
```

做：

- multiple seeds；
- stronger validation；
- ensemble feasibility。

---

## Stage 4

冻结最终 configuration。

---

# 23. Model Ensemble

如果 top configs 在 inner domains 存在互补性，允许：

- weighted ensemble；
- stacking；
- ridge-on-predictions；
- constrained non-negative weights。

权重必须：

> inner training/validation 学习。

禁止 outer test 学 ensemble weight。

---

# 24. Formal Outer Evaluation

每个 outer domain：

```text
5 domains
→ full exploration + model selection

1 domain
→ final untouched evaluation
```

一旦 outer result 产生：

> 该 fold 不再根据结果返回修改配置。

---

# 25. Primary Success Criteria

最低成功：

\[
MAE<0.089636
\]

---

正式 positive：

\[
MAE\le0.085154
\]

即：

\[
\ge5\%
\]

improvement。

---

强 positive：

\[
MAE\le0.082465
\]

即：

\[
\ge8\%
\]

improvement。

---

理想目标：

\[
MAE<0.080
\]

---

# 26. Domain Criteria

至少：

```text
4/6 outer domains improve
```

优先目标：

```text
5/6
```

强结果：

```text
6/6
```

同时重点记录：

```text
worst-domain improvement
```

---

# 27. Secondary Success Mode

如果总体 MAE 改善不足 5%，但满足：

- worst-domain MAE 显著降低；
- domain SD 大幅下降；
- 5/6 或 6/6 域轻微改善；

则保留为：

> **cross-domain robustness improvement**

但不称 primary accuracy breakthrough。

---

# 28. 如果 P6 Residual Pilot 失败

不要立刻停止。

允许正式训练 residual diffusion：

\[
p_\theta(R|S)
\]

因为 P6 原模型不是为 nuisance modeling 设计的。

但是触发条件应为：

> Pilot 至少存在某些 inner-domain positive trend，或者分析证明 P6 residual 与目标 nuisance distribution 明显不匹配。

---

# 29. Formal Residual Diffusion Architecture

优先：

```text
input resolution: 64x64 or equivalent
small UNet
base channels: 32 / 64
limited attention
conditional input: morphology S
output: residual R
```

探索：

- DDPM；
- DDIM；
- EDM-style noise schedule；
- flow-matching residual generator。

但不要引入巨大 foundation diffusion。

---

# 30. Diffusion Training Objective

允许探索：

- epsilon prediction；
- v prediction；
- residual prediction；
- conditional denoising；
- spectral-weighted loss。

特别建议探索：

\[
L
=
L_{\rm diffusion}
+
\lambda_{spec}L_{\rm spectrum}
+
\lambda_{LP}L_{\rm morphology}
\]

其中 morphology loss 强制生成 residual 不改变低频结构。

---

# 31. Frequency-Aware Diffusion

由于现有 P3/P5/P6 强烈暗示：

> spatial organization important, fine detail partly redundant

优先探索：

> **frequency-weighted diffusion**

例如：

- diffusion only models high band；
- diffusion models mid-high；
- different noise schedule per band；
- residual spectrum normalization。

这是本轮最重要的理论方向之一。

---

# 32. Optional Strong Extension — Morphology-Adaptive Residual Strength

不同 specimen 可能允许不同 residual perturbation。

学习：

\[
\alpha_i
=
g(S_i)
\]

但 g 必须基于训练域确定。

可用简单规则：

- damage area；
- eccentricity；
- spatial entropy；
- frequency ratio。

不要直接使用 CAI。

---

# 33. 25% Sparse Replication

如果 D8 full internal-only 成功，则在：

```text
25% sparse internal-only
```

上复制。

当前参考约：

```text
MAE ≈ 0.0901
```

目标：

> D8 同样能提高 sparse internal representation。

如果 full + sparse 两个设置都成功，论文价值明显提高。

---

# 34. 不要重新做的事情

不得继续投入：

```text
surface distillation
surface + internal fusion
DINO backbone chasing
full-image diffusion reconstruction
PSNR-oriented diffusion
CAI-conditioned image generation
```

除非新的 D8 evidence 明确指出这些方向重新有必要。

---

# 35. Scientific Interpretation if Successful

如果 D8 成功，不把结论写成：

> diffusion is better than ResNet.

而写成：

> **The cross-domain CAI value of C-scan depends on stable mesoscale morphology, while diffusion-based marginalization of fine-scale appearance variability improves robustness by suppressing non-mechanical domain-specific information.**

核心关键词：

- mechanically relevant morphology；
- nuisance variability；
- cross-domain invariance；
- information preservation；
- not reconstruction.

---

# 36. CPB Story after D8

整篇逻辑：

```text
Global damage dimensions
        ↓
insufficient

Full C-scan
        ↓
strong CAI value

Spatial destruction
        ↓
large loss

Reduced sampling
        ↓
most mechanical value retained

Full-image diffusion reconstruction
        ↓
unnecessary

Morphology-preserving diffusion marginalization
        ↓
remove non-mechanical variability
        ↓
better cross-domain CAI
```

最终观点：

> **Preserve mechanically relevant spatial morphology and marginalize non-mechanical C-scan appearance instead of reconstructing the entire ultrasonic field.**

---

# 37. First Mandatory Deliverable

开始任何正式训练前，先生成：

```text
docs/D8_RESULT_ORIENTED_EXPLORATION_PLAN.md
```

必须包括：

1. baseline reconstruction；
2. exact data split；
3. available P6 checkpoint audit；
4. decomposition candidates；
5. residual candidates；
6. Optuna search space；
7. objective function；
8. model candidates；
9. ensemble strategy；
10. compute budget；
11. pilot stage；
12. formal stage；
13. failure modes；
14. leakage audit。

---

# 38. Experiment Tracking

所有 trial 必须记录。

建议：

```text
results/d8_search/
    trial_index.csv
    study.db
    best_inner_configs/
    formal_outer/
```

trial_index 至少记录：

- trial id；
- outer fold；
- inner score；
- worst-domain score；
- decomposition；
- diffusion config；
- alpha；
- K；
- loss；
- regressor；
- seed；
- runtime；
- status。

失败 trial 可以保留在实验日志，不要求都写入主论文。

---

# 39. Formal Result Package

最终生成：

```text
results/d8_final/
    aggregate_metrics.csv
    domain_metrics.csv
    bootstrap.csv
    selected_configs.csv
    ablation.csv
    morphology_audit.csv
    search_summary.csv
    REPORT.md
```

---

# 40. Codex Execution Priority

按照：

```text
Step 1
Reproduce 0.089636 exactly

Step 2
Audit P6 diffusion outputs

Step 3
Build multi-scale residual bank

Step 4
Run cheap D8 pilot

Step 5
Launch broad inner-domain Optuna search

Step 6
Rerank top configurations

Step 7
Test ensembles

Step 8
Freeze one final pipeline

Step 9
Run untouched outer-domain evaluation

Step 10
Bootstrap + domain analysis

Step 11
If positive, replicate on 25% sparse setting

Step 12
Update claim-evidence matrix
```

---

# 41. 最终执行哲学

本轮开发允许：

> **训练和内部验证阶段高度成果导向。**

也就是说：

- 可以调；
- 可以筛；
- 可以组合；
- 可以 ensemble；
- 可以自动搜索；
- 可以淘汰无效 diffusion 形式；
- 可以寻找最优频段和最优 residual 建模方式。

但是最终 outer-domain 评价必须保持可信，否则即使 MAE 很低也不能支撑 CPB 投稿。

最终目标不是证明：

> diffusion 很先进。

而是找到一个真正成立的：

\[
\boxed{
\text{diffusion-based mechanism}
}
\]

使：

\[
\boxed{
MAE<0.089636
}
\]

最好达到：

\[
\boxed{
MAE\le0.085154
}
\]

并最终解释为：

> **Diffusion improves CAI prediction by marginalizing mechanically irrelevant cross-domain image variability while preserving the spatial damage morphology already proven to matter.**

如果 diffusion-specific 方法最终不如 Gaussian / spectrum randomization，则必须如实识别：

> augmentation/invariance 是有效机制，而非 diffusion-specific prior。

反之，如果 diffusion 在相同 morphology constraint 下稳定超过所有随机 residual controls，则可以将：

> **morphology-preserving diffusion marginalization**

晋升为论文的新方法贡献。