# Codex Prompt — New Positive Experiment: Mechanics-Consistent Multi-View Regression for Cross-Domain CAI

## 0. Role and Goal

你现在继续接管当前 CFRP ultrasonic C-scan → CAI 项目。

目标期刊优先考虑：

> **Advanced Engineering Informatics**

当前项目已有 A0–A5 结果已经证明：

- full C-scan 对 CAI 有很强预测价值；
- FULL / 50% / 25% reduced-sensing views 都能保留较强 CAI 信息；
- 但强制学习 feature-level observation-invariant representation 会损失 CAI 信息；
- A4 stable representation MAE = **0.09784**，明显劣于 raw full-C-scan baseline **0.0896358**；
- 因此 MASI feature-invariance 假设正式进入 `FACTORISATION_NO_GO`。

本轮禁止继续优化：

```text
feature-level invariance
repeated-measure stable subspace
MASI-LD
pixel residual diffusion
full-image diffusion reconstruction
```

新的科学假设改为：

> **不同 C-scan sensing views 不需要共享相同的 feature representation；它们可以通过不同的表示路径编码同一个机械状态。合理的约束应放在 CAI prediction 层，同时保留各 view 的互补信息。**

核心问题：

> **Are FULL, 50% and 25% C-scan views mechanically consistent but predictively complementary?**

最终目标：

\[
\boxed{
\text{prediction consistency}
+
\text{view complementarity}
\rightarrow
\text{better cross-domain CAI}
}
\]

---

# 1. 当前必须击败的 Baseline

正式 primary baseline：

\[
\boxed{
MAE_{FULL}=0.0896358
}
\]

来自：

```text
full measured C-scan
→ frozen ResNet18
→ fold-local PCA
→ registered regressor
→ strict LODO CAI
```

后续所有新方法必须与该结果使用完全相同：

- 276 specimens；
- 6 domains；
- outer leave-one-domain-out；
- specimen-level split；
- frozen ResNet18；
- target definition；
- equal-domain MAE；
- common bootstrap / statistical protocol。

---

# 2. 本轮参考方法

优先阅读并借鉴以下方法，不要重新发明已有 multi-view regression 理论。

## Reference A — Cooperative Learning for Multi-view Analysis

论文：

> Daisy Yi Ding et al., *Cooperative Learning for Multi-view Analysis*, PNAS, 2022.

官方代码仓库：

```text
dingdaisy/cooperative-learning
```

重点阅读：

```text
regularized_cooperative_regression/
general_cooperative_learning/
more_than_two_data_views/
```

需要借鉴：

> prediction loss + cross-view agreement penalty。

不要照搬其数据结构。

本项目是三个 C-scan sensing views：

```text
FULL
BILINEAR_50
BILINEAR_25
```

---

## Reference B — GMvR

论文：

> *A generalized framework for multi-view regression with diverse extensions*, Neurocomputing, 2026.

重点学习其三个原则：

\[
\boxed{
Prediction
+
Consistency
+
Complementarity
}
\]

本项目的新实验逻辑必须显式区分：

- shared mechanical prediction；
- view-specific useful information。

---

## Reference C — Deep Mutual Learning

论文：

> Zhang et al., *Deep Mutual Learning*, CVPR 2018.

官方代码：

```text
YingZhangDUT/Deep-Mutual-Learning
```

借鉴：

> 多个 peer experts 独立学习，同时在 prediction level 互相提供约束。

不要使用原分类 KL loss。

改造成 continuous regression agreement。

---

## Reference D — DiCoM

论文：

> *Diverse and Consistent Multi-view Networks for Semi-supervised Regression*.

只借鉴一个重要原则：

> **consistency 不能导致 view collapse；准确的 ensemble 需要 consistency 和 diversity 同时存在。**

本项目不需要照搬 semi-supervised protocol。

---

# 3. 核心新实验链

严格按照：

```text
E1 Cross-view predictive audit
        ↓
E2 Cooperative multi-view regression
        ↓
E3 Consistency + complementarity
        ↓
E4 Dynamic mixture-of-experts
        ↓
E5 Optional distributional / diffusion extension
```

每一步必须有 GO / NO-GO。

禁止直接跳到 E4/E5。

---

# 4. E1 — Cross-View Predictive Equivalence and Complementarity Audit

这是第一优先级。

## 4.1 Views

统一使用已经通过 semantic audit 的：

```text
FULL
BILINEAR_50
BILINEAR_25
```

暂时不要加入：

```text
12.5%
PCHIP
Bicubic
```

避免重新引入 P5/P6 semantic ambiguity。

---

# 5. 每个 View 单独建立完整 Predictor

对于 specimen \(i\)：

\[
D_i^F,\quad D_i^{50},\quad D_i^{25}
\]

使用相同 frozen ResNet18：

\[
f_i^v=E(D_i^v)
\]

但每个 view 允许拥有**独立的 fold-local**：

- scaler；
- PCA；
- regressor；
- hyperparameters。

得到：

\[
\hat y_i^F,
\quad
\hat y_i^{50},
\quad
\hat y_i^{25}.
\]

原因：

> 不再强制三个 view 共享相同 representation geometry。

---

# 6. E1 必须输出 Strict OOF Predictions

生成：

```text
results/multiview/e1_audit/oof_predictions.csv
```

至少包含：

```text
specimen_id
domain_id
y_true

pred_full
pred_50
pred_25

err_full
err_50
err_25
```

所有预测必须来自真正 held-out outer / inner evaluation。

禁止使用 in-sample prediction 做 complementarity 分析。

---

# 7. E1-A — Individual Performance

报告：

```text
FULL MAE
50% MAE
25% MAE
```

以及：

- per-domain MAE；
- worst-domain MAE；
- domain SD；
- R²；
- RMSE。

首先确认 P5/P7 已有结果能够在统一 protocol 下重现。

---

# 8. E1-B — Prediction Agreement

计算：

\[
\rho(
\hat y^F,\hat y^{50}
)
\]

\[
\rho(
\hat y^F,\hat y^{25}
)
\]

\[
\rho(
\hat y^{50},\hat y^{25}
)
\]

同时计算：

\[
MAE(
\hat y^F,
\hat y^{25}
)
\]

等 prediction disagreement。

目标：

> 判断各 view 是否在 specimen level 表达近似机械判断。

---

# 9. E1-C — Residual Correlation

这是本阶段最重要的诊断之一。

定义：

\[
e_v=y-\hat y_v
\]

计算：

\[
\rho(e_F,e_{50})
\]

\[
\rho(e_F,e_{25})
\]

\[
\rho(e_{50},e_{25})
\]

解释：

### residual correlation 很高

说明：

> views 高度冗余，犯类似错误。

multi-view fusion 上限可能有限。

### residual correlation 中低

说明：

> views 存在 complementary error patterns。

允许进入 E3。

---

# 10. E1-D — Oracle Complementarity

仅作为诊断，不作为正式预测结果。

计算：

\[
E_{oracle}
=
\frac1N
\sum_i
\min_v
|\hat y_{iv}-y_i|
\]

得到 oracle MAE。

同时计算：

> oracle 相对于 FULL 0.0896358 的理论最大改善空间。

注意：

```text
oracle result must never be reported as deployable performance.
```

---

# 11. E1-E — Win/Loss Map

对每个 specimen：

```text
best_view ∈ {FULL,50,25}
```

分析：

- FULL 最优比例；
- 50 最优比例；
- 25 最优比例。

再按：

- domain；
- ply count；
- layup；
- damage descriptors；

统计。

目标：

> 判断 view 优势是否与 specimen condition 有系统关系。

---

# 12. E1 GO / NO-GO

进入 multi-view cooperative learning 的最低条件：

满足以下任意一种：

## Case A — Predictive Equivalence

三个 view：

- prediction correlation 高；
- performance 接近；

支持：

> 不同 representation 可以表达相同 mechanical output。

## Case B — Complementarity

至少两个 view：

- individual MAE 均有价值；
- residual correlation 不高；
- oracle MAE 明显低于 0.0896358。

优先希望出现：

\[
Oracle\ improvement\ge10\%
\]

这只是判断是否值得继续 fusion，不是论文最终门槛。

---

# 13. E2 — Cooperative Multi-View Regression

若 E1 GO，则实现 Cooperative Learning 风格 regression。

不要做 feature alignment。

每个 view 保留独立 predictor：

\[
g_F
\]

\[
g_{50}
\]

\[
g_{25}.
\]

---

# 14. Prediction Loss

对于三个 view：

\[
L_{\text{pred}}
=
\frac13
\sum_v
\ell(
\hat y_v,y
)
\]

第一版：

```text
MSE
Huber
```

inner selection。

---

# 15. Cooperative Agreement Loss

定义：

\[
L_{\rm cons}
=
\frac13
[
(\hat y_F-\hat y_{50})^2+
(\hat y_F-\hat y_{25})^2+
(\hat y_{50}-\hat y_{25})^2
]
\]

总损失：

\[
\boxed{
L
=
L_{\rm pred}
+
\lambda_c L_{\rm cons}
}
\]

搜索：

```text
lambda_c:
0
1e-3
3e-3
1e-2
3e-2
0.1
0.3
1.0
```

也可在 inner search 中使用 log-uniform continuous range。

---

# 16. 非常重要：不要让 Consistency Collapse

必须同时监控：

\[
Var_v(\hat y_v)
\]

以及：

\[
\rho(e_v,e_u)
\]

如果 lambda 增大导致：

```text
三个 predictor 完全一样
但整体 MAE 不降
```

则属于 collapse。

禁止因为 agreement 更高就认定模型更好。

---

# 17. E2 Outputs

至少比较：

```text
FULL only
50 only
25 only

Independent 3-expert mean
Independent validation-weighted mean

Cooperative λ=selected
```

---

# 18. E2 要验证的科学问题

不是：

> 多视图一定更好。

而是：

\[
\boxed{
\text{Can prediction-level agreement improve CAI
without feature-level invariance?}
}
\]

如果 E2：

\[
MAE<0.0896358
\]

且至少：

```text
4/6 domains improve
```

则说明：

> prediction-level consistency 是比 feature invariance 更合理的机制。

---

# 19. E2 Success Targets

Minimum positive：

\[
MAE<0.0896358
\]

Formal positive：

\[
MAE\le0.08515
\]

约 5% improvement。

Strong：

\[
MAE\le0.08247
\]

约 8%。

但即使平均改善只有 3–5%，若：

- worst-domain 明显下降；
- 5/6 或 6/6 domain 改善；

也保留为 robustness-positive result。

---

# 20. E3 — Consistency + Complementarity

E2 不应该是终点。

GMvR 的核心启示是：

\[
\boxed{
Consistency
+
Complementarity
}
\]

所以必须测试：

> agreement 是否应该是软约束，而不是所有 view 完全一致。

---

# 21. E3-A — Equal Late Fusion

最简单：

\[
\hat y
=
\frac13
(
\hat y_F+
\hat y_{50}+
\hat y_{25}
)
\]

这是必须 baseline。

---

# 22. E3-B — Validation-Weighted Fusion

学习：

\[
w_F,w_{50},w_{25}
\]

满足：

\[
w_v\ge0
\]

\[
\sum_vw_v=1
\]

最小化 inner validation MAE。

最终：

\[
\hat y
=
w_F\hat y_F
+
w_{50}\hat y_{50}
+
w_{25}\hat y_{25}.
\]

所有权重：

> inner domains only。

---

# 23. E3-C — Stacking

使用：

\[
[
\hat y_F,
\hat y_{50},
\hat y_{25}
]
\]

作为 level-1 输入。

meta-regressor 仅使用：

```text
Ridge
non-negative Ridge
Huber
```

不要第一轮上复杂网络。

训练必须使用 strict OOF source predictions，避免 stacking leakage。

---

# 24. E3-D — GMvR-Style Objective

实现一个轻量 consistency + complementarity objective。

形式不必逐字复现 MvLSR-2C，但必须借鉴其原则：

\[
L
=
L_{\rm pred}
+
\lambda_cL_{\rm consistency}
+
\lambda_rL_{\rm complementarity}
\]

其中 complementarity 不要通过强制 feature 不同来定义。

优先采用：

- view-specific regression coefficients；
- view-specific prediction contribution；
- learned non-negative view weights。

---

# 25. E3 关键问题

验证：

\[
\boxed{
\text{Do different sensing views contain complementary
mechanical predictive information?}
}
\]

最重要比较：

\[
MAE_{\rm fusion}
<
\min(
MAE_F,
MAE_{50},
MAE_{25}
)
\]

如果成立：

> multi-view complementarity 得到直接支持。

---

# 26. Complementarity Success Gate

至少：

\[
MAE_{\rm best\ fusion}
<
0.0896358
\]

并：

```text
>=4/6 outer domains improve
```

Formal target：

\[
\le0.08515
\]

Strong target：

\[
\le0.08247.
\]

---

# 27. E4 — Dynamic Mixture of Experts

只有 E3 证明：

> 不同 views 确实存在 complementarity

才允许进入。

不要提前做。

---

# 28. E4 Goal

不是：

> 让一个大模型融合三个 view。

而是：

> **针对不同 specimen 动态选择更可信的 view / expert。**

三个 expert：

\[
g_F,g_{50},g_{25}
\]

输出：

\[
\hat y_F,\hat y_{50},\hat y_{25}.
\]

gating：

\[
w_i=
softmax(h(q_i))
\]

最终：

\[
\hat y_i
=
\sum_v
w_{iv}\hat y_{iv}.
\]

---

# 29. Gate Input 必须非常小

由于只有 276 specimen，禁止 giant transformer。

候选：

```text
expert predictions
prediction disagreement
expert feature norms
damage descriptors
low-dimensional PCA features
```

gating model：

```text
linear softmax
small MLP
decision tree / gradient boosting
```

优先简单模型。

---

# 30. E4 必须有 Oracle Upper Bound

比较：

```text
oracle view selection
learned gating
static weighted fusion
```

如果 learned gating 与 oracle 上限差距巨大：

> 不继续增加 gate capacity。

说明样本不足以学习 selection rule。

---

# 31. E5 — Optional Diffusion / Distributional Cross-View Modeling

Diffusion 暂时不进入第一阶段。

只有同时满足：

1. E1 证明 cross-view relation 非平凡；
2. E3/E4 证明 views complementary；
3. deterministic fusion 仍明显低于 oracle upper bound；

才允许研究 Diffusion。

---

# 32. Diffusion 的新角色

禁止再次做：

```text
image reconstruction
nuisance removal
feature invariance
```

如果进入 E5，Diffusion 只允许研究：

> **distributional cross-view transport / uncertainty。**

例如：

\[
p(
f_F|f_{25}
)
\]

或者：

\[
p(
\Delta f_{25\rightarrow F}
\mid
f_{25}
)
\]

目标：

> 从一个有效 sensing view 推断另一个 view 的可能 representation 分布。

不是生成 C-scan 图像。

---

# 33. E5 必须先和简单 Mapping 比

比较：

```text
linear mapping
ridge mapping
MLP mapping
Gaussian conditional model
normalizing flow
latent diffusion
```

只有 deterministic mapping 明显不足且 distributional model 有优势时：

> Diffusion 才允许晋升。

---

# 34. Cross-View Reliability Metric

新增：

\[
V_i
=
Std(
\hat y_i^F,
\hat y_i^{50},
\hat y_i^{25}
)
\]

称：

> cross-view prediction dispersion。

测试：

\[
corr(
V_i,
|\hat y_i-y_i|
)
\]

如果显著正相关：

> cross-view disagreement 可以作为 prediction reliability signal。

这对 AEI 非常有价值。

---

# 35. Reliability Experiment

根据：

\[
V_i
\]

按低 → 高分组。

报告：

```text
lowest 25% disagreement
middle 50%
highest 25%
```

各自 prediction error。

如果：

```text
high disagreement
→
high CAI error
```

则得到一个新的工程解释：

> sensing-view disagreement can indicate unreliable CAI assessment.

---

# 36. Reference Method Audit

建立：

```text
docs/MULTIVIEW_REFERENCE_METHOD_AUDIT.md
```

记录：

## Cooperative Learning

Paper：

```text
Cooperative Learning for Multi-view Analysis
PNAS 2022
```

Repo：

```text
dingdaisy/cooperative-learning
```

借：

```text
agreement penalty
multi-view regression structure
cross-validation of agreement strength
>2-view formulation
```

不借：

```text
their datasets
their feature extractors
```

---

## GMvR

Paper：

```text
A generalized framework for multi-view regression with diverse extensions
Neurocomputing 2026
```

借：

```text
prediction
consistency
complementarity
```

作为 E3 理论框架。

---

## Deep Mutual Learning

Paper：

```text
Deep Mutual Learning
CVPR 2018
```

Repo：

```text
YingZhangDUT/Deep-Mutual-Learning
```

借：

> peers collaboratively constrain predictions。

需要重新实现 regression version。

禁止把分类 KL loss 直接复制。

---

## DiCoM

借：

> consistency alone can cause collapse；需要 diversity/complementarity。

不需要照搬 semi-supervised architecture。

---

# 37. Code Structure

新建：

```text
src/cmc_bbdm/aei_multiview_regression/
```

建议：

```text
view_experts.py
oof_predictions.py
agreement_audit.py
complementarity.py

cooperative_regression.py
mutual_regression.py

late_fusion.py
stacking.py
gmvr_regression.py

moe_gate.py
reliability.py

cross_view_transport.py
latent_diffusion_transport.py

search.py
formal_outer.py
artifacts.py
replay.py
```

不要修改 A0–A5 原始结果。

---

# 38. Independent Unit

所有 views 仍属于同一 specimen。

任何：

```text
FULL
50
25
```

都不能被当作三个独立 specimen。

所有：

- split；
- bootstrap；
- stacking；
- gating；
- statistical inference；

必须：

\[
\boxed{
\text{specimen grouped}
}
\]

---

# 39. Stacking Leakage Rule

如果训练 meta-regressor：

绝对不能使用 source model 对自身训练数据的 in-sample prediction。

必须：

```text
source domains
→ inner OOF predictions
→ meta model
```

正式 outer test：

```text
refit base models on all source
→ outer predictions
→ frozen meta model
```

---

# 40. Evaluation Protocol

Primary：

```text
strict six-domain LODO
```

另外保留：

```text
leave-ply-count-out
leave-layup-family-out
```

作为 engineering stress tests。

---

# 41. Primary Metric

\[
\boxed{
equal-domain CAI ratio MAE
}
\]

baseline：

\[
0.0896358.
\]

Secondary：

```text
per-domain MAE
worst-domain MAE
domain SD
R²
RMSE
```

如可恢复 MPa：

同时输出：

```text
MAE MPa
RMSE MPa
```

---

# 42. 新增 Multi-View Metrics

必须报告：

```text
prediction correlation
residual correlation
prediction disagreement
oracle MAE
best-view frequency
fusion gain
cross-view dispersion
dispersion-error correlation
```

---

# 43. 第一轮 Desired Result Pattern

理想情况下，希望出现：

```text
FULL        ~0.0896
50%         ~similar
25%         ~similar

equal fusion        < 0.0896
cooperative         < equal fusion
weighted fusion     < best single
consistency+comp    best
```

注意：

> 这是期望验证的趋势，不允许伪造或强行得到。

---

# 44. 正式 Promotion Targets

最低：

\[
MAE<0.0896358
\]

并：

```text
>=4/6 domains improve
```

较强：

\[
MAE\le0.08515
\]

强：

\[
MAE\le0.08247.
\]

如果：

\[
MAE<0.080
\]

属于非常强结果。

---

# 45. 科学 Success Pattern

即使最终 MAE 只改善 3–5%，如果发现：

```text
feature representations differ strongly
prediction outputs agree
residual errors are partially complementary
fusion improves worst-domain performance
view disagreement predicts error
```

同样属于非常有价值的 AEI positive result。

因为新科学结论是：

> **mechanical consistency does not require feature invariance.**

---

# 46. Stop Rules

## E1 没有 Complementarity

如果：

```text
residual correlation extremely high
oracle improvement negligible
```

则：

> E3/E4 NO-GO。

只保留 predictive-equivalence analysis。

---

## E2 Cooperative 不改善

如果：

```text
lambda=0
```

始终最好，

说明：

> prediction agreement regularization 无价值。

停止继续增加 consistency complexity。

---

## E3 Fusion 不改善

如果所有：

```text
equal fusion
weighted fusion
stacking
```

均不超过 FULL，

则：

> views 在 CAI 上基本冗余。

停止 MoE。

---

## E4 MoE 不接近 Oracle

如果 oracle 很强但 learned gate 无法捕获：

> 数据量不足以学习 dynamic selection。

停止扩大 gate。

---

## E5 Diffusion

只有 deterministic cross-view models 明显存在 distributional gap 后才允许。

---

# 47. 第一批必须生成的文档

开始正式 E2 前：

```text
docs/AEI_MULTIVIEW_SCIENTIFIC_PROTOCOL.md
docs/MULTIVIEW_REFERENCE_METHOD_AUDIT.md
docs/E1_CROSS_VIEW_AUDIT_PROTOCOL.md
docs/AEI_MULTIVIEW_CLAIM_EVIDENCE_MATRIX.md
```

---

# 48. 第一阶段执行顺序

严格：

```text
STEP 1
Reproduce FULL 0.0896358.

STEP 2
Generate canonical FULL / 50 / 25 feature banks.

STEP 3
Train independent view predictors.

STEP 4
Generate strict specimen-level OOF predictions.

STEP 5
Run prediction agreement analysis.

STEP 6
Run residual-correlation analysis.

STEP 7
Compute oracle complementarity.

STEP 8
Analyse best-view frequency by domain/ply/layup.

STEP 9
Issue E1 GO / NO-GO.

IF E1 GO:
    implement Cooperative Learning regression.

STEP 10
Run λ consistency search.

STEP 11
Compare independent/equal-average/cooperative.

STEP 12
Implement late fusion / stacking.

STEP 13
Implement consistency + complementarity objective.

STEP 14
Formal outer evaluation.

IF complementarity confirmed:
    implement small MoE gate.

IF remaining oracle gap is large:
    investigate distributional cross-view mapping.

Only then consider Diffusion.
```

---

# 49. 新论文的 Positive Story

如果实验成功，新的论文故事应该变成：

```text
A/W/H
↓
insufficient

Full C-scan
↓
strong CAI value

Reduced C-scan
↓
similar mechanical predictive capability

Feature invariance
↓
fails

Therefore:
same mechanical state does not require same feature representation

Different sensing views
↓
distinct representations
+
consistent CAI predictions
+
complementary errors

Cooperative multi-view regression
↓
prediction-level agreement
without representation collapse

Complementary fusion
↓
improved cross-domain CAI
```

---

# 50. 核心科学结论

本轮真正要验证：

> **Different ultrasonic sensing views may encode the same mechanical state through distinct feature representations. Reliable CAI assessment should therefore enforce consistency at the prediction level while preserving view-specific complementarity, rather than forcing feature-level invariance.**

中文：

> **不同 C-scan 观测形式可能通过不同的特征路径编码同一机械状态。因此更合理的方法是在机械预测层建立一致性，同时保留各观测形式的互补信息，而不是强迫它们共享同一个特征表示。**

---

# 51. 方法 Novelty 应如何写

不要声称：

> We invent multi-view consistency.

这是已有理论。

正确 novelty：

> **We introduce mechanically validated C-scan sensing transformations as multi-view observations and show that feature-level invariance is detrimental, motivating prediction-level cooperation that explicitly balances mechanical consistency and view complementarity for cross-domain CAI assessment.**

即：

### 已有轮子

```text
Cooperative Learning
Multi-view regression
Mutual learning
Consistency + complementarity
```

### 我们的新东西

```text
mechanics-validated sensing views
+
feature-invariance failure evidence
+
cross-domain CAI setting
+
mechanical prediction consistency
+
engineering reliability analysis
```

---

# 52. Final Principle

整个新阶段必须遵守：

\[
\boxed{
\text{Do not force the views to look the same.}
}
\]

\[
\boxed{
\text{Ask whether they mean the same mechanically.}
}
\]

以及：

\[
\boxed{
\text{Preserve disagreement when it contains useful information.}
}
\]

Diffusion 暂时退出主线。

只有当后续证明：

> cross-view relation 是复杂 distributional mapping，并且 Gaussian/Flow/确定性映射无法解释时，

Diffusion 才重新进入。

本轮第一目标是：

\[
\boxed{
0.0896358
\rightarrow
\text{lower MAE through prediction-level cooperation}
}
\]

而不是继续制造新的 feature invariance 方法。