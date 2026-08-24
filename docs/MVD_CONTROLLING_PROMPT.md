# Codex Execution Prompt — MVD Feasibility Gates for AEI-Oriented Mechanical-Value Ultrasonic Acquisition

## 0. 你的角色

你现在继续接管以下研究仓库：

```bash
git@github.com:Orangekostar/diff.git
````

当前基准提交：

```text
d0e0ebfca1f1de6b04e9cb43a5065de3435aee5b
```

当前已经正式完成：

* CFRP C-scan → CAI 基线
* P1 / P3 / P5 / P6 / P7
* MVA A0–A3
* MVA A4 Global Mechanical Mask
* MVA A5 Sequential Imitation Policy

冻结结论：

```text
A3: mechanical oracle headroom exists
A4: NO_GO
A5: NO_GO
A6: NOT AUTHORIZED
```

你不能修改这些历史 gate。

本轮不是“A5 优化”，也不是继续堆模型。

本轮唯一目标是判断：

> **Mechanical-Value Distilled One-Shot Acquisition 是否具备继续开发成 AEI 正式方法的科学前提。**

---

# 1. 最终 AEI 工程目标

最终论文希望建立的工程能力不是：

> prediction accuracy improvement on full C-scan

而是：

> **在有限 ultrasonic measurement budget 下，利用 coarse survey 为当前 specimen 生成 task-oriented refinement plan，从而以更少测量保持可靠 CAI assessment。**

最终目标能力：

```text
Coarse ultrasonic survey
        ↓
specimen-specific mechanical-value inference
        ↓
budget-constrained refinement plan
        ↓
additional ultrasonic measurements
        ↓
CAI assessment
```

最终希望证明：

```text
same sensing budget:
MVD > Uniform
MVD > Reconstruction-driven

or

same CAI fidelity:
MVD requires less measurement budget
```

本轮还没有资格声称这个结论。

---

# 2. 当前已有证据

必须先阅读并理解历史结果，禁止脱离已有 evidence 重新设计问题。

## 2.1 Full C-scan baseline

当前 authoritative internal-only full C-scan baseline：

[
MAE_{\text{FULL}} \approx 0.0896358
]

当前 strong evaluator：

```text
Frozen ImageNet ResNet18
→ final 512-D embedding
→ fold-local processing / PCA
→ registered CAI regressor
```

禁止本轮更换 backbone。

---

## 2.2 P3 — Spatial organization matters

已有 spatial intervention 证明：

> spatial organization 对 CAI prediction 有真实贡献。

因此 acquisition strategy 不能只优化：

```text
number of measurements
```

还必须考虑：

```text
where measurements are acquired
```

---

## 2.3 P5/P7 — Dense sensing contains redundancy

已有 25% sparse internal observation：

```text
MAE ≈ 0.090116
```

与 FULL：

```text
0.0896358
```

非常接近。

这支持：

> dense spatial acquisition contains substantial redundancy for CAI assessment.

---

## 2.4 P6 — Reconstruction fidelity is not mechanical fidelity

已有实验表明：

```text
diffusion / learned reconstruction
```

并不比简单 interpolation 更能保存 downstream CAI information。

因此新方法不得以：

```text
PSNR / SSIM maximization
```

作为主目标。

---

## 2.5 A3 — Mechanical acquisition headroom exists

Mechanical oracle 明显优于：

```text
Uniform
Reconstruction oracle
Appearance heuristic
```

并且 Mechanical Value map 与 reconstruction-value map 相关性很低。

因此已有证据支持：

[
\text{reconstruction-important measurement}
\neq
\text{CAI-important measurement}
]

但 A3 oracle 使用 true CAI，因此：

```text
A3 = diagnostic upper bound
```

不是 deployable method。

---

## 2.6 A4 — Global mechanical pattern failed

A4 Global Mechanical Mask：

```text
NO_GO
```

说明当前数据不支持：

> one universal mechanical-value spatial mask works for all specimens.

因此未来方法必须：

```text
specimen-conditioned
```

而不是 global fixed mask。

---

## 2.7 A5 — Sequential imitation failed

A5：

```text
NO_GO
```

总体 AUEBC improvement 很弱，
只有部分 domains 改善，
且性能接近简单 observed-uncertainty heuristic。

因此：

> 不允许把 A5 换成 Transformer / GNN / RL 后继续碰运气。

当前只能把 A5 当：

```text
historical sequential baseline
```

---

# 3. 新候选方法

暂命名：

# MVD

## Mechanical-Value Distilled One-Shot Acquisition

核心区别：

A5：

```text
score
→ select 1 cell
→ reveal
→ update state
→ rescore
→ select
→ ...
```

MVD：

```text
build coarse state S0 once
        ↓
predict all candidate mechanical values once
        ↓
freeze ranking / subset
        ↓
select all refinement regions under budget
        ↓
perform refinement
        ↓
final CAI prediction
```

即：

[
\boxed{\text{one-shot specimen-specific acquisition}}
]

禁止 rollout 中重新评分。

---

# 4. 为什么当前不能直接开发 MVD

现在还有两个尚未回答的科学问题。

## Gate M0

> **即使得到完美的 initial Mechanical Value Map，一次性选区是否真的有效？**

A3 是 sequential oracle。

不能自动推出：

```text
initial one-shot oracle is also strong
```

---

## Gate M1

> **Initial Mechanical Value Map 能否仅从当前 coarse observations 中预测出来？**

Mechanical Value 使用 true CAI 生成，是 privileged training knowledge。

如果 partial observation 无法预测：

[
V^{CAI}
]

则 MVD 无法部署。

因此本轮只允许验证：

```text
M0 — One-shot Oracle Feasibility
M1 — Mechanical-Value Observability
```

不允许直接开发最终 MVD proposed model。

---

# 5. 第一阶段：Repository Authority Audit

首先完整审计当前 MVA package。

重点阅读：

```text
src/cmc_bbdm/mva/
```

至少确认以下模块及其 authoritative semantics：

```text
acquisition_grid.py
measurement_state.py
refinement_simulator.py
interpolation.py

encoder_session.py
cai_evaluator.py
crossfit.py

mechanical_value.py

a4_candidate_bank.py
a4_source_labels.py

candidate_features.py
policy_state.py
ranking_policy.py
a5_deployment.py

budget_metrics.py
statistics.py
artifacts.py
```

输出：

```text
docs/MVD_REPOSITORY_AUTHORITY_AUDIT.md
```

---

# 6. Authority audit 必须回答

逐项回答：

1. 当前 acquisition grid native shape；
2. initial budgets；
3. 8×8 candidate-cell semantics；
4. level0 / level1 / level2 nested lattice；
5. exact unique-measurement counting；
6. bilinear reconstruction semantics；
7. measured-point restoration；
8. frozen ResNet18 hash；
9. full baseline reproduction；
10. CandidateBank feature dimensions；
11. CandidateBank 中 candidate embedding 是否代表“真实 refine 后状态”；
12. A4 source labels 如何保证 strict OOF；
13. source query domain 是否从 predictor training 中排除；
14. outer target 是否完全隔离；
15. A5 policy 当前输入；
16. A5 supervision 当前只保存 selected action 还是完整 candidate values；
17. historical A4/A5 artifact hashes；
18. 可直接复用代码；
19. 禁止修改的冻结代码；
20. 新 MVD 应新建哪些模块。

---

# 7. 强制复用的代码

不要重新实现以下逻辑：

```text
AcquisitionGrid
MeasurementState
apply_action
budget accounting

bilinear reconstruction
measured-value restoration

frozen ResNet18 encoder
registered CAI evaluator

A4 CandidateBank
A4 strict-OOF mechanical labels

Mechanical Value definition

AUEBC
B_5%
bootstrap
artifact hashing
replay
```

---

# 8. 禁止修改历史 MVA

新代码必须放：

```text
src/cmc_bbdm/mvd/
```

禁止把新的 one-shot 方法偷偷写进：

```text
mva/a4_*.py
mva/a5_*.py
```

因为：

```text
A4 NO_GO
A5 NO_GO
```

必须永久保持可重放。

---

# 9. 新代码建议结构

```text
src/cmc_bbdm/mvd/
    __init__.py

    authority.py

    initial_value_dataset.py
    action_cost_audit.py
    interaction_audit.py

    one_shot_oracle.py

    observability_dataset.py
    observability_models.py
    observability_metrics.py

    value_model.py
    value_losses.py
    subset_selection.py

    one_shot_deployment.py

    evaluation.py
    statistics.py

    artifacts.py
    replay.py
    cli.py
```

本轮主要实现到：

```text
one_shot_oracle.py
observability_*.py
```

为止。

最终 MVD student 暂时 LOCKED。

---

# 10. M0 — One-Shot Oracle Feasibility

## 10.1 科学问题

A3 sequential oracle 已证明：

> state-dependent adaptive acquisition has large hindsight headroom.

现在必须独立回答：

> **一个仅在初始 coarse state S0 上计算一次的 perfect Mechanical Value ranking，能否形成有效的 one-shot acquisition plan？**

---

# 11. M0 数据来源

不要重新跑完整 oracle。

直接消费：

```text
A4 CandidateBank
A4 SourceLabelResult
```

每个 source specimen 应得到：

```text
specimen_id
dataset_id

initial_embedding       # 512D

candidate_embeddings    # 64 × 512
mechanical_values       # 64

added_measurements      # 64

initial_budget
grid hash
predictor hash
candidate-bank hash
```

所有数组：

```text
readonly
hash-bound
```

---

# 12. M0 前必须做 Action-Cost Audit

当前不能假设：

```text
one cell = same cost
```

生成：

```text
results/mvd/m0_one_shot_oracle/action_cost_audit.csv
```

逐：

```text
specimen
initial budget
cell
```

记录：

```text
added_measurements
added_fraction
row
column
boundary/interior
```

统计：

```text
min
max
mean
std
CV
unique costs
```

---

# 13. Action cost 决策规则

如果所有 candidate refinement：

```text
exact added measurement count identical
```

则 one-shot oracle 可以按：

```text
descending Mechanical Value
```

进行 Top-K。

如果 costs 不完全一致：

禁止简单 Top-K。

需要使用：

```text
budget-aware greedy selection
```

第一版按：

```text
predicted/oracle value descending
```

遍历 candidate：

```text
if action fits exact checkpoint:
    select
```

不要立即写复杂 0-1 knapsack。

---

# 14. Initial Mechanical Value Ranking

每个 specimen：

[
V_{1:64}^{S_0}
]

只在：

[
S_0
]

计算一次。

生成：

[
r_i=\operatorname{argsort}(V_i,\ descending)
]

整个 acquisition 过程中：

```text
ranking remains frozen
```

绝对禁止：

```text
after first action:
recompute Mechanical Value
```

否则不再是 one-shot oracle。

---

# 15. M0 Evaluation

使用历史注册 budget checkpoints。

至少：

```text
6.25%
9.375%
12.5%
18.75%
25%
```

如果 initial survey 已经达到某 checkpoint，则按现有 MVA semantics 处理。

对于每个 budget：

```text
initial state S0
→ use frozen initial oracle ranking
→ add maximum legal actions under budget
→ reconstruct
→ CAI prediction
```

---

# 16. M0 Comparators

至少：

```text
Uniform
Random
Reconstruction one-shot ranking
Global Mechanical A4
One-shot Mechanical Oracle
Sequential A3 Mechanical Oracle
FULL
```

注意：

```text
Sequential A3 oracle
```

只作为 theoretical ceiling。

主比较是：

```text
One-shot Mechanical Oracle
vs
Uniform
vs
Reconstruction
```

---

# 17. M0 Interaction Audit

不能默认：

[
V(A)=\sum_{k\in A}V_k
]

。

因此对 source specimens 计算：

[
J(A)
====

## |y-\hat y(S_0)|

|y-\hat y(S_0+A)|
]

并比较：

[
J(A)
]

与：

[
\sum_{k\in A} V_k.
]

不穷举全部 subsets。

只审计：

```text
one-shot top-value sets
uniform sets
reconstruction-ranked sets
random sets
```

在注册 budgets：

```text
6.25
9.375
12.5
18.75
25%
```

报告：

```text
Pearson
Spearman
bias
MAE between predicted additive gain and actual joint gain
```

该 audit 的目的只是理解 interaction。

M0 GO 不依赖“必须完全可加”。

---

# 18. M0 主指标

继续复用：

[
AUEBC
]

和：

[
B_{5%}.
]

此外报告：

```text
per-budget CAI MAE
per-domain MAE
worst-domain effects
```

---

# 19. M0 GO / NO-GO

## GO

One-shot Mechanical Oracle 必须：

[
AUEBC_{\rm one-shot-oracle}
<
AUEBC_{\rm uniform}
]

且：

[
AUEBC_{\rm one-shot-oracle}
<
AUEBC_{\rm reconstruction}
]

并且 domain-level 方向不能由极少数 domain 驱动。

建议：

```text
>=4/6 source-held-out domains same direction
```

同时 one-shot oracle 相对 sequential oracle 不能丢失绝大多数 headroom。

需要明确报告：

[
HeadroomRetention
=================

\frac{
AUEBC_{baseline}-AUEBC_{one-shot}
}{
AUEBC_{baseline}-AUEBC_{sequential-oracle}
}
]

---

## STRONG GO

如果：

```text
HeadroomRetention >= 50%
```

属于很强信号。

---

## NO-GO

如果：

```text
one-shot oracle ≈ uniform
```

或者：

```text
one-shot oracle ≈ reconstruction
```

则：

```text
MVD_ONE_SHOT_NO_GO
```

解释：

> sequential A3 headroom cannot be converted into a fixed initial refinement plan.

立即停止 MVD。

不得训练 value student。

---

# 20. M0 输出

```text
results/mvd/m0_one_shot_oracle/
```

至少：

```text
action_cost_audit.csv
interaction_audit.csv

uniform_curve.csv
reconstruction_curve.csv
one_shot_oracle_curve.csv
sequential_oracle_curve.csv

domain_metrics.csv
budget_metrics.csv
bootstrap.csv

summary.json
REPORT.md
artifact_manifest.json
CHECKSUMS.sha256
```

---

# 21. M1 — Mechanical-Value Observability Audit

只有：

```text
M0 = GO
```

才允许执行。

---

# 22. M1 科学问题

Mechanical Value：

[
V_k^{CAI}
]

由：

```text
full training information
+
true CAI
```

生成。

部署阶段这些信息不存在。

所以必须回答：

> **当前 coarse observation 中是否包含足够信息，使模型能够预测不同 candidate regions 的 Mechanical Value ranking？**

如果答案 NO：

```text
privileged knowledge is not deployable
```

MVD 必须停止。

---

# 23. M1 禁止直接优化 final CAI MAE

第一阶段只看：

```text
Mechanical Value predictability
```

因为先要回答：

> knowledge observable?

而不是：

> final acquisition performance?

---

# 24. M1 数据集

创建：

```text
initial_value_dataset.py
```

每条 specimen 包含：

```text
specimen_id
dataset_id

initial_embedding      # 512D current coarse observation

current_prediction

candidate_features     # 64 × 8 observed-only

mechanical_values      # 64 strict-OOF teacher values

candidate_costs        # 64

authority hashes
```

禁止 student 读取：

```text
candidate refined embedding
full image
unobserved RGB
true CAI
oracle future states
```

---

# 25. 必须新增泄漏测试

至少：

```text
test_mvd_student_never_reads_true_cai.py
test_mvd_student_never_reads_candidate_embedding.py
test_mvd_student_never_reads_full_scan.py
test_mvd_student_never_reads_unobserved_rgb.py
```

尤其：

```text
candidate_embeddings[N,64,512]
```

只能生成 oracle labels。

Deployable model 绝对不能读取。

---

# 26. M1 Observability Baselines

不要直接上一个“大网络”。

必须比较：

## O0 — Global mechanical ranking

A4 historical baseline。

---

## O1 — Candidate-only model

输入：

```text
8D observed-only candidate features
```

回答：

> 只靠 candidate geometry/local observed state 能预测多少 Mechanical Value？

---

## O2 — Global + Candidate

Primary。

输入：

```text
512D initial coarse embedding
current CAI prediction
8D candidate features
```

---

## O3 — A5-style state

输入：

```text
historical A5 state
+
candidate features
```

作为 compatibility baseline。

---

# 27. M1 第一版模型

先跑：

```text
Ridge
small MLP
```

不要：

```text
Transformer
GNN
CNN decoder
Diffusion
RL
```

MLP 共享 candidate scorer，例如：

```text
global:
513 → 64 → 32

candidate:
8 → 32 → 16

concat:
48 → 32 → 1
```

参数控制：

```text
<100k preferred
```

---

# 28. M1 训练监督

必须使用完整：

[
V_{1:64}
]

而不是只保留：

```text
top-1 action
```

。

需要比较：

## L0

A5-style top-1 / pairwise baseline。

## L1

Huber value regression：

[
L_v=
Huber(\hat V,V)
]

## L2

Value regression + margin-weighted ranking：

[
L=
L_v+\lambda L_r
]

其中 pair 权重与：

[
|V_i-V_j|
]

相关。

---

# 29. M1 Model Selection

只能：

```text
source-domain held-out
strict grouped/domain CV
```

禁止 outer target domain：

* 选 architecture；
* 选 loss；
* 选 hidden dim；
* 选 lambda；
* 选 features。

---

# 30. M1 主指标

逐 specimen 计算：

## Spearman

[
\rho(
\hat V_{1:64},
V_{1:64}
)
]

---

## NDCG

例如：

```text
NDCG@5
NDCG@10
```

---

## Top-K Recall

```text
Recall@5
Recall@10
```

---

## Regret

例如：

[
Regret@1
========

## V_{\max}

V_{\text{selected}}
]

以及：

```text
budgeted-set regret
```

---

## Mechanical value captured

selected candidate set 的真实：

[
V^{CAI}
]

相对于 oracle set 可捕获多少。

---

# 31. M1 最重要的比较

Observability model 必须优于：

```text
A4 global ranking
```

并最好优于：

```text
simple observed uncertainty heuristic
```

不能只证明：

```text
Spearman > 0
```

就算成功。

---

# 32. M1 GO / NO-GO

## GO

需要同时看到：

1. source-held-out Mechanical Value ranking 有稳定可预测性；
2. O2/O3 明显优于 A4 global ranking；
3. majority domains 同方向；
4. synchronized bootstrap lower bound 支持 positive rank association；
5. budgeted top-K regret 明显低于 global/random ranking。

---

## Strong GO

如果 one-shot predicted ranking 能捕获：

```text
>=30–40% of one-shot oracle advantage
```

可认为值得开发完整 MVD。

这里 advantage 应定义为：

[
AUEBC_{baseline}
----------------

AUEBC_{\text{oracle/predicted}}
]

而不是人为发明一个单独 accuracy。

---

## NO-GO

如果：

```text
Mechanical Value ranking
cannot be predicted better than global/uncertainty baselines
```

则：

```text
MVD_OBSERVABILITY_NO_GO
```

解释：

> hindsight Mechanical Value is not sufficiently observable from deployable coarse observations.

立即停止。

禁止：

```text
Transformer rescue
GNN rescue
RL rescue
Diffusion rescue
```

---

# 33. M1 输出

```text
results/mvd/m1_observability/
```

至少：

```text
observability_predictions.parquet

model_metrics.csv
domain_metrics.csv
ranking_metrics.csv
regret_metrics.csv
bootstrap.csv

summary.json
REPORT.md
artifact_manifest.json
CHECKSUMS.sha256
```

---

# 34. 本轮禁止执行 M2

即使 M0/M1 GO，

本轮也先停止。

不要实现正式 proposed MVD deployment。

先把：

```text
M0 result
M1 result
```

提交给人工 review。

之后再决定是否授权：

```text
M2 student development
M3 formal evaluation
```

---

# 35. 并行任务 E0 — External Dataset Feasibility Audit

这一项可与 M0/M1 并行。

但：

[
\boxed{\text{禁止运行任何 MVD performance}}
]

只允许：

```text
download
unpack
hash
inspect
pair
count
document
```

---

# 36. External Dataset 1 — Imperial RSS

官方数据：

```text
Tailorable through-thickness fibre reinforcement in CFRP laminates
with AFP via Repeated Segment Stacking
```

Mendeley：

```text
10.17632/wg4dmwddjy.2
```

官方 metadata 已说明：

```text
raw HVI data
raw LVI data
raw CAI data
post-impact C-scan data
CC BY 4.0
```

任务：

实际下载并建立：

```text
specimen_id
cscan_path
cai_raw_path
material
layup/group
impact_condition
specimen_geometry
possible CAI target
```

最重要输出：

[
N_{\rm paired\ Cscan+CAI}
]

不能从摘要猜。

---

# 37. External Dataset 2 — Imperial Interlock

Zenodo：

```text
10.5281/zenodo.1476887
```

官方数据说明已明确：

```text
post-impact C-scans
+
subsequent CAI force data
+
displacement
+
strain-gauge data
```

实际下载后必须确认：

```text
paired specimen count
file naming
CAI target derivability
groups
geometry
license
```

---

# 38. TU Delft

数据：

```text
10.4121/21621381
```

已知只有：

```text
3 CAI specimens
```

且有：

```text
C-scan JPG
force-displacement CSV
```

只建立 manifest。

明确标记：

```text
MICRO_CASE_VALIDATION_ONLY
```

禁止把 n=3 当正式 statistical external benchmark。

---

# 39. External Audit 输出

创建：

```text
docs/EXTERNAL_CAI_DATA_FEASIBILITY_AUDIT.md
```

并生成：

```text
artifacts/external_data/
    imperial_rss_manifest.csv
    imperial_interlock_manifest.csv
    tudelft_manifest.csv

    EXTERNAL_DATA_MANIFEST.json
    CHECKSUMS.sha256
```

最终表：

| Dataset | Paired N | C-scan | CAI target | Groups | License | Role |
| ------- | -------: | ------ | ---------- | ------ | ------- | ---- |

---

# 40. External Dataset Discipline

本轮禁止：

```text
run Uniform
run Reconstruction
run MVD
run CAI predictor
inspect method performance
```

这些数据若满足条件，应继续保持：

```text
sealed external replication data
```

直到 MVD formal protocol 完全冻结。

---

# 41. 并行任务 R0 — Cranfield Raw PA Audit

数据：

```text
CompInnova WP2
10.5281/zenodo.4405277
```

已知官方 archive 含：

```text
5 MHz wheel PA:
raw CSV + processed PNG

10 MHz PA:
raw CSV + processed TIFF
```

任务：

解析 raw CSV：

```text
spatial coordinates
scan indexing
waveform/sample dimensions
header format
measurement ordering
```

并与：

```text
PNG/TIFF C-scan
```

建立对应。

---

# 42. Cranfield Audit 目标

回答：

1. raw CSV 中一个 record 对应什么 physical acquisition；
2. 是否可恢复 spatial grid；
3. scan spacing；
4. processed C-scan 与 raw grid 如何对应；
5. 是否可把当前 8×8 normalized refinement cells 映射到 raw spatial positions；
6. sparse mask 是否能定义为 raw measurement subset；
7. measurement fraction 是否可按真实 raw locations 精确计算。

---

# 43. Cranfield 不能做什么

禁止声称：

```text
CAI prediction validated
```

因为没有 CAI label。

禁止声称：

```text
real scanner time reduced by X%
```

因为没有真正控制 scanner 执行。

可以支持：

> spatial acquisition masks correspond to realizable raw phased-array measurement locations.

---

# 44. Cranfield 输出

```text
docs/CRANFIELD_RAW_PA_ACQUISITION_AUDIT.md
```

以及：

```text
artifacts/external_data/cranfield_wp2/
    raw_file_manifest.csv
    scan_pair_manifest.csv
    grid_schema.json
    example_mapping.json
    CHECKSUMS.sha256
```

---

# 45. Tests

至少新增：

```text
test_mvd_reuses_authoritative_grid.py

test_mvd_candidate_bank_binding.py
test_mvd_source_labels_strict_oof.py

test_mvd_action_cost_exact.py
test_mvd_one_shot_scores_once.py
test_mvd_no_sequential_recompute.py

test_mvd_student_never_reads_true_cai.py
test_mvd_student_never_reads_candidate_embedding.py
test_mvd_student_never_reads_full_scan.py
test_mvd_student_never_reads_unobserved_rgb.py

test_mvd_outer_domain_never_trains_observability_model.py

test_external_manifest_has_no_model_results.py

test_cranfield_raw_pairing.py

test_mvd_replay.py
```

---

# 46. Claim-Evidence Matrix

创建：

```text
docs/MVD_CLAIM_EVIDENCE_MATRIX.md
```

初始：

| Claim                                                | Current status |
| ---------------------------------------------------- | -------------- |
| Full C-scan predicts CAI                             | PROVEN         |
| Spatial organization matters                         | PROVEN         |
| Dense spatial sensing is redundant                   | PROVEN         |
| Reconstruction value differs from mechanical value   | PROVEN / A3    |
| Sequential hindsight oracle has large headroom       | PROVEN / A3    |
| A universal mechanical mask works                    | REFUTED / A4   |
| Current sequential BC reliably realizes oracle value | REFUTED / A5   |
| Initial one-shot Mechanical Value plan has headroom  | TO TEST / M0   |
| Mechanical Value is observable from coarse scan      | TO TEST / M1   |
| One-shot MVD improves deployable acquisition         | LOCKED         |
| Imperial external replication possible               | TO AUDIT       |
| Raw PA acquisition semantics realizable              | TO AUDIT       |

---

# 47. 本轮最终执行顺序

严格：

```text
STEP 0
Checkout d0e0ebf main.

STEP 1
Reproduce existing MVA authority and FULL baseline.

STEP 2
Write MVD_REPOSITORY_AUTHORITY_AUDIT.md.

STEP 3
Bind A4 CandidateBank + SourceLabelResult.

STEP 4
Implement action-cost audit.

STEP 5
Implement fixed-initial-ranking one-shot Mechanical Oracle.

STEP 6
Run one-shot oracle vs Uniform / Reconstruction.

STEP 7
Run interaction audit.

STEP 8
Issue M0 GO / NO-GO.

IF M0 NO_GO:
    stop all MVD model development.

IF M0 GO:
    continue.

STEP 9
Build strict initial Mechanical Value observability dataset.

STEP 10
Run candidate-only baseline.

STEP 11
Run global+candidate Ridge.

STEP 12
Run lightweight MLP.

STEP 13
Compare top1 / Huber / value+ranking supervision.

STEP 14
Compute Spearman/NDCG/Recall/Regret/value-capture.

STEP 15
Issue M1 GO / NO-GO.

STEP 16
STOP.

Do not implement M2/M3 in this execution.
```

Parallel：

```text
EXTERNAL-A:
download + audit Imperial RSS / Interlock / TU Delft

EXTERNAL-B:
download + audit Cranfield WP2 raw PA

NO METHOD PERFORMANCE ON EXTERNAL CAI DATA.
```

---

# 48. 本轮最终交付

代码：

```text
src/cmc_bbdm/mvd/
```

文档：

```text
docs/MVD_REPOSITORY_AUTHORITY_AUDIT.md
docs/MVD_M0_PROTOCOL.md
docs/MVD_M1_OBSERVABILITY_PROTOCOL.md
docs/MVD_CLAIM_EVIDENCE_MATRIX.md

docs/EXTERNAL_CAI_DATA_FEASIBILITY_AUDIT.md
docs/CRANFIELD_RAW_PA_ACQUISITION_AUDIT.md
```

结果：

```text
results/mvd/m0_one_shot_oracle/
results/mvd/m1_observability/
```

测试：

```text
full focused test suite PASS
Ruff PASS
replay PASS
checksum PASS
```

Git：

```text
clean working tree
local == origin/main
```

---

# 49. 结果报告必须回答的核心问题

最终 `REPORT.md` 不允许只说 GO/NO-GO。

必须明确回答：

## M0

1. Sequential A3 oracle headroom 有多少能被 initial one-shot plan 保留？
2. One-shot oracle 是否稳定超过 uniform？
3. 是否稳定超过 reconstruction-driven acquisition？
4. Mechanical values 是否存在严重 interaction / non-additivity？
5. 是否值得训练 student？

## M1

1. Mechanical Value ranking 是否可从 coarse observations 中预测？
2. candidate-only、global+candidate 哪种有效？
3. candidate Mechanical Values 是连续可预测，还是只有 top-level weak ranking signal？
4. predicted top-K 的 regret 多大？
5. 能捕获多少 one-shot oracle headroom？
6. 是否真正优于 A4 global ranking？
7. 是否值得开发正式 MVD？

## External

1. Imperial RSS paired C-scan+CAI N 是多少？
2. Imperial Interlock paired N 是多少？
3. 哪套数据足够 formal external replication？
4. TU Delft 是否只适合 case-level validation？
5. Cranfield raw PA 是否能恢复真实 spatial measurement grid？

---

# 50. Explicit Stop Rules

如果 M0 失败：

```text
DO NOT train MVD.
```

如果 M1 失败：

```text
DO NOT increase network capacity.
DO NOT add Transformer.
DO NOT add GNN.
DO NOT add RL.
DO NOT add Diffusion.
```

如果 Imperial paired N 太少：

```text
DO NOT claim statistical external validation.
```

如果 Cranfield raw grid 无法可靠恢复：

```text
DO NOT claim raw-acquisition realizability.
```

所有论文 claim 必须服从真实 evidence。

---

# 51. 本轮最重要的科研纪律

不要再遵循：

```text
positive phenomenon
→ assume a mechanism
→ build a network
```

必须遵循：

```text
positive phenomenon
→ formulate minimum testable hypothesis
→ test hypothesis
→ only then authorize method
```

当前：

```text
A3 oracle headroom
```

只允许推出两个问题：

```text
Does one-shot headroom exist?
Can Mechanical Value be observed?
```

不能直接推出：

```text
MVD will work.
```

---

# 52. Final Scientific Decision

只有满足：

```text
M0 = GO
M1 = GO
```

下一轮才允许正式开发：

# Mechanical-Value Distilled One-Shot Acquisition

最终 AEI 方法目标才可以定义为：

> **A specimen-specific ultrasonic acquisition framework that distills training-time CAI-informed measurement knowledge into a deployable coarse-to-refinement inspection plan, allowing limited sensing resources to be allocated according to their value for structural-integrity assessment.**

当前阶段不要声称：

```text
preserves full-scan CAI fidelity
reduces measurement budget
generalizes externally
```

这些必须由后续 M2/M3 + external validation 证明。

本轮只有一个目标：

[
\boxed{
\text{Determine whether this method is scientifically worth building.}
}
]
