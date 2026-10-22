<!-- block: TITLE -->

# Learning region selection and acquisition order in C-scan images for efficient compression-after-impact strength assessment of composite laminates

<!-- block: B001 -->

# Abstract

<!-- block: B002 -->

Compression-after-impact (CAI) strength assessment depends on which internal-image observations are supplied to the predictor and when they become available. We propose strength-assessment-driven multimodal sequential observation to learn C-scan region selection and acquisition order from specimen-level strength supervision. Initially available surface descriptors, acquired internal regions and the current CAI estimate form an observable decision state. A regional policy requests the next feasible region, and a common predictor updates strength after each observation. A trajectory objective combines process-average and terminal absolute errors to value useful estimates throughout acquisition. In offline regional replay on 50 development-validation specimens, also used for checkpoint selection, endpoint mean absolute error (MAE) was 42.385 MPa at an acquisition-area cap of 0.25, 9.65% below geometry-spread. The 0.0625 prefix of this same total-cap path reached MAE 43.324 MPa. On the five-cap reporting grid, first attainment of cohort MAE at most 47 MPa required a cap of 0.0625 versus 0.125 for geometry-spread. Recorded events showed larger early net error reductions. Fixed-final-set replay isolated arrival within selected content: the primary permutation-minus-original process contrast was +1.486 MPa, with a 95% interval of [−0.658, 3.384] MPa, leaving its average advantage uncertain. These results support task-driven observation organization for earlier useful CAI estimates within the selected development configuration.

<!-- block: B003 -->

**Keywords:** compression-after-impact strength; composite laminates; C-scan images; sequential acquisition; region selection; task-driven assessment.

<!-- block: B004 -->

# 1. Introduction

<!-- block: B005 -->

Compression-after-impact (CAI) strength connects the observed damage of a composite laminate to its residual compressive capacity. For carbon-fiber-reinforced polymer laminates, surface images and internal C-scan images provide complementary descriptions of the impacted specimen. Public paired-image datasets and measured CAI labels make this connection accessible to data-driven assessment \[1, 2\]. Image-based regression has further shown how C-scan spatial information can support strength prediction \[3\]. These developments create an opportunity to improve not only the predictor, but also the observations supplied to it.

<!-- block: B006 -->

In sequential assessment, internal C-scan information enters prediction region by region, so the current strength estimate depends on the content acquired so far. Region selection determines what is observed; acquisition order determines when it becomes available. A useful observation obtained early can inform more subsequent estimates, making information arrival relevant to prediction quality throughout acquisition. CAI assessment can therefore be formulated as a coordinated task of region selection, observation update and strength prediction.

<!-- block: B007 -->

The central design challenge is to assign regional observations value through the mechanical assessment they support. Specimen-level strength provides a direct task signal, whereas the usefulness of an individual region depends on its interaction with the information already available. The same final observed set can also support different intermediate estimates when its regions arrive in different orders. An assessment-oriented acquisition method thus needs to learn both which content to request and how to bring useful content into the prediction process early. A prescribed acquisition-area cap provides a common scale for comparing prediction quality, while a fixed prediction-quality target reveals how much internal information is needed to reach that quality.

<!-- block: B008 -->

Sequential inspection and active feature acquisition provide foundations for decisions based on an evolving information state \[4--6\]. Autonomous ultrasonic inspection links measurement selection to damage indication \[4\], while active feature acquisition learns to request information for downstream prediction \[5, 6\]. For CAI assessment, this decision structure must connect spatial image requests to a continuous mechanical output at every observed prefix. The unresolved question is how strength-supervised regional selection and information arrival can work together to improve the sequence of estimates, rather than treating acquisition and regression as separate stages.

<!-- block: B009 -->

We introduce strength-assessment-driven multimodal sequential observation. Initially available surface descriptors establish specimen-specific context. A regional policy combines this context with acquired internal descriptors, spatial position, measurement status, acquisition history, progress and the current CAI estimate to select the next feasible region. Each observation updates a common strength predictor and the state used for the next decision (Fig. 1). Training combines process-average absolute error with terminal error, assigning value to both early useful estimates and final prediction quality.

<!-- block: B010 -->

The study makes three connected contributions. First, it formulates regional C-scan acquisition as a strength-supervised decision task in which every observed prefix produces a mechanical estimate. Second, the multimodal observable state and process-plus-terminal objective jointly organize which regions are requested and when their information enters a common predictor. Third, a frozen common-predictor evaluation links matched-cap accuracy, matched-quality information requirements and fixed-final-set order interventions, while retaining conditional uncertainty and adverse cases. These contributions concern task formulation, observation organization and supporting evidence.

<!-- block: B011 -->

The study evaluates this formulation through regional image replay on a paired composite-laminate dataset. Section 2 positions the work in image-informed CAI prediction and sequential acquisition. Section 3 presents the representation, predictor and policy-learning objective; Section 4 specifies the experimental protocol. Section 5 examines accuracy, acquisition efficiency and the behavior underlying earlier useful estimates, followed by the conclusions in Section 6.

<!-- block: B012 -->

# 2. Related work

<!-- block: B013 -->

## 2.1. Image-informed compression-after-impact strength assessment

<!-- block: B014 -->

Image-informed strength assessment learns the relationship between an observed damage state and the subsequent mechanical response. Hasebe and colleagues released paired surface and internal-damage images after impact, together with subsequent CAI measurements \[1, 2\]. These resources support specimen-level investigation of complementary image modalities: the surface describes externally visible morphology, while C-scan images provide a spatial description of internal damage. Their pairing with mechanical labels makes it possible to evaluate how the supplied image information contributes to quantitative strength estimation.

<!-- block: B015 -->

Mack and colleagues used a ResNet18-based model to predict impact energy and CAI strength from C-scan images \[3\]. This work establishes a close engineering precedent for learning image-to-strength relationships. We extend the assessment question from reading available image content to selecting the regional content that becomes available during prediction. A common strength predictor provides the readout for this comparison, so the evaluated change is the organization of its observations. Final accuracy describes the selected content at completion, and the prediction trajectory describes its usefulness as acquisition progresses.

<!-- block: B016 -->

## 2.2. Sequential inspection and task-driven acquisition

<!-- block: B017 -->

Sequential inspection treats the next measurement as a decision informed by previous observations. Fuentes and colleagues combined Bayesian optimization with robust outlier analysis to select ultrasonic observations from an evolving novelty field \[4\]. Their observation--update--decision loop connects spatial sampling to damage indication and probabilistic inspection. The present method adopts this regional-request perspective and links the decision criterion to measured CAI strength, making the sequence of continuous strength estimates the object of optimization.

<!-- block: B018 -->

Sequential inspection couples each request to an evolving information state. For strength assessment, the quality of that state can be evaluated through a continuous mechanical output after every completed regional observation. This task criterion motivates the present comparison between spatial allocation rules and a specimen-conditioned acquisition policy under a common predictor.

<!-- block: B020 -->

## 2.3. Active feature acquisition and prediction with partial observations

<!-- block: B021 -->

Active feature acquisition learns which information to request for a downstream prediction task. Janisch and colleagues formulated costly-feature classification through sequential requests [5]. Covert and colleagues studied dynamic feature selection through conditional mutual information and amortized learning [6]. These approaches provide a decision perspective for assessing the value of information before all inputs are observed. Here, the requested units are C-scan regions and the output is continuous CAI strength throughout acquisition.

<!-- block: B022 -->

Partial observation also requires distinguishing available evidence from missing input. Residual-network image representations [7] and attention-based contextualization [8] supply building blocks for representing visible content and comparing candidate requests. The present method uses surface context and progressively revealed internal information for this purpose. Section 3 defines its specific masks, tokens and prediction interface; the experiments assess the complete observation-selection mechanism.

<!-- block: B023 -->

# 3. Strength-assessment-driven multimodal sequential observation

<!-- block: B024 -->

## 3.1. Framework overview: selecting regions and organizing their arrival

<!-- block: B025 -->

The proposed method couples a regional acquisition policy with a strength predictor to produce an ordered observation set and a CAI prediction trajectory. Before internal acquisition, surface descriptors supply specimen-specific context and support an initial strength estimate. The policy scores feasible unmeasured regions, requests one region and receives its internal content. The predictor updates the CAI estimate, while the measurement mask, acquisition history and acquired-area fraction update the state for the next decision. Figure 1 organizes this loop into surface representation, regional policy, observation/state update and strength assessment.

<!-- block: B026 -->

Specimen-level strength supervises this sequence through its assessment cost. Each region gains training value from its effect on the estimates produced after acquisition, allowing regional preferences to be learned from specimen-level labels. The policy outputs a regional index, and the predictor outputs strength in MPa. Both models have fixed weights during evaluation; adaptation occurs through newly available observations and updated regional scores. Region selection and order emerge from the same decisions: requesting a region determines both its membership in the observed set and the prefixes in which it will be available.

<!-- block: B027 -->

![Figure 1](figures/F1.png){width=6.5in}

<!-- block: B028 -->

**Figure 1.** Assessment-driven sequential observation. With fixed model weights, a regional request reveals internal content, updates the observable state and produces a new CAI estimate. During learning, measured specimen strength evaluates process-average and terminal errors. This task cost updates the policy and value head while the image encoder and strength readouts remain frozen. Execution uses the observable state; strength labels are used for training and retrospective evaluation.

<!-- block: B029 -->

## 3.2. Multimodal observable-state representation

<!-- block: B030 -->

A fixed regional representation connects requests to the same specimen locations throughout acquisition. Each specimen has a registered surface image and a paired two-dimensional C-scan raster. Orientation and outer cropping establish correspondence in normalized specimen coordinates. Both modalities are partitioned into an $8 \times 8$ grid with row-major indices $\mathcal{I} = \{ 0,\ldots,63\}$. These indices retain their spatial meaning throughout the episode, including when the native raster sizes differ. Each acquisition reveals one complete internal cell.

<!-- block: B031 -->

A frozen ResNet18 encoder maps each independently cropped region to a 512-dimensional descriptor \[7\]. Regional cropping precedes encoding, preserving the local support of each requested descriptor. Surface preprocessing fixes orientation and limits the longest edge to 1024 pixels. All 64 surface descriptors $S = \left( s_{i} \right)_{i \in \mathcal{I}}$ are available initially; internal descriptors $x_{i}$ become available through regional requests. The replay environment stores the descriptor bank, and a measurement mask controls which internal descriptors enter the models.

<!-- block: B032 -->

Native pixel area defines the acquisition scale. For an internal crop of $H_{img} \times W_{img}$ pixels, region $i$ has support $\Omega_{i}$ and cost $w_{i} = \left| \Omega_{i} \right|/\left( H_{img}W_{img} \right)$, with $\sum_{i}^{}w_{i} = 1$. Let $\mathcal{M}_{t}$ be the observed set after $t$ acquisitions, $M_{t,i}$ its binary indicator and $c_{t} = \sum_{i \in \mathcal{M}_{t}}^{}w_{i}$ the acquired fraction. Under a prescribed cap $B$, feasible actions are

<!-- block: B033 -->

$$\mathcal{L}_{t} = \{ i \in \mathcal{I}:M_{t,i} = 0,\ c_{t} + w_{i} \leq B\}.\quad\quad(1)$$

<!-- block: B034 -->

Integer raster boundaries make cell areas slightly unequal, so the cap $B = 0.25$ admits 15 or 16 whole cells. Selection uses the known cell geometry and the remaining allowance; acquisition ends when no whole unmeasured cell fits. The history value $H_{t,i}$ is zero for an unmeasured region and otherwise equals its acquisition ordinal divided by 64. The observable policy state is

<!-- block: B035 -->

$$z_{t} = \left( S,X_{obs,t},M_{t},H_{t},{\widehat{y}}_{t},c_{t},B - c_{t} \right).\quad\quad(2)$$

<!-- block: B036 -->

The current estimate ${\widehat{y}}_{t}$ is computed from the available inputs. For regional scoring, a learned unknown descriptor $u$ represents unavailable internal content through ${\widetilde{x}}_{t,i} = M_{t,i}x_{i} + \left( 1 - M_{t,i} \right)u$, alongside the explicit measurement indicator. Surface descriptors establish initial context, acquired internal descriptors add evidence, position identifies requests, and history and progress describe the sequence. Together with the current strength estimate, these inputs allow priorities to depend on the specimen and its observation state (Fig. 2).

<!-- block: B037 -->

![Figure 2](figures/F2.png){width=6.5in}

<!-- block: B038 -->

**Figure 2.** Observable-state representation and regional scoring. The actor combines known surface features, measured internal content or a learned unknown descriptor, mask, position and ordinal. Sixty-four regional tokens and one global token are contextualized by a two-layer, four-head Transformer. The 64 scores are restricted by a mask of unmeasured cells affordable within the remaining native area. The independent set-based predictor supplies the current strength estimate. Area and measured-cell fractions remain distinct; the two constant-zero channels per input block are detailed in S2. Diagrammatic connections specify information flow, not new data.

<!-- block: B039 -->

## 3.3. Strength estimation throughout the observation process

<!-- block: B040 -->

A common predictor converts each observed prefix into a CAI estimate, including the initial surface-only state. Its input is ${\widehat{y}}_{t} = f_{\phi}\left( S,X_{obs,t},M_{t},c_{t} \right)$. Separate affine projections $P_{s}\left( s_{i} \right) = W_{s}s_{i} + b_{s}$ and $P_{x}\left( M_{t,i}x_{i} \right) = W_{x}\left( M_{t,i}x_{i} \right) + b_{x}$ map the two modalities to width 64. Internal descriptors are masked before projection, retaining the affine bias. With normalized coordinates $p_{i} = \left( r_{i}/7,k_{i}/7 \right)$, a two-layer GELU multilayer perceptron produces a fused regional representation:

<!-- block: B041 -->

$$h_{t,i} = \psi_{\phi}\left( \left\lbrack P_{s}\left( s_{i} \right),P_{x}\left( M_{t,i}x_{i} \right),M_{t,i},p_{i} \right\rbrack \right).\quad\quad(3)$$

<!-- block: B042 -->

Two complementary summaries support regression. The all-cell mean preserves surface context and the represented observation state over the grid. The observed-cell mean focuses on acquired regions and is zero before internal acquisition. A regression head combines these summaries with the acquired fraction and converts its normalized output to MPa using training-only target statistics $\mu_{y}$ and $\sigma_{y}$:

<!-- block: B043 -->

$${\overline{h}}_{t} = \frac{1}{64}\sum_{i}^{}h_{t,i},\quad\quad{\overline{h}}_{t}^{obs} = \frac{\sum_{i}^{}M_{t,i}h_{t,i}}{\max\left( 1,\sum_{i}^{}M_{t,i} \right)}.\quad\quad(4)$$

<!-- block: B044 -->

$${\widehat{y}}_{t} = \mu_{y} + \sigma_{y}\rho_{\phi}\left( \left\lbrack {\overline{h}}_{t},{\overline{h}}_{t}^{obs},c_{t} \right\rbrack \right).\quad\quad(5)$$

<!-- block: B045 -->

Predictor training covers the information levels encountered during acquisition. The mask distribution allocates 10% to empty internal inputs, 60% to 1--16 cells from random or fixed geometric orders, 20% to 17--48 randomly selected cells and 10% to complete internal input. A Huber loss on the standardized residual trains the regressor. Three capture-group cross-fitted predictors provide training-side trajectory costs, with each training specimen evaluated by the predictor that excludes its fold. Evaluation uses one common frozen predictor fitted on the training partition.

<!-- block: B046 -->

Both means in Eq. (4) summarize a set: the all-cell mean includes the masked-input affine bias, and the observed-cell mean uses only measured cells. Acquisition history enters the policy, not this predictor. Identical final sets and acquired fractions therefore produce identical terminal inputs. Endpoint equality in the fixed-set intervention is a structural invariant; order can affect accuracy only at intermediate prefixes.

<!-- block: B047 -->

## 3.4. State-conditioned regional scoring

<!-- block: B048 -->

The regional policy compares candidates in the context of evidence already obtained elsewhere. It constructs 64 regional tokens and one global token, then contextualizes them jointly with a spatial Transformer \[8\]. Each regional token combines a surface projection, the visible or unknown internal projection, normalized coordinates, measurement status and acquisition history. Fixed position indices maintain each candidate's relation to the observed grid. The global token supplies task context through the current strength estimate and acquisition progress.

<!-- block: B049 -->

Surface and internal projections each have width 48. Regional inputs are mapped to 128-dimensional tokens, and global inputs are mapped separately to the same width. The global values comprise acquired-area fraction, remaining allowance, standardized strength prediction and measured-cell fraction. Retaining both area and cell fractions represents the slightly unequal native cell costs. Two encoder layers, four attention heads and a feed-forward width of 256 contextualize the 65 tokens. The actor uses the learned unknown descriptor for missing internal content; the predictor uses the pre-projection zero mask described in Section 3.3.

<!-- block: B050 -->

A shared scoring head receives the concatenated contextual regional and global tokens and returns $s_{\theta}\left( z_{t},i \right)$. A separate value head reads the global token during training. At evaluation, the next requested region is the highest-scoring feasible candidate:

<!-- block: B051 -->

$$a_{t} = {argmax}_{i \in \mathcal{L}_{t}}s_{\theta}\left( z_{t},i \right).\quad\quad(6)$$

<!-- block: B052 -->

The feasibility mask excludes measured cells and cells that exceed the remaining allowance. Training samples actions from a categorical softmax over feasible scores, allowing regional preferences to be explored. These probabilities describe selection preferences, and the value head estimates remaining policy cost. New observations, acquisition history and the current estimate can revise candidate priorities; the recorded-action analysis in Section 5.4 characterizes the revisions realized during execution.

<!-- block: B053 -->

## 3.5. Learning assessment relevance and information timing

<!-- block: B054 -->

### 3.5.1. Task supervision and trajectory objective

<!-- block: B055 -->

A training episode samples a specimen, acquires feasible regions and updates strength estimates through its cross-fitted predictor. The measured training strength $y$ evaluates the resulting trajectory through $e_{t} = \left| {\widehat{y}}_{t} - y \right|$. Predictions update when a region is fully observed and remain constant between observation completions. If acquisition finishes at $c_{T} \leq B$, the last estimate is carried through the remaining interval to $B$. The acquisition-process average absolute error is

<!-- block: B056 -->

$$A(B) = \frac{1}{B}\int_{0}^{B}e(c)\, dc = \frac{\sum_{t = 0}^{T - 1}\left( c_{t + 1} - c_{t} \right)e_{t} + \left( B - c_{T} \right)e_{T}}{B}.\quad\quad(7)$$

<!-- block: B057 -->

The learning objective combines this process criterion with terminal accuracy:

<!-- block: B058 -->

$$J = A(B) + \lambda_{T}e_{T},\quad\quad\lambda_{T} = 0.25.\quad\quad(8)$$

<!-- block: B059 -->

Both terms are expressed in MPa before training-scale normalization. The carry interval gives every trajectory the same integration range, including when the remaining allowance accommodates no whole cell. The terminal term preserves an incentive for final accuracy, including the effect of regions completed near the cap. This objective learns assessment relevance directly from the ability of the acquired observations to support prediction of measured specimen strength.

<!-- block: B060 -->

### 3.5.2. Process value of earlier useful observations

<!-- block: B061 -->

The trajectory objective makes the value of information arrival explicit. Let $\Delta e_{t} = e_{t - 1} - e_{t}$ denote the signed error decrease after the observation completed at $c_{t}$. Expanding the interval sum gives

<!-- block: B062 -->

$$A(B) = e_{0} - \sum_{t = 1}^{T}\left( 1 - \frac{c_{t}}{B} \right)\Delta e_{t},\quad\quad(9)$$

<!-- block: B063 -->

$$J = \left( 1 + \lambda_{T} \right)e_{0} - \sum_{t = 1}^{T}\left( 1 - \frac{c_{t}}{B} + \lambda_{T} \right)\Delta e_{t}.\quad\quad(10)$$

<!-- block: B064 -->

For a fixed specimen and predictor, the initial error is constant. An error decrease achieved at a smaller acquired fraction receives a larger process weight because it remains available over a longer part of the acquisition interval. Signed increases in error enter with the opposite effect. At $c_{t} = B$, the process weight is zero and the terminal weight remains $\lambda_{T}$. Thus, the objective favors useful information early while continuing to optimize endpoint quality.

<!-- block: B065 -->

Supplementary S2 derives Eqs. (9–10) from Eq. (7), including the final carry interval. Their signed event contributions accommodate unequal native cell areas and prediction changes in either direction. Section 5.3 tests realized changes using archived events and fixed-set replay.

<!-- block: B066 -->

### 3.5.3. Updating regional preferences from task cost

<!-- block: B067 -->

For a sampled decision at state $t$, the normalized remaining cost includes subsequent interval errors and terminal error:

<!-- block: B068 -->

$$G_{t} = \frac{1}{\sigma_{y}}\left\lbrack \sum_{k = t}^{T - 1}\frac{c_{k + 1} - c_{k}}{B}e_{k} + \left( \frac{B - c_{T}}{B} + \lambda_{T} \right)e_{T} \right\rbrack.\quad\quad(11)$$

<!-- block: B069 -->

With value estimate $v_{\theta}\left( z_{t} \right)$, the actor minimizes the score-function surrogate

<!-- block: B070 -->

$$\mathcal{L}_{actor} = \mathbb{E}\left\lbrack \sum_{t = 0}^{T - 1}\log\pi_{\theta}\left( a_{t} \mid z_{t} \right)\, stopgrad\left( G_{t} - v_{\theta}\left( z_{t} \right) \right) \right\rbrack.\quad\quad(12)$$

<!-- block: B071 -->

The positive sign implements cost minimization: a sampled action whose cost exceeds the value baseline receives a probability-reducing update. Actor terms are summed over decisions within an episode and averaged across the batch. Value squared errors and categorical entropy are averaged within episodes before batch aggregation. The total loss is $\mathcal{L}_{actor} + 0.5\mathcal{L}_{value} - \beta\mathcal{H}$, with $\beta$ decreasing linearly from 0.01 to zero. The encoder and strength predictors remain frozen, so policy updates learn which observations to supply to a fixed assessment model.

<!-- block: B072 -->

## 3.6. Sequential execution and prediction updates

<!-- block: B073 -->

Execution begins with an empty internal mask, zero history and zero acquired fraction. The surface-only estimate provides the initial task state. At each iteration, the policy constructs the feasible set, scores candidates, requests the highest-scoring region and receives its descriptor. The predictor then updates CAI, and the new observation, acquisition ordinal, completion fraction and prediction are recorded. Acquisition ends when the feasible set is empty. Since each cell is requested at most once, an episode has at most 64 actions.

<!-- block: B074 -->

**Algorithm 1. Learning and sequential execution**

<!-- block: B075 -->

**1.** Fit the common predictor and three capture-group cross-fitted training predictors on the designated training data; freeze the image encoder and readouts.

<!-- block: B076 -->

**2.** Train the regional policy on sampled feasible trajectories using $G_{t}$, the value baseline and entropy regularization. Select its checkpoint by development-validation $A$.

<!-- block: B077 -->

**3.** For execution, fix all weights. Receive $S$, cell costs and $B$; initialize $M_{0} = H_{0} = 0$, $c_{0} = 0$ and ${\widehat{y}}_{0} = f_{\phi}(S,0,0,0)$.

<!-- block: B078 -->

**4.** While $\mathcal{L}_{t} \neq \varnothing$, select $a_{t}$ by Eq. (6), obtain $x_{a_{t}}$ and update $M_{t + 1,a_{t}} = 1$, $H_{t + 1,a_{t}} = (t + 1)/64$, $c_{t + 1} = c_{t} + w_{a_{t}}$.

<!-- block: B079 -->

**5.** Update ${\widehat{y}}_{t + 1}$ and $z_{t + 1}$; record the requested region, completion fraction and prediction. Return the ordered observed set and complete prediction trajectory.

<!-- block: B080 -->

Policy learning uses subsequent task cost to update sampled choices through the stopped-gradient advantage in Eq. (12); the value baseline reduces variation in this feedback. Evaluation uses deterministic feasible-score selection with fixed parameters, giving a reproducible sequence for a fixed specimen, checkpoint, preprocessing and cap. Measured strength enters training or retrospective scoring, while execution reads only the observable state. In the present implementation, each request reveals a stored two-dimensional C-scan regional descriptor.

<!-- block: B081 -->

# 4. Experimental setup

<!-- block: B082 -->

## 4.1. Paired images, mechanical labels and evaluation units

<!-- block: B083 -->

The study uses publicly released surface and internal-damage images of impacted carbon-fiber-reinforced polymer laminates, paired with measured CAI strength \[1, 2, 9\]. The assembled cohort contains 276 physical specimens from six source groups covering cross-ply and quasi-isotropic layups with 8, 16 and 24 layers. Each record connects a specimen identifier, a registered image pair and a mechanical label in MPa. Regional acquisition is evaluated through offline revelation of the two-dimensional C-scan raster; the strength labels are the source authors' mechanical measurements.

<!-- block: B084 -->

Capture groups preserve dependencies associated with shared source images, paths or crop identities. Groups are formed before partitioning, and all members remain in one partition. Deterministic ordering within each source group assigns approximately 60%, 20% and 20% of capture groups to training, development validation and a reserved partition. This produces 161, 50 and 65 physical specimens, respectively, comprising 152, 48 and 59 capture groups. Table 1 reports the specimen counts for each configuration.

<!-- block: B085 -->

**Table 1.** Data configurations and physical-specimen split. Capture groups remain within partitions; all reported method comparisons use the same 50 development-validation specimens.

<!-- block: B086 -->

| Laminate | Train | Dev. | Reserved | Total |
| --- | --- | --- | --- | --- |
| 8-layer cross-ply | 25 | 9 | 11 | 45 |
| 24-layer quasi-isotropic | 29 | 9 | 11 | 49 |
| 16-layer quasi-isotropic | 25 | 8 | 10 | 43 |
| 24-layer cross-ply | 35 | 9 | 15 | 59 |
| 16-layer cross-ply | 25 | 8 | 9 | 42 |
| 8-layer quasi-isotropic | 22 | 7 | 9 | 38 |
| Total | 161 | 50 | 65 | 276 |

<!-- block: B087 -->

All reported effects use the 50 development-validation specimens, which also support checkpoint selection. The estimates therefore characterize the selected development configuration. The 65 reserved specimens are outside the reported evaluation. The specimen cohort and six diagnostic cases are retained in full. Random-order repetitions, image cells and prediction prefixes remain repeated observations of these specimens. Supplementary S1 records specimen identities, capture groups and partition membership.

<!-- block: B088 -->

## 4.2. Acquisition strategies and shared prediction conditions

<!-- block: B089 -->

Five controls provide distinct ways of allocating the same allowed internal-image area (Table 2). Center-first sorts regions by distance to the grid center. Geometry-spread starts from a corner and repeatedly maximizes the minimum squared distance to previously selected cells, using the region index to break ties. Serpentine alternates horizontal direction between neighboring rows. Random uses five stored seeded permutations of all 64 cells. Learned-static optimizes 64 shared positional logits, giving all specimens the same initial order. The proposed policy uses the multimodal observable state for specimen-conditioned scoring.

<!-- block: B090 -->

**Table 2.** Selection information and execution rules. All sequential strategies share the regional encoder, strength predictor, native-area costs and cap.

<!-- block: B091 -->

| Strategy | Selection input | Rule |
| --- | --- | --- |
| Center-first | Grid position | Center-distance order |
| Geometry-spread | Position and observed locations | Farthest-point expansion |
| Serpentine | Grid position | Alternating row order |
| Random | Saved seeded permutations | Five repeats over 64 cells |
| Learned-static | 64 shared learned logits | Shared position order |
| Proposed | Surface, observed internal content, mask, history, prediction and progress | State-conditioned feasible argmax |
| Full-input reference | All 64 internal cells | Common predictor at fraction 1.0 |

<!-- block: B092 -->

All sequential methods share the frozen regional encoder, common strength predictor, grid geometry, native-area costs and total cap $B = 0.25$. Their strength estimates use the same complete surface information and the internal regions actually acquired. This fixes the readout while varying observation selection. A separate full-input reference supplies all 64 internal cells to the same predictor at area fraction 1.0. It provides an endpoint information-scale reference. Whole-cell acquisition retains each strategy's actual acquired fraction under the common cap.

<!-- block: B093 -->

## 4.3. Accuracy, efficiency and mechanism evaluation

<!-- block: B094 -->

Accuracy is reported at caps $\mathcal{B}_{5} = \{ 0,0.0625,0.125,0.1875,0.25\}$. Each cap reads the last completed prediction at or below that value from the path planned with total cap 0.25. Thus, smaller-cap results query prefixes of the same execution, with the policy's original total-cap input retained. Endpoint MAE and RMSE pool physical specimens, and $R^{2}$ uses the same measured-target variation. For Random, absolute and squared losses are averaged within specimen before pooling; repeat-specific $R^{2}$ values are averaged.

<!-- block: B095 -->

Process error $A$ and early process error $A(0.0625)$ use a six-source-group-equal estimate: repeats are averaged within specimen, specimens within source group, and the six group means equally. Event-stage quantities use the same weighting. This differs from specimen-pooled endpoint MAE because group sizes differ. Prediction curves retain all saved completion breakpoints and hold the current estimate between events, including increases as well as decreases in error. Actual acquired fractions and repeat-level losses are reported in Supplementary S3.

<!-- block: B096 -->

Matched-quality efficiency is the earliest supported common cap at which cohort MAE reaches a target $q$. We evaluate the five-cap grid and the union of recorded event breakpoints separately, denoting either by $\mathcal{B}$. The first-attainment cap and relative cap reduction are

<!-- block: B097 -->

$$b^{\text{*}}(q) = \min\{ b \in \mathcal{B}:MAE(b) \leq q\},\quad\quad R(q) = 1 - \frac{b_{proposed}^{\text{*}}(q)}{b_{control}^{\text{*}}(q)}.\quad\quad(13)$$

<!-- block: B098 -->

The target set contains the inherited integer range 41--61 MPa and the full-input MAE anchor. Relative reduction is defined when both methods reach the target and the control cap is positive. We record first attainment and later recrossings separately. These targets describe cohort prediction quality at a common acquisition scale; they are analysis criteria rather than certified engineering acceptance thresholds or specimen-level stopping decisions.

<!-- block: B099 -->

Event analysis uses 500 archived trajectories: 250 Random episodes and 50 from each of the other five strategies, yielding 7,849 acquisition events. Each event contributes a signed raw decrease $\Delta e_{t}$ and a process-weighted contribution $\left( 1 - c_{t}/B \right)\Delta e_{t}$. Completion fractions assign events to four right-closed stages: (0, 0.0625\], (0.0625, 0.125\], (0.125, 0.1875\] and (0.1875, 0.25\]. Events are summed within stage, then aggregated using the repeat, specimen and source-group scheme above. Paired stage contrasts compare the proposed policy with each control.

<!-- block: B100 -->

The frozen fixed-final-set intervention retains the 15 or 16 cells selected by the proposed policy for each of the same 50 specimens. Original order, complete reversal and five preset hash-determined permutations provide 350 replay conditions. The archived intervention evaluated each intermediate mask with the common frozen predictor. Cumulative area uses integer native pixel counts divided by total crop pixels. The primary contrast is the within-specimen five-permutation mean minus original order; reversal minus original order is the prespecified secondary contrast. These repeated conditions do not increase the independent specimen count. Supplementary S4 provides the saved interventions, direction definitions and endpoint checks.

<!-- block: B101 -->

Paired 95% percentile intervals use 5,000 capture-group bootstrap draws sampled within source groups. Each sampled capture group carries all its specimens with the same multiplicity. MAE retains specimen pooling, while process and stage metrics retain equal source-group weighting. The five fixed-set permutations are averaged within specimen before resampling. These intervals are conditional on the selected checkpoints and are unadjusted for checkpoint selection or multiple comparisons. The independent sampling unit is the capture group, with repeated trajectories retaining their specimen identities.

<!-- block: B102 -->

## 4.4. Training and checkpoint selection

<!-- block: B103 -->

The common mean-aggregation predictor completes 2,000 updates, with update 1,750 selected for evaluation. Three cross-fitted predictors supply the policy's training costs. The actor contains 378,978 parameters, including its training value head. It is optimized with AdamW, learning rate $3 \times 10^{- 4}$, weight decay $10^{- 4}$, batch size 16 and gradient-norm limit 1. The stored architecture retains two zero-valued placeholder channels in each of its two input blocks. Parameterization and model identities are documented in Supplementary S2.

<!-- block: B104 -->

The main policy initialization uses seed 2026091302 and completes 1,250 updates. Candidate checkpoints are compared every 250 updates using development-validation $A$. An earlier candidate is retained unless improvement exceeds $10^{- 12}$, selecting update 1,000. All reported policy results use this single selected initialization. Event and order analyses reuse the frozen predictor and policy choices, with the corresponding candidate scores and settings recorded in Supplementary S2.

<!-- block: B105 -->

# 5. Results and discussion

<!-- block: B106 -->

## 5.1. Task-driven region selection improves prediction at matched acquisition scales

<!-- block: B107 -->

Task-driven observation selection improved CAI prediction under the common readout. At cap 0.25, the proposed method obtained the lowest point estimates of process error, early process error, endpoint MAE and RMSE, and the highest endpoint $R^{2}$, among the six sequential strategies (Table 3). Endpoint MAE decreased from 46.910 MPa for geometry-spread to 42.385 MPa, a reduction of 4.525 MPa or 9.65%. The paired 95% interval for this reduction was \[0.670, 8.276\] MPa. Against learned-static, endpoint MAE improved by 4.608 MPa \[0.656, 8.628\]. These comparisons demonstrate the value of selecting observations for the strength-assessment task within the evaluated configuration.

<!-- block: B108 -->

**Table 3.** Prediction performance. Errors are in MPa. Process $A$ and early $A$ use equal weighting of the six source groups; endpoint metrics pool specimens. Bold values identify the best sequential point estimates. The full-input reference uses area fraction 1.0 and has no acquisition-path metric.

<!-- block: B109 -->

| Strategy | A | Early A | MAE | RMSE | R² |
| --- | --- | --- | --- | --- | --- |
| Center-first | 48.475 | 51.821 | 48.027 | 65.223 | 0.577 |
| Geometry-spread | 47.311 | 51.880 | 46.910 | 62.278 | 0.614 |
| Serpentine | 48.911 | 52.170 | 48.013 | 63.041 | 0.605 |
| Random | 49.594 | 54.117 | 46.886 | 61.941 | 0.618 |
| Learned-static | 47.977 | 52.946 | 46.993 | 62.168 | 0.616 |
| **Proposed** | **43.597** | **48.771** | **42.385** | **56.534** | **0.682** |
| Full-input reference (area 1.0) | — | — | 41.690 | 53.944 | 0.711 |

<!-- block: B111 -->

Useful estimates emerged at the first nonzero reporting cap. All strategies started from the same surface-only cohort MAE of 58.551 MPa. At cap 0.0625, the proposed method reached 43.324 MPa with a mean actual acquired fraction of 0.057700, already below all five controls' endpoint MAEs at cap 0.25 (46.886--48.027 MPa). Its MAEs at caps 0.125, 0.1875 and 0.25 were 43.164, 42.928 and 42.385 MPa. This establishes an early prediction advantage along the common-cap evaluation path.

<!-- block: B110 -->

The trajectory-level point estimates also favored the proposed policy. Process error $A$ was 43.597 MPa versus 47.311 MPa for geometry-spread, a reduction of 3.714 MPa or 7.85%. Its paired 95% interval, [−0.337, 7.283] MPa, crosses zero, so this process contrast remains uncertain despite the favorable mean. Relative to learned-static, the reduction was 4.380 MPa [0.199, 8.405]. Table 3 distinguishes this six-source-group-equal process metric from specimen-pooled endpoint MAE.

<!-- block: B112 -->

Figure 3 shows how this advantage develops across the recorded observation events. Individual updates can raise as well as lower error: geometry-spread reached 46.189 MPa at cap 0.125 and ended at 46.910 MPa, while learned-static rose from 46.465 MPa at 0.0625 to 48.064 MPa at 0.125 before improving. The proposed policy improved at its four nonzero reporting points, with local reversals also visible on the finer event curve. Process and endpoint metrics capture these complementary aspects of assessment. The proposed mean final acquired fraction was 0.247174, compared with approximately 0.250 for geometry-spread.

<!-- block: B113 -->

![Figure 3](figures/F3.png){width=6.5in}

<!-- block: B114 -->

**Figure 3.** Prediction quality throughout acquisition on 50 development-validation specimens. (a) Six-method full event curves. (b) A local view of proposed recrossings, with the 42 MPa target dashed. Step curves preserve all recorded observation-completion breakpoints; dots mark the five reporting caps. Cohort MAE pools physical specimens, with Random losses averaged within specimen over five saved orders. All methods share the predictor and surface inputs. The horizontal reference is full-input MAE at area fraction 1.0. Process $A$ in Table 3 uses equal source-group weighting.

<!-- block: B115 -->

## 5.2. Selected observations reach matched prediction quality earlier

<!-- block: B116 -->

The accuracy trajectories translate into lower acquisition requirements at defined quality targets. For a cohort MAE target of 47 MPa, the first supported five-grid cap was 0.0625 for the proposed method, 0.125 for geometry-spread and 0.25 for Random (Table 4). The required cap was therefore 50% smaller than geometry-spread and 75% smaller than Random. Learned-static also reached this target at 0.0625 on the five-cap grid. At the more demanding target of 44 MPa, the proposed method reached the first nonzero reporting cap, while all five controls remained above the target over the reported range.

<!-- block: B117 -->

**Table 4.** First attainment of cohort-MAE targets. Five-cap and event grids are reported separately. NR indicates that the target was not reached within cap 0.25. Entries describe first crossings under a common cap.

<!-- block: B118 -->

| Strategy | 47 MPa, five caps | 44 MPa, five caps | 47 MPa, events |
| --- | --- | --- | --- |
| Proposed | 6.25% | 6.25% | 3.1203% |
| Center-first | NR | NR | NR |
| Geometry-spread | 12.5% | NR | 9.3426% |
| Serpentine | NR | NR | NR |
| Random | 25% | NR | 24.9630% |
| Learned-static | 6.25% | NR | 6.2407% |

<!-- block: B119 -->

The event grid resolves information arrival more finely. At the same 47 MPa target, the first cap was 0.031203 for the proposed method, 0.093426 for geometry-spread, 0.062407 for learned-static and 0.249630 for Random. The relative reduction against geometry-spread was approximately 66.60% at this resolution. Each value applies one common cap to the cohort before calculating MAE. The coarse and event grids therefore provide different resolutions of the same prediction trajectories.

<!-- block: B120 -->

The full target range reveals where this advantage matters (Fig. 4). At 55 MPa, all six strategies first met the five-grid criterion at 0.0625. Tighter targets differentiated their information requirements. The proposed event curve first reached 43 MPa at 0.062591, compared with 0.1875 on the five-cap grid. It also crossed 42 MPa at 0.078844 before rising again and ending at 42.385 MPa; the five-cap grid recorded no 42 MPa crossing. These observations distinguish first useful attainment from sustained attainment and retain the target dependence of information efficiency. The full-input anchor of 41.690 MPa was not reached within cap 0.25.

<!-- block: B121 -->

![Figure 4](figures/F4.png){width=6.5in}

<!-- block: B122 -->

**Figure 4.** Acquisition scale required for matched cohort prediction quality. (a) First supported cap on the five-point grid. (b) First supported cap on the event grid. Both panels retain all preset integer targets from 41 to 61 MPa. The NR band explicitly marks nonattainment; initially satisfied targets are shown at zero cost. Lines guide comparisons between evaluated targets. The full-input anchor is reported separately in the text.

<!-- block: B123 -->

## 5.3. Earlier useful estimates arise from early error reduction and information arrival

<!-- block: B124 -->

### 5.3.1. Early observations produce larger realized error reductions

<!-- block: B125 -->

Early acquisition events produced larger realized improvements under the proposed policy. For observations completed in (0, 0.0625\], the unweighted net error decrease exceeded geometry-spread by 6.865 MPa, with a paired 95% interval of \[0.362, 12.502\] MPa, and exceeded Random by 9.707 MPa \[4.862, 14.131\] (Fig. 5a). These stage sums retain both favorable and adverse events. They show that useful early observations generated larger actual reductions, providing empirical support for the early-process value encoded in the learning objective.

<!-- block: B126 -->

Process weighting connects these realized changes to the integrated assessment gain. The first-stage weighted contribution improved by 5.926 MPa \[−0.117, 11.095\] against geometry-spread and 8.617 MPa \[4.194, 12.678\] against Random. Against geometry-spread, the four stage gains were +5.926, −2.475, +0.199 and +0.063 MPa (Fig. 5b); their full-precision sum recovered the 3.714 MPa reduction in process $A$. Geometry-spread recovered part of the initial gap in the second stage, while the proposed sequence retained a lower average process error.

<!-- block: B127 -->

For the proposed method, the four weighted contributions were 13.069, 0.196, 0.075 and 0.111 MPa. Most of the integrated gain relative to its source-group-equal initial error was established in the first stage. The raw event changes establish the realized reduction, while the weights account for how long each change remains useful. Together, they explain how early information contributes to the observed trajectory advantage.

<!-- block: B128 -->

### 5.3.2. Arrival order changes process quality with final content fixed

<!-- block: B129 -->

Holding the final selected content fixed exposed the effect of its arrival order on intermediate assessment. Original order gave process $A = 43.597$ MPa and early $A = 48.771$ MPa. The five-permutation mean was 45.083 and 50.399 MPa, and complete reversal gave 47.526 and 52.099 MPa, respectively. All three conditions had the same endpoint MAE of 42.385 MPa. Their intermediate trajectories differed and converged at the identical terminal input (Fig. 5c).

<!-- block: B130 -->

The primary permutation-mean-minus-original process contrast was +1.486 MPa, with a paired 95% interval of [−0.658, 3.384] MPa (Fig. 5d). Thus, the lower mean under original order does not establish a resolved average advantage against the preset permutations. The prespecified secondary reversal contrast was +3.929 MPa [0.145, 7.405], supporting an order effect for that intervention. Early-process contrasts were +1.628 MPa [−2.190, 5.180] for permutation and +3.328 MPa [−1.527, 7.979] for reversal; both intervals cross zero. These contrasts describe arrival within the selected sets and do not partition the total policy gain into selection and order percentages.

<!-- block: B131 -->

Original-order replay reproduced the archived predictions with a maximum absolute deviation of $6.104 \times 10^{- 5}$ MPa, and all 350 terminal-set checks preserved the terminal input and prediction. The observed process differences thus came from predictions recomputed on the corresponding intermediate prefixes. This intervention shows how the same final regional content can support different assessment trajectories through its arrival sequence.

<!-- block: B132 -->

![Figure 5](figures/F5.png){width=6.5in}

<!-- block: B133 -->

**Figure 5.** Evidence linking information arrival to earlier useful assessment. (a) Proposed-minus-control gains in unweighted net error decrease for completions in (0, 0.0625\]. (b) Proposed-minus-geometry-spread weighted gains in stages 1--4. (c) Recomputed cohort-MAE trajectories for original, permutation-mean and reversed arrivals of each identical final selected set. (d) Primary permutation-mean-minus-original and secondary reverse-minus-original contrasts in process $A$. Positive values in (a,b) favor the proposed method; positive values in (d) favor original order. Error bars are paired 95% capture-group percentile intervals from 5,000 draws. Panels (a,b,d) use equal source-group weighting; (c) uses specimen-pooled MAE.

<!-- block: B135 -->

Together, the event and fixed-set results connect earlier useful estimates to realized early reductions and the timing of selected content, with the primary permutation contrast remaining uncertain.

<!-- block: B136 -->

## 5.4. Specimen-specific initialization and selective later reordering

<!-- block: B137 -->

The fixed diagnostic cases show how specimen information enters regional decisions. The first selected cells were 62, 31, 14, 27, 63 and 63 for c8-16, q24-48, q16-29, c24-9t, c16-24 and q8-22, respectively. These choices differed before any internal region was visible, demonstrating specimen-specific initialization from the initially available state. Figure 6 follows c8-16 and q24-48 through the registered surface image, internal-image reference with acquired-cell order, and strength updates. All six fixed cases are documented in Supplementary S5.

<!-- block: B138 -->

The subsequent decisions showed selective reordering of the initial preferences. Each recorded action was compared with the highest initially scored cell still feasible on the actual path, accounting for removal of observed and unaffordable cells. Nine of 94 decisions differed from this reference: two for c8-16, two for q16-29, four for c16-24 and one for q8-22. The q24-48 and c24-9t cases retained their initial feasible ranking throughout. The observed behavior therefore combined specimen-specific starting priorities with selective adjustment during acquisition. These nine departures concern only the six fixed cases. Because the changing state jointly includes measured content, history, progress and prediction, they cannot be attributed causally to internal-image feedback alone.

<!-- block: B139 -->

The prediction traces expose both the benefit of acquisition and the residual error of the strength model. For c8-16, measured CAI was 264.163 MPa; the estimate changed from 301.366 to 232.095 MPa, reducing absolute error from 37.203 to 32.068 MPa through updates in both directions. For q24-48, the estimate moved from 302.099 to 258.932 MPa against measured strength of 136.680 MPa. Acquisition reduced its initial error, but the final residual remained 122.252 MPa. These fixed cases make the complete request--observation--prediction process visible, including an instance where improved information supply left substantial prediction error.

<!-- block: B140 -->

![Figure 6](figures/F6.png){width=6.5in}

<!-- block: B141 -->

**Figure 6.** Real-image evidence and CAI trajectories for c8-16 (a–c) and q24-48 (d–f). Each row shows the registered surface view, retrospective full-image C-scan reference with observed cells and acquisition ordinals overlaid, and saved prediction updates. Unnumbered cells were unavailable to the policy; the displayed full C-scan was never an execution input. Ordinals follow the native row-major grid. The final prediction is carried to cap 0.25; the q24-48 final residual of 122.252 MPa is retained. Images: S. Hasebe, R. Higuchi, T. Yokozeki and S. Takeda, Mendeley Data v1 (2022), CC BY 4.0 [10,11]. Surface images follow the frozen clockwise-90° registration and display resize; no C-scan contrast or rotation is added.

<!-- block: B144 -->

Information-use diagnostics complement the executed trajectories. Action probabilities quantify selection preferences. Surface attribution measures local sensitivity of the selected action's log probability, including the route through the current strength prediction. Attention summarizes global-to-regional interactions among fused tokens. Supplementary S5 defines these quantities separately. The maps describe how the policy uses its represented inputs, while the strength traces measure the resulting assessment outcome; the diagnostic values are not regional mechanical labels or calibrated damage probabilities.

<!-- block: B145 -->

## 5.5. Groupwise performance and the partial-to-full information trade-off

<!-- block: B146 -->

The groupwise results locate the configurations contributing to the overall improvement (Table 5). Relative to geometry-spread, the 24-layer quasi-isotropic group improved process $A$ by 13.061 MPa and endpoint MAE by 16.440 MPa. The 16-layer cross-ply group improved these metrics by 8.157 and 5.875 MPa. The 8-layer cross-ply group also improved both metrics. Other configurations showed smaller or mixed changes: 24-layer cross-ply had a lower process error but 2.097 MPa higher endpoint MAE, and the 8-layer quasi-isotropic group was worse on both metrics. The 16-layer quasi-isotropic group had nearly unchanged process error and a modest endpoint improvement.

<!-- block: B147 -->

**Table 5.** Groupwise performance in MPa. Gains are geometry-spread minus the proposed method, so positive values favor the proposed method. These are source-group strata within the evaluated partition.

<!-- block: B148 -->

| Laminate | n | Proposed A | Proposed MAE | Gain in A | Gain in MAE |
| --- | --- | --- | --- | --- | --- |
| 8-layer cross-ply | 9 | 31.330 | 28.655 | +0.893 | +5.634 |
| 24-layer quasi-isotropic | 9 | 68.749 | 64.067 | +13.061 | +16.440 |
| 16-layer quasi-isotropic | 8 | 47.030 | 44.088 | −0.105 | +1.615 |
| 24-layer cross-ply | 9 | 48.163 | 46.433 | +0.760 | −2.097 |
| 16-layer cross-ply | 8 | 48.059 | 48.408 | +8.157 | +5.875 |
| 8-layer quasi-isotropic | 7 | 18.250 | 18.124 | −0.485 | −1.922 |

<!-- block: B149 -->

At specimen level, the proposed method improved process $A$ for 33 of 50 specimens and endpoint absolute error for 29 of 50 relative to geometry-spread. These results characterize a useful but heterogeneous selection policy. The source-group strata combine differences in imagery, labels and acquisition source with laminate configuration; they describe within-partition performance rather than leave-one-configuration-out transfer. Their variation identifies concrete settings in which observation selection helps and settings requiring further investigation.

<!-- block: B150 -->

The full-input reference places the result on an information-scale axis. The proposed method acquired a mean 24.7174% of internal pixels, with specimen fractions ranging from 0.234492 to 0.250000, and reached MAE 42.385 MPa. Full input gave 41.690 MPa, a partial-minus-full difference of 0.695 MPa with paired interval \[−4.009, 5.327\] MPa. RMSE was 56.534 versus 53.944 MPa, and $R^{2}$ was 0.682 versus 0.711. These results quantify the prediction--information trade-off under one readout: selected partial input approached the full-input mean error while retaining differences in endpoint metrics. This is a difference estimate; equivalence or noninferiority was not tested with a prespecified margin.

<!-- block: B151 -->

## 5.6. Implications for inspection-and-assessment workflows

<!-- block: B152 -->

The central engineering contribution is to make information provision an explicit, learnable part of strength assessment. Regional requests determine the observations available to a common predictor, and the trajectory objective connects these requests to both immediate and final assessment quality. Accuracy comparisons establish the value of observation organization; matched-quality analysis expresses that value as an information requirement; and event and fixed-set analyses explain how information arrival contributes to the process. Together, they describe an assessment workflow in which acquisition is guided by the mechanical quantity being estimated.

<!-- block: B153 -->

This formulation extends image-informed CAI regression \[3\] by optimizing the input sequence as well as evaluating its readout. It also connects the observation--update--decision structure of autonomous inspection \[4\] and active feature acquisition \[5, 6\] to a continuous strength trajectory. The common-predictor design attributes the evaluated difference to the complete selection mechanism. Surface conditioning, observed internal content, history and global task state operate together in that mechanism; their separate contributions would require matched component interventions. Similarly, the fixed-set experiment evaluates arrival within the selected content, rather than assigning additive percentages of the total gain to selection and order.

<!-- block: B154 -->

The present evidence concerns regional revelation of registered two-dimensional images. Native-area fraction measures the quantity of internal-image information supplied to assessment. Extending the request interface to physical inspection would require accounting for scanner movement, coupling, setup and acquisition latency, and validating an execution-time stopping criterion. The method supplies a concrete interface for that extension: regional indices specify requests, completed observations update the state, and the predictor supplies a strength estimate after each update.

<!-- block: B155 -->

Independent evaluation would establish how this capability transfers beyond the selected development configuration. The present cohort also selected checkpoints, and the policy results use one initialization; the groupwise reversals and residual cases further identify variability that broader evaluation should resolve. Independent specimens and repeated initializations can test the stability of the accuracy and information-efficiency gains, while physical regional acquisition can test their translation into inspection performance. The demonstrated result is a task-driven way to obtain useful CAI estimates earlier along regional image acquisition and to use the supplied internal information more effectively.

<!-- block: B156 -->

# 6. Conclusions

<!-- block: B157 -->

We presented strength-assessment-driven multimodal sequential observation for efficient CAI estimation. The method uses initially available surface descriptors and progressively acquired internal C-scan regions to learn specimen-conditioned requests. A common predictor updates strength after each observation, and a trajectory objective combines process-average and terminal errors to organize regional content and its arrival around the assessment task. The event identity explains why useful error reductions obtained earlier contribute more to process quality.

<!-- block: B158 -->

On the 50 development-validation specimens used for selection and effect estimation, endpoint MAE reached 42.385 MPa at cap 0.25, 9.65% below geometry-spread; the 0.0625 prefix reached 43.324 MPa. The five-grid cap first meeting a 47 MPa cohort-MAE target was 0.0625 versus 0.125 for geometry-spread. Larger early net reductions and fixed-set trajectories support the value of information arrival, although the primary permutation contrast crosses zero and endpoint equality follows from the set-based predictor. These conditional results establish observation organization as an optimization target for image-based assessment. Independent specimens, repeated initializations and physical acquisition remain necessary to establish broader reliability and inspection-time benefits.

<!-- block: B159 -->

# Data availability

<!-- block: B160 -->

The original surface and internal images are available in the six versioned Mendeley Data collections \[10--15\], and the CAI measurements are available in the corresponding strength-testing collection \[9\]. Supplementary S1--S5 describe specimen and capture-group manifests, model identities, frozen prediction trajectories, full-precision results, fixed-case diagnostic records and the scripts used to regenerate the manuscript displays. Source-file hashes identify the frozen evidence associated with the reported analyses.

<!-- block: B161 -->

# References

<!-- block: B162 -->

\[1\] Saki Hasebe, Ryo Higuchi, Tomohiro Yokozeki, and Shin-ichi Takeda. Dataset for surface and internal damage after impact on CFRP laminates. Data in Brief, 43:108462, 2022. doi: 10.1016/j.dib.2022.108462.

<!-- block: B163 -->

\[2\] Saki Hasebe, Ryo Higuchi, Tomohiro Yokozeki, and Shin-ichi Takeda. Dataset of compression after impact testing on carbon fiber reinforced plastic laminates. Data in Brief, 60:111509, 2025. doi: 10.1016/j.dib.2025.111509.

<!-- block: B164 -->

\[3\] Jason P. Mack, Faizan Mirza, Zhong-Hui Duan, Saki Hasebe, Ryo Higuchi, Tomohiro Yokozeki, and K. T. Tan. Deep learning for predicting impact energy and compression after impact strength of composite materials using C-scan images. Advanced Engineering Informatics, 72:104518, 2026. doi: 10.1016/j.aei.2026.104518.

<!-- block: B165 -->

\[4\] R. Fuentes, P. Gardner, C. Mineo, T. J. Rogers, S. G. Pierce, K. Worden, N. Dervilis, and E. J. Cross. Autonomous ultrasonic inspection using Bayesian optimisation and robust outlier analysis. Mechanical Systems and Signal Processing, 145:106897, 2020. doi: 10.1016/j.ymssp.2020.106897.

<!-- block: B166 -->

\[5\] Jaromír Janisch, Tomáš Pevný, and Viliam Lisý. Classification with Costly Features Using Deep Reinforcement Learning. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01):3959--3966, 2019. doi: 10.1609/aaai.v33i01.33013959.

<!-- block: B167 -->

\[6\] Ian Connick Covert, Wei Qiu, Mingyu Lu, Na Yoon Kim, Nathan J. White, and Su-In Lee. Learning to Maximize Mutual Information for Dynamic Feature Selection. Proceedings of the 40th International Conference on Machine Learning, 202:6424--6447, 2023. https://proceedings.mlr.press/v202/covert23a.html.

<!-- block: B168 -->

\[7\] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. arXiv, 2015. doi: 10.48550/arXiv.1512.03385.

<!-- block: B169 -->

\[8\] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention Is All You Need. arXiv, 2017. doi: 10.48550/arXiv.1706.03762.

<!-- block: B170 -->

\[9\] Saki Hasebe, Ryo Higuchi, Tomohiro Yokozeki, and Shin-ichi Takeda. Datasets on CFRP specimens subjected to compression after impact tests. Mendeley Data, 2025, version 3. https://data.mendeley.com/datasets/8scdmfdcfb/3.

<!-- block: B171 -->

\[10\] Saki Hasebe, Ryo Higuchi, Tomohiro Yokozeki, and Shin-ichi Takeda. Datasets on CFRP specimens after low-velocity impact tests: 8-layer cross-ply laminates. Mendeley Data, 2022, version 1. [8-layer cross-ply dataset, version 1](https://data.mendeley.com/datasets/74t7kcdgkr/1).

<!-- block: B172 -->

\[11\] Saki Hasebe, Ryo Higuchi, Tomohiro Yokozeki, and Shin-ichi Takeda. Datasets on CFRP specimens after low-velocity impact tests: 24-layer quasi-isotropic laminates. Mendeley Data, 2022, version 1. [24-layer quasi-isotropic dataset, version 1](https://data.mendeley.com/datasets/cgtnjyggtm/1).

<!-- block: B173 -->

\[12\] Saki Hasebe, Ryo Higuchi, Tomohiro Yokozeki, and Shin-ichi Takeda. Datasets on CFRP specimens after low-velocity impact tests: 16-layer quasi-isotropic laminates. Mendeley Data, 2022, version 1. [16-layer quasi-isotropic dataset, version 1](https://data.mendeley.com/datasets/w68dtmpfyf/1).

<!-- block: B174 -->

\[13\] Saki Hasebe, Ryo Higuchi, Tomohiro Yokozeki, and Shin-ichi Takeda. Datasets on CFRP specimens after low-velocity impact tests: 24-layer cross-ply laminates. Mendeley Data, 2022, version 1. [24-layer cross-ply dataset, version 1](https://data.mendeley.com/datasets/xcmzfsbd9t/1).

<!-- block: B175 -->

\[14\] Saki Hasebe, Ryo Higuchi, Tomohiro Yokozeki, and Shin-ichi Takeda. Datasets on CFRP specimens after low-velocity impact tests: 16-layer cross-ply laminates. Mendeley Data, 2022, version 1. [16-layer cross-ply dataset, version 1](https://data.mendeley.com/datasets/yfxyg8jm46/1).

<!-- block: B176 -->

\[15\] Saki Hasebe, Ryo Higuchi, Tomohiro Yokozeki, and Shin-ichi Takeda. Datasets on CFRP specimens after low-velocity impact tests: 8-layer quasi-isotropic laminates. Mendeley Data, 2022, version 1. [8-layer quasi-isotropic dataset, version 1](https://data.mendeley.com/datasets/ykhs7s2dck/1).
