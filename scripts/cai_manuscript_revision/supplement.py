"""Assemble substantive S1–S5 from frozen records and declared methods."""
from pathlib import Path
from collections import Counter
import shutil
from evidence import read_json,read_csv,write_json,write_csv,display_names,METHODS,LABELS,ORDER,DOCS,DIAG
from text import emit,md_table


def prepare_supplement(paper,art,main_shared):
    paper=Path(paper);art=Path(art);repo=paper.parents[1];data=read_json(paper/'figure_data/frozen_data.json');cases=read_json(paper/'figure_data/cases.json');facts=read_json(repo/DOCS/'verified_facts.json')
    blocks=[]
    def add(en,zh=None,shared=False):
        blocks.append(dict(id=f'S{len(blocks)+1:03d}',en=en,zh=en if shared else zh,shared=shared))
        if blocks[-1]['zh'] is None: raise ValueError('missing SI translation')
    def table(en,zh,rows,column_widths=None):
        add(en,zh); add(md_table(rows,column_widths=column_widths),shared=True)
    def subset(rows,mapping,decimals=3):
        out=[]
        for r in rows:
            item={}
            for dest,key in mapping.items():
                v=r[key]
                if isinstance(v,float):v=f'{v:.{decimals}f}'
                item[dest]=display_names(v)
            out.append(item)
        return out
    add('# Supplementary material','# 补充材料')
    add('This supplement accompanies the revised manuscript on region selection and acquisition order for CAI assessment. All numerical results, intervals, trajectories and diagnostic arrays were already saved. This revision performs deterministic source binding, formatting and plotting only. No optimizer update, research-model forward, new bootstrap draw, seed, reserved-partition evaluation or checkpoint selection was performed.',
        '本补充材料配套 CAI 评估区域选择与采集次序修订稿。全部数值结果、区间、轨迹及诊断数组均已保存。本次修订仅执行确定性的来源绑定、排版与绘图，未执行优化器更新、研究模型前向、新 bootstrap 抽样、新种子、预留分区评价或 checkpoint 选择。')
    add('# S1. Data identity, partitions and evaluation units','# S1. 数据身份、划分与评价单位')
    add('The six paired-image collections and strength labels are the existing references [1,2,9–15] in the main article. A specimen record binds its laminate configuration, specimen ID, registered image pair and CAI label. Capture groups were formed before splitting from shared source/crop identities, then retained within partitions. The split contains 276 physical specimens (161 training, 50 development validation and 65 reserved), with 152/48/59 capture groups. The reserved partition was not evaluated. The same 50 development specimens selected checkpoints and estimated effects; all intervals are conditional on that selection.',
        '六份配对图像集合及强度标签对应主文既有参考文献 [1,2,9–15]。试样记录绑定层合板配置、试样编号、登记图像对及 CAI 标签。捕获组在划分前按共享来源或裁切身份建立，并在划分内保持完整。总计 276 件物理试样，训练、开发验证及预留分区分别为 161/50/65 件，对应 152/48/59 个捕获组。预留分区未评价。同一 50 件开发试样用于选模及效果估计，所有区间均条件于该选择。')
    split=read_csv(repo/'results/cai_agent_v3/new_protocol/split_manifest.csv',unique=['specimen_key']);write_csv(paper/'tables/S1_all_partition_metadata.csv',split)
    assert Counter(r['split'] for r in split)=={'TRAIN':161,'VALID':50,'TEST':65}
    table('**Table S1.** The complete 50-specimen evaluation cohort. Specimens are identified by layer count, layup and specimen number; dimensions and area clocks remain in cohort_manifest.csv. The full 276-row partition metadata are provided as S1_all_partition_metadata.csv. Reading split labels is not reserved-set evaluation.',
          '**表 S1。** 完整 50 件评价试样。试样以层数、铺层及试样编号标识；原生尺寸及面积计量见 cohort_manifest.csv。全部 276 条划分元数据见 S1_all_partition_metadata.csv。读取划分标签不构成预留分区评价。',
          subset(data['cohort_manifest'],{'Specimen':'specimen_key','Capture group':'capture_group_id','CAI (MPa)':'target_mpa'}))
    add('Random contributes five saved trajectories per specimen; the other five methods contribute one each. Thus 500 archived trajectories and 7,849 acquisition events are repeated observations of these 50 specimens, not additional independent samples. Fixed-final-set analysis adds seven conditions per specimen (original, reverse, five permutations), giving 350 conditions on the same cohort. Paired intervals resample capture groups within six source groups, carrying each group’s specimens together.',
        '随机策略每件试样有五条已保存轨迹，其余五策略各一条，因此 500 条归档轨迹与 7,849 个采集事件是同一 50 件试样上的重复观测，不是额外独立样本。固定最终集合分析每件试样包含原序、逆序及五次置换七个条件，共 350 个条件，仍使用同一队列。配对区间在六来源组内重采样捕获组，并整体携带各捕获组的试样。')
    add('# S2. Model identity, representation and learning protocol','# S2. 模型身份、表示与学习协议')
    add('The frozen actor is SpatialCAIActor with use_vlm=False and use_feedback=True. The method key is `NO_VLM_SPATIAL_FEEDBACK`; its seed is 2026091302, selected update 1000 after 1250 completed updates. The frozen readout is MeanSCPredictor (MEAN_SC), selected at update 1750 after 2000 updates. The actor contains 378,978 parameters including its training value head. The model-identity lock records the checkpoint and state-dictionary identities and whether local checkpoint bytes could be rehashed. A manifest-only checkpoint verification is not represented as a new model load.',
        '冻结策略为 SpatialCAIActor，use_vlm=False、use_feedback=True，方法键 NO_VLM_SPATIAL_FEEDBACK，种子 2026091302，共完成 1250 次更新，选定第 1000 次。冻结读出为 MeanSCPredictor（MEAN_SC），完成 2000 次更新，选定第 1750 次。策略含训练价值头共 378,978 个参数。MODEL_IDENTITY_LOCK.json 记录 checkpoint、状态字典身份及本地 checkpoint 字节是否可重新哈希；仅核对清单不等同于重新加载模型。')
    table('**Table S2.** Frozen architecture and observable-input contract.','**表 S2。** 冻结架构与可观测输入合同。',[
      {'Component':'Local encoder','Specification':'Frozen ResNet18; independently crop each cell before encoding; 512 features per region'},
      {'Component':'Grid / native area','Specification':'8 × 8 row-major, indices 0–63; edges rint(linspace(0, native dimension, 9)); integer pixels'},
      {'Component':'Surface registration','Specification':'Frozen orientation; clockwise 90° once; maximum edge 1024 before local extraction'},
      {'Component':'Actor local projections','Specification':'Surface 512 → 48; visible/learned-unknown internal 512 → 48'},
      {'Component':'Actor regional input (102)','Specification':'48 surface + 48 internal + 2 coordinates + 1 measured + 1 ordinal/64 + 2 zero VLM placeholders'},
      {'Component':'Actor global input (6)','Specification':'Acquired area, remaining area, standardized CAI, measured-cell fraction, 2 zero VLM placeholders'},
      {'Component':'Actor contextualizer','Specification':'64 regional + 1 global tokens; width 128; 2 layers; 4 heads; feed-forward 256; dropout 0'},
      {'Component':'Actor heads','Specification':'Concatenate local/global tokens → 128 → 1 score; separate global value head during training'},
      {'Component':'Readout local cell','Specification':'Two 512 → 64 affine projections; concatenate with mask and coordinates (131 inputs); 131 → 64 → 64 GELU'},
      {'Component':'Readout pooling / output','Specification':'Mean of all 64 cells + mean of measured cells + acquired area (129 inputs); 129 → 64 → 1; training-only target scaling'},
      {'Component':'Readout missing input','Specification':'Internal descriptor is zero-masked BEFORE affine projection, retaining bias; measured-cell mean is zero at empty mask'},
      {'Component':'Readout order dependence','Specification':'Set-based; no ordinal or history input. Same final set and area imply the same terminal input.'}])
    add('The actor uses a learned unknown internal descriptor and an explicit measurement indicator. It never treats an unmeasured internal cell as known image evidence. Its two reserved regional channels and two reserved global channels are constant zero in this model. The predictor instead masks raw internal descriptors before its affine projection, so an unmeasured cell still contributes the projection bias to the all-cell summary. These are different missing-input mechanisms. Surface descriptors are available initially; measured CAI enters training costs or retrospective metrics only, never an execution decision.',
        '策略用学习到的未知内部描述符及显式测量指示表示缺失输入，未测内部格不会作为已知图像证据。该模型的两个区域保留通道及两个全局保留通道恒为零。预测器则在仿射投影前对原始内部描述符施加零掩膜，因此未测格仍向全格汇总贡献投影偏置。这是两种不同缺失输入机制。表面描述符最初可用，实测 CAI 仅用于训练代价或事后指标，从不参与执行决策。')
    add('Predictor mask training uses the frozen mixture: 10% empty masks; 60% masks with 1–16 cells drawn from random or fixed geometric orders; 20% random masks with 17–48 cells; 10% full masks. The standardized residual is fitted with Huber loss. Three capture-group cross-fitted training predictors provide policy cost: a training specimen is scored only by the predictor excluding its fold. The common evaluation predictor is fitted on the training partition. This description documents the completed training; the revision does not execute it.',
        '预测器掩膜训练沿用冻结混合分布：10% 空掩膜，60% 为随机或固定几何次序的 1–16 格，20% 为随机 17–48 格，10% 为完整输入。使用标准化残差的 Huber 损失。三份按捕获组交叉拟合的训练预测器提供策略代价，每件训练试样仅由排除其所属折的预测器评分；共同评价预测器在训练分区拟合。这里描述的是既有训练，本次修订不执行该过程。')
    add('Policy optimization used AdamW with learning rate $3 \\times 10^{-4}$, weight decay $10^{-4}$, batch size 16 and gradient-norm limit 1. Training sampled feasible categorical actions. Evaluation used deterministic feasible-score argmax. Candidate checkpoints were compared every 250 updates on development process A; an earlier candidate was replaced only for an improvement exceeding $10^{-12}$. The following scores are the existing job history, not a new candidate comparison.',
        '策略优化采用 AdamW，学习率 $3 \\times 10^{-4}$、权重衰减 $10^{-4}$、批次 16、梯度范数上限 1。训练采样可行分类动作，评价采用确定性的可行评分最大值。每 250 次按开发过程 A 比较候选，仅当改善超过 $10^{-12}$ 时替换较早候选。下表成绩来自既有任务历史，不构成新候选比较。')
    job=read_json(repo/'results/cai_agent_v3/w3_pilot/r1_0e11452a/jobs/NO_VLM_SPATIAL_FEEDBACK.json')
    table('**Table S3.** Archived actor-selection history.','**表 S3。** 归档策略选模历史。',subset(job['progress'],{'Update':'update','Development A (MPa)':'validation_area_mpa'}))
    add('## S2.1. Derivation and cost-minimization sign','## S2.1. 推导与代价最小化符号')
    add('All 13 original numbered display equations remain in the main article; equation_map.csv records their locations. To derive Eqs. (9–10), substitute the signed error identity below into the interval sum in Eq. (7). Each completed event contributes to every later interval, with total length B − c_j including final carry. Unequal cell areas and negative decreases are retained.',
        '全部 13 个原编号陈列公式保留在主文，equation_map.csv 记录位置。推导式（9–10）时，将下列带符号误差恒等式代入式（7）的区间求和。每个完成事件贡献于之后全部区间，总长度为 B − c_j，包含最后延续区间，并保留不等格面积及负下降。')
    add(r'$$e_t=e_0-\sum_{j=1}^{t}\Delta e_j,\qquad \sum_{k=j}^{T-1}(c_{k+1}-c_k)+(B-c_T)=B-c_j.$$',shared=True)
    add(r'$$A(B)=e_0-\sum_{j=1}^{T}(1-c_j/B)\Delta e_j,\qquad J=(1+\lambda_T)e_0-\sum_{j=1}^{T}(1-c_j/B+\lambda_T)\Delta e_j.$$',shared=True)
    add('The trajectory baseline e₀ is constant for a fixed specimen and predictor. An earlier favorable event has more remaining integration length, whereas an adverse event contributes negatively. At c_j = B the process contribution vanishes and the terminal weight λT = 0.25 remains. Equation (11) divides remaining process-plus-terminal cost by the training target scale. Equation (12) has a positive log-probability sign because it minimizes cost: above-baseline sampled cost reduces that action’s probability under gradient descent. The advantage is stopped for actor differentiation; actor terms sum decisions within an episode and average episodes, while squared-value errors and entropy average within episodes before batch aggregation. Total loss adds 0.5 times value loss and subtracts entropy weight, linearly reduced from 0.01 to zero.',
        '对固定试样及预测器，初始基线 e₀ 为常数。较早有利事件具有更长剩余积分区间，不利事件作负贡献；在 c_j = B 时过程贡献消失，终点权重 λT = 0.25 保留。式（11）将剩余过程与终点代价除以训练目标尺度。式（12）对数概率项为正号，因为优化目标是最小化代价：梯度下降使高于基线代价的采样动作概率降低。策略求导时优势停止梯度，策略项在轨迹内对决策求和后平均轨迹，价值平方误差及熵则先在轨迹内平均再聚合批次。总损失增加 0.5 倍价值损失并减去熵权重，后者从 0.01 线性降为零。')
    add('# S3. Caps, acquired area, intervals and quality targets','# S3. 比例上限、获取面积、区间与质量目标')
    add('A requested reporting cap queries the latest completed prediction at or below that cap on the single path planned for total B = 0.25. No policy is replanned for a smaller cap. Predictions are right-continuous step functions; the initial or final estimate is carried until the next event or total cap. MAE and RMSE pool physical specimens; Random averages absolute and squared losses within specimen before pooling. Process A and event-stage quantities average repeats within specimen, then specimens within source group, then the six source groups equally. These weights differ because development group sizes are 9,9,8,9,8,7.',
        '报告上限查询总上限 B = 0.25 的单一路径中，不超过该值的最近完成预测，不为较小上限重新规划策略。预测为右连续阶梯函数，初始或最后估计延续至下一事件或总上限。MAE、RMSE 按物理试样池化；随机策略先在试样内平均绝对或平方损失再池化。过程 A 与事件阶段量先在试样内平均重复，再平均来源组内试样，最后六来源组等权。开发组大小为 9、9、8、9、8、7，故两种权重不同。')
    maincaps=[r for r in data['same_cap_metrics'] if r['identity_family']=='ARCHIVED_METHODS' and r['series'] in METHODS]
    rows=[{'Method':LABELS[METHODS.index(r['series'])],'Cap':r['requested_cost'],'MAE':f"{float(r['mae_mpa']):.3f}",'Mean area':f"{float(r['actual_cost_mean']):.6f}",'Min area':f"{float(r['actual_cost_min']):.6f}",'Max area':f"{float(r['actual_cost_max']):.6f}"} for r in maincaps]
    table('**Table S4.** All five-cap results and actual acquired fractions (MAE in MPa). Full-precision RMSE and R² remain in same_cap_metrics.csv.','**表 S4。** 全部五点上限及实际获取比例，MAE 单位为 MPa。全精度 RMSE 与 R² 见 same_cap_metrics.csv。',rows)
    paired=read_csv(paper/'tables/paired_contrasts_saved.csv')
    paired=[{'Control':r['control'].replace('GEOMETRY_SPREAD','Geometry').replace('LEARNED_STATIC_TRUE','Static').replace('CENTER_FIRST','Center').replace('FULL_SCAN_P_ALL','Full').replace('SERPENTINE','Serpentine').replace('RANDOM','Random'),'Metric':r['metric'],'Estimate':f"{float(r['proposed_minus_control']):.3f}",'95% low':f"{float(r['lower']):.3f}",'95% high':f"{float(r['upper']):.3f}",'Improved n':r['improved_n']} for r in paired]
    table('**Table S5.** Saved paired differences, proposed minus control. Negative error differences favor proposed; these are existing 95% intervals, conditional on selection.','**表 S5。** 已保存配对差“本文减对照”。误差差为负有利于本文；区间为既有、条件于选模的 95% 区间。',paired)
    add('The percentile intervals use the existing 5,000 paired capture-group draws within source groups. Neither checkpoint selection nor multiple comparisons is adjusted. No interval is newly generated. The partial-minus-full MAE contrast of +0.695 MPa [−4.009, 5.327] is a difference estimate; no prespecified equivalence/noninferiority margin was tested. The area-cost axis measures supplied image pixels rather than measured scanner time.',
        '百分位区间采用已有的来源组内 5000 次配对捕获组抽样，未校正 checkpoint 选择及多重比较，本次未生成新区间。部分减完整输入 MAE 差为 +0.695 MPa [−4.009, 5.327]，这是差异估计，未按预设界限检验等效或非劣效。面积成本轴衡量提供的图像像素，不是实测扫描时间。')
    for grid,title,zh in [('FIVE_FIXED_CAPS','Five-cap','五点上限'),('FULL_EVENT_UNION','Event-union','事件并集')]:
        rows=[]
        for q in data['quality_targets']:
            val=float(q['target_mpa']);rout={'Target':f'{val:.3f}' if not val.is_integer() else str(int(val))}
            for m,lab in zip(METHODS,LABELS):
                r=next(r for r in data['matched_quality'] if r['identity_family']=='ARCHIVED_METHODS' and r['grid_type']==grid and float(r['target_mpa'])==val and (r['main_series']==m or r['comparator_series']==m))
                pfx='main' if r['main_series']==m else 'comparator';v=r[pfx+'_cost'];cross=r[pfx+'_later_recrosses_target']
                rout[lab]=('NR' if not v else f'{float(v):.6f}')+('†' if str(cross).lower()=='true' else '')
            rows.append(rout)
        table(f'**Table S6{ "a" if grid=="FIVE_FIXED_CAPS" else "b"}.** {title} first-attainment caps for every preset target, including the full-input anchor. NR = not reached by 0.25; 0 = initially met; † = later recrossing. Relative saving is undefined if either method is NR or control cost is zero. All statuses and savings are preserved in matched_quality.csv.',
              f'**表 S6{ "a" if grid=="FIVE_FIXED_CAPS" else "b"}。** {zh}中全部预设目标的首次达标上限，含完整输入锚点。NR 表示 0.25 内未达标；0 表示最初满足；† 表示随后回升。任一方法未达标或对照成本为零时，相对节约不定义。全部状态与节约值保存在 matched_quality.csv。',rows)
    add('At q = 47 MPa, five-cap attainment is 0.0625 for proposed and learned-static, 0.125 for geometry and 0.25 for Random. The 50% reduction applies to geometry on this grid only. At q = 42 MPa the proposed event curve first crosses at 0.078844 and later recrosses; the five-cap grid is NR. The 41.690 MPa full-input anchor is not attained by 0.25. These are cohort first crossings, not an execution-time specimen-level stopping rule or engineering certification.',
        'q = 47 MPa 时，五点网格首次达标分别为本文与学习固定排序 0.0625、几何分散 0.125、随机 0.25。50% 减少仅适用于该网格的几何分散对照。q = 42 MPa 时，本文事件曲线在 0.078844 首次交叉后回升，五点网格为 NR。41.690 MPa 完整输入锚点在 0.25 内未达到。这些是队列首次交叉，不是执行时的试样级停止规则或工程认证。')
    add('# S4. Event contributions and fixed-set order interventions','# S4. 事件贡献与固定集合次序干预')
    add('Stage 1–4 completion intervals are (0,.0625], (.0625,.125], (.125,.1875] and (.1875,.25]. A raw decrease is previous absolute error minus updated absolute error. Its weighted contribution multiplies that decrease by (1 − completion fraction / B). Both retain adverse events. The sum of weighted decreases equals initial error minus process A. Archived methods use their original float64 area clock; fixed-set order intervention uses integer cumulative native pixels divided by total crop pixels. The two identity families remain separate; small native-replay differences from archived predictions are not silently mixed.',
        '阶段 1–4 按完成区间定义为 (0,.0625]、(.0625,.125]、(.125,.1875]、(.1875,.25]。原始下降为之前绝对误差减更新后绝对误差，加权贡献再乘以（1 − 完成比例 / B），两者均保留不利事件。加权下降和等于初始误差减过程 A。归档方法保留原 float64 面积计量，固定集合干预使用整数累计原生像素除以总裁切像素。两种身份家族保持分离，不混用原序回放与归档预测之间的微小数值差。')
    rows=[{'Control':LABELS[METHODS.index(r['comparator'])],'Stage':r['stage'],'Type':'weighted' if r['metric']=='weighted_net_mpa' else 'raw','Gain':f"{float(r['estimate_mpa']):.3f}",'95% low':f"{float(r['ci_low_mpa']):.3f}",'95% high':f"{float(r['ci_high_mpa']):.3f}"} for r in data['stage_paired_contrasts']]
    table('**Table S7.** All saved paired stage contrasts (MPa), proposed minus control; positive favors proposed. No direction change is applied to this already correct source table.','**表 S7。** 全部已保存配对阶段差（MPa），方向为本文减对照，正值有利于本文。这份原表方向正确，不作反号。',rows)
    # Full frozen intervention ledger, without importing model code or recomputing predictions.
    fields=['order_id','specimen_key','variant','repeat','cells','costs','predictions_mpa','target_mpa','final_cost','final_error_mpa','left_error_area_mpa','early_left_error_area_mpa']
    write_csv(paper/'tables/S4_all_350_order_conditions.csv',[{k:r[k] for k in fields} for r in data['reorder_trajectories']])
    fields=['method','run','specimen_key','dataset_id','capture_group_id','cells','costs','predictions_mpa','target_mpa','left_error_area_mpa','early_left_error_area_mpa','final_error_mpa']
    write_csv(paper/'tables/S4_all_500_archived_trajectories.csv',[{k:r[k] for k in fields} for r in data['archived_cohort_episodes']])
    rows=[]
    for r in data['order_contrasts']:
        rows.append({'Contrast':'Primary permutation' if r['contrast'].startswith('PRIMARY') else 'Secondary reverse','Metric':{'area_mpa':'A','early_area_mpa':'Early A','objective_mpa':'J','final_error_mpa':'Final error',**{f'weighted_stage_{i}_mpa':f'Weighted S{i}' for i in range(1,5)}}[r['metric']],'Direction':'N−C contribution' if r['metric'].startswith('weighted_stage') else 'C−N error','Estimate':f"{float(r['estimate_mpa']):.3f}",'95% low':f"{float(r['ci_low_mpa']):.3f}",'95% high':f"{float(r['ci_high_mpa']):.3f}"})
    table('**Table S8.** Normalized fixed-set contrasts. N = native/original; C = comparator. Error rows are comparator minus native. Contribution rows are native minus comparator. All positive values favor native under these distinct definitions.','**表 S8。** 归一化固定集合对比。N 为原序，C 为对照。误差行为对照减原序，贡献行为原序减对照，按各自定义正值均有利于原序。',rows)
    add('The legacy order_paired_contrasts.csv contains eight weighted-stage rows labeled comparator-minus-native. Only those contribution rows are negated in the derived normalized table, with CI [lo,hi] transformed to [−hi,−lo]. Eight error/objective rows remain unchanged. source_* columns preserve the original values and source Git blob 23a65f8d481797298b25bb228d1b2fd91181c65f. Applying normalization twice is identical to applying it once. The four normalized stage differences sum to the corresponding A difference within $10^{-8}$ MPa. Because final error is invariant within each fixed set, J − A differences also close. DIRECTION_CORRECTION.json records residuals.',
        '旧 order_paired_contrasts.csv 的八条加权阶段贡献行标为“对照减原序”。派生归一化表仅对这些贡献行反号，将区间 [lo,hi] 变为 [−hi,−lo]，其余八条误差或目标行保持不变。source_* 列保留原值及源 Git blob 23a65f8d481797298b25bb228d1b2fd91181c65f。归一化重复执行结果不变；四阶段差的和在 $10^{-8}$ MPa 内等于对应 A 差。固定集合的终点误差不变，故 J 与 A 的差值也闭合。DIRECTION_CORRECTION.json 记录残差。')
    add('The original-order intervention reproduced the archived predictions within the saved maximum deviation 6.104 × $10^{-5}$ MPa. All 350 terminal-set checks preserved the same terminal input and prediction. This is expected from a set-based predictor and validates the intervention; it is not an endpoint accuracy gain caused by order. The primary permutation A interval [−0.658,3.384] crosses zero, whereas the secondary reverse interval [0.145,7.405] is positive. Both early-A intervals cross zero. No component-specific causal claim or additive percentage decomposition follows.',
        '原序干预复现归档预测，已保存最大偏差为 6.104 × $10^{-5}$ MPa。全部 350 项终点集合核验保持相同终点输入与预测，这是集合式预测器的预期性质，用于验证干预，不是排序带来的终点准确性增益。主要置换 A 区间 [−0.658,3.384] 跨零，辅助逆序区间 [0.145,7.405] 为正，两项早期 A 区间均跨零，由此不能推出组件单独因果贡献或可加百分比分解。')
    add('# S5. Six fixed cases and saved information-use diagnostics','# S5. 六个固定案例与已保存信息使用诊断')
    add('The six cases were fixed before this revision and are all retained, including adverse residuals. Initial preferences are feasible-action softmax probabilities, not regional damage probabilities. The attribution maps are already saved L1 sensitivity of the selected action’s log probability to local surface descriptors, including the path through the current strength prediction. Attention maps are the saved global-query-to-regional attention among fused tokens. These three quantities have different meanings and units; none is a regional mechanical ground truth. The displays below use only the initial, NO_VLM model’s saved arrays. No attribution or attention was regenerated.',
        '六个案例在修订前已经固定，本次全部保留，包括不利残差。初始偏好是可行动作 softmax 概率，不是区域损伤概率。归因图为已保存的所选动作对数概率对局部表面描述符的 L1 敏感性，包含经过当前强度预测的路径；注意力图为融合 token 中已保存的全局查询到区域的注意力。这三种量的含义和单位不同，均不是区域力学真值。以下仅使用 NO_VLM 模型初始状态的已保存数组，未重新生成归因或注意力。')
    rows=[]
    for case in cases:
        rows.append({'Specimen':display_names(case['specimen_key']),'First cell':case['actions'][0],'Actions':len(case['actions']),'CAI target':f"{case['target_mpa']:.3f}",'Initial':f"{case['predictions_mpa'][0]:.3f}",'Final':f"{case['predictions_mpa'][-1]:.3f}",'Final abs. error':f"{abs(case['predictions_mpa'][-1]-case['target_mpa']):.3f}"})
    table('**Table S9.** All six cases; CAI values and errors in MPa.','**表 S9。** 全部六个案例；CAI 与误差单位为 MPa。',rows,column_widths=[30,8,8,9,9,9,10]);write_csv(paper/'tables/S5_case_summary.csv',rows)
    add('Nine of 94 recorded decisions departed from the highest initially scored cell still feasible on the actual path: c8-16 (2), q16-29 (2), c16-24 (4), q8-22 (1); q24-48 and c24-9t retained their initial feasible ranking. The comparison removes measured and unaffordable cells at every actual state. This describes specimen-specific initialization and selective reordering in six cases only. Changing internal evidence, history, progress and prediction are coupled, so the departures do not isolate internal-feedback causality.',
        '94 个记录决策中有九次偏离实际路径上仍可行的初始最高评分格：c8-16 两次、q16-29 两次、c16-24 四次、q8-22 一次；q24-48 与 c24-9t 保持初始可行排序。该比较在每个实际状态排除已测与超出剩余面积的格，仅描述六例的试样相关初始化及选择性重排序。变化的内部证据、历史、进度及预测相互耦合，故偏离不能隔离内部反馈的因果作用。')
    for i,c in enumerate(cases,1):
        add(f'## S5.{i}. {c["specimen_key"]}',f'## S5.{i}. {c["specimen_key"]}')
        add(f'![Saved diagnostic plate {i}](figures/S5_{i}.png){{width=6.5in}}',shared=True)
        add(f'**Figure S5.{i}.** Initial NO_VLM state and saved CAI trajectory. (a) Feasible-action preference; (b) surface L1 sensitivity for the selected-action query (first saved target); (c) first-layer global-query regional attention (first saved layer); (d) saved CAI predictions with measured strength dashed. Maps retain all 64 cells and use independent ranges. Values and target/layer indices are in figure_data/cases.json; selected cell {c["actions"][0]}. Final absolute error: {abs(c["predictions_mpa"][-1]-c["target_mpa"]):.3f} MPa.',
            f'**图 S5.{i}。** NO_VLM 初始状态与保存的 CAI 轨迹。(a) 可行动作偏好；(b) 所选动作查询（首个保存目标）的表面 L1 敏感性；(c) 第一层全局查询区域注意力；(d) 保存的 CAI 预测，虚线为实测强度。每张图保留全部 64 格，使用独立范围。数值及目标、层编号见 figure_data/cases.json；选择格为 {c["actions"][0]}，终点绝对误差为 {abs(c["predictions_mpa"][-1]-c["target_mpa"]):.3f} MPa。')
    add('## S5.7. Real-image provenance and reproducible displays','## S5.7. 真实图像来源与可复现图稿')
    add('Main Fig. 6 uses the c8-16 (8-layer cross-ply) and q24-48 (24-layer quasi-isotropic) image pairs from the corresponding Mendeley Data version 1 collections (Hasebe, Higuchi, Yokozeki and Takeda, 2022; CC BY 4.0). The candidate manifest resolves each surface and registered C-scan path with SHA256 and native dimensions. Surface orientation follows the frozen clockwise-90° transform and maximum-edge-1024 display resize. The registered C-scan pixels are unchanged; overlays use rint native grid edges and the saved action ordinals. Grey overlays mark unavailable cells. The full-image backdrop is explicitly retrospective and was never supplied to the actor. assets.json records source URLs, hashes, dimensions, transformations and attribution. No damage outline, scale calibration or physical length measurement is invented.',
        '主图 6 使用 c8-16（8-layer cross-ply）与 q24-48（24-layer quasi-isotropic）图像对，分别来自相应的 Mendeley Data 第 1 版数据集（Hasebe 等，2022，CC BY 4.0）。候选清单以 SHA256 与原生尺寸解析每张表面图及登记 C-scan 路径。表面方向遵循冻结的顺时针 90° 变换及最长边 1024 显示缩放；登记 C-scan 像素保持不变，覆盖层采用 rint 原生网格边界及保存动作序号，灰覆盖表示不可用格。完整图像背景明确为事后参照，从未提供给策略。assets.json 记录来源网址、哈希、尺寸、变换及署名，不虚构损伤轮廓、标尺校准或物理长度测量。')
    add('The delivered scripts bind, prepare, render, build and verify this frozen-evidence package. Machine-readable trajectory ledgers provide all 500 archived paths and all 350 fixed-set conditions, without duplicating weights or feature shards. The eight document outputs provide English and aligned bilingual main text and supplement. Reference numbering is inherited from the main article; this supplement introduces no new references. Author names, affiliations, funding, competing interests and final journal submission metadata require author confirmation before submission; the package is a revision delivery, not a journal submission.',
        '交付脚本执行冻结证据包的绑定、准备、绘图、构建及核验。机器可读轨迹台账提供全部 500 条归档路径及 350 个固定集合条件，不重复打包权重或特征分片。八份文档输出提供英文及对齐双语的正文与补充材料。参考文献编号沿用主文，补充材料不引入新文献。作者姓名、单位、资助、利益冲突及最终投稿元数据仍须作者确认；本包是修订交付，不是期刊投稿。')
    # Give exported table identities the same names as the manuscript.
    # Exact source keys and paths remain bound in the frozen-data/source metadata.
    for path in (paper/'tables').glob('*.csv'):
        rows=read_csv(path)
        renamed=[]
        for row in rows:
            renamed.append({{'dataset_id':'Laminate','specimen_key':'Specimen'}.get(key,key):display_names(value) if key in {'dataset_id','specimen_key','order_id'} else value for key,value in row.items()})
        write_csv(path,renamed)
    shared=emit(paper,'supplementary',blocks)
    write_json(paper/'source/alignment.json',{'manuscript':{'shared_ids':main_shared},'supplementary':{'shared_ids':shared}})
    return {'blocks':len(blocks),'sections':['S1','S2','S3','S4','S5']}
