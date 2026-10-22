"""Editorial transformation of the supplied manuscript; no research execution."""
from pathlib import Path
import re, shutil
from evidence import read_json, read_csv, write_json, write_csv, DOCS

TITLE='Learning region selection and acquisition order in C-scan images for efficient compression-after-impact strength assessment of composite laminates'


def parse_original(text):
    # Tables are structural blocks, not blank-line-delimited paragraphs.
    tables=[]
    def table(m):
        tables.append(m.group()); return f'\n\n@@TABLE{len(tables)}@@\n\n'
    text=re.sub(r'^  -{20,}[ \t]*\n.*?^  -{20,}[ \t]*$',table,text,flags=re.M|re.S)
    chunks=[p.strip() for p in re.split(r'\n\s*\n',text) if p.strip()]
    blocks=[]; i=0
    while i<len(chunks):
        p=chunks[i]; shared=False
        if p.startswith('#'):
            en,sep,zh=p.partition(' / ')
            zh=re.match(r'#+',en).group()+' '+zh if sep else en
        elif p.startswith(('$$','![]','@@TABLE','\\[')):
            en=zh=p; shared=True
        else:
            en=p
            if i+1>=len(chunks) or not re.search(r'[\u4e00-\u9fff]',chunks[i+1]):
                raise ValueError(f'No Chinese counterpart for {p[:90]}')
            i+=1; zh=chunks[i]
        blocks.append({'id':f'B{len(blocks)+1:03d}','en':en,'zh':zh,'shared':shared,'original':en})
        i+=1
    return blocks


def md_table(rows,columns=None):
    if not rows: raise ValueError('empty table')
    columns=columns or list(rows[0])
    def clean(v): return str(v).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(clean,columns))+' |','| '+' | '.join('---' for c in columns)+' |',*['| '+' | '.join(clean(r.get(c,'')) for c in columns)+' |' for r in rows]])


def emit(paper,stem,blocks):
    for lang in ['en','zh']:
        (paper/'source'/f'{stem}_{lang}.md').write_text('\n\n'.join(f"<!-- block: {b['id']} -->\n\n{b[lang]}" for b in blocks)+'\n',encoding='utf-8')
    return [b['id'] for b in blocks if b.get('shared')]


def prepare_text(paper,art):
    paper=Path(paper); art=Path(art); repo=paper.parents[1]
    baseline=(repo/DOCS/'original_extracted.md').read_text()
    blocks=parse_original(baseline); changes=[]
    def find(prefix):
        found=[b for b in blocks if b.get('original','').startswith(prefix)]
        if len(found)!=1: raise ValueError(f'Anchor {prefix}: {len(found)}')
        return found[0]
    def replace(prefix,en,zh,reason):
        b=find(prefix); b['en']=en; b['zh']=zh
        changes.append(dict(block_id=b['id'],original_anchor=prefix,action='REVISE',reason=reason))
    def drop(prefix,reason):
        b=find(prefix); blocks.remove(b); changes.append(dict(block_id=b['id'],original_anchor=prefix,action='REMOVE',reason=reason))
    blocks.insert(0,dict(id='TITLE',en='# '+TITLE,zh='# 通过学习 C-scan 区域选择与采集次序高效评估复合材料层合板冲击后压缩强度',shared=False))
    # Protocol first: retain exact cohort, loss, prefix and conditional-interval definitions.
    replace('The fixed-final-set intervention retains',
      'The frozen fixed-final-set intervention retains the 15 or 16 cells selected by the proposed policy for each of the same 50 specimens. Original order, complete reversal and five preset hash-determined permutations provide 350 replay conditions. The archived intervention evaluated each intermediate mask with the common frozen predictor. Cumulative area uses integer native pixel counts divided by total crop pixels. The primary contrast is the within-specimen five-permutation mean minus original order; reversal minus original order is the prespecified secondary contrast. These repeated conditions do not increase the independent specimen count. Supplementary S4 provides the saved interventions, direction definitions and endpoint checks.',
      '冻结的固定最终集合干预保留同一 50 件试样中本文策略选定的 15 或 16 格。原序、完全逆序及五次预设哈希置换提供 350 个回放条件。归档干预曾使用共同冻结预测器评价各中间掩膜；本次修订直接复用这些结果。累计面积采用整数原生像素数除以总裁切像素数。主要对比为试样内五次置换均值减原序，预设辅助对比为逆序减原序。这些重复条件不增加独立试样数。补充 S4 给出已保存的干预、方向定义及终点核验。','T02: distinguish frozen repeated conditions from independent units')
    # Method boundary and reproducibility.
    replace('The identities follow by substituting',
      'Supplementary S2 derives Eqs. (9–10) from Eq. (7), including the final carry interval. Their signed event contributions accommodate unequal native cell areas and prediction changes in either direction. Section 5.3 tests realized changes using archived events and fixed-set replay.',
      '补充 S2 从式（7）推导式（9–10），并保留最后延续区间。带符号事件贡献适用于不等原生格面积及双向预测变化。第 5.3 节利用归档事件与固定集合回放检验实际变化。','T03: move detailed algebra to S2, retain all 13 numbered equations')
    replace('The readout depends on the available set',
      'Both means in Eq. (4) summarize a set: the all-cell mean includes the masked-input affine bias, and the observed-cell mean uses only measured cells. Acquisition history enters the policy, not this predictor. Identical final sets and acquired fractions therefore produce identical terminal inputs. Endpoint equality in the fixed-set intervention is a structural invariant; order can affect accuracy only at intermediate prefixes.',
      '式（4）的两个均值均汇总集合：全格均值包含掩膜输入的仿射偏置，已测格均值只使用已测格。采集历史进入策略而不进入该预测器。因此，相同最终集合与获取比例对应相同终点输入。固定集合干预中的终点一致性是结构不变量，次序只能影响中间前缀的准确性。','T03: distinguish set-based readout and history-conditioned actor')
    # Results: endpoint -> early prefix -> process, conditional intervals explicit.
    early=find('Useful estimates emerged at the first nonzero reporting cap.')
    process=find('The improvement extended to the acquisition trajectory.')
    blocks.remove(early); blocks.insert(blocks.index(process),early)
    replace('The improvement extended to the acquisition trajectory.',
      'The trajectory-level point estimates also favored the proposed policy. Process error $A$ was 43.597 MPa versus 47.311 MPa for geometry-spread, a reduction of 3.714 MPa or 7.85%. Its paired 95% interval, [−0.337, 7.283] MPa, crosses zero, so this process contrast remains uncertain despite the favorable mean. Relative to learned-static, the reduction was 4.380 MPa [0.199, 8.405]. Table 3 distinguishes this six-source-group-equal process metric from specimen-pooled endpoint MAE.',
      '轨迹级点估计也有利于本文策略。过程误差 $A$ 为 43.597 MPa，几何分散为 47.311 MPa，减少 3.714 MPa，即 7.85%。配对 95% 区间为 [−0.337, 7.283] MPa，跨越零，因此该过程对比虽有利于本文，仍存在不确定性。相对学习固定排序，减少量为 4.380 MPa [0.199, 8.405]。表 3 区分了六来源组等权的过程指标与试样池化的终点 MAE。','T04: endpoint/early/process hierarchy and zero-crossing uncertainty')
    # Narrow order evidence and remove redundant summary subsection.
    replace('The primary permutation-mean-minus-original contrast was',
      'The primary permutation-mean-minus-original process contrast was +1.486 MPa, with a paired 95% interval of [−0.658, 3.384] MPa (Fig. 5d). Thus, the lower mean under original order does not establish a resolved average advantage against the preset permutations. The prespecified secondary reversal contrast was +3.929 MPa [0.145, 7.405], supporting an order effect for that intervention. Early-process contrasts were +1.628 MPa [−2.190, 5.180] for permutation and +3.328 MPa [−1.527, 7.979] for reversal; both intervals cross zero. These contrasts describe arrival within the selected sets and do not partition the total policy gain into selection and order percentages.',
      '主要“置换均值减原序”的过程对比为 +1.486 MPa，配对 95% 区间为 [−0.658, 3.384] MPa（图 5d）。因此，原序均值更低尚不能确立其相对预设置换的平均优势。预设辅助逆序对比为 +3.929 MPa [0.145, 7.405]，支持该干预下的次序作用。早期过程对比分别为置换 +1.628 MPa [−2.190, 5.180] 和逆序 +3.328 MPa [−1.527, 7.979]，两者区间均跨零。这些对比描述选定集合内部的到达顺序，不将总体策略改善分解为选择与排序的百分比。','T06: primary uncertainty, secondary scope and no additive attribution')
    drop('### 5.3.3.','T06: remove redundant closing subsection')
    replace('The objective, event analysis and fixed-set intervention describe',
      'Together, the event and fixed-set results connect earlier useful estimates to realized early reductions and the timing of selected content, with the primary permutation contrast remaining uncertain.',
      '事件与固定集合结果共同将较早的有用估计联系到实际早期下降及选定内容的到达时机，同时主要置换对比仍具有不确定性。','T06: one bounded closing sentence')
    b=find('The subsequent decisions showed selective reordering')
    b['en']+=' These nine departures concern only the six fixed cases. Because the changing state jointly includes measured content, history, progress and prediction, they cannot be attributed causally to internal-image feedback alone.'
    b['zh']+=' 这九次偏离仅涉及六个固定案例。由于变化状态同时包含已测内容、历史、进度和预测，不能将这些偏离因果归因于内部图像反馈这一单独因素。'
    changes.append(dict(block_id=b['id'],action='REVISE',reason='T07: restrict 9/94 and causal scope'))
    b=find('The fixed diagnostic cases show how specimen information')
    b['en']=b['en'].replace('registered surface image, initial selection preference, acquired-cell order and strength updates','registered surface image, internal-image reference with acquired-cell order, and strength updates')
    b['zh']=b['zh'].replace('登记表面图、初始选择偏好、已获取格次序及强度更新','登记表面图、叠加已获取格次序的内部图参照及强度更新')
    # Introduction / related work after protocol and results are locked.
    replace('The method makes three connected contributions.',
      'The study makes three connected contributions. First, it formulates regional C-scan acquisition as a strength-supervised decision task in which every observed prefix produces a mechanical estimate. Second, the multimodal observable state and process-plus-terminal objective jointly organize which regions are requested and when their information enters a common predictor. Third, a frozen common-predictor evaluation links matched-cap accuracy, matched-quality information requirements and fixed-final-set order interventions, while retaining conditional uncertainty and adverse cases. These contributions concern task formulation, observation organization and supporting evidence.',
      '本文形成三项相互衔接的贡献。首先，将区域 C-scan 采集建模为强度监督决策任务，使每个观测前缀均产生力学估计。其次，通过多模态可观测状态及过程与终点联合目标，共同组织请求哪些区域以及其信息何时进入共同预测器。最后，以冻结的共同预测器评价连接相同上限准确性、等质量信息需求与固定最终集合的次序干预，同时保留条件不确定性及不利案例。三项贡献分别对应任务建模、观测组织与支持证据。','T09: task/method/evidence contributions')
    replace('Active feature acquisition similarly learns',
      'Sequential inspection couples each request to an evolving information state. For strength assessment, the quality of that state can be evaluated through a continuous mechanical output after every completed regional observation. This task criterion motivates the present comparison between spatial allocation rules and a specimen-conditioned acquisition policy under a common predictor.',
      '序贯检查将每次请求连接到不断演化的信息状态。对于强度评估，可以在每个区域观测完成后，通过连续力学输出评价该状态的质量。这一任务准则引出了本文在共同预测器下，对空间分配规则与试样条件化采集策略的比较。','T09: keep sequential context in 2.2')
    drop('The acquisition controls distinguish spatial allocation', 'T09: controls are specified in 4.2, remove duplicate from related work')
    replace('## 2.3.', '## 2.3. Active feature acquisition and prediction with partial observations', '## 2.3. 主动特征获取与部分观测预测','T09: conceptual related-work scope')
    replace('The temporal asymmetry between modalities shapes',
      'Active feature acquisition learns which information to request for a downstream prediction task. Janisch and colleagues formulated costly-feature classification through sequential requests [5]. Covert and colleagues studied dynamic feature selection through conditional mutual information and amortized learning [6]. These approaches provide a decision perspective for assessing the value of information before all inputs are observed. Here, the requested units are C-scan regions and the output is continuous CAI strength throughout acquisition.',
      '主动特征获取学习为下游预测任务请求哪些信息。Janisch 等将高成本特征分类建模为序贯请求 [5]；Covert 等通过条件互信息与摊销学习研究动态特征选择 [6]。这些研究提供了在全部输入尚未观测时评价信息价值的决策视角。本文请求单元为 C-scan 区域，输出为贯穿采集过程的连续 CAI 强度。','T09: restore cited active-feature context without new prior-art claims')
    replace('Attention-based contextualization allows',
      'Partial observation also requires distinguishing available evidence from missing input. Residual-network image representations [7] and attention-based contextualization [8] supply building blocks for representing visible content and comparing candidate requests. The present method uses surface context and progressively revealed internal information for this purpose. Section 3 defines its specific masks, tokens and prediction interface; the experiments assess the complete observation-selection mechanism.',
      '部分观测还要求区分已有证据与缺失输入。残差网络图像表示 [7] 与基于注意力的上下文化 [8]，为表示可见内容及比较候选请求提供基础构件。本文据此利用表面上下文与逐步揭示的内部信息，第 3 节定义具体掩膜、token 及预测接口，实验评价完整观测选择机制。','T09: move implementation detail out of related work')
    # Abstract / conclusion last.
    abstract='Compression-after-impact (CAI) strength assessment depends on which internal-image observations are supplied to the predictor and when they become available. We propose strength-assessment-driven multimodal sequential observation to learn C-scan region selection and acquisition order from specimen-level strength supervision. Initially available surface descriptors, acquired internal regions and the current CAI estimate form an observable decision state. A regional policy requests the next feasible region, and a common predictor updates strength after each observation. A trajectory objective combines process-average and terminal absolute errors to value useful estimates throughout acquisition. In offline regional replay on 50 development-validation specimens, also used for checkpoint selection, endpoint mean absolute error (MAE) was 42.385 MPa at an acquisition-area cap of 0.25, 9.65% below geometry-spread. The 0.0625 prefix of this same total-cap path reached MAE 43.324 MPa. On the five-cap reporting grid, first attainment of cohort MAE at most 47 MPa required a cap of 0.0625 versus 0.125 for geometry-spread. Recorded events showed larger early net error reductions. Fixed-final-set replay isolated arrival within selected content: the primary permutation-minus-original process contrast was +1.486 MPa, with a 95% interval of [−0.658, 3.384] MPa, leaving its average advantage uncertain. These results support task-driven observation organization for earlier useful CAI estimates within the selected development configuration.'
    replace('Compression-after-impact (CAI) strength assessment can benefit',abstract,
      '冲击后压缩（CAI）强度评估取决于向预测器提供哪些内部图像观测，以及这些信息何时可用。本文提出强度评估驱动的多模态序贯观测方法，利用试样级强度监督学习 C-scan 区域选择与采集次序。初始表面描述符、已获取内部区域及当前 CAI 估计构成可观测状态；区域策略请求下一可行区域，共同预测器在每次观测后更新强度。轨迹目标联合过程平均绝对误差与终点误差，评价整个采集过程中的有用估计。在 50 件同时用于 checkpoint 选择的开发验证试样上进行离线区域回放时，方法于 0.25 采集面积上限达到 42.385 MPa 终点平均绝对误差（MAE），比几何分散低 9.65%；同一总上限路径的 0.0625 前缀达到 43.324 MPa。在五点报告网格上，首次达到不高于 47 MPa 的队列 MAE，本文所需上限为 0.0625，几何分散为 0.125。归档事件显示更大的早期净误差下降。固定最终集合回放检验选定内容内部的到达顺序：主要“置换减原序”的过程差为 +1.486 MPa，95% 区间 [−0.658, 3.384] MPa，平均优势仍不确定。结果支持在已选开发配置中，以任务驱动的观测组织较早形成有用 CAI 估计。','T10: 200–250 words, same-path early result and primary uncertainty')
    replace('On 50 development-validation specimens, the method reached',
      'On the 50 development-validation specimens used for selection and effect estimation, endpoint MAE reached 42.385 MPa at cap 0.25, 9.65% below geometry-spread; the 0.0625 prefix reached 43.324 MPa. The five-grid cap first meeting a 47 MPa cohort-MAE target was 0.0625 versus 0.125 for geometry-spread. Larger early net reductions and fixed-set trajectories support the value of information arrival, although the primary permutation contrast crosses zero and endpoint equality follows from the set-based predictor. These conditional results establish observation organization as an optimization target for image-based assessment. Independent specimens, repeated initializations and physical acquisition remain necessary to establish broader reliability and inspection-time benefits.',
      '在同时用于选择与效果估计的 50 件开发验证试样上，方法于 0.25 上限达到 42.385 MPa 终点 MAE，比几何分散低 9.65%；0.0625 前缀达到 43.324 MPa。首次达到 47 MPa 队列 MAE 的五点网格上限为 0.0625，几何分散为 0.125。更大的早期净下降与固定集合轨迹支持信息到达的价值，但主要置换对比区间跨零，终点一致性来自集合式预测器结构。这些条件性结果使观测组织成为图像评估的优化对象。更广可靠性与检查时间收益仍需独立试样、重复初始化及物理采集检验。','T10: two-paragraph conclusion with evidence boundaries')
    # Replace figures with a single self-contained plate per figure.
    fig_index=0; captions={}
    for b in list(blocks):
        if b['en'].startswith('![]'):
            fig_index+=1
            if fig_index==7:
                blocks.remove(b); continue
            b['en']=b['zh']=f'![Figure {fig_index}](figures/F{fig_index}.png){{width=6.5in}}'
        if b['en'].startswith('**Figure 6a'):
            b['en']='**Figure 6.** Real-image evidence and CAI trajectories for c8-16 (a–c) and q24-48 (d–f). Each row shows the registered surface view, retrospective full-image C-scan reference with observed cells and acquisition ordinals overlaid, and saved prediction updates. Unnumbered cells were unavailable to the policy; the displayed full C-scan was never an execution input. Ordinals follow the native row-major grid. The final prediction is carried to cap 0.25; the q24-48 final residual of 122.252 MPa is retained. Images: S. Hasebe, R. Higuchi, T. Yokozeki and S. Takeda, Mendeley Data v1 (2022), CC BY 4.0 [10,11]. Surface images follow the frozen clockwise-90° registration and display resize; no C-scan contrast or rotation is added.'
            b['zh']='**图 6。** c8-16（a–c）与 q24-48（d–f）的真实图像证据及 CAI 轨迹。每行依次为登记表面图、叠加已测格及获取序号的事后完整 C-scan 参照、已保存的预测更新。未编号格对策略不可用，图示完整 C-scan 从未作为执行输入。序号遵循原生行优先网格，最后预测延续至 0.25 上限，保留 q24-48 的 122.252 MPa 终点残差。图像来源：Hasebe 等，Mendeley Data v1（2022），CC BY 4.0 [10,11]。表面图遵循冻结的顺时针 90° 登记与显示缩放，C-scan 不额外调整对比度或旋转。'
        if b['en'].startswith('**Figure 6e'):
            blocks.remove(b); continue
        if b['en'].startswith('**Figure 1.'):
            b['en']=b['en'].replace('(a) With','With').replace('(b) During','During')
            b['zh']=b['zh'].replace('(a) 在','在').replace('(b) 学习','学习')
        if b['en'].startswith('**Figure 2.'):
            b['en']='**Figure 2.** Observable-state representation and regional scoring. The actor combines known surface features, measured internal content or a learned unknown descriptor, mask, position and ordinal. Sixty-four regional tokens and one global token are contextualized by a two-layer, four-head Transformer. The 64 scores are restricted by a mask of unmeasured cells affordable within the remaining native area. The independent set-based predictor supplies the current strength estimate. Area and measured-cell fractions remain distinct; the two constant-zero channels per input block are detailed in S2. Diagrammatic connections specify information flow, not new data.'
            b['zh']='**图 2。** 可观测状态表示与区域评分。策略结合已知表面特征、已测内部内容或学习到的未知描述符、掩膜、位置及序号。64 个区域 token 与 1 个全局 token 经两层四头 Transformer 上下文化，再按剩余原生面积可容纳的未测格掩膜限制 64 个评分。独立的集合式预测器提供当前强度估计，面积比例与已测格比例保持区分；每个输入块的两个恒零通道详见 S2。示意连线规定信息流，不表示新数据。'
            b['en']=b['en'].replace('(a) The fixed', 'The fixed').replace('(b) Sixty-four','Sixty-four')
            b['zh']=b['zh'].replace('(a) 固定','固定').replace('(b) 64','64')
        if b['en'].startswith('**Figure 3.'):
            b['en']=b['en'].replace('Step curves preserve','(a) Six-method full event curves. (b) A local view of proposed recrossings, with the 42 MPa target dashed. Step curves preserve')
            b['zh']=b['zh'].replace('阶梯曲线保留','(a) 六方法完整事件曲线；(b) 本文局部回升，虚线表示 42 MPa 目标。阶梯曲线保留')
        if b['en'].startswith('**Figure 4.'):

            b['en']=b['en'].replace('Missing points indicate nonattainment; initially satisfied targets have zero cost.','The NR band explicitly marks nonattainment; initially satisfied targets are shown at zero cost.')
            b['zh']=b['zh'].replace('缺失点表示未达标','NR 区带显式表示未达标')
        if b['en'].startswith('**Figure '):
            num=re.search(r'Figure (\d+)',b['en']).group(1); captions['F'+num]={'en':b['en'],'zh':b['zh']}
        if b['en'].startswith('@@TABLE'):
            n=int(re.search(r'TABLE(\d+)',b['en']).group(1)); rows=read_csv(paper/f'tables/table{n}.csv')
            if n==3:
                rows=[{'Strategy':r['strategy'],'A':f"{float(r['A']):.3f}" if r['A'] else '—','Early A':f"{float(r['early_A']):.3f}" if r['early_A'] else '—','MAE':f"{float(r['MAE']):.3f}",'RMSE':f"{float(r['RMSE']):.3f}",'R²':f"{float(r['R2']):.3f}"} for r in rows]
            if n==3:
                rows[5]={k:'**'+v+'**' for k,v in rows[5].items()}
            if n in [1,2,4]:
                rows=[{k.split(' / ')[0]:v.split(' / ')[0] for k,v in r.items()} for r in rows]
            b['en']=b['zh']=md_table(rows)
    # Reference entries are shared, preserve exact fifteen, no synthetic references.
    reference_blocks=[b for b in blocks if b.get('original','').startswith('\\[')]
    assert len(reference_blocks)==15
    bib=repo/'paper_cai_aei/r2_c_331f5295/references.bib'
    entries={re.match(r'@\w+\{([^,]+)',v).group(1):v.strip() for v in re.split(r'(?m)(?=^@)',bib.read_text()) if v.strip()}
    keys=['hasebe2022data','hasebe2025data','mack2026','fuentes2020','janisch2019','covert2023','he2015','vaswani2017','data_cai','data_74t7kcdgkr','data_cgtnjyggtm','data_w68dtmpfyf','data_xcmzfsbd9t','data_yfxyg8jm46','data_ykhs7s2dck']
    (paper/'source/references.bib').write_text('\n\n'.join(entries[k] for k in keys)+'\n')
    contexts={i:[] for i in range(1,16)}
    section='Abstract'
    for block in blocks:
        text=block['en'].replace(r'\[','[').replace(r'\]',']')
        if text.startswith('#'):section=text.lstrip('# ')
        if block in reference_blocks:continue
        numbers=set()
        for group in re.findall(r'\[([0-9]+(?:\s*(?:,|--|–|-)\s*[0-9]+)*)\]',text):
            for part in group.split(','):
                ends=re.split(r'\s*(?:--|–|-)\s*',part.strip())
                numbers.update(range(int(ends[0]),int(ends[-1])+1))
        for number in numbers:
            if number in contexts:contexts[number].append((section,block['id']+': '+text))
    reference_rows=[]
    for number,(key,block) in enumerate(zip(keys,reference_blocks),1):
        if not contexts[number]:raise ValueError(f'Unbound current citation {number}: {key}')
        license_read=number in (10,11)
        reference_rows.append({'number':number,'citekey':key,
          'current_citation_section':' | '.join(dict.fromkeys(s for s,_ in contexts[number])),
          'current_citation_context':' | '.join(t for _,t in contexts[number]),
          'original_source':block['en'],
          'read_scope':'Supplied manuscript reference and frozen BibTeX'+('; Mendeley Data v1 landing page and CC BY 4.0 license read' if license_read else ''),
          'verified_status':'BASELINE_REFERENCE_PRESERVED; DATA_LICENSE_PAGE_VERIFIED' if license_read else 'PRESERVED_BASELINE_NOT_FULLTEXT_REVERIFIED',
          'citation_scope':'Existing assertions preserved; no new priority or literature-absence claim'})
    write_csv(paper/'reference_map.csv',reference_rows)
    shared=emit(paper,'manuscript',blocks)
    write_json(paper/'source/figure_captions.json',captions)
    write_json(art/'TEXT_CHANGE_MAP.json',{'workflow':['4 protocol','3 methods','5 results','1/2 context','6/abstract'],'changes':changes,'abstract_word_count':len(abstract.split()),'original_display_equations':13,'retained_display_equations':sum(b['en'].startswith('$$') for b in blocks),'table_count':5,'figure_count':6,'references':15})
    # An explicit equation-location ledger avoids renumbering ambiguity.
    write_csv(paper/'tables/equation_map.csv',[{'original_number':i,'revised_number':i,'location':'4.3' if i==13 else '3.2' if i<=2 else '3.3' if i<=5 else '3.4' if i==6 else '3.5','status':'RETAINED','derivation':'S2 for 7–12' if 7<=i<=12 else 'main text'} for i in range(1,14)])
    return shared
