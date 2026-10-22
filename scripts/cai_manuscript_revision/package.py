"""Write review-facing source maps and delivery instructions."""
from pathlib import Path
from evidence import read_json,write_json,write_csv,digest,ORDER,EVID,DIAG,DOCS,SHA,TASK

def prepare_package(paper,art):
 p=Path(paper);a=Path(art);repo=p.parents[1]
 bindings=read_json(a/'SOURCE_BINDINGS.json');by={r['path']:r for r in bindings['records']}
 claims=[]
 specs=[
 ('C01','4.1','50 specimens / 48 capture groups evaluated; 276 total in 161/50/65 partitions',ORDER+'/cohort_manifest.csv','VALID50_CAPTURE48_DOMAIN6','cohort metadata','counts','Same development partition for selection and evaluation'),
 ('C02','3/4.4','Proposed is NO_VLM@1000, MEAN_SC@1750',ORDER+'/MODEL_SELECTION_LOCK.json','actor / predictor','model identity','update','No reselection; actual byte-verification status in model lock'),
 ('C03','3.3','Internal zero mask precedes affine projection; two set means exclude history','src/cmc_bbdm/cai_agent_v3/models.py','MeanSCPredictor._forward_weighted','code semantics','features','No claim that zero mask removes affine bias'),
 ('C04','3.4','Actor unknown descriptor, 102 regional / 6 global inputs, two zero channels each','src/cmc_bbdm/cai_agent_v3/models.py','SpatialCAIActor','code semantics','features','use_vlm=False and use_feedback=True'),
 ('C05','3.5','Positive log-probability sign minimizes remaining task cost','src/cmc_bbdm/cai_agent_v3/actor_training.py','actor loss / returns','training definition','normalized cost','Describes completed training only'),
 ('C06','5.1/Abstract','Endpoint MAE 42.385 MPa at cap .25; geometry 46.910 (9.65% lower)',EVID+'/method_summary.csv','NO_VLM_SPATIAL_FEEDBACK / endpoint_mae_mpa','ARCHIVED_METHODS','MAE specimen pooled','Selected development configuration, one initialization'),
 ('C07','5.1','Geometry endpoint gain 4.525 [0.670,8.276] MPa','paper_cai_aei/r3_positive_review_r1/tables/paired_contrasts_saved.csv','GEOMETRY_SPREAD / MAE','saved writing-evidence intervals','paired specimen-pooled error','Existing 5000-draw conditional interval; no selection/multiplicity adjustment'),
 ('C08','5.1/Abstract','Same .25 path at .0625 prefix: MAE 43.324 MPa',ORDER+'/same_cap_metrics.csv','ARCHIVED_METHODS / NO_VLM_SPATIAL_FEEDBACK / .0625','ARCHIVED_METHODS','MAE specimen pooled','Prefix query, no smaller-cap replanning'),
 ('C09','5.1','A 43.597; geometry gain 3.714 [-.337,7.283] MPa','paper_cai_aei/r3_positive_review_r1/tables/paired_contrasts_saved.csv','GEOMETRY_SPREAD / A','saved writing-evidence intervals','six-source-group equal','Interval crosses zero'),
 ('C10','5.2','q47 five caps: proposed/static .0625, geometry .125, random .25',ORDER+'/matched_quality.csv','FIVE_FIXED_CAPS / target47 / ARCHIVED_METHODS','ARCHIVED_METHODS','cohort first attainment','50% reduction applies to geometry on this reporting grid'),
 ('C11','5.2','q42 event first cap .078844 later recrosses; full anchor not attained',ORDER+'/matched_quality.csv','FULL_EVENT_UNION / targets42,41.69001007080078','ARCHIVED_METHODS','cohort first attainment','No sustained attainment or specimen stopping rule'),
 ('C12','5.3.1','Early raw gains: geometry 6.865 [.362,12.502], random 9.707 [4.862,14.131]',ORDER+'/stage_paired_contrasts.csv','stage1 / raw_net_mpa / geometry,random','ARCHIVED_METHODS','six-source-group equal','Positive proposed-minus-control decrease; adverse events retained'),
 ('C13','5.3.1','Weighted first-stage geometry gain 5.926 [-.117,11.095]',ORDER+'/stage_paired_contrasts.csv','stage1 / weighted_net_mpa / GEOMETRY_SPREAD','ARCHIVED_METHODS','six-source-group equal','Interval crosses zero; four stages sum to A gain'),
 ('C14','5.3.2/Abstract','Primary permutation A +1.486 [-.658,3.384]',ORDER+'/order_paired_contrasts.csv','PRIMARY_PERMUTED_MEAN_MINUS_NATIVE / area_mpa','FIXED_SET_ORDERS','six-source-group equal','Primary interval crosses zero; average permutation advantage uncertain'),
 ('C15','5.3.2','Secondary reversal A +3.929 [.145,7.405]; both early intervals cross zero',ORDER+'/order_paired_contrasts.csv','SECONDARY_REVERSE_MINUS_NATIVE / area_mpa; both early_area_mpa','FIXED_SET_ORDERS','six-source-group equal','Secondary support does not replace unresolved primary contrast'),
 ('C16','5.3.2','Endpoint equality is a structural invariant across the fixed final sets',ORDER+'/endpoint_invariance.csv','350 terminal checks','FIXED_SET_ORDERS','terminal input equality','Not an endpoint accuracy gain or additive causal decomposition'),
 ('C17','5.4','9/94 departures from initial feasible ranking in six fixed cases','paper_cai_aei/r3_positive_review_r1/tables/S5_initial_feasible_ranking.csv','six locked case_keys; per-decision initial feasible score argmax versus actual','saved diagnostic trajectories','six-case behavior','No cohort-wide rate or feedback-only causal attribution'),
 ('C18','5.4','q24-48 retains final residual 122.252 MPa',DIAG+'/trajectories/cgtnjyggtm_q24-48/N_NATIVE.json','target_mpa / predictions_mpa[-1]','saved NO_VLM case','specimen absolute error','Fixed adverse case retained'),
 ('C19','5.5','Geometry-relative improvements: 33/50 A and 29/50 endpoint; heterogeneous groups','paper_cai_aei/r3_positive_review_r1/tables/paired_contrasts_saved.csv','GEOMETRY_SPREAD / improved_n','saved paired contrasts','physical specimens','Within-partition strata, no configuration-transfer test'),
 ('C20','5.5','Partial-minus-full MAE +.695 [-4.009,5.327] MPa','paper_cai_aei/r3_positive_review_r1/tables/paired_contrasts_saved.csv','FULL_SCAN / MAE','saved paired contrasts','specimen pooled','Difference only; no equivalence or noninferiority margin'),
 ('C21','5.6','Area measures supplied information; physical scanner time was not measured',ORDER+'/ORDER_MECHANISM_SCOPE.json','scope / nonclaims','protocol boundary','native-pixel fraction','No deployment, time-saving or hardware-performance claim')]
 chinese_claims=[
 '共276件，划分为161/50/65件；效果评价为50件、48个捕获组',
 '主方法锁定NO_VLM@1000，共同预测器锁定MEAN_SC@1750',
 '内部特征先置零掩膜再仿射投影；两个集合均值不读取历史',
 '策略使用未知描述符，区域/全局输入为102/6维，各含两个恒零占位通道',
 '正号对数概率项用于最小化后续任务代价，仅描述已完成训练',
 '0.25上限下终点MAE为42.385 MPa；几何分散为46.910 MPa，降低9.65%',
 '相对几何分散终点改善4.525 MPa，95%区间[0.670,8.276]',
 '同一0.25路径的0.0625前缀MAE为43.324 MPa，不重新规划',
 '过程A为43.597 MPa；相对几何分散改善3.714，区间[-0.337,7.283]跨零',
 '47 MPa五点网格首次达标：本文/静态0.0625，几何0.125，随机0.25',
 '42 MPa事件网格首次达标0.078844，随后回升；未达到完整输入锚点',
 '早期原始净下降相对几何/随机改善6.865 [0.362,12.502]及9.707 [4.862,14.131] MPa',
 '首阶段加权改善相对几何为5.926 [-0.117,11.095] MPa，区间跨零',
 '主要置换减原序的过程A差为+1.486 [-0.658,3.384] MPa，区间跨零',
 '辅助逆序A差为+3.929 [0.145,7.405] MPa；两项早期区间均跨零',
 '固定最终集合下350个终点检查一致，这是集合型读出的结构性质',
 '六个固定案例94次决策中9次偏离初始可行排序，不外推为全队列比例',
 '保留q24-48终点122.252 MPa的大残差案例',
 '相对几何分散，33/50件改善过程A、29/50件改善终点，组间存在不利变化',
 '部分减完整输入的MAE差为+0.695 [-4.009,5.327] MPa，不作等效结论',
 '面积衡量供应图像信息量，未测量物理扫描时间'
 ]
 for index,(cid,section,text,path,key,family,estimand,boundary) in enumerate(specs):
  source=repo/path
  # Derived saved intervals retain their separately bound original provenance.
  for lang in ['en','zh']:
   claims.append(dict(claim_id=cid,section=section,language=lang,claim_text=text if lang=='en' else chinese_claims[index],source_ref=SHA if path in by and by[path]['git_blob_sha'] else 'frozen derived copy; parent recorded in SOURCE_BINDINGS',source_path=path,source_key_or_symbol=key,source_hash=digest(source),evidence_family=family,estimand=estimand,units='MPa or stated protocol units',claim_boundary=boundary,status='SOURCE_BOUND_SCOPE_REVIEWED'))
 write_csv(a/'CLAIM_EVIDENCE_MAP.csv',claims)
 maps=[]
 figures={
 'F1':[('loop','Observable request/update/assessment boundary','src/cmc_bbdm/cai_agent_v3/models.py','MeanSCPredictor / SpatialCAIActor','Schematic; actual one-update c8-16 values from cases.json','fixed weights; y training/scoring only')],
 'F2':[('representation','Known versus unknown and independent predictor','src/cmc_bbdm/cai_agent_v3/models.py','SpatialCAIActor and MeanSCPredictor','Schematic only, explicit dimensions','102 regional + 6 global; 65 width-128 tokens')],
 'F3':[('a','When does useful cohort accuracy appear?',ORDER+'/cost_error_curves.csv.gz','ARCHIVED_METHODS / six methods','post-event step; five cap dots; no smoothing','repeat loss mean then specimen pool'),('b','Does the proposed curve locally recross?',ORDER+'/cost_error_curves.csv.gz','NO_VLM_SPATIAL_FEEDBACK / .06–.14','crop saved step curve, preserve held boundaries','same pooled MAE')],
 'F4':[('a','Which caps first reach every target?',ORDER+'/matched_quality.csv','FIVE_FIXED_CAPS / ARCHIVED_METHODS','all integer targets 41–61; NR band and zero','cohort first crossing, no interpolation'),('b','How does event resolution change attainment?',ORDER+'/matched_quality.csv','FULL_EVENT_UNION / ARCHIVED_METHODS','all targets 41–61; NR and zero','cohort first crossing, recrossing preserved in table')],
 'F5':[('a','Did early events reduce actual error?',ORDER+'/stage_paired_contrasts.csv','stage1 / raw_net_mpa / geometry,random','existing estimate and 95% CI; proposed minus control','six-source-group equal, positive favors proposed'),('b','Which stages account for integrated change?',ORDER+'/stage_paired_contrasts.csv','weighted_net_mpa / geometry / stages1–4','existing estimates and 95% CI; no sign change','six-source-group equal, four stages close to A gain'),('c','Does arrival alter paths within fixed content?',ORDER+'/cost_error_curves.csv.gz','FIXED_SET_ORDERS / native,permuted_mean,reverse','post-event steps, separate identity family','specimen pool; permutation losses before means'),('d','How resolved is the order effect?',ORDER+'/order_paired_contrasts.csv','PRIMARY and SECONDARY / area_mpa','error-row signs unchanged; paired95%CI','comparator minus native A; six-source-group equal')],
 'F6':[]}
 assets=read_json(p/'figure_data/assets.json')
 for i,asset in enumerate(assets):
  key=asset['specimen_key'];letter='abc' if i==0 else 'def'
  for panel,kind in [(letter[0],'surface'),(letter[1],'cscan')]:
   v=asset[kind];figures['F6'].append((panel,'Where were actual observations acquired?',v['source_path'],key+' / '+kind,v['transform']+'; overlays only for C-scan; no added contrast','native rint edges; saved ordinals; retrospective full C-scan only'))
  figures['F6'].append((letter[2],'What strength updates and residual remain?',DIAG+'/trajectories/'+key.replace(':','_')+'/N_NATIVE.json',key,'saved post-event step and final carry','measured strength is retrospective reference only'))
 for fig,items in figures.items():
  for panel,question,path,key,transform,rule in items:
   maps.append(dict(figure_id=fig,panel_id=panel,question=question,source_ref=SHA if not path.startswith('/') else 'Mendeley Data v1; source hash in assets.json',source_path=path,source_key=key,transform=transform,output_path=str((p/'figures'/f'{fig}.pdf').relative_to(repo)),coordinate_or_aggregation_rule=rule,asset_status='RESOLVED',qa_status='AUTOMATED_NO_FAIL; manual panel review in FIGURE_PANEL_REVIEW.csv'))
 write_csv(a/'FIGURE_SOURCE_MAP.csv',maps)
 # Requirement completion is anchored to concrete files; publication remains receipt-dependent.
 descriptions=['Input/source binding','Model identity and zero-computation boundary','Evidence, estimands and 8-row direction repair','Protocol chapter 4','Method chapter 3 and 13 equations','Results chapter 5 and boundaries','Context, abstract and conclusions','Six figures and source data','S1–S5 and eight document builds','Finite checks and one primary review','Whitelisted content, handoff and actual publication']
 targets=['SOURCE_BINDINGS.json','MODEL_IDENTITY_LOCK.json','DIRECTION_CORRECTION.json','source/manuscript_en.md §4','source/manuscript_en.md §3; tables/equation_map.csv','source/manuscript_en.md §5','source/manuscript_en.md §§1/2/6/Abstract','FIGURE_SOURCE_MAP.csv; figures/F1–F6','BUILD_REPORT.json; source/supplementary_en.md','LOCAL_VALIDATION.json; REVIEW.md','DELIVERY_MANIFEST.json; PUBLICATION_RECEIPT.json (written only after push)']
 write_csv(a/'REQUIREMENT_MATRIX.csv',[dict(task_id=f'T{i:02}',requirement=desc,evidence=target,status='CONTENT_COMPLETE_REMOTE_RECEIPT_PENDING' if i==10 else 'COMPLETE') for i,(desc,target) in enumerate(zip(descriptions,targets))])
 changes=read_json(a/'TEXT_CHANGE_MAP.json')
 (p/'change_log.md').write_text('# Revision change log\n\nFrozen evidence: '+SHA+'.\n\n'+ '\n'.join('- '+r['reason'] for r in changes['changes'])+'\n\nFigures F1–F6 were redrawn from declared source data. Fig. 6 adds two source-verified registered C-scan references. All six diagnostic cases and adverse effects remain. The supplement now supplies S1–S5, all target statuses and the 500/350 trajectory ledgers.\n')
 (p/'README.md').write_text('''# CAI positive review R1

Task: CAI_POSITIVE_REVIEW_R1. Scientific scope: selected development configuration only.

The `build` folder contains English and aligned bilingual manuscript and supplement, each in editable DOCX and PDF. Canonical bilingual blocks are in `source`; shared tables, equations and figures appear once in the combined version. `figures` contains six main SVG/PDF/PNG figures and six saved-diagnostic plates. `tables` and `figure_data` contain frozen source subsets and explicit source keys.

Run from the repository root (Python, matplotlib/numpy/Pillow, Pandoc and XeLaTeX required):

```bash
python scripts/cai_manuscript_revision/run.py prepare-evidence --paper-root paper_cai_aei/r3_positive_review_r1 --artifact-root artifacts/cai_agent_v3/manuscript/r3_positive_review_r1
python scripts/cai_manuscript_revision/run.py render-figures --paper-root paper_cai_aei/r3_positive_review_r1 --artifact-root artifacts/cai_agent_v3/manuscript/r3_positive_review_r1
python scripts/cai_manuscript_revision/run.py build --paper-root paper_cai_aei/r3_positive_review_r1 --artifact-root artifacts/cai_agent_v3/manuscript/r3_positive_review_r1 --languages en,bilingual
python scripts/cai_manuscript_revision/run.py verify --paper-root paper_cai_aei/r3_positive_review_r1 --artifact-root artifacts/cai_agent_v3/manuscript/r3_positive_review_r1
```

Rendering and building use already supplied arrays and image subsets, without loading research weights. Initial `bind` additionally needs `--repo . --packet <extracted-packet-path>` and the precisely resolved local image pairs. Re-preparing evidence and text requires the private local originals and extractions under the docs root; these are ignored by Git and excluded from the delivery inventory. Building the supplied canonical sources does not require those private originals. No research computation is authorized. `SOURCE_BINDINGS.json`, `MODEL_IDENTITY_LOCK.json`, `REVIEW.md` and the handoff under the artifact root document evidence and QA. The publication receipt records actual remote verification after the content commit.

The main title, six chapters, five main tables, thirteen numbered equations and fifteen references are retained. Author identity, affiliations, funding and competing interests require author confirmation before submission. Independent evaluation, multiple training initializations, component interventions and physical scanner-time validation were not performed.
''')
 (repo/DOCS/'README.md').write_text('''# Frozen manuscript revision inputs

The two input originals retain their supplied bytes locally. `inputs/`, `original_extracted.md`, `original_media/` and `original_tables.json` are private local inputs: Git ignores them and the delivery inventory excludes their contents. They must not be staged or uploaded to GitHub. Source bindings retain only path/hash metadata for verification.

`original_extracted.md` is the Pandoc structural extraction used for deterministic editorial anchors; it does not replace the originals. `original_media` retains the seven original image objects. The execution packet files preserve task scope. New manuscript files, evidence maps and publication status live in the separate paper/artifact roots. Re-preparing text requires these private local files; building the supplied canonical sources does not. No research source or frozen result was edited.
''')
