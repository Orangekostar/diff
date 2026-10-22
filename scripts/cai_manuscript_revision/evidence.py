"""Read-only adapters for CAI_POSITIVE_REVIEW_R1 frozen evidence."""
from __future__ import annotations
import bisect
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from zipfile import ZipFile

TASK='CAI_POSITIVE_REVIEW_R1'
SHA='6082b69f2792130a321c74b0efcb198f5d65cefb'
ORDER='results/cai_agent_v3/order_mechanism/r1_37b3c404'
EVID='results/cai_agent_v3/c_render_retrain/r1_331f5295/evidence'
DIAG='results/cai_agent_v3/actor_c0_diagnostic/r2_9e765b04'
DOCS='docs/cai/positive_review_r1'
METHODS=['CENTER_FIRST','GEOMETRY_SPREAD','SERPENTINE','RANDOM','LEARNED_STATIC_TRUE','NO_VLM_SPATIAL_FEEDBACK']
LABELS=['Center-first','Geometry-spread','Serpentine','Random','Learned-static','Proposed']
ZH_LABELS=['中心优先','几何分散','蛇形','随机','学习固定排序','本文方法']
CONFIGS=['8-layer cross-ply','24-layer quasi-isotropic','16-layer quasi-isotropic','24-layer cross-ply','16-layer cross-ply','8-layer quasi-isotropic']
LAMINATES=dict(zip(['74t7kcdgkr','cgtnjyggtm','w68dtmpfyf','xcmzfsbd9t','yfxyg8jm46','ykhs7s2dck'],CONFIGS))


def display_names(text):
 # Keep genuine hyperlink destinations intact while giving datasets readable labels.
 parts=re.split(r'(https?://[^\s)]+)',str(text))
 for i,part in enumerate(parts):
  if i%2:
   match=re.fullmatch(r'https://data\.mendeley\.com/datasets/([a-z0-9]+)/(\d+)([.,;]?)',part)
   if match and match[1] in LAMINATES and not parts[i-1].endswith(']('):
    url=part[:-1] if match[3] else part
    parts[i]=f'[{LAMINATES[match[1]]} dataset, version {match[2]}]({url}){match[3]}'
  else:
   for key,label in LAMINATES.items():part=part.replace(key+':',label+' / ').replace(key,label)
   parts[i]=part
 return ''.join(parts)


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write_json(path,value):
 p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); return p

def read_csv(path,required=(),unique=()):
 p=Path(path)
 with (gzip.open(p,'rt',encoding='utf-8-sig',newline='') if p.suffix=='.gz' else p.open(encoding='utf-8-sig',newline='')) as handle:
  reader=csv.DictReader(handle)
  if not set(required).issubset(reader.fieldnames or []): raise ValueError(f'{TASK}: {p}: missing columns {set(required)-set(reader.fieldnames or [])}')
  rows=list(reader)
 if unique:
  keys=[tuple(r[k] for k in unique) for r in rows]
  if len(keys)!=len(set(keys)): raise ValueError(f'{TASK}: {p}: duplicate key {unique}')
 return rows

def write_csv(path,rows,fields=None):
 p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
 fields=fields or list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',encoding='utf-8',newline='') as handle:
  writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
 return p

def git(repo,*args): return subprocess.check_output(['git',*args],cwd=repo,text=True).strip()

def validate_roots(paper,art):
 paper=Path(paper).resolve(); art=Path(art).resolve()
 repo=Path(git(paper if paper.exists() else paper.parent,'rev-parse','--show-toplevel')).resolve()
 for path,relative in [(paper,'paper_cai_aei/r3_positive_review_r1'),(art,'artifacts/cai_agent_v3/manuscript/r3_positive_review_r1')]:
  if path!=repo/relative: raise ValueError(f'{TASK}: output root must be {repo/relative}, got {path}')
 return repo,paper,art

def bind(repo,packet,paper,art):
 repo=Path(repo).resolve(); packet=Path(packet).resolve(); _,paper,art=validate_roots(paper,art)
 if git(repo,'rev-parse','HEAD')!=SHA: raise ValueError(f'{TASK}: initial evidence base must be {SHA}')
 docs=repo/DOCS; docs.mkdir(parents=True,exist_ok=True)
 records=[]
 def record(path,purpose,scope='ALL_BYTES',original=True):
  p=Path(path); rel=str(p.relative_to(repo)) if p.is_relative_to(repo) else str(p)
  blob=None
  if p.is_relative_to(repo):
   result=subprocess.run(['git','rev-parse',f'{SHA}:{rel}'],cwd=repo,capture_output=True,text=True)
   if result.returncode==0:
    blob=result.stdout.strip()
    raw=subprocess.check_output(['git','show',f'{SHA}:{rel}'],cwd=repo)
    if raw!=p.read_bytes(): raise ValueError(f'{TASK}: frozen source changed: {rel}')
  records.append(dict(path=rel,repo='Orangekostar/diff' if blob else None,ref=SHA if blob else None,git_blob_sha=blob,sha256=digest(p),bytes=p.stat().st_size,read_scope=scope,purpose=purpose,kind='ORIGINAL' if original else 'DERIVED',status='VERIFIED_READ'))
  return p
 for name,expected in [('current_manuscript.pdf','6e7369955e378592c04758d5b8297391eac0a4fa89babbed255061cee47188c3'),('current_manuscript.docx','277a673c2235f2196781fa1b78cad90310fd43cc8820b15fca1e45bc36264160')]:
  p=packet/'inputs'/name
  if digest(p)!=expected: raise ValueError(f'{TASK}: input hash mismatch {p}')
  dest=docs/'inputs'/name; dest.parent.mkdir(exist_ok=True); shutil.copyfile(p,dest); record(dest,'T00 current manuscript baseline')
 for name in ['CODEX_CAI_REVISION_EXECUTION.md','FILE_TASK_BINDINGS.md','REVIEW_AND_PUBLICATION.md','PROMPT_AUDIT_REPORT.md']:
  dest=docs/name; shutil.copyfile(packet/name,dest); record(dest,'T00–T10 task contract')
 shutil.copyfile(packet/'reference/verified_facts.json',docs/'verified_facts.json')
 shutil.copyfile(packet/'reference/manuscript_tables.json',docs/'original_tables.json')
 for name in ['verified_facts.json','original_tables.json','original_extracted.md']:
  if (docs/name).is_file():record(docs/name,'T00 supplied reference or derived structural extraction',original=False)
 for rel in [f'{ORDER}/{n}' for n in ['MODEL_SELECTION_LOCK.json','ORDER_MECHANISM_SCOPE.json','order_summary.csv','order_paired_contrasts.csv','stage_paired_contrasts.csv','stage_summary.csv','cohort_manifest.csv','cost_error_curves.csv.gz','same_cap_metrics.csv','matched_quality.csv','quality_targets.csv','reorder_trajectories.csv.gz','archived_cohort_episodes.csv.gz','acceptance_results.json','input_bindings.json','endpoint_invariance.csv','native_reproduction.csv']]+[f'{EVID}/{n}' for n in ['method_summary.csv','full_scan_reference.json','full_scan_predictions.csv','same_cost_metrics_by_domain.csv']]+['src/cmc_bbdm/cai_agent_v3/models.py','src/cmc_bbdm/cai_agent_v3/checkpoint_selection.py','scripts/cai_actor_c0_diagnostic/diagnose.py','src/cmc_bbdm/cai_agent_v3/metrics.py','src/cmc_bbdm/cai_agent_v3/actor_training.py','results/cai_agent_v3/w3_pilot/r1_0e11452a/jobs/NO_VLM_SPATIAL_FEEDBACK.json','results/cai_agent_v3/w3_pilot/r1_0e11452a/initialization_summary.csv','results/cai_agent_v3/new_protocol/feature_bank_index.csv','results/cai_agent_v3/new_protocol/candidate_queue.csv','results/cai_agent_v3/new_protocol/feature_bank_manifest.json','results/cai_agent_v3/new_protocol/split_manifest.csv',f'{DIAG}/selected_cases.csv',f'{DIAG}/diagnostic_lock.json',f'{DIAG}/model_bindings.json','paper_cai_aei/r2_c_331f5295/README.md','paper_cai_aei/r2_c_331f5295/references.bib']:
  record(repo/rel,'T01–T08 frozen model/data/figure binding')
 facts=read_json(docs/'verified_facts.json'); lock=read_json(repo/ORDER/'MODEL_SELECTION_LOCK.json')
 job=read_json(repo/'results/cai_agent_v3/w3_pilot/r1_0e11452a/jobs/NO_VLM_SPATIAL_FEEDBACK.json')
 # Read the actual source symbols, without importing any research module.
 models=(repo/'src/cmc_bbdm/cai_agent_v3/models.py').read_text(); training=(repo/'src/cmc_bbdm/cai_agent_v3/actor_training.py').read_text()
 if 'class MeanSCPredictor' not in models or 'class SpatialCAIActor' not in models or 'NO_VLM_SPATIAL_FEEDBACK' not in training: raise ValueError(f'{TASK}: model source mismatch')
 model_lock=dict(task_id=TASK,evidence_sha=SHA,actor=facts['actor'],predictor=facts['predictor'],new_model_selection=0,research_model_forwards=0,optimizer_updates=0,new_bootstrap_draws=0,new_seeds=0,new_test_evaluation=0,weight_byte_checks={})
 common=Path(git(repo,'rev-parse','--git-common-dir')).resolve().parent
 for role in ['actor','predictor']:
  item=facts[role]; p=repo/item['checkpoint_path']
  if not p.exists(): p=common/item['checkpoint_path']
  if p.exists():
   actual=digest(p)
   if actual!=item['file_sha256']: raise ValueError(f'{TASK}: {role} checkpoint hash conflict')
   model_lock['weight_byte_checks'][role]={'status':'BYTE_HASH_VERIFIED','sha256':actual,'resolved_path':str(p),'bytes':p.stat().st_size,'read_scope':'Raw byte hash only; no tensor deserialization or forward'}
  else: model_lock['weight_byte_checks'][role]={'status':'checkpoint_bytes_not_reverified','manifest':f'{ORDER}/MODEL_SELECTION_LOCK.json'}
 # Preserve seven original image objects, for visual comparison and the fallback contract.
 with ZipFile(docs/'inputs/current_manuscript.docx') as z:
  for name in z.namelist():
   if name.startswith('word/media/'):
    dest=docs/'original_media/media'/Path(name).name; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(z.read(name))
 # Only two precisely bound real image pairs are resolved. No source dataset scan.
 queue=read_csv(repo/'results/cai_agent_v3/new_protocol/candidate_queue.csv',unique=['specimen_key'])
 encoder_root=Path(read_json(repo/'results/cai_agent_v3/new_protocol/feature_bank_manifest.json')['encoder_execution_root'])
 assets=[]
 from PIL import Image
 for row in queue:
  if row['specimen_key'] not in facts['case_keys'][:2]: continue
  key=row['specimen_key']; image_record={'specimen_key':key,'status':'RESOLVED','source_dataset_url':f"https://data.mendeley.com/datasets/{row['dataset_id']}/1",'license':'CC BY 4.0','license_verified_utc':'2026-10-21','attribution':'Saki Hasebe, Ryo Higuchi, Tomohiro Yokozeki and Shin-ichi Takeda, Mendeley Data, version 1 (2022)'}
  for field,hashfield,label in [('registered_cscan_crop_path','registered_cscan_crop_sha256','cscan'),('impacted_surface_path','surface_sha256','surface')]:
   p=encoder_root/row[field]
   if not p.exists() or digest(p)!=row[hashfield]: image_record['status']='OPTIONAL_CSCAN_OVERLAY_UNRESOLVED'; continue
   if label=='cscan' and Image.open(p).size!=(int(row['registered_cscan_width_px']),int(row['registered_cscan_height_px'])): raise ValueError(f'{TASK}: image geometry mismatch {key}')
   target=paper/'figure_data/assets'/f'{key.replace(":","_")}_{label}.png'; target.parent.mkdir(parents=True,exist_ok=True)
   if label=='surface':
    # This matches the frozen CLOCKWISE_90_ONCE_MAX_EDGE_1024 preprocessing.
    im=Image.open(p).convert('RGB').transpose(Image.Transpose.ROTATE_270); im.thumbnail((1024,1024),Image.Resampling.LANCZOS); im.save(target)
   else: shutil.copyfile(p,target)
   record(p,f'T07 Fig6 {key} {label}')
   image_record[label]={'source_path':str(p),'source_sha256':row[hashfield],'output_path':str(target.relative_to(paper)),'native_size':list(Image.open(p).size),'transform':'clockwise90; maxedge1024' if label=='surface' else 'existing registered crop; no added rotation'}
  assets.append(image_record)
 # Read exactly the six N_NATIVE trajectories and initial archived maps.
 case_data=[]
 import numpy as np
 for key in facts['case_keys']:
  stem=key.replace(':','_'); p=repo/DIAG/'trajectories'/stem/'N_NATIVE.json'; record(p,f'T05/S5 {key} saved NO_VLM actions')
  raw=read_json(p); item={k:raw[k] for k in ['specimen_key','actions','costs','predictions_mpa','target_mpa']}; item['initial_logits']=raw['trace'][0]['raw_logits']
  states=repo/DIAG/'states'/stem
  for d in sorted(states.iterdir()):
   metadata=read_json(d/'metadata.json')
   if metadata['action_count']!=0: continue
   item['diagnostics']={}
   for fname in ['attention.npz','surface_attribution.npz']:
    p=d/'N'/fname; record(p,f'S5 {key} archived {fname}')
    with np.load(p,allow_pickle=False) as archive:
     keep=('query_to_cells','selected_action') if fname=='attention.npz' else ('target_actions','total_l1','target_labels')
     item['diagnostics'][fname]={k:archive[k].tolist() for k in keep}
   break
  case_data.append(item)
 write_json(paper/'figure_data/cases.json',case_data); write_json(paper/'figure_data/assets.json',assets)
 # Existing paper-level conditional intervals are retained, never redrawn.
 saved=common/'docs/1008/aei_cscan_v6/evidence/legacy_tables/paired_contrasts.csv'
 if saved.exists():
  record(saved,'T05/S3 saved NO_VLM paired intervals; local writing-evidence table',original=False)
  dest=paper/'tables/paired_contrasts_saved.csv'; dest.parent.mkdir(exist_ok=True); shutil.copyfile(saved,dest)
 bindings=dict(task_id=TASK,evidence_sha=SHA,execution_base_sha=git(repo,'rev-parse','HEAD'),branch=git(repo,'branch','--show-current'),repo=str(repo),packet=str(packet),records=records,read_only_research=True,activities={k:0 for k in ['optimizer_updates','research_model_forwards','new_bootstrap_draws','new_seeds','new_test_evaluation','new_checkpoint_selection']})

 for role in ['actor','predictor']: model_lock[role].update(forward_allowed=False,loaded=False)
 write_json(art/'SOURCE_BINDINGS.json',bindings); write_json(art/'MODEL_IDENTITY_LOCK.json',model_lock)
 return bindings


def normalize_order(rows,source_blob):
 result=[]
 for raw in rows:
  row=dict(raw)
  for name in ['estimate_mpa','ci_low_mpa','ci_high_mpa','difference_definition']:
   row.setdefault('source_'+name,raw[name])
  row.setdefault('source_blob_sha',source_blob)
  metric=row['metric']; direction=row['difference_definition']
  if re.fullmatch(r'weighted_stage_[1-4]_mpa',metric):
   if direction=='COMPARATOR_MINUS_NATIVE_POSITIVE_FAVORS_NATIVE':
    row['estimate_mpa']=-float(raw['estimate_mpa']); row['ci_low_mpa']=-float(raw['ci_high_mpa']); row['ci_high_mpa']=-float(raw['ci_low_mpa'])
    row['difference_definition']='NATIVE_MINUS_COMPARATOR_POSITIVE_FAVORS_NATIVE'
   elif direction!='NATIVE_MINUS_COMPARATOR_POSITIVE_FAVORS_NATIVE': raise ValueError(f'{TASK}: unknown contribution direction {direction}')
  elif direction!='COMPARATOR_MINUS_NATIVE_POSITIVE_FAVORS_NATIVE': raise ValueError(f'{TASK}: unknown error direction {direction}')
  if float(row['ci_low_mpa'])>float(row['ci_high_mpa']): raise ValueError(f'{TASK}: reversed interval {metric}')
  result.append(row)
 return result


def held(costs,predictions,cap): return predictions[bisect.bisect_right(costs,cap)-1]
def left_area(costs,predictions,target,budget):
 points=[0.,*[x for x in costs if 0<x<budget],budget]
 return sum((b-a)*abs(held(costs,predictions,a)-target) for a,b in zip(points,points[1:]))/budget

def first_quality(costs,errors,target):
 hits=[i for i,e in enumerate(errors) if e<=target]
 if not hits: return {'cost':None,'recrosses':None}
 i=hits[0]; return {'cost':costs[i],'recrosses':any(x>target for x in errors[i+1:])}
def relative_saving(main,control): return None if main is None or control in (None,0) else 1-main/control

def prepare_evidence(paper,art):
 repo,paper,art=validate_roots(paper,art); bindings=read_json(art/'SOURCE_BINDINGS.json')
 for item in bindings['records']:
  p=Path(item['path']); p=p if p.is_absolute() else repo/p
  if digest(p)!=item['sha256']: raise ValueError(f'{TASK}: bound source changed {p}')
 tables=read_json(repo/DOCS/'original_tables.json')
 summary=read_csv(repo/EVID/'method_summary.csv',['method','endpoint_mae_mpa'],['method'])
 selected=[next((r for r in summary if r['method']==m),None) for m in METHODS]
 if any(r is None for r in selected): raise ValueError(f'{TASK}: missing main method')
 full=read_json(repo/EVID/'full_scan_reference.json')
 t3=[]
 for label,row in zip(LABELS,selected): t3.append(dict(strategy=label,method=row['method'],A=row['area_mpa'],early_A=row['early_area_mpa'],MAE=row['endpoint_mae_mpa'],RMSE=row['endpoint_rmse_mpa'],R2=row['endpoint_r2']))
 t3.append(dict(strategy='Full-input reference (area 1.0)',method='FULL_SCAN_P_ALL',A='',early_A='',MAE=full['mae'],RMSE=full['rmse'],R2=full['r2']))
 write_csv(paper/'tables/table3.csv',t3)
 for table in tables:
  if table['table']==3: continue
  cells=table['rows']
  if table['table']==1:
   for r in cells[1:-1]:
    if display_names(r[0])!=r[1].split(' / ')[0]:raise ValueError('Laminate name differs from the supplied source mapping')
   cells=[cells[0][1:]]+[[r[1] or r[0],*r[2:]] for r in cells[1:]]
  if table['table']==5:
   cells=[['Laminate','n','Proposed A','Proposed MAE','Gain in A','Gain in MAE']]+[[display_names(r[0]),*r[1:]] for r in cells[1:]]
  write_csv(paper/f"tables/table{table['table']}.csv",[dict(zip(cells[0],r)) for r in cells[1:]],cells[0])
 raw=read_csv(repo/ORDER/'order_paired_contrasts.csv',unique=['contrast','metric'])
 blob=git(repo,'rev-parse',f'{SHA}:{ORDER}/order_paired_contrasts.csv')
 normalized=normalize_order(raw,blob); write_csv(paper/'tables/order_paired_contrasts_normalized.csv',normalized)
 changed=sum(float(a['estimate_mpa'])!=float(b['estimate_mpa']) for a,b in zip(raw,normalized))
 if changed!=8: raise ValueError(f'{TASK}: expected 8 direction corrections, got {changed}')
 closure={}
 for contrast in dict.fromkeys(r['contrast'] for r in normalized):
  local=[r for r in normalized if r['contrast']==contrast]
  area=next(float(r['estimate_mpa']) for r in local if r['metric']=='area_mpa')
  gain=sum(float(r['estimate_mpa']) for r in local if r['metric'].startswith('weighted_stage_'))
  if abs(area-gain)>1e-8: raise ValueError(f'{TASK}: stage-area closure failed {contrast}')
  closure[contrast]=abs(area-gain)
 data={'method_summary':selected,'full_reference':full,'order_contrasts':normalized}
 for name in ['stage_paired_contrasts.csv','stage_summary.csv','cohort_manifest.csv','same_cap_metrics.csv','matched_quality.csv','quality_targets.csv','order_summary.csv','cost_error_curves.csv.gz','archived_cohort_episodes.csv.gz','reorder_trajectories.csv.gz']:
  data[name.split('.')[0]]=read_csv(repo/ORDER/name)
 for name in ['same_cost_metrics_by_domain.csv','full_scan_predictions.csv']:
  data[name.split('.')[0]]=read_csv(repo/EVID/name)
 write_json(paper/'figure_data/frozen_data.json',data)
 # Deterministic verification of the six already recorded action paths.
 ranking=[]
 for case in read_json(paper/'figure_data/cases.json'):
  cohort=next(r for r in data['cohort_manifest'] if r['specimen_key']==case['specimen_key'])
  costs=[int(v)/int(cohort['total_pixels']) for v in cohort['cell_pixel_counts'].split(';')]; seen=set()
  for t,action in enumerate(case['actions']):
   feasible=[i for i in range(64) if i not in seen and case['costs'][t]+costs[i]<=.25+1e-12]
   initial=max(feasible,key=lambda i:case['initial_logits'][i])
   ranking.append(dict(specimen_key=case['specimen_key'],decision=t+1,actual_cell=action,initial_feasible_top_cell=initial,departure=int(action!=initial))); seen.add(action)
 if len(ranking)!=94 or sum(r['departure'] for r in ranking)!=9: raise ValueError(f'{TASK}: six-case ranking audit differs from frozen finding')
 write_csv(paper/'tables/S5_initial_feasible_ranking.csv',ranking)

 for name in ['cohort_manifest','same_cap_metrics','matched_quality','quality_targets','stage_paired_contrasts','stage_summary','order_summary']:
  write_csv(paper/f'tables/{name}.csv',data[name])
 write_json(art/'DIRECTION_CORRECTION.json',{'task_id':TASK,'source_blob':blob,'source_sha256':digest(repo/ORDER/'order_paired_contrasts.csv'),'changed_contribution_rows':changed,'unchanged_error_rows':8,'closure_residual_mpa':closure,'source_unchanged':True})
 return data
