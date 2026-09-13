"""Bounded independent review of derived tables; no models or new bootstrap draws."""
from pathlib import Path
import csv
import json
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
RUN=ROOT/'results/cai_agent_v3/paper_evidence/r1_e2a11154'
ART=Path(__file__).resolve().parent
MAIN='VLM_SPATIAL_FEEDBACK'
read=lambda n:list(csv.DictReader((RUN/n).open()))
index=read('frozen_episode_index.csv');assert len(index)==650
assert len({r['specimen_key'] for r in index})==50 and len({r['capture_group_id'] for r in index})==48
full=read('full_scan_predictions.csv');assert len(full)==50 and {r['specimen_key'] for r in full}=={r['specimen_key'] for r in index}
y=np.array([float(r['target_mpa']) for r in full]);p=np.array([float(r['full_prediction_mpa']) for r in full]);err=p-y
ref=json.loads((RUN/'full_scan_reference.json').read_text())
assert ref['area'] is None and ref['cost_support']==[1.0]
for name,value in [('mae',np.abs(err).mean()),('rmse',np.sqrt(np.square(err).mean())),('r2',1-np.square(err).sum()/np.square(y-y.mean()).sum())]:assert abs(value-ref[name])<1e-12
same=read('same_cost_metrics.csv'); assert len(same)==46
assert len([r for r in same if r['method']=='FULL_SCAN_P_ALL'])==1
assert next(r for r in same if r['method']=='FULL_SCAN_P_ALL')['area']==''
with np.load(RUN/'bootstrap_group_weights.npz',allow_pickle=False) as z:
 weights=z['weights'];keys=list(z['specimen_keys']);domains=z['domains'];groups=z['capture_groups']
assert weights.shape==(5000,50)
for g in set(groups):assert np.all(weights[:,groups==g]==weights[:,groups==g][:,:1])
for d in set(domains):
 representatives=[np.flatnonzero(groups==g)[0] for g in set(groups[domains==d])]
 assert np.all(weights[:,representatives].sum(axis=1)==len(representatives))
losses=read('same_cost_paired_losses.csv');summary=read('same_cost_paired_summary.csv');assert len(losses)==2000 and len(summary)==40
max_ci_difference=0.
for row in summary:
 group={r['specimen_key']:float(r['gain_mpa']) for r in losses if r['comparator']==row['comparator'] and r['budget']==row['budget']}
 values=np.array([group[k] for k in keys]);draws=(weights*values[None,:]).sum(axis=1)/weights.sum(axis=1)
 assert abs(values.mean()-float(row['mae_gain_mpa']))<1e-12
 for k,v in zip(['ci_low','ci_high'],np.quantile(draws,[.025,.975])):max_ci_difference=max(max_ci_difference,abs(float(row[k])-v))
 assert row['simultaneous']=='False' and row['inference_scope']=='POSTHOC_SELECTED_VALID_CONDITIONAL'
area=read('paired_area_losses.csv')
for row in read('mechanism_area_intervals.csv'):
 field='early_area_gain' if row['estimand']=='early_area' else 'area_gain'
 by_key={r['specimen_key']:float(r[field]) for r in area if r['comparator']==row['comparator']};values=np.array([by_key[k] for k in keys])
 draws=np.zeros(5000)
 for d in sorted(set(domains)):
  mask=domains==d;draws+=(weights[:,mask]*values[mask]).sum(axis=1)/weights[:,mask].sum(axis=1)/6
 for k,v in zip(['ci_low','ci_high'],np.quantile(draws,[.025,.975])):max_ci_difference=max(max_ci_difference,abs(float(row[k])-v))
 assert float(row['ci_low'])<0<float(row['ci_high'])
assert max_ci_difference<1e-12
curves=read('cost_error_event_curves.csv');assert len(curves)==599*9
curve={}
for row in curves:curve.setdefault(('SECONDARY_EVENT_GRID',row['method']),[]).append((float(row['budget']),float(row['mae'])))
for row in same:
 if row['method']!='FULL_SCAN_P_ALL':curve.setdefault(('PRIMARY_FIVE_BUDGETS',row['method']),[]).append((float(row['budget']),float(row['mae'])))
for grid in ['PRIMARY_FIVE_BUDGETS','SECONDARY_EVENT_GRID']:curve[(grid,'FULL_SCAN_P_ALL')]=[(1.,ref['mae'])]
qmin=int(np.floor(min([float(r['mae']) for r in curves]+[ref['mae']])));qmax=int(np.ceil(max([float(r['mae']) for r in curves]+[ref['mae']])))
targets=read('quality_targets.csv');assert {int(float(r['target_mae'])) for r in targets if r['target_type']=='UNIFORM_1_MPA'}==set(range(qmin,qmax+1))
assert len([r for r in targets if r['target_type']!='UNIFORM_1_MPA'])==21
not_reached=negative=zero=0
for name in ['equal_quality_grid.csv','equal_quality_anchors.csv']:
 rows=read(name);assert len(rows)==420
 for row in rows:
  q=float(row['target_mae']);series=curve[(row['grid'],row['method'])];hits=[(i,c,v) for i,(c,v) in enumerate(series) if v<=q]
  if not hits:assert row['cost']=='' and row['status']=='NOT_REACHED_WITHIN_OBSERVED_RANGE';not_reached+=1
  else:
   i,c,v=hits[0];assert float(row['cost'])==c and float(row['attained_mae'])==v
   assert (row['later_recrosses_target']=='True')==any(val>q for _,val in series[i+1:])
  mh=[c for c,v in curve[(row['grid'],MAIN)] if v<=q]
  assert row['main_cost']=='' if not mh else float(row['main_cost'])==mh[0]
  if not hits or not mh:assert row['relative_saving']==''
  elif hits[0][1]==0:assert row['relative_saving']=='';zero+=1
  else:
   expected=1-mh[0]/hits[0][1];assert abs(float(row['relative_saving'])-expected)<1e-14;negative+=expected<0
assert not_reached>0 and zero>0 and negative>0
for stem in ['same_cost_metrics','same_cost_paired_summary','equal_quality_anchors','same_cost_metrics_by_domain']:
 md=(RUN/(stem+'.md')).read_text().splitlines();tex=(RUN/(stem+'.tex')).read_text();assert len(md)-2==len(read(stem+'.csv'))
 columns=[x.strip() for x in md[0].strip('|').split('|')]
 for row,line in zip(read(stem+'.csv'),md[2:]):
  shown=[x.strip() for x in line.strip('|').split('|')]
  for col,text in zip(columns,shown):
   raw=row.get(col,'')
   if raw=='':assert text=='NA'
   else:
    try:value=float(raw)
    except ValueError:assert text==raw
    else:assert abs(value-float(text))<=.500001e-6
  escaped=[s.replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&') for s in shown]
  assert ' & '.join(escaped) in tex
assert not subprocess.check_output(['git','diff','e2a11154','--name-only','--','results/cai_agent_v3/new_protocol','results/cai_agent_v3/w2_replay/r1_292b1c74','results/cai_agent_v3/w3_pilot/r1_0e11452a','artifacts/cai_agent_v3/w3_pilot/r1_0e11452a'])
report={'status':'PASS','full_source_mae':ref['mae'],'frozen_episode_count':650,'capture_groups':48,'same_cost_rows':46,'paired_rows':2000,'paired_summary_rows':40,'shared_bootstrap_shape':[5000,50],'max_interval_recompute_difference':max_ci_difference,'quality_grid_range':[qmin,qmax],'quality_rows_total':840,'not_reached_rows':not_reached,'negative_saving_rows':int(negative),'zero_denominator_rows':zero,'table_markdown_latex_rows_match':True,'old_outputs_unchanged':True,'new_model_forwards':0,'new_bootstrap_draws_in_review':0,'review_mode':'SELF_CHECK_WITH_EXTERNAL_NUMERIC_ORACLE'}
(ART/'SAVED_EVIDENCE_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
