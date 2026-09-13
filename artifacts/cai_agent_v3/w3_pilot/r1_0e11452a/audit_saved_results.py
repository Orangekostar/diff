"""Independent saved-trajectory arithmetic and bounded W3 evidence audit, no forward."""
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import cmc_bbdm
ROOT=Path(__file__).resolve().parents[4]
cmc_bbdm.__path__.append(str(ROOT/'src/cmc_bbdm'))
from cmc_bbdm.cai_agent_v3.feature_bank import load_feature_bank
from cmc_bbdm.cai_agent_v3.actor_training import _state_dict_sha256
from cmc_bbdm.cai_agent_v3.actor_selection import inspect_actor_archive
from cmc_bbdm.cai_agent_v3.files import write_json
from cmc_bbdm.cai_agent_v3.w3_pilot import ART, RUN, DATA


def read(path):
    with gzip.open(path,'rt') as f:return list(csv.DictReader(f))


def vals(text,kind=float):return [kind(x) for x in text.split(';') if x]


def oracle(costs,predictions,y,B):
    total=0.
    for i,left in enumerate(costs):
        if left>=B:break
        right=min(costs[i+1] if i+1<len(costs) else B,B)
        total+=(right-left)*abs(predictions[i]-y)
    return total/B


bank=load_feature_bank(project_root=ROOT)
key_index={k:i for i,k in enumerate(bank.specimen_keys)}
features={r['specimen_key']:r for r in csv.DictReader((ROOT/DATA/'vlm_actor_features_fit.csv').open())}
valid={bank.specimen_keys[i] for i in bank.indices('VALID')}
assert np.isnan(bank.targets_mpa[bank.indices('TEST')]).all()
maximum_difference=0.


def verify_episodes(episodes):
    global maximum_difference
    grouped=defaultdict(list)
    for row in episodes:
        key=row['specimen_key'];assert key in valid;i=key_index[key]
        y=float(row['target_mpa']);assert y==float(bank.targets_mpa[i])
        cells=vals(row['cells'],int);cost=vals(row['costs']);pred=vals(row['predictions_mpa'])
        assert len(cost)==len(pred)==len(cells)+1 and cost[0]==0
        assert len(cells)==len(set(cells)) and len(cells)>0
        h,w=bank.native_shapes[i];h=int(h);w=int(w);total=h*w
        dy=np.diff(np.rint(np.linspace(0,h,9)).astype(int));dx=np.diff(np.rint(np.linspace(0,w,9)).astype(int));pixels=np.outer(dy,dx).ravel()
        assert cost[-1]<=.25+1e-12
        np.testing.assert_allclose(cost[1:],np.cumsum(pixels[cells])/total,rtol=0,atol=1e-12)
        assert vals(row['new_pixels'],int)==pixels[cells].tolist()
        trace=json.loads(row['execution_trace']);assert len(trace)==len(cells)
        indicator=np.array(vals(features[key]['region_indicator']),np.float32)
        confidence=np.array(vals(features[key]['confidence']),np.float32)
        is_fixed=row['actor_state_dict_sha256']=='FIXED_ORDER'
        use_vlm=row['method'] in ('VLM_SPATIAL_FEEDBACK','VLM_SPATIAL_OPEN_LOOP','VLM_MEAN_FEEDBACK')
        for step,entry in enumerate(trace):
            legal=np.ones(64,bool);legal[cells[:step]]=False;legal &= (cost[step]+pixels/total<=.25+1e-12)
            assert entry['actor_call_index']==(None if is_fixed else step+1)
            assert entry['action_index']==step+1 and entry['cell']==cells[step]
            assert entry['visible_cells_before']==cells[:step]
            assert entry['environment_legal']==''.join('1' if x else '0' for x in legal)
            proposed=legal.copy();reason='FIXED_ORDER'
            if not is_fixed:
                if step>0:reason='C0_RELEASED_AFTER_FIRST_ACTION'
                elif not use_vlm:reason='METHOD_DOES_NOT_USE_VLM_C0'
                elif features[key]['vlm_available']!='True':reason='VLM_UNAVAILABLE'
                elif features[key]['no_reliable_cue']=='True':reason='VLM_NO_RELIABLE_CUE'
                else:
                    eligible=(indicator>0)&(confidence>=np.float32(2/3))
                    if not eligible.any():reason='NO_MEDIUM_OR_HIGH_CONFIDENCE' if (indicator>0).any() else 'NO_VLM_REGION_CANDIDATES'
                    else:
                        restricted=legal&eligible&(confidence==confidence[eligible].max())
                        if restricted.any():proposed=restricted;reason='HIGHEST_RELIABLE_CONFIDENCE_C0'
                        else:reason='C0_NO_AFFORDABLE_LEGAL_CELL'
            assert entry['proposal_legal']==''.join('1' if x else '0' for x in proposed)
            assert entry['c0_reason']==reason and proposed[cells[step]]
            assert entry['before_prediction_mpa']==pred[step] and entry['after_prediction_mpa']==pred[step+1]
            assert entry['new_pixels']==int(pixels[cells[step]]) and entry['cumulative_pixels']==sum(pixels[cells[:step+1]])
            assert entry['before_cost']==cost[step] and entry['after_cost']==cost[step+1]
        remaining=np.ones(64,bool);remaining[cells]=False
        assert not (remaining&(cost[-1]+pixels/total<=.25+1e-12)).any()
        A=oracle(cost,pred,y,.25);early=oracle(cost,pred,y,.0625);J=A+.25*abs(pred[-1]-y)
        for field,value in [('left_error_area_mpa',A),('early_left_error_area_mpa',early),('trajectory_objective_mpa',J),('final_error_mpa',abs(pred[-1]-y))]:
            difference=abs(float(row[field])-value);maximum_difference=max(maximum_difference,difference)
            assert difference<1e-12,(key,field,difference)
        grouped[(row['dataset_id'],key)].append(A)
    domains=defaultdict(list)
    for (domain,key),scores in grouped.items():domains[domain].append(float(np.mean(scores)))
    return float(np.mean([np.mean(v) for v in domains.values()]))


mode=sys.argv[1]
output=ROOT/RUN
fixed=read(output/'fixed_episodes.csv.gz')
assert len(fixed)==400
fixed_reports={}
for method in ('CENTER_FIRST','GEOMETRY_SPREAD','SERPENTINE','RANDOM'):
    group=[r for r in fixed if r['method']==method];assert {r['specimen_key'] for r in group}==valid
    assert len(group)==(250 if method=='RANDOM' else 50)
    fixed_reports[method]=verify_episodes(group)
report={'fixed_scores':fixed_reports,'model_forward_calls':0,'optimizer_updates':0}
if mode=='final':
    manifests=json.loads((output/'actor_manifests.json').read_text())['actor_manifests']
    assert len(manifests)==5
    authorization=json.loads((ROOT/'docs/cai/w3_valid_pilot/W3_PILOT_AUTHORIZATION.json').read_text())
    specifications={m['method']:m for m in authorization['models']}
    candidate_count=0;models=[]
    for m in manifests:
        path=ROOT/m['selection_archive'];job=json.loads((output/'jobs'/f"{m['method']}.json").read_text())
        archive=inspect_actor_archive(path,environment=job['environment'])
        specification=specifications[m['method']]
        assert m['training_seed']==specification['training_seed'] and m['seed_panel']==1
        assert archive['max_updates']==specification['max_updates'] and archive['validation_interval']==250 and archive['patience']==4
        assert m['updates_completed']<=specification['max_updates']
        assert m['initial_state_dict_sha256']==archive['metadata']['initial_state_dict_sha256']
        best=None
        for candidate in archive['candidates']:
            group=read(path/candidate['episodes']);assert len(group)==50 and {r['specimen_key'] for r in group}==valid
            payload=torch.load(path/candidate['checkpoint'],map_location='cpu',weights_only=False)
            state_hash=_state_dict_sha256(payload['state_dict'])
            assert all(r['actor_state_dict_sha256']==state_hash for r in group)
            assert payload['manifest']['environment']==job['environment'] and payload['manifest']['run_id']==job['run_id']
            score=verify_episodes(group)
            assert abs(score-candidate['metrics']['left_error_area_mpa'])<1e-12
            if best is None or score<best[1]-1e-12:best=(candidate['update'],score,candidate)
            candidate_count+=1
        assert best[0]==m['selected_update'] and abs(best[1]-m['validation_area_mpa'])<1e-12
        final=torch.load(ROOT/m['checkpoint_path'],map_location='cpu',weights_only=False)
        winner=torch.load(path/best[2]['checkpoint'],map_location='cpu',weights_only=False)
        assert all(torch.equal(v,winner['state_dict'][k]) for k,v in final['state_dict'].items())
        latest=torch.load(path/'latest_training_state.pt',map_location='cpu',weights_only=False)
        assert latest['update']==m['updates_completed'] and latest['best_update']==m['selected_update']
        assert latest['run_id']==job['run_id'] and latest['environment']==job['environment']
        assert latest['optimizer']['state'] and len(latest['torch_cuda_rng'])==1
        assert (ROOT/m['selected_episodes']).read_bytes()==(path/best[2]['episodes']).read_bytes()
        models.append({'method':m['method'],'candidate_count':len(archive['candidates']),'selected_update':m['selected_update'],'actual_updates':m['updates_completed'],'weight_reload_equal':True})
    episodes=read(output/'policy_validation_episodes.csv.gz');assert len(episodes)==650
    assert len({r['specimen_key'] for r in episodes})==50
    zeros=defaultdict(set)
    for row in episodes:zeros[row['specimen_key']].add(vals(row['predictions_mpa'])[0])
    assert all(len(v)==1 for v in zeros.values()), 'methods do not share the same initial predictor state'
    table={r['method']:r for r in csv.DictReader((output/'policy_pilot_metrics.csv').open())}
    expected_scores={**fixed_reports,**{m['method']:m['validation_area_mpa'] for m in manifests}}
    assert set(table)==set(expected_scores)
    for name,score in expected_scores.items():
        assert abs(float(table[name]['validation_area_mpa'])-score)<1e-12
        group=[r for r in episodes if r['method']==name]
        by_specimen=defaultdict(list)
        for r in group:by_specimen[r['specimen_key']].append(vals(r['predictions_mpa'])[-1]-float(r['target_mpa']))
        mae=np.mean([np.mean(np.abs(v)) for v in by_specimen.values()])
        rmse=np.sqrt(np.mean([np.mean(np.square(v)) for v in by_specimen.values()]))
        assert abs(float(table[name]['endpoint_pooled_mae_mpa'])-mae)<1e-12
        assert abs(float(table[name]['endpoint_pooled_rmse_mpa'])-rmse)<1e-12
    nonadaptive=['CENTER_FIRST','GEOMETRY_SPREAD','SERPENTINE','RANDOM','LEARNED_STATIC_TRUE']
    best=min(nonadaptive,key=lambda n:(expected_scores[n],n))
    gate=json.loads((output/'pilot_gate.json').read_text())
    expected_gate=expected_scores['VLM_SPATIAL_FEEDBACK']<=.98*expected_scores[best] and expected_scores['VLM_SPATIAL_FEEDBACK']<expected_scores['VLM_SPATIAL_OPEN_LOOP']
    assert gate['best_nonadaptive']==best and gate['pilot_expansion_condition_met']==expected_gate
    assert gate['automatically_execute_next_stage'] is False
    report.update(models=models,candidate_count=candidate_count,final_episode_count=650,final_table_verified=True,pilot_gate_independently_verified=True)
report.update(status='SAVED_EVIDENCE_PASS',max_metric_absolute_difference=maximum_difference)
write_json(ROOT/ART/f'saved_evidence_{mode}.json',report)
print(json.dumps(report,indent=2))
