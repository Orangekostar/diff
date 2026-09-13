"""Bounded W2 audit from saved predictions; no model inference or optimization."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import cmc_bbdm

ROOT = Path(__file__).resolve().parents[4]
cmc_bbdm.__path__.append(str(ROOT / 'src/cmc_bbdm'))
from cmc_bbdm.cai_agent_v3.checkpoint_selection import inspect_archive
from cmc_bbdm.cai_agent_v3.feature_bank import load_feature_bank
from cmc_bbdm.cai_agent_v3.files import write_csv, write_json
from cmc_bbdm.cai_agent_v3.w2_replay import ART, RUN

parser = argparse.ArgumentParser()
parser.add_argument('stage', choices=['A', 'B'])
args = parser.parse_args()
run = ROOT / RUN
art = ROOT / ART
bank = load_feature_bank(project_root=ROOT)
binding = json.loads((run / 'input_reuse_manifest.json').read_text())['validation_identity']
assert np.isnan(bank.targets_mpa[bank.indices('TEST')]).all()
gate = json.loads((run / ('predictor_gate.json' if args.stage == 'A' else 'oof_readiness.json')).read_text())
models = gate['candidate_manifests' if args.stage == 'A' else 'fold_manifests']
report = []
conditions = []
for manifest in models:
    name = manifest['model']
    key = name if args.stage == 'A' else f"fold{manifest['fold']}"
    archive_path = ROOT / manifest['selection_archive']
    archive = json.loads((archive_path / 'selection.json').read_text())
    inspect_archive(archive_path, validation=binding)
    expected_updates = list(range(250, manifest['updates_completed'] + 1, 250))
    assert [c['update'] for c in archive['candidates']] == expected_updates
    assert archive['status'] == 'COMPLETE'
    metadata = archive['run_metadata']
    fit_keys = set(metadata['fit_specimen_keys'])
    fit = np.array([i for i, k in enumerate(bank.specimen_keys) if k in fit_keys])
    assert len(fit) == manifest['fit_physical_n']
    assert all(bank.splits[i] == 'TRAIN' for i in fit)
    targets = bank.targets_mpa[fit].astype(np.float64)
    constants = metadata['constants']
    valid_targets = bank.targets_mpa[bank.indices('VALID')].astype(np.float64)
    expected_constants = {
        'train_target_mean_mpa': targets.mean(),
        'train_target_median_mpa': np.median(targets),
        'train_target_std_mpa': targets.std(),
        'train_median_full_mae_mpa': np.abs(valid_targets - np.median(targets)).mean(),
        'train_mean_full_mse_mpa2': ((valid_targets - targets.mean()) ** 2).mean(),
    }
    for field, value in expected_constants.items():
        np.testing.assert_allclose(constants[field], value, rtol=0, atol=1e-12)
    if args.stage == 'B':
        fold = manifest['fold']
        folds = list(csv.DictReader((run / 'oof_fold_manifest.csv').open()))
        query = {r['specimen_key'] for r in folds if int(r['fold']) == fold}
        expected_fit = {r['specimen_key'] for r in folds if int(r['fold']) != fold}
        assert fit_keys == expected_fit and not query & fit_keys
        query_groups = {r['capture_group_id'] for r in folds if r['specimen_key'] in query}
        assert not query_groups & set(metadata['fit_capture_group_ids'])
        assert manifest['seed'] == 2026091211 + fold
        assert manifest['model'] == gate['selected_structure']
    else:
        assert manifest['seed'] == {'MEAN_SC':2026091201, 'SPATIAL_SC':2026091202, 'SPATIAL_C':2026091203}[name]
    best = None
    max_error = 0.
    for candidate in archive['candidates']:
        update = candidate['update']
        file = run / 'candidate_state_predictions' / f'{args.stage}_{key}' / f'update_{update:06d}.npz'
        with np.load(file) as d:
            assert set(d['specimen_indices']) == set(bank.indices('VALID'))
            assert set(d['full_specimen_indices']) == set(bank.indices('VALID'))
            assert d['costs'].dtype == np.float64
            assert d['costs'].max() <= .25 + 1e-12
            groups = defaultdict(list)
            for j, (i, route) in enumerate(zip(d['specimen_indices'], d['route_names'])):
                groups[(int(i), str(route))].append(j)
            specimen_areas = defaultdict(list)
            route_errors = defaultdict(list)
            zero_errors = {}
            for (i, route), js in groups.items():
                js = sorted(js, key=lambda j:int(d['state_indices'][j]))
                cost = d['costs'][js]
                errors = np.abs(d['predictions_mpa'][js].astype(np.float64) - d['targets_mpa'][js].astype(np.float64))
                assert cost[0] == 0 and np.all(np.diff(cost) > 0)
                area = (np.sum(np.diff(cost) * errors[:-1]) + (.25-cost[-1])*errors[-1]) / .25
                specimen_areas[i].append(area)
                route_errors[route].append(errors[-1])
                zero_errors.setdefault(i, errors[0])
            by_domain = defaultdict(list)
            for i, areas in specimen_areas.items():
                assert len(areas) == 4
                by_domain[bank.dataset_ids[i]].append(np.mean(areas))
            errors = d['full_predictions_mpa'].astype(np.float64) - d['full_targets_mpa'].astype(np.float64)
            observed = {
                'valid_area_mpa':np.mean([np.mean(v) for v in by_domain.values()]),
                'zero_mae_mpa':np.mean(list(zero_errors.values())),
                'full_mae_mpa':np.abs(errors).mean(),
                'full_mse_mpa2':(errors**2).mean(),
                'full_rmse_mpa':np.sqrt((errors**2).mean()),
                'full_r2':1-(errors**2).sum()/((d['full_targets_mpa'].astype(np.float64)-d['full_targets_mpa'].astype(np.float64).mean())**2).sum(),
                'center_endpoint_mae_mpa':np.mean(route_errors['CENTER_FIRST']),
                'geometry_endpoint_mae_mpa':np.mean(route_errors['GEOMETRY_SPREAD']),
            }
            for field, value in observed.items():
                difference = abs(value - candidate['metrics'][field])
                max_error = max(max_error, difference)
                np.testing.assert_allclose(value, candidate['metrics'][field], rtol=0, atol=1e-12)
        if best is None or candidate['metrics']['valid_area_mpa'] < best['metrics']['valid_area_mpa'] - 1e-12:
            best = candidate
    assert best['update'] == manifest['selected_update']
    for field, value in best['metrics'].items():
        np.testing.assert_allclose(value, manifest['metrics'][field], rtol=0, atol=1e-12)
    selected = torch.load(ROOT / manifest['checkpoint_path'], map_location='cpu', weights_only=False)
    original = torch.load(archive_path / best['checkpoint'], map_location='cpu', weights_only=False)
    assert all(torch.equal(v, original['state_dict'][k]) for k,v in selected['state_dict'].items())
    latest = torch.load(archive_path/'latest_training_state.pt', map_location='cpu', weights_only=False)
    assert latest['update'] == manifest['updates_completed'] and latest['best_update'] == manifest['selected_update']
    assert latest['validation_identity'] == binding and len(latest['torch_cuda_rng']) == 1
    job = json.loads((run/'jobs'/f'{args.stage}_{key}.json').read_text())
    assert latest['run_id'] == job['run_id']
    assert latest['optimizer']['state']
    m = manifest['metrics']
    checks = {
        'full_mae_below_median': m['full_mae_mpa'] < constants['train_median_full_mae_mpa'],
        'full_mse_below_mean': m['full_mse_mpa2'] < constants['train_mean_full_mse_mpa2'],
        'full_at_least_2pct_better_than_zero':m['full_mae_mpa'] <= .98*m['zero_mae_mpa'],
        'center_below_zero':m['center_endpoint_mae_mpa'] < m['zero_mae_mpa'],
        'geometry_below_zero':m['geometry_endpoint_mae_mpa'] < m['zero_mae_mpa'],
    }
    ready = all(checks.values())
    if args.stage == 'B':
        assert ready == (manifest['readiness_status'] == 'PREDICTOR_READY')
    conditions.append({'stage':args.stage,'model':name,'fold':manifest.get('fold',''),'fit_n':len(fit),'query_n':manifest.get('query_physical_n',50),'query_scope':'VALID' if args.stage=='A' else 'TRAIN_OOF','valid_n':50,'selected_update':manifest['selected_update'],'actual_updates':manifest['updates_completed'],'uses_surface':name!='SPATIAL_C','scale_lower_bound':1.0,'archive_complete':True,'ready':ready,**checks,**m})
    report.append({'model':key,'archive_candidates':len(expected_updates),'selected_update':manifest['selected_update'],'max_metric_absolute_difference':max_error,'weight_reload_equal':True,'fit_constants_verified':True,'ready':ready})
if args.stage == 'A':
    eligible = [m for m, c in zip(models, conditions) if c['ready']]
    if eligible:
        score = min(m['metrics']['valid_area_mpa'] for m in eligible)
        tied = [m for m in eligible if abs(m['metrics']['valid_area_mpa']-score)<=1e-8]
        expected = min(tied,key=lambda m:(m['parameter_count'],m['model']))['model']
    else:
        expected = None
    assert gate['selected_p_all'] == expected
else:
    assert (gate['status']=='REWARD_MODELS_READY') == all(c['ready'] for c in conditions)
write_csv(run/f'preparation_conditions_{args.stage}.csv', conditions)
result = {'status':'SAVED_EVIDENCE_PASS','stage':args.stage,'models':report,'test_predictions':0,'model_forward_calls':0,'optimizer_updates':0}
write_json(art/f'saved_evidence_review_{args.stage}.json',result)
print(json.dumps(result,indent=2))
