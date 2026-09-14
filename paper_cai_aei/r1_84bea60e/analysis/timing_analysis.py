"""One saved-event pass. Standard library only; immutable evidence inputs."""
import csv
import hashlib
import json
import runpy
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
EVID = ROOT / 'results/cai_agent_v3/paper_evidence/r1_e2a11154'
REFERENCE = ROOT / 'docs/cai/aei_manuscript/reference/timing_identity_reference.py'
_ref = runpy.run_path(str(REFERENCE))
decompose = _ref['decompose']
bin_terms = _ref['bin_terms']


def aggregate(rows, fields):
    """Episode totals -> specimen repeat mean -> domain mean -> equal domains."""
    specimen = defaultdict(list)
    for row in rows:
        specimen[(row['method'], row['dataset_id'], row['specimen_key'])].append(row)
    domain = defaultdict(list)
    for (method, dataset, _), values in specimen.items():
        domain[(method, dataset)].append({f:mean(r[f] for r in values) for f in fields})
    method_rows = defaultdict(list)
    for (method, _), values in domain.items():
        method_rows[method].append({f:mean(r[f] for r in values) for f in fields})
    return {method:{f:mean(r[f] for r in values) for f in fields}
            for method, values in method_rows.items()}


def read(name):
    with (EVID/name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    if (OUT/'timing_identity_checks.json').exists():
        raise SystemExit('Saved analysis exists; refusing to repeat or overwrite')
    index = read('frozen_episode_index.csv')
    events = read('acquisition_events.csv')
    group = defaultdict(list)
    key = lambda r:(r['method'],r['specimen_key'],r['run'])
    for row in events:
        group[key(row)].append(row)
    assert set(group) == {key(r) for r in index}
    episode = []
    max_residual = 0.
    for entry in index:
        rows = sorted(group[key(entry)], key=lambda r:int(r['action_index']))
        assert len(rows) == int(entry['action_count'])
        costs = [0.]
        errors = [float(rows[0]['error_before_mpa'])]
        for i,r in enumerate(rows,1):
            assert int(r['action_index']) == i
            assert abs(float(r['before_cost'])-costs[-1]) < 1e-12
            assert abs(float(r['error_before_mpa'])-errors[-1]) < 1e-9
            costs.append(float(r['after_cost']))
            errors.append(float(r['error_after_mpa']))
        assert abs(costs[-1]-float(entry['final_cost'])) < 1e-12
        result = decompose(costs,errors)
        bins = bin_terms(costs,result['terms'])
        max_residual = max(max_residual,abs(result['area_direct']-result['area_from_terms']))
        episode.append(dict(method=entry['method'],dataset_id=entry['dataset_id'],
                            specimen_key=entry['specimen_key'],e0=errors[0],
                            area=result['area_direct'], total=sum(result['terms']),
                            **{f'stage{i+1}':v for i,v in enumerate(bins)}))
    totals = aggregate(episode,['e0','area','total','stage1','stage2','stage3','stage4'])
    saved = {r['method']:float(r['area']) for r in read('method_summary.csv')}
    saved_residual = {method:values['area']-saved[method] for method,values in totals.items()}
    main_values = totals['VLM_SPATIAL_FEEDBACK']
    pair_checks = {}
    for method,v in totals.items():
        pair_checks[method] = ((v['e0']-main_values['e0']) + main_values['total']-v['total']
                               - (saved[method]-saved['VLM_SPATIAL_FEEDBACK']))
    assert max_residual < 1e-9
    assert max(map(abs,saved_residual.values())) < 1e-9
    assert max(map(abs,pair_checks.values())) < 1e-9
    output=[]
    edges=[0,.0625,.125,.1875,.25]
    for method,v in totals.items():
        for i in range(4):
            output.append(dict(method=method,stage=i+1,lower=edges[i],upper=edges[i+1],
                               contribution_mpa=v[f'stage{i+1}'],initial_error_mpa=v['e0'],
                               total_contribution_mpa=v['total'],area_mpa=v['area']))
    with (OUT/'timing_contributions.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(output[0]))
        writer.writeheader();writer.writerows(output)
    checks=dict(status='PASS', scope='DESCRIPTIVE_ALGEBRAIC_DECOMPOSITION_OF_FROZEN_VALID',
                episodes=len(index),events=len(events),methods=len(totals),
                physical_n=len({r['specimen_key'] for r in index}),
                max_episode_identity_residual=max_residual,
                aggregate_minus_saved_area=saved_residual,comparison_identity_residual=pair_checks,
                bins='right_closed_by_action_completion_cost',
                aggregation='episode_sum_then_repeat_mean_then_within_domain_specimen_mean_then_domain_equal',
                model_calls=0,bootstrap_replicates=0,optimizer_updates=0,
                inputs={name:hashlib.sha256((EVID/name).read_bytes()).hexdigest()
                        for name in ['acquisition_events.csv','frozen_episode_index.csv','method_summary.csv']})
    (OUT/'timing_identity_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))

if __name__ == '__main__':
    main()
