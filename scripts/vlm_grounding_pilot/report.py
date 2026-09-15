"""CPU-only export from saved answers; no localization GT or model execution."""
import json
import textwrap
from collections import Counter

import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use('Agg')
from common import (
    OUT,
    TASK,
    VARIANTS,
    NativeCellGrid,
    csvout,
    dump,
    parse_contract,
    perception,
    policy,
    sha,
    slug,
    vlm_perception,
)
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

font_manager.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
plt.rcParams['font.family'] = ['DejaVu Sans', 'WenQuanYi Zen Hei']


def relative(path):
    return str(path.relative_to(OUT))


def analyze(percept, case):
    parsed = perception.parse_surface_percept(json.dumps(percept)) if percept else None
    indicator, confidence = vlm_perception._features(parsed) if parsed else (np.zeros(64, dtype=np.float32), np.zeros(64, dtype=np.float32))
    grid = NativeCellGrid.from_shape((int(case['registered_cscan_height_px']), int(case['registered_cscan_width_px'])))
    legal = grid.legal_mask(set(), endpoint_budget=.25)
    proposal, reasons = policy.vlm_first_action_mask(legal=torch.tensor(legal[None]), use_vlm=True, action_count=torch.tensor([0]), indicator=torch.tensor(indicator[None]), confidence=torch.tensor(confidence[None]), available=torch.tensor([parsed is not None]), no_reliable=torch.tensor([parsed.no_reliable_cue if parsed else False]))
    return {'cells': np.flatnonzero(indicator).tolist(), 'indicator': indicator.tolist(), 'confidence': confidence.tolist(), 'c0': torch.nonzero(proposal[0]).flatten().tolist(), 'c0_reason': reasons[0], 'legal': np.flatnonzero(legal).tolist(), 'available': parsed is not None, 'no_cue': parsed.no_reliable_cue if parsed else None, 'action_count': 0, 'measured_cells': [], 'budget': .25}


def figures(case, variant, percept, state, values):
    idir = OUT / 'inputs' / slug(case['specimen_key'])
    odir = OUT / 'overlays' / slug(case['specimen_key'])
    odir.mkdir(parents=True, exist_ok=True)
    clean = Image.open(idir / 'clean.png')
    grid = NativeCellGrid.from_shape((clean.height, clean.width))
    regions = percept['regions'] if percept else []
    fig, ax = plt.subplots(figsize=(8, 9.8))
    fig.subplots_adjust(left=.025, right=.975, top=.92, bottom=.24)
    ax.imshow(clean, origin='upper', interpolation='none')
    for cell in grid.cells:
        ax.add_patch(Rectangle((cell.col_start-.5, cell.row_start-.5), cell.col_stop-cell.col_start, cell.row_stop-cell.row_start, fill=False, linewidth=.4, edgecolor='#aaaaaa'))
        ax.text(cell.col_start+3, cell.row_start+3, str(cell.index), fontsize=6, va='top', color='white', bbox={'facecolor': 'black', 'alpha': .5, 'pad': .4, 'edgecolor': 'none'})
    colors = ['#ff9100', '#db3da8']
    for i, region in enumerate(regions):
        for c in region['cells']:
            cell = grid.cells[c]
            ax.add_patch(Rectangle((cell.col_start-.5, cell.row_start-.5), cell.col_stop-cell.col_start, cell.row_stop-cell.row_start, fill=False, linewidth=2.4, edgecolor=colors[i]))
    ax.axis('off')
    fig.suptitle(case['specimen_key']+' | '+variant+'\n'+state, fontsize=11)
    details = ['Candidate overlay: diagnostic reference, NOT actual numbered input.']
    for i, r in enumerate(regions):
        details.extend(textwrap.wrap(f"R{i+1} ({['orange','magenta'][i]}) {r['cells']} [{r['confidence']}] cue: {r['cue']}; alternative: {r['alternative']}", width=72))
    if not regions:
        details.append('No regions' if values['available'] else 'UNAVAILABLE / invalid response, not valid no-cue')
    details.extend(textwrap.wrap(f"Diagnostic C0 ({len(values['c0'])} cells): {values['c0_reason']}; no Actor/action executed.", width=72))
    fig.text(.035, .215, '\n'.join(details), va='top', fontsize=8)
    overlay = odir / (variant + '.png')
    fig.savefig(overlay, dpi=140, facecolor='white')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(np.asarray(values['confidence']).reshape(8, 8), vmin=0, vmax=1, cmap='Blues', interpolation='nearest', origin='upper')
    for i in range(64):
        ax.text(i%8, i//8, f"{i}\n{values['confidence'][i]:.2f}", ha='center', va='center', fontsize=7, color='white' if values['confidence'][i]>.6 else 'black')
    ax.set_title(variant+' | '+state+'\ndecoded ordinal confidence\nNOT attention / damage probability', fontsize=10)
    fig.colorbar(im, ax=ax, fraction=.046)
    ordinal = odir / (variant + '_ordinal.png')
    fig.savefig(ordinal, dpi=120, bbox_inches='tight')
    plt.close(fig)
    rows = []
    for c in grid.cells:
        rows.append({'cell_id': c.index, 'row': c.row, 'column': c.column, 'x0': c.col_start, 'y0': c.row_start, 'x1_exclusive': c.col_stop, 'y1_exclusive': c.row_stop, 'indicator': values['indicator'][c.index], 'ordinal_confidence': values['confidence'][c.index], 'diagnostic_c0': c.index in values['c0'], 'legal': c.index in values['legal'], 'available': values['available']})
    features = odir / (variant + '_features_64.csv')
    csvout(features, rows)
    dump(odir / (variant + '_c0.json'), values)
    return relative(overlay), relative(ordinal), relative(features)


def compare(key, left, right, records):
    a, b = records[left], records[right]
    va, vb = a['values'], b['values']
    sa, sb = set(va['cells']), set(vb['cells'])
    valid = va['available'] and vb['available']
    union = sa | sb
    return {'specimen_key': key, 'comparison': right+'-'+left, 'left_status': a['status'], 'right_status': b['status'], 'comparison_status': 'UNAVAILABLE' if not valid else ('BOTH_EMPTY' if not union else 'AVAILABLE'), 'jaccard': len(sa & sb)/len(union) if valid and union else '', 'added_cells': json.dumps(sorted(sb-sa)) if valid else '', 'removed_cells': json.dumps(sorted(sa-sb)) if valid else '', 'ordinal_changed_cells': json.dumps([i for i in range(64) if va['confidence'][i] != vb['confidence'][i]]) if valid else '', 'ordinal_l1_change': sum(abs(x-y) for x,y in zip(va['confidence'],vb['confidence'])) if valid else '', 'c0_added': json.dumps(sorted(set(vb['c0'])-set(va['c0']))) if valid else '', 'c0_removed': json.dumps(sorted(set(va['c0'])-set(vb['c0']))) if valid else '', 'raw_text_identical': a['raw_text'] == b['raw_text'], 'regions_cue_confidence_identical': json.dumps(a['percept'], sort_keys=True) == json.dumps(b['percept'], sort_keys=True), 'interpretation': 'OUTPUT_SENSITIVITY_NOT_LOCALIZATION_ACCURACY'}


def main():
    from html_view import build_html
    torch.set_num_threads(4)
    lock = json.loads((OUT / 'experiment_lock.json').read_text())
    session = json.loads((OUT / 'gpu_session.json').read_text())
    payload = {'status': session['status'], 'cases': []}
    summary, changes, historical, human, attempt_rows = [], [], [], [], []
    for case in lock['cases']:
        key = case['specimen_key']
        idir = OUT / 'inputs' / slug(key)
        cp = {'specimen_key': key, 'split': case['split'], 'clean': relative(idir/'clean.png'), 'R0': relative(idir/'R0.png'), 'R1': relative(idir/'R1.png'), 'panels': []}
        records = {}
        for variant in [*VARIANTS, 'H00_CACHED']:
            raw_links, attempts = [], []
            if variant == 'H00_CACHED':
                path = idir / 'H00_CACHED.json'
                old = json.loads(path.read_text()) if path.exists() else None
                checked = parse_contract(old['raw_text']) if old else {'contract_valid': False}
                percept = checked['percept'] if checked['contract_valid'] else None
                state = {'status': 'HISTORICAL_CACHED' if percept else 'HISTORICAL_UNAVAILABLE'}
                if old:
                    raw_links = [relative(idir/'H00_raw.txt')]
                render_id = 'R0'
                raw = old['raw_text'] if old else ''
            else:
                directory = OUT / 'runs' / slug(key) / variant
                path = directory / 'state.json'
                if path.exists():
                    state = json.loads(path.read_text())
                    if state['status'] == 'STARTED':
                        state['status'] = 'INTERRUPTED'
                        dump(path, state)
                else:
                    job = next(j for j in lock['jobs'] if j['specimen_key']==key and j['variant']==variant)
                    state = {'specimen_key': key, 'variant': variant, 'signature': job['signature'], 'status': 'NOT_EXECUTED_RESOURCE_LIMIT', 'attempts': []}
                    dump(path, state)
                percept = state.get('final_percept')
                attempts = state['attempts']
                raw_links = [relative(directory/f'raw_{i}.txt') for i in range(1,len(attempts)+1) if (directory/f'raw_{i}.txt').exists()]
                render_id = VARIANTS[variant][1]
                raw = attempts[-1].get('text','') if attempts else ''
                attempt_rows.extend(dict(a, specimen_key=key, variant=variant, run_id=session.get('run_id')) for a in attempts)
            values = analyze(percept, case)
            overlay, ordinal, features = figures(case, variant, percept, state['status'], values)
            first_status = ('CONTRACT_VALID' if attempts[0].get('contract_valid') else 'INVALID_OR_FAILED') if attempts else 'HISTORICAL_FINAL_ONLY'
            panel = {'variant': variant, 'status': state['status'], 'overlay': overlay, 'numbered': relative(idir/(render_id+'.png')), 'regions': percept['regions'] if percept else [], 'c0': values['c0'], 'c0_reason': values['c0_reason'], 'raw_links': raw_links, 'features': features, 'ordinal': ordinal, 'first_pass_status': first_status, 'repair_dependent': state['status']=='VALID_AFTER_REPAIR', 'no_reliable_cue': values['no_cue'] if values['available'] else 'UNAVAILABLE', 'first_pass_detail': json.dumps(attempts[0].get('percept'), ensure_ascii=False, indent=2) if attempts else 'H00只有历史最终回答；未重建历史首答'}
            cp['panels'].append(panel)
            records[variant] = {'values': values, 'percept': percept, 'status': state['status'], 'raw_text': raw}
            for i, region in enumerate(panel['regions'] or [None]):
                human.append({'specimen_key': key, 'variant': variant, 'region_index': i if region else '', 'reviewer_alias': '', 'visible_cue_match': '', 'cell_location_match': '', 'annotation_artifact_suspected': '', 'reference_cells_optional': '', 'review_status': 'PENDING', 'notes': ''})
            if variant != 'H00_CACHED':
                levels = Counter(r['confidence'] for r in panel['regions'])
                summary.append({'specimen_key': key, 'split': case['split'], 'variant': variant, 'prompt_id': VARIANTS[variant][0], 'render_id': render_id, 'clean_sha256': case['clean_sha256'], 'numbered_sha256': case[render_id+'_sha256'], 'status': state['status'], 'first_pass_status': first_status, 'first_pass_percept': json.dumps(attempts[0].get('percept'), ensure_ascii=False) if attempts else '', 'first_pass_parser_valid': attempts[0].get('parser_valid', '') if attempts else '', 'first_pass_contract_valid': attempts[0].get('contract_valid', '') if attempts else '', 'repair_dependent': panel['repair_dependent'], 'region_count': len(panel['regions']) if values['available'] else '', 'unique_cell_count': len(values['cells']) if values['available'] else '', 'cells': json.dumps(values['cells']) if values['available'] else '', 'confidence_counts': json.dumps(dict(levels)), 'no_reliable_cue': values['no_cue'], 'c0_size': len(values['c0']), 'c0_cells': json.dumps(values['c0']), 'c0_reason': values['c0_reason'], 'calls': len(attempts), 'repair_calls': max(0,len(attempts)-1), 'elapsed_seconds': sum(a.get('elapsed_seconds',0) for a in attempts), 'output_tokens': sum(a.get('output_tokens',0) for a in attempts), 'forward_calls': sum(a.get('forward_calls',0) for a in attempts), 'raw_paths': json.dumps(raw_links), 'overlay': overlay, 'human_review': 'PENDING_HUMAN_REVIEW'})
        for a,b in [('A_P0_R0','B_P1_R0'),('A_P0_R0','C_P0_R1'),('B_P1_R0','D_P1_R1'),('C_P0_R1','D_P1_R1'),('A_P0_R0','D_P1_R1')]:
            changes.append(compare(key,a,b,records))
        historical.append(compare(key,'H00_CACHED','A_P0_R0',records))
        payload['cases'].append(cp)
    for name, rows in [('summary.csv',summary),('candidate_changes.csv',changes),('historical_replay_comparison.csv',historical),('human_review_template.csv',human)]:
        csvout(OUT/name,rows)
    with (OUT/'attempts.jsonl').open('w') as f:
        for row in attempt_rows:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    (OUT/'HOW_TO_REVIEW_ZH.md').write_text('# 作者评价方法\n\n打开index.html，先看clean，逐区域核对cue是否可见及是否覆盖所报cells；切换真实编号图检查格号。分别评价两类线索，不只看讨论中的线条。\n\n在human_review_template.csv填写reviewer_alias、visible_cue_match和cell_location_match（MATCH/PARTIAL/MISMATCH/UNCERTAIN），以及annotation_artifact_suspected、可选reference_cells_optional、notes，真实完成后将review_status由PENDING改为CONFIRMED。无区域行用于记录漏报或不确定；空reference不是空GT。\n\n先看首次raw回答，再看格式修复后回答；修复可能改变内容。H00只有历史最终回答，不能假设历史首次回答可得。候选变化、no-cue增多、C0大小都不是准确率；C0为预算0.25、空已测集合、action_count=0的CPU规则结果，不是新动作。当前未收到真实人评，不自动选择赢家或接入生产。\n')
    (OUT/'USER_HYPOTHESES.md').write_text('# 用户目视假设（非GT）\n\n任务包转述：q24-48旋转后某条线状痕迹可能覆盖23、31。只作为待作者核对的观察，不进入任何模型prompt，也不作为自动评分依据。首先确认回答中的line是否指同一条线，圆状线索另行评价。\n')
    dump(OUT/'report_payload.json',payload)
    build_html(payload,OUT)
    manifest = {'task_id': TASK,'status': session['status'],'human_review': 'PENDING_HUMAN_REVIEW','case_count':6,'primary_jobs':24,'generation_attempts':len(attempt_rows),'repair_calls':sum(r.get('kind')=='FORMAT_REPAIR' for r in attempt_rows),'files':[]}
    for path in sorted(OUT.rglob('*')):
        if path.is_file() and path.name!='final_manifest.json':
            manifest['files'].append({'path':relative(path),'bytes':path.stat().st_size,'sha256':sha(path.read_bytes())})
    dump(OUT/'final_manifest.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'}))


if __name__ == '__main__':
    main()
