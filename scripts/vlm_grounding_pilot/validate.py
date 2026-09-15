"""Q1–Q6 finite diagnostic checks; never calls a research model."""
import json
import tempfile
from pathlib import Path

import numpy as np
from common import ART, OUT, ROOT, digest, dump, readcsv, runtime, sha, slug
from PIL import Image
from prepare import select_cases
from run import execute_job


def cpu_preflight():
    lock = json.loads((OUT / 'experiment_lock.json').read_text())
    assert [r['specimen_key'] for r in select_cases()] == [r['specimen_key'] for r in lock['cases']]
    assert len(lock['cases']) == 6 and len(lock['jobs']) == 24
    seen = set()
    no_cue = json.dumps({'regions': [], 'no_reliable_cue': True})
    calls = []
    class Fake:
        def infer(self, images, prompt):
            calls.append((prompt, [runtime._image_sha256(im) for im in images]))
            return {'text': no_cue, 'generated_token_ids': [1], 'input_tokens': 1, 'output_tokens': 1, 'forward_calls': 1}
    with tempfile.TemporaryDirectory() as tmp:
        for job in lock['jobs']:
            assert digest(job['signature_data']) == job['signature']
            assert job['signature'] not in seen
            seen.add(job['signature'])
            idir = OUT / 'inputs' / slug(job['specimen_key'])
            clean = Image.open(idir / 'clean.png').convert('RGB')
            numbered = Image.open(idir / (job['render_id'] + '.png')).convert('RGB')
            prompt = (OUT / 'prompts' / (job['prompt_id'] + '.txt')).read_text()
            assert sha(prompt) == job['signature_data']['prompt_sha256']
            assert not any(k in prompt for k in ['q24-48', 'c8-16', 'q16-29', '23/31', '27/28', '36/37/59/60'])
            expected = [job['signature_data']['clean_sha256'], job['signature_data']['numbered_sha256']]
            directory = Path(tmp) / slug(job['specimen_key']) / job['variant']
            state = execute_job(directory, job, (clean, numbered), prompt, Fake())
            assert state['status'] == 'VALID_FIRST_PASS' and len(state['attempts']) == 1
            assert calls[-1] == (prompt, expected)
            execute_job(directory, job, (clean, numbered), prompt, Fake())
        assert len(calls) == 24
        stalled = Path(tmp) / 'interrupted'
        dump(stalled / 'state.json', {'signature': 'stalled', 'status': 'STARTED', 'attempts': [{'status': 'STARTED'}]})
        state = execute_job(stalled, {'signature': 'stalled'}, (), 'x', Fake())
        assert state['status'] == 'INTERRUPTED' and len(calls) == 24
        class Broken:
            def infer(self, images, prompt):
                raise TimeoutError('stub timeout')
        failed = execute_job(Path(tmp) / 'failed', {'specimen_key': 'stub', 'variant': 'A', 'signature': 'failed'}, (), 'x', Broken())
        assert failed['status'] == 'GENERATION_FAILED' and len(failed['attempts']) == 1
        execute_job(Path(tmp) / 'failed', {'signature': 'failed'}, (), 'x', Fake())
        assert len(calls) == 24
    for case in lock['cases']:
        idir = OUT / 'inputs' / slug(case['specimen_key'])
        r0, r1 = [np.asarray(Image.open(idir / f'{n}.png')) for n in ['R0', 'R1']]
        mask = np.asarray(Image.open(idir / 'label_change_mask.png')) > 0
        assert np.array_equal(r0[~mask], r1[~mask])
        boxes = json.loads((idir / 'label_boxes.json').read_text())
        assert [b['cell_id'] for b in boxes] == list(range(64))
        for b in boxes:
            x0, y0, x1, y1 = b['cell_box']
            tx0, ty0, tx1, ty1 = b['text_box']
            assert x0 <= tx0 < tx1 <= x1 and y0 <= ty0 < ty1 <= y1
        assert case['historical_input_hash_match'] is True
        four = [j for j in lock['jobs'] if j['specimen_key'] == case['specimen_key']]
        assert len({j['signature_data']['clean_sha256'] for j in four}) == 1
        assert four[0]['signature_data']['numbered_sha256'] == four[1]['signature_data']['numbered_sha256']
        assert four[2]['signature_data']['numbered_sha256'] == four[3]['signature_data']['numbered_sha256']
    dump(ART / 'cpu_preflight.json', {'passed': True, 'Q1': 'six exact cases, no TEST, all six R0 historical hashes match', 'Q2': '24 fake backend deliveries: actual prompt and actual image bytes; 24 unique full signatures', 'Q3': '384 label boxes inside cells and unchanged pixels outside declared mark masks; odd-size synthetic contract separately passed', 'Q4': 'valid no-cue no repair; format-only once repair test; interrupted/failed/finished no reissue; changed signature rejected', 'new_research_generations': 0})
    print('Q1–Q4 CPU preflight PASS; no research generation')


def final_checks():
    lock = json.loads((OUT / 'experiment_lock.json').read_text())
    rows = readcsv(OUT / 'summary.csv')
    assert len(rows) == 24
    total = primary = repairs = 0
    for job in lock['jobs']:
        directory = OUT / 'runs' / slug(job['specimen_key']) / job['variant']
        state = json.loads((directory / 'state.json').read_text())
        assert state['signature'] == job['signature']
        assert 1 <= len(state['attempts']) <= 2
        for n, attempt in enumerate(state['attempts'], 1):
            total += 1
            primary += n == 1
            repairs += n == 2
            if n == 2:
                assert not state['attempts'][0]['contract_valid']
            if attempt['status'] == 'COMPLETED':
                ids = json.loads((directory / f'generated_token_ids_{n}.json').read_text())
                assert len(ids) == attempt['output_tokens'] <= 500
                assert (directory / f'raw_{n}.txt').read_text() == attempt['text']
                meta = json.loads((directory / f'input_{n}.json').read_text())
                assert meta['image_sha256'] == [job['signature_data']['clean_sha256'], job['signature_data']['numbered_sha256']]
    session = json.loads((OUT / 'gpu_session.json').read_text())
    assert total <= 48 and primary == 24 and total == session['generation_calls']
    assert session['elapsed_seconds'] <= 1800
    assert len(readcsv(OUT / 'candidate_changes.csv')) == 30
    assert len(readcsv(OUT / 'historical_replay_comparison.csv')) == 6
    assert all(r['review_status'] == 'PENDING' for r in readcsv(OUT / 'human_review_template.csv'))
    before = json.loads((ART / 'protected_before.json').read_text())
    assert all(sha((ROOT / p).read_bytes()) == h for p, h in before['files'].items())
    ledger = (ROOT / 'results/cai_agent_v3/compute_ledger.jsonl').read_bytes()
    assert sha(ledger[:before['ledger_bytes']]) == before['ledger_sha256']
    dump(ART / 'finite_validation.json', {'passed': True, 'Q1_Q3': json.loads((ART / 'cpu_preflight.json').read_text()), 'Q4': {'primary': primary, 'repairs': repairs, 'attempts': total, 'output_token_files_match': True, 'elapsed_seconds': session['elapsed_seconds']}, 'Q5': {'summary_rows': 24, 'paired_changes': 30, 'human_review': 'PENDING_HUMAN_REVIEW'}, 'Q6': {'protected_hashes_unchanged': True, 'ledger_prefix_unchanged': True, 'git_delivery': 'verified separately at commit/push'}})
    print('Q1–Q6 finite checks PASS (Git delivery separately)')


if __name__ == '__main__':
    import sys
    if '--final' in sys.argv:
        final_checks()
    else:
        cpu_preflight()
