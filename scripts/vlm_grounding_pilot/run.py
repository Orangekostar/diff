"""Bounded isolated generation; never calls the historical cache resolver."""
import json
import os
import signal
import time
import uuid
from pathlib import Path

from common import (
    ART,
    OUT,
    TASK,
    digest,
    dump,
    parse_contract,
    repair_text,
    runtime,
    sha,
    slug,
    vlm,
)


def execute_job(directory, job, images, prompt, backend):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / 'state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['signature'] != job['signature']:
            raise ValueError('Frozen signature mismatch; no cache reuse')
        if state['status'] == 'STARTED':
            state['status'] = 'INTERRUPTED'
            dump(state_path, state)
        return state
    state = {'specimen_key': job['specimen_key'], 'variant': job['variant'], 'signature': job['signature'], 'status': 'STARTED', 'attempts': []}
    dump(state_path, state)
    for number in (1, 2):
        actual = prompt if number == 1 else repair_text(state['attempts'][0]['text'])
        (directory / f'prompt_{number}.txt').write_text(actual)
        entry = {'attempt_number': number, 'kind': 'PRIMARY' if number == 1 else 'FORMAT_REPAIR', 'status': 'STARTED', 'started_unix': time.time(), 'prompt_sha256': sha(actual)}
        state['attempts'].append(entry)
        dump(state_path, state)
        started = time.monotonic()
        try:
            result = backend.infer(images, actual)
            entry.update(result)
            entry.update(parse_contract(result['text']))
            entry['status'] = 'COMPLETED'
            (directory / f'raw_{number}.txt').write_text(result['text'])
            dump(directory / f'generated_token_ids_{number}.json', result['generated_token_ids'])
        except Exception as error:  # noqa: BLE001 -- retain every failed authorized attempt
            entry.update(status='FAILED', error=f'{type(error).__name__}: {error}')
            state['status'] = 'GENERATION_FAILED'
        finally:
            entry['elapsed_seconds'] = time.monotonic() - started
            dump(directory / f'attempt_{number}.json', entry)
            dump(state_path, state)
        if entry['status'] == 'FAILED':
            break
        if entry['contract_valid']:
            state['status'] = 'VALID_FIRST_PASS' if number == 1 else 'VALID_AFTER_REPAIR'
            state['final_percept'] = entry['percept']
            break
        state['status'] = 'SCHEMA_INVALID_AFTER_ONE_REPAIR' if number == 2 else 'STARTED'
    dump(state_path, state)
    return state


class RecordedBackend:
    def __init__(self, backend, deadline):
        self.backend = backend
        self.deadline = deadline
        self.total_forwards = 0
        self.calls = 0
        self.current_dir = None
        self.current_number = 0
        self.backend._model.register_forward_pre_hook(self._count)

    def _count(self, module, args):
        self.total_forwards += 1

    def infer(self, images, prompt):
        import torch
        if self.calls >= 48 or time.monotonic() >= self.deadline:
            raise RuntimeError('CUMULATIVE_RESOURCE_LIMIT')
        self.calls += 1
        self.current_number += 1
        n = self.current_number
        processor = self.backend._processor
        model = self.backend._model
        text = processor.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'image'}, {'type': 'text', 'text': prompt}]}], tokenize=False, add_generation_prompt=True)
        (self.current_dir / f'chat_{n}.txt').write_text(text)
        before = self.total_forwards
        def expired(signum, frame):
            raise TimeoutError('GENERATION_120S_OR_TOTAL_1800S_LIMIT')
        previous = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, max(.001, min(120, self.deadline - time.monotonic())))
        try:
            inputs = processor(text=[text], images=list(images), padding=True, return_tensors='pt').to('cuda:0')
            meta = {'chat_sha256': sha(text), 'image_sha256': [runtime._image_sha256(im) for im in images], 'image_grid_thw': inputs['image_grid_thw'].tolist(), 'input_tokens': int(inputs['attention_mask'].sum()), 'input_token_ids': inputs['input_ids'][0].tolist(), 'generation_kwargs': {'max_new_tokens': 500, 'do_sample': False, 'temperature': None, 'use_cache': True}}
            dump(self.current_dir / f'input_{n}.json', meta)
            with torch.inference_mode():
                generated = model.generate(**inputs, **meta['generation_kwargs'])
            torch.cuda.synchronize()
            ids = generated[0, inputs['input_ids'].shape[1]:].tolist()
            answer = processor.batch_decode([ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return {'text': answer, 'generated_token_ids': ids, 'input_tokens': meta['input_tokens'], 'output_tokens': len(ids), 'forward_calls': self.total_forwards - before, 'input_metadata': f'input_{n}.json', 'chat_text': f'chat_{n}.txt'}
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


def audit_first_case(lock):
    checks = []
    key = lock['cases'][0]['specimen_key']
    for job in [j for j in lock['jobs'] if j['specimen_key'] == key]:
        directory = OUT / 'runs' / slug(key) / job['variant']
        state = json.loads((directory / 'state.json').read_text())
        for n, attempt in enumerate(state['attempts'], 1):
            if attempt['status'] != 'COMPLETED':
                continue
            meta = json.loads((directory / f'input_{n}.json').read_text())
            assert meta['image_sha256'] == [job['signature_data']['clean_sha256'], job['signature_data']['numbered_sha256']]
            if n == 1:
                assert sha((directory / 'prompt_1.txt').read_text()) == job['signature_data']['prompt_sha256']
            assert (directory / f'raw_{n}.txt').exists() and (directory / f'generated_token_ids_{n}.json').exists()
        checks.append({'variant': job['variant'], 'state': state['status'], 'wiring_and_saved_files_checked': True})
    dump(ART / 'first_case_wiring.json', {'case': key, 'checks': checks, 'semantic_quality_not_a_gate': True})


def main():
    import subprocess

    import torch
    from PIL import Image
    lock = json.loads((OUT / 'experiment_lock.json').read_text())
    if not (ART / 'cpu_preflight.json').exists():
        raise RuntimeError('CPU preflight required')
    assert json.loads((ART / 'cpu_preflight.json').read_text())['passed']
    resource = OUT / 'gpu_session.json'
    if resource.exists():
        # At most one load: a prior live or interrupted session never receives a new budget.
        raise RuntimeError('GPU session already recorded. Export saved states; never reload implicitly.')
    for job in lock['jobs']:
        assert job['signature'] == digest(job['signature_data'])
        assert sha((OUT / 'prompts' / (job['prompt_id'] + '.txt')).read_text()) == job['signature_data']['prompt_sha256']
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.free,memory.total', '--format=csv,noheader'], text=True)
    dump(ART / 'gpu_preflight.json', {'query': gpu, 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'), 'minimum_free_gib': 34})
    free_mib = int(gpu.splitlines()[0].split(',')[-2].strip().split()[0])
    if free_mib < 34 * 1024:
        dump(resource, {'status': 'PARTIAL_RESOURCE_LIMIT', 'reason': 'INSUFFICIENT_FREE_VRAM', 'free_mib': free_mib, 'required_mib': 34816, 'generation_calls': 0, 'elapsed_seconds': 0})
        return
    torch.set_num_threads(4)
    started = time.monotonic()
    session = {'task_id': TASK, 'run_id': str(uuid.uuid4()), 'pid': os.getpid(), 'started_unix': time.time(), 'status': 'STARTED', 'budget_seconds': 1800, 'device': 'cuda:0', 'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES', '0')}
    dump(resource, session)
    backend = None
    recorder = None
    try:
        backend = vlm.QwenVLBackend(lock['config']['model_path'], device='cuda:0', dtype='bfloat16', max_new_tokens=500)
        backend.load()
        backend._model.requires_grad_(False)
        assert sha(backend._processor.chat_template) == lock['config']['chat_template_sha256']
        recorder = RecordedBackend(backend, started + 1800)
        for i, job in enumerate(lock['jobs']):
            if time.monotonic() >= started + 1800:
                session['status'] = 'PARTIAL_RESOURCE_LIMIT'
                break
            idir = OUT / 'inputs' / slug(job['specimen_key'])
            images = (Image.open(idir / 'clean.png').convert('RGB'), Image.open(idir / (job['render_id'] + '.png')).convert('RGB'))
            assert [runtime._image_sha256(im) for im in images] == [job['signature_data']['clean_sha256'], job['signature_data']['numbered_sha256']]
            directory = OUT / 'runs' / slug(job['specimen_key']) / job['variant']
            recorder.current_dir = directory
            recorder.current_number = 0
            state = execute_job(directory, job, images, (OUT / 'prompts' / (job['prompt_id'] + '.txt')).read_text(), recorder)
            print(json.dumps({'case': job['specimen_key'], 'variant': job['variant'], 'status': state['status'], 'calls': recorder.calls, 'elapsed': time.monotonic() - started}), flush=True)
            if state['status'] == 'GENERATION_FAILED':
                session['status'] = 'PARTIAL_RESOURCE_LIMIT'
                break
            if i == 3:
                audit_first_case(lock)
        else:
            session['status'] = 'RUN_COMPLETE_REVIEW_PENDING'
    except Exception as error:
        session.update(status='PARTIAL_RESOURCE_LIMIT', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        session.update(elapsed_seconds=time.monotonic() - started, generation_calls=recorder.calls if recorder else 0, actual_qwen_forward_calls=recorder.total_forwards if recorder else 0, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30, training_updates=0, other_research_model_forwards=0, attention_forwards=0, test_access=0)
        dump(resource, session)
        entries = []
        for path in sorted((OUT / 'runs').glob('*/*/state.json')):
            state = json.loads(path.read_text())
            entries.extend(dict(a, specimen_key=state['specimen_key'], variant=state['variant'], run_id=session['run_id']) for a in state['attempts'])
        with (OUT / 'attempts.jsonl').open('w') as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
