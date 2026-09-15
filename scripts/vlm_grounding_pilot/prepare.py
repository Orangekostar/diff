"""Freeze authorized cases, actual inputs and signatures before any generation."""
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from common import (
    ART,
    BASE,
    DATA,
    MODULES,
    OUT,
    ROOT,
    TASK,
    VARIANTS,
    csvout,
    digest,
    dump,
    perception,
    readcsv,
    runtime,
    sha,
    slug,
    vlm_perception,
)
from PIL import Image, ImageDraw, ImageFont

FONT = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
FIXED = ['cgtnjyggtm:q24-48', '74t7kcdgkr:c8-16', 'w68dtmpfyf:q16-29']
DOMAINS = ['xcmzfsbd9t', 'yfxyg8jm46', 'ykhs7s2dck']
COLUMNS = ['specimen_key', 'dataset_id', 'split', 'capture_group_id', 'impacted_surface_path', 'surface_sha256', 'identity_status', 'p0r_roster_status', 'registered_cscan_height_px', 'registered_cscan_width_px']


def grid_only(clean):
    image = clean.copy()
    draw = ImageDraw.Draw(image, 'RGBA')
    w, h = image.size
    for i in range(1, 8):
        x, y = round(i * w / 8), round(i * h / 8)
        width = max(1, round(max(w, h) / 512))
        draw.line((x, 0, x, h - 1), fill=(0, 255, 255, 110), width=width)
        draw.line((0, y, w - 1, y), fill=(0, 255, 255, 110), width=width)
    return image


def render_readable(clean):
    image = grid_only(clean)
    w, h = image.size
    side = min(min(round((i + 1) * w / 8) - round(i * w / 8) for i in range(8)), min(round((i + 1) * h / 8) - round(i * h / 8) for i in range(8)))
    size = max(12, round(24 * side / 128))
    inset = max(3, round(10 * side / 128))
    probe = ImageDraw.Draw(image)
    while size > 0:
        font = ImageFont.truetype(str(FONT), size)
        boxes = []
        for c in range(64):
            x0, y0 = round((c % 8) * w / 8), round((c // 8) * h / 8)
            x1, y1 = round((c % 8 + 1) * w / 8), round((c // 8 + 1) * h / 8)
            bx0, by0, bx1, by1 = probe.textbbox((0, 0), str(c), font=font)
            tx, ty = x0 + inset, y0 + inset
            box = [tx, ty, tx + bx1 - bx0, ty + by1 - by0]
            panel = [box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2]
            boxes.append({'cell_id': c, 'cell_box': [x0, y0, x1, y1], 'text_box': box, 'panel_box': panel, 'draw_origin': [tx - bx0, ty - by0]})
        if all(b['cell_box'][0] <= b['panel_box'][0] and b['cell_box'][1] <= b['panel_box'][1] and b['panel_box'][2] <= b['cell_box'][2] and b['panel_box'][3] <= b['cell_box'][3] for b in boxes):
            break
        size -= 1
    if size == 0:
        raise ValueError('Cannot fit uniformly inset labels')
    mask = Image.new('L', clean.size)
    md = ImageDraw.Draw(mask)
    draw = ImageDraw.Draw(image, 'RGBA')
    for b in boxes:
        x0, y0, x1, y1 = b['panel_box']
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=(0, 0, 0, 170))
        draw.text(tuple(b['draw_origin']), str(b['cell_id']), font=font, fill=(255, 255, 255, 230))
        md.rectangle((x0, y0, x1 - 1, y1 - 1), fill=255)
        # Declare exact original default-font mark area, including its stroke.
        cx, cy = b['cell_box'][:2]
        old = probe.textbbox((cx + 2, cy + 1), str(b['cell_id']), stroke_width=1)
        md.rectangle((old[0], old[1], old[2] - 1, old[3] - 1), fill=255)
        b['r0_label_box'] = list(old)
    return image, boxes, mask, {'font_path': str(FONT), 'font_name': font.getname(), 'font_sha256': sha(FONT.read_bytes()), 'font_size': size, 'inset': inset, 'panel_rgba': [0, 0, 0, 170], 'text_rgba': [255, 255, 255, 230]}


def select_cases():
    import csv
    with (DATA / 'candidate_queue.csv').open() as f:
        reader = csv.reader(f)
        header = next(reader)
        indices = [header.index(k) for k in COLUMNS]
        rows = [dict(zip(COLUMNS, (r[i] for i in indices))) for r in reader]
    chosen = [next(r for r in rows if r['specimen_key'] == k) for k in FIXED]
    for domain in DOMAINS:
        candidates = [r for r in rows if r['dataset_id'] == domain and r['split'] == 'TRAIN']
        chosen.append(min(candidates, key=lambda r: (sha('VLM_GROUNDING_PILOT_R1|' + r['specimen_key']), r['specimen_key'])))
    for i, row in enumerate(chosen):
        row['selection'] = 'FIXED_VALID' if i < 3 else 'HASH_MIN_TRAIN'
        row['selection_hash'] = sha('VLM_GROUNDING_PILOT_R1|' + row['specimen_key'])
        assert row['split'] == ('VALID' if i < 3 else 'TRAIN')
    return chosen


def main():
    if (OUT / 'experiment_lock.json').exists():
        raise RuntimeError('Experiment already frozen; do not regenerate inputs')
    from transformers import AutoProcessor
    pdir = OUT / 'prompts'
    pdir.mkdir(parents=True, exist_ok=True)
    p1 = (ART / 'spec/prompts/P1_SPATIAL_GROUNDING_ZH.txt').read_text().rstrip() + '\nJSON schema:\n' + json.dumps(perception.SURFACE_PERCEPT_SCHEMA, sort_keys=True, separators=(',', ':'))
    prompts = {'P0': perception.SURFACE_PERCEPT_PROMPT, 'P1': p1}
    for name, text in prompts.items():
        (pdir / (name + '.txt')).write_text(text)
    (pdir / 'FORMAT_REPAIR.txt').write_text(perception.FORMAT_REPAIR_PROMPT)
    dump(pdir / 'repair_context.json', {'prefix': perception.FORMAT_REPAIR_CONTEXT_PREFIX, 'suffix': perception.FORMAT_REPAIR_CONTEXT_SUFFIX})
    processor = AutoProcessor.from_pretrained(vlm_perception._MODEL_PATH, local_files_only=True, use_fast=False, min_pixels=200704, max_pixels=1003520)
    config = {'model_repository': 'Qwen/Qwen2.5-VL-7B-Instruct', 'model_revision': vlm_perception._MODEL_REVISION, 'model_path': str(vlm_perception._MODEL_PATH), 'dtype': 'bfloat16', 'attention': 'sdpa', 'processor': {'use_fast': False, 'min_pixels': 200704, 'max_pixels': 1003520}, 'generation': {'max_new_tokens': 500, 'do_sample': False, 'temperature': None, 'use_cache': True}, 'batch_size': 1, 'image_order': ['clean', 'numbered'], 'chat_template': processor.chat_template, 'chat_template_sha256': sha(processor.chat_template), 'versions': {x: importlib.metadata.version(x) for x in ['torch', 'transformers', 'Pillow', 'numpy']}, 'repair_sha256': sha(perception.FORMAT_REPAIR_PROMPT + perception.FORMAT_REPAIR_CONTEXT_PREFIX + perception.FORMAT_REPAIR_CONTEXT_SUFFIX)}
    cases = select_cases()
    keys = {r['specimen_key'] for r in cases}
    features = {r['specimen_key']: r for r in readcsv(DATA / 'vlm_actor_features_fit.csv') if r['specimen_key'] in keys}
    cachekeys = {r['cache_key'] for r in features.values()}
    cache = {}
    for line in (DATA / 'vlm_surface_percepts.jsonl').open():
        row = json.loads(line)
        if row['cache_key'] in cachekeys:
            cache[row['cache_key']] = row
    source_root = Path(json.loads((DATA / 'feature_bank_manifest.json').read_text())['encoder_execution_root'])
    jobs = []
    for case in cases:
        key = case['specimen_key']
        idir = OUT / 'inputs' / slug(key)
        idir.mkdir(parents=True, exist_ok=True)
        path = source_root / case['impacted_surface_path']
        case['source_path_resolved'] = str(path)
        feature = features.get(key, {})
        historical = cache.get(feature.get('cache_key'))
        case['historical_cache_available'] = historical is not None
        if historical:
            dump(idir / 'H00_CACHED.json', historical)
            (idir / 'H00_raw.txt').write_text(historical['raw_text'])
        if not path.is_file():
            case['input_status'] = 'INPUT_MISSING'
            continue
        assert sha(path.read_bytes()) == case['surface_sha256']
        source = Image.open(path).convert('RGB')
        render = runtime.render_surface_inputs(source, max_edge=1024)
        r1, boxes, mask, rcfg = render_readable(render.clean)
        assert not np.any(np.asarray(r1)[np.asarray(mask) == 0] != np.asarray(render.gridded)[np.asarray(mask) == 0])
        history_match = None
        if historical:
            history_match = render.clean_sha256 == historical['request']['clean_image_sha256'] and render.gridded_sha256 == historical['request']['gridded_image_sha256']
            assert history_match, f'Historical input drift: {key}'
        for name, image in [('clean', render.clean), ('R0', render.gridded), ('R1', r1), ('label_change_mask', mask)]:
            image.save(idir / (name + '.png'), optimize=False, compress_level=9)
        dump(idir / 'label_boxes.json', boxes)
        csvout(idir / 'label_boxes.csv', [{k: json.dumps(v) if isinstance(v, list) else v for k, v in b.items()} for b in boxes])
        case.update(input_status='READY', clean_sha256=render.clean_sha256, R0_sha256=render.gridded_sha256, R1_sha256=runtime._image_sha256(r1), width=render.clean.width, height=render.clean.height, historical_input_hash_match=history_match)
        dump(idir / 'render_metadata.json', {'source_size': source.size, 'render_size': render.clean.size, 'rotation': 'ROTATE_270 once before resize', 'R1': rcfg, 'outside_declared_mark_mask_equal': True})
        for variant, (prompt_id, render_id) in VARIANTS.items():
            signature_data = dict(config, prompt_sha256=sha(prompts[prompt_id]), clean_sha256=render.clean_sha256, numbered_sha256=case[render_id + '_sha256'], render_id=render_id, render_config=rcfg if render_id == 'R1' else {'implementation': 'original render_surface_inputs'})
            jobs.append({'specimen_key': key, 'variant': variant, 'prompt_id': prompt_id, 'render_id': render_id, 'signature': digest(signature_data), 'signature_data': signature_data})
    csvout(OUT / 'cases.csv', cases, list(dict.fromkeys(k for r in cases for k in r)))
    protected = ['results/cai_agent_v3/new_protocol/vlm_surface_percepts.jsonl', 'results/cai_agent_v3/new_protocol/vlm_actor_features_fit.csv', 'src/cmc_bbdm/learned_cscan/perception.py', 'src/cmc_bbdm/vlm_cscan/runtime.py', 'src/cmc_bbdm/vlm_cscan/vlm.py', 'src/cmc_bbdm/cai_agent_v3/vlm_perception.py', 'src/cmc_bbdm/cai_agent_v3/policy.py']
    ledger = ROOT / 'results/cai_agent_v3/compute_ledger.jsonl'
    dump(ART / 'protected_before.json', {'files': {p: sha((ROOT / p).read_bytes()) for p in protected}, 'ledger_bytes': ledger.stat().st_size, 'ledger_sha256': sha(ledger.read_bytes())})
    dump(OUT / 'experiment_lock.json', {'task_id': TASK, 'base_commit': BASE, 'cases': cases, 'jobs': jobs, 'config': config, 'limits': {'primary': 24, 'attempts': 48, 'per_job': 2, 'gpu_seconds': 1800, 'attempt_seconds': 120}, 'modules': MODULES, 'review_status': 'PENDING_HUMAN_REVIEW'})
    print(json.dumps({'cases': [r['specimen_key'] for r in cases], 'ready_jobs': len(jobs), 'new_generations': 0}))


if __name__ == '__main__':
    main()
