"""Isolated source bindings and file contracts, without model execution."""
import os

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'
import csv
import hashlib
import json
import sys
import types
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/cai_agent_v3/vlm_grounding_pilot/r1_e8d9ef01'
ART = ROOT / 'artifacts/cai_agent_v3/vlm_grounding_pilot/r1_e8d9ef01'
DATA = ROOT / 'results/cai_agent_v3/new_protocol'
TASK = 'VLM_GROUNDING_PILOT_R1_e8d9ef01'
BASE = 'e8d9ef0171ab6d0e1d14187a56cdbd7ecc579750'
sys.path.insert(0, str(ROOT / 'src'))
package = types.ModuleType('cmc_bbdm')
package.__path__ = [str(ROOT / 'src/cmc_bbdm')]
sys.modules['cmc_bbdm'] = package
from cmc_bbdm.cai_active_image.environment import (
    NativeCellGrid,  # noqa: F401 -- public source binding
)
from cmc_bbdm.cai_agent_v3 import policy, vlm_perception
from cmc_bbdm.learned_cscan import perception
from cmc_bbdm.vlm_cscan import runtime, vlm

MODULES = {m.__name__: m.__file__ for m in (perception, runtime, vlm, vlm_perception, policy)}
assert all(Path(p).is_relative_to(ROOT) for p in MODULES.values())
VARIANTS = {'A_P0_R0': ('P0', 'R0'), 'B_P1_R0': ('P1', 'R0'), 'C_P0_R1': ('P0', 'R1'), 'D_P1_R1': ('P1', 'R1')}


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def csvout(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def readcsv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def slug(key):
    return key.replace(':', '__')


def parse_contract(text):
    try:
        parsed = perception.parse_surface_percept(text)
    except (ValueError, TypeError) as error:
        return {'parser_valid': False, 'contract_valid': False, 'error': str(error), 'percept': None}
    cells = [c for r in parsed.regions for c in r.cells]
    valid = len(cells) == len(set(cells)) and (bool(parsed.regions) != parsed.no_reliable_cue)
    return {'parser_valid': True, 'contract_valid': valid, 'error': None if valid else 'DUPLICATE_CELLS_OR_EMPTY_NO_CUE_CONTRADICTION', 'percept': asdict(parsed)}


def repair_text(raw):
    return f'{perception.FORMAT_REPAIR_PROMPT}\n{perception.FORMAT_REPAIR_CONTEXT_PREFIX}{raw}{perception.FORMAT_REPAIR_CONTEXT_SUFFIX}'
