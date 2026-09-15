"""Finite CPU contracts only; fake backend never loads research models."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


class Contracts(unittest.TestCase):
    def test_geometry(self):
        self.assertIsNotNone(importlib.util.find_spec('prepare'), 'preparation module missing')
        from common import NativeCellGrid
        from prepare import render_readable
        grid = NativeCellGrid.from_shape((773, 899))
        pixels = np.zeros((773, 899, 3), dtype=np.uint8)
        for c in grid.cells:
            pixels[c.row_start:c.row_stop, c.col_start:c.col_stop] = [c.index, c.index * 3, c.index * 4]
        clean = Image.fromarray(pixels)
        image, boxes, mask, config = render_readable(clean)
        self.assertEqual(image.size, clean.size)
        self.assertEqual([b['cell_id'] for b in boxes], list(range(64)))
        for b in boxes:
            self.assertEqual(int(pixels[(b['cell_box'][1]+b['cell_box'][3])//2, (b['cell_box'][0]+b['cell_box'][2])//2, 0]), b['cell_id'])
            x0, y0, x1, y1 = b['cell_box']
            a, c, d, e = b['text_box']
            self.assertTrue(x0 <= a < d <= x1 and y0 <= c < e <= y1)
        self.assertEqual(np.asarray(mask).shape, (773, 899))
        self.assertGreater(config['font_size'], 0)

    def test_comparison_normalizes_json_sequences(self):
        from report import compare
        values = {'cells': [0], 'confidence': [0.0]*64, 'c0': [0], 'available': True}
        a = {'values': values, 'status': 'VALID', 'raw_text': 'same', 'percept': {'regions': ({'cells': (0,)},)}}
        b = dict(a, percept={'regions': [{'cells': [0]}]})
        self.assertTrue(compare('stub', 'a', 'b', {'a': a, 'b': b})['regions_cue_confidence_identical'])

    def test_prompt_repair_resume(self):
        self.assertIsNotNone(importlib.util.find_spec('run'), 'runner module missing')
        from common import parse_contract, sha
        from run import execute_job
        good = json.dumps({'regions': [], 'no_reliable_cue': True})
        calls = []
        class Fake:
            def infer(self, images, prompt):
                calls.append((tuple(sha(im.tobytes()) for im in images), prompt))
                return {'text': 'bad' if len(calls) == 1 else good, 'generated_token_ids': [1], 'input_tokens': 1, 'output_tokens': 1, 'forward_calls': 1}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            ims = (Image.new('RGB', (16, 16)), Image.new('RGB', (16, 16), 'white'))
            job = {'specimen_key': 'stub', 'variant': 'A_P0_R0', 'signature': 'x'}
            first = execute_job(p, job, ims, 'unique actual prompt', Fake())
            self.assertEqual(first['status'], 'VALID_AFTER_REPAIR')
            self.assertEqual(calls[0][1], 'unique actual prompt')
            self.assertIn('<invalid_response>\nbad\n</invalid_response>', calls[1][1])
            self.assertEqual(calls[0][0], calls[1][0])
            execute_job(p, job, ims, 'unique actual prompt', Fake())
            self.assertEqual(len(calls), 2)
            with self.assertRaises(ValueError):
                execute_job(p, dict(job, signature='different'), ims, 'unique actual prompt', Fake())
        duplicate = json.dumps({'regions': [{'cells': [0], 'cue': 'x', 'alternative': 'y', 'confidence': 'low'}] * 2, 'no_reliable_cue': False})
        self.assertTrue(parse_contract(duplicate)['parser_valid'])
        self.assertFalse(parse_contract(duplicate)['contract_valid'])


if __name__ == '__main__':
    unittest.main()
