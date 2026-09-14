"""Six bounded checks specified by the manuscript package; no model imports."""
import importlib.util
import math
from pathlib import Path

script = Path(__file__).with_name('timing_analysis.py')
assert script.exists(), 'Missing authorized timing-analysis implementation'
spec = importlib.util.spec_from_file_location('timing_analysis', script)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def close(a, b):
    assert math.isclose(a, b, abs_tol=1e-10), (a, b)

# 1: boundary-at-B; would fail if endpoint observation were given area weight.
x = m.decompose([0,.125,.25], [10,6,2])
close(x['area_direct'], 8); close(x['objective'],8.5)
assert x['terms'] == (2.,0.)
# 2: held tail; would fail if unspent budget were discarded.
x = m.decompose([0,.1,.2], [10,6,2])
close(x['area_direct'],6.8)
# 3: negative gain; would fail under positive clipping.
x = m.decompose([0,.0625,.125], [10,15,5])
assert x['terms'] == (-3.75,5.)
close(x['area_direct'],8.75)
# 4: all right-closed bins, including a zero endpoint term.
c = [0,.0625,.125,.1875,.25]
x = m.decompose(c,[10,9,8,7,6])
assert m.bin_terms(c,x['terms']) == (.75,.5,.25,0.)
# 5: unequal repeat/domain sizes must not weight runs as specimens.
rows = [dict(method='M', dataset_id='a', specimen_key='a1', value=v) for v in [0,4]]
rows += [dict(method='M', dataset_id='a', specimen_key='a2', value=6),
         dict(method='M', dataset_id='b', specimen_key='b1', value=10)]
close(m.aggregate(rows, ['value'])['M']['value'], 7)
# 6: different initial errors must remain in the comparison identity.
a=m.decompose([0,.03,.11,.24],[10,8,12,4])
b=m.decompose([0,.05,.10,.20],[11,9,7,6])
close(b['area_direct']-a['area_direct'],1+sum(a['terms'])-sum(b['terms']))
print('PASS: six prescribed numerical checks; no research computation')
