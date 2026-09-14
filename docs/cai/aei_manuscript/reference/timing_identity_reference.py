#!/usr/bin/env python3
"""Standalone exact checks, not a real-data experiment or repository runner.

No models, network, files containing study data, or external packages are used.
Run: python timing_identity_reference.py --self-test
"""
from __future__ import annotations
import argparse
import bisect
import math
from collections.abc import Sequence


def decompose(costs: Sequence[float], errors: Sequence[float],
              budget: float = 0.25, terminal_weight: float = 0.25) -> dict:
    c, e = tuple(map(float, costs)), tuple(map(float, errors))
    if (not c or len(c) != len(e) or not math.isfinite(budget) or budget <= 0
            or not math.isfinite(terminal_weight) or terminal_weight < 0
            or any(not math.isfinite(x) for x in c + e)
            or c[0] != 0 or c[-1] > budget or any(x < 0 for x in e)
            or any(b <= a for a, b in zip(c[:-1], c[1:]))):
        raise ValueError('Finite aligned increasing costs and nonnegative errors required')
    direct = sum((c[j+1]-c[j])*e[j] for j in range(len(c)-1))
    direct = (direct + (budget-c[-1])*e[-1]) / budget
    terms = tuple((1-c[j]/budget)*(e[j-1]-e[j]) for j in range(1,len(c)))
    return {'area_direct': direct, 'area_from_terms': e[0]-sum(terms),
            'terms': terms, 'objective': direct+terminal_weight*e[-1]}


def bin_terms(costs: Sequence[float], terms: Sequence[float],
              edges: Sequence[float] = (0,.0625,.125,.1875,.25)) -> tuple[float,...]:
    if (len(costs) != len(terms)+1 or len(edges)<2
            or any(b<=a for a,b in zip(edges[:-1],edges[1:]))):
        raise ValueError('Invalid event or bin dimensions')
    totals=[0.0]*(len(edges)-1)
    for c, term in zip(costs[1:],terms):
        if not edges[0]<c<=edges[-1]:
            raise ValueError('Acquisition completion outside covered bins')
        i=bisect.bisect_left(edges,c)-1
        totals[i]+=term
    return tuple(totals)


def _close(a,b):
    if not math.isclose(a,b,rel_tol=0,abs_tol=1e-10):
        raise AssertionError(f'{a} != {b}')


def self_test() -> None:
    # 1. Measurement completed exactly at B affects endpoint, not the preceding area.
    x=decompose([0,.125,.25],[10,6,2])
    _close(x['area_direct'],8); _close(x['objective'],8.5)
    assert x['terms']==(2.,0.)
    # 2. The unspent tail is held at the last prediction.
    x=decompose([0,.1,.2],[10,6,2])
    _close(x['area_direct'],6.8); _close(x['terms'][0],2.4); _close(x['terms'][1],.8)
    # 3. Negative contributions must not be discarded.
    x=decompose([0,.0625,.125],[10,15,5])
    assert x['terms']==(-3.75,5.)
    _close(x['area_direct'],8.75)
    # 4. All completion-boundary events go to the right-closed interval.
    c=[0,.0625,.125,.1875,.25]
    x=decompose(c,[10,9,8,7,6])
    assert bin_terms(c,x['terms'])==(.75,.5,.25,0.)
    # 5. Even a no-observation trajectory has the same identity.
    x=decompose([0],[7])
    _close(x['area_direct'],7); _close(x['area_from_terms'],7)
    # 6. Identity and comparison identity across deterministic toy trajectories.
    main=decompose([0,.03,.11,.24],[10,8,12,4])
    ctrl=decompose([0,.05,.10,.20],[11,9,7,6])
    for x in (main,ctrl):
        _close(x['area_direct'],x['area_from_terms'])
    _close(ctrl['area_direct']-main['area_direct'],
           (11-10)+sum(main['terms'])-sum(ctrl['terms']))
    print('6 independent mathematical checks passed; no study data processed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test()
    else:
        parser.print_help()
