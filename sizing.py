"""Dome-consistent sizing of the hydrogen vessels H1..H8.

Netting theory balances the barrel only. Here the helical wall is scaled by
k so that the largest fibre-stress ratio in the dome (beyond the junction
band, s > 2 sqrt(Rt), where the junction bending field has decayed below
about 5%; boss excluded) equals the mid-barrel value, R_D(k) = 1; the hoop wall is
re-sized from netting hoop equilibrium with the thicker helical wall. The
reference transition (e = 0, l_h = 0.5 sqrt(Rt)) is used; the dome region
s > sqrt(Rt) is outside its influence.
"""
import json
import os
import sys
import numpy as np
import p1_study as PS
import tr_eval as T


def RD(vn, k):
    b = PS.base(vn, k_hel=k)
    w = PS.wj(b)
    c = dict(b, name=f'siz_{vn}_{k:.4f}', hoop_end=0.0, hoop_taper=0.5*w)
    o = T.run_case(c, tag='siz')
    pr = [p for p in o['prof'] if p['zone'] != 'boss']
    dome = [p for p in pr if p['s'] > 2.0*w]       # beyond the junction boundary layer
    pk = max(dome, key=lambda p: p['rho'])
    return pk['rho']/o['mid']['rho'], pk, o, b


def size(vn, tol=0.01):
    hist = []
    k0, k1 = 1.0, 1.5
    r0 = RD(vn, k0)[0]; hist.append((k0, r0))
    r1 = RD(vn, k1)[0]; hist.append((k1, r1))
    for _ in range(8):
        if abs(r1 - 1) < tol:
            break
        k2 = k1 + (1 - r1)*(k1 - k0)/(r1 - r0)
        k2 = float(np.clip(k2, 0.5, 4.0))
        k0, r0 = k1, r1
        k1 = k2
        r1 = RD(vn, k1)[0]; hist.append((k1, r1))
    R, pk, o, b = RD(vn, k1)
    t = sum(e[-1] for e in b['stack'])
    return dict(k=k1, RD=R, hist=hist, t=t, t_h=b['t_h'], t_hel=b['t_hel'], hoop_frac=b['t_h']/t,
                dome_peak_s=pk['s']/PS.wj(b), dome_peak_ply=pk['rho_ply'], pf_ref=o['pf'],
                pf_mid=o['pf_barrel'], mass=o['mass'])


if __name__ == '__main__':
    out = json.load(open('sizing.json')) if os.path.exists('sizing.json') else {}
    for vn in sys.argv[1:]:
        r = size(vn)
        out[vn] = r
        json.dump(out, open('sizing.json', 'w'), indent=1, default=float)
        print(vn, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != 'hist'},
              [(round(a, 3), round(b_, 3)) for a, b_ in r['hist']], flush=True)
