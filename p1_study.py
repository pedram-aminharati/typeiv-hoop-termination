"""Paper 1: full study matrix.

Vessels
  H1..H8 : Type IV hydrogen vessels (R = 150 mm, L/D = 3) sized by netting
           theory for a burst pressure of 2.25 x NWP (UN GTR 13), carbon/epoxy
           ply of the companion study; full factorial on the nominal working
           pressure (35, 70 MPa), dome compactness h_d/R and polar-opening
           ratio r_b/R.
  GFRP, CFRP : open benchmark vessels of Agne et al. (2025).
Design variables: hoop end e, drop-off l_h, stacking order, hybrid layer, and
combinations at the friction limit.
"""
import itertools
import json
import os
import numpy as np
from multiprocessing import Pool
from scipy.optimize import brentq
import core
import vessel_tr as vt
import tr_eval as T

OUT = 'results_p1full'
R0 = 150.0
E_GRID = [-1.0, -0.5, -0.25, 0.0, 0.125, 0.25, 0.375, 0.5]
L_GRID = [0.1, 0.25, 0.5, 1.0]
ORDERS = ['H-F-F x2 (ref.)', 'F-F-H x2', 'H block inside', 'H block middle', 'H block outside']


def netting(R, rb_R, P):
    P_B = 2.25*P
    Xt = core.MAT['Xt']
    a1 = np.degrees(np.arcsin(rb_R))
    a2 = 30.0                                   # turnaround at 0.5 R
    c1, c2 = np.cos(np.radians(a1))**2, np.cos(np.radians(a2))**2
    s1, s2 = 1 - c1, 1 - c2
    t_hel = P_B*R/(Xt*(c1 + c2))                # axial equilibrium, equal split
    t_h = P_B*R/Xt - 0.5*t_hel*(s1 + s2)        # hoop equilibrium
    return a1, a2, t_hel, t_h


K_HEL = {}          # dome-consistent helical scale factors (filled from sizing.json)
_SZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sizing.json')
if os.path.exists(_SZ):
    K_HEL = {k: v['k'] for k, v in json.load(open(_SZ)).items()}


def h2_vessel(name, R, hd_R, rb_R, P, order='H-F-F x2 (ref.)', k_hel=None):
    a1, a2, t_hel, t_h = netting(R, rb_R, P)
    k = K_HEL.get(name, 1.0) if k_hel is None else k_hel
    if k != 1.0:
        # dome-consistent sizing: helical wall scaled by k, hoop re-sized from
        # netting hoop equilibrium with the thicker helical wall
        P_B = 2.25*P
        s1, s2 = np.sin(np.radians(a1))**2, np.sin(np.radians(a2))**2
        t_hel = k*t_hel
        t_h = P_B*R/core.MAT['Xt'] - 0.5*t_hel*(s1 + s2)
    h = ('H', t_h/2)
    f0 = [(0, +1, t_hel/8), (0, -1, t_hel/8)]
    f1 = [(1, +1, t_hel/8), (1, -1, t_hel/8)]
    st = {'H-F-F x2 (ref.)': [h] + f0 + f1 + [h] + f0 + f1,
          'F-F-H x2': f0 + f1 + [h] + f0 + f1 + [h],
          'H block inside': [h, h] + f0 + f1 + f0 + f1,
          'H block middle': f0 + f1 + [h, h] + f0 + f1,
          'H block outside': f0 + f1 + f0 + f1 + [h, h]}[order]
    return dict(name=name, ri=R, L=6.0*R, hd=hd_R*R, fam=[a1, a2], stack=st,
                mat=dict(core.MAT), p=P, R=R, hd_R=hd_R, rb_R=rb_R, P=P,
                t_hel=t_hel, t_h=t_h, k_hel=k)


VESSELS = {}
for i, (P, hd, rb) in enumerate(itertools.product((35.0, 70.0), (0.6, 0.8), (0.15, 0.25))):
    VESSELS[f'H{i+1}'] = dict(P=P, hd_R=hd, rb_R=rb)
# intermediate dome compactness series (70 MPa, r_b/R = 0.15) for the design rule
SERIES = {'S65': 0.65, 'S70': 0.70, 'S75': 0.75}
for nm, hd in SERIES.items():
    VESSELS[nm] = dict(P=70.0, hd_R=hd, rb_R=0.15)


def base(vn, order=None, k_hel=None):
    if vn in ('GFRP', 'CFRP'):
        return T.agne_case(vn, order or 'hoop_middle')
    v = VESSELS[vn]
    return h2_vessel(vn, R0, v['hd_R'], v['rb_R'], v['P'], order or 'H-F-F x2 (ref.)', k_hel)


def wj(b):
    return np.sqrt(b['ri']*sum(e[-1] for e in b['stack'] if e[0] != 'P'))


def e_crit(b, mu):
    w = wj(b)
    return brentq(lambda e: vt.slippage(b['ri'], b['hd'], e*w) - mu, 1e-5, 3.0)


def cases():
    out = []
    for vn in [v for v in VESSELS if v not in SERIES] + ['GFRP', 'CFRP']:
        b = base(vn)
        w = wj(b)
        for e in E_GRID:
            for l in L_GRID:
                out.append(dict(b, name=f'{vn}_e{e:+.3f}_l{l:.2f}', hoop_end=e*w, hoop_taper=l*w,
                                meta=dict(vessel=vn, study='hoop', e_w=e, l_w=l, order='ref')))
        orders = ORDERS[1:] if vn in VESSELS else ['hoop_inside', 'hoop_outside']
        for o in orders:
            bo = base(vn, o)
            out.append(dict(bo, name=f'{vn}_stack_{o.replace(" ", "_")}', hoop_end=0.0,
                            hoop_taper=0.5*w, meta=dict(vessel=vn, study='stack', order=o,
                                                         e_w=0.0, l_w=0.5)))
        for mu in (0.1, 0.2, 0.3):
            ec = e_crit(b, mu)
            for o in ([None, 'H block outside'] if vn in VESSELS else [None]):
                bo = base(vn, o)
                tag = 'ref' if o is None else 'out'
                out.append(dict(bo, name=f'{vn}_combo_mu{mu}_{tag}', hoop_end=ec*w, hoop_taper=0.5*w,
                                meta=dict(vessel=vn, study='combo', mu=mu, e_w=ec, l_w=0.5,
                                          order=o or 'ref')))
        if vn in VESSELS:
            t0 = sum(e[-1] for e in b['stack'] if e[0] == 0)
            for rtp in (0.4, 0.55, 0.7):
                for sp in (-0.5, -1.0, -2.0):
                    st = list(b['stack']) + [('P', +1, t0/4), ('P', -1, t0/4)]
                    out.append(dict(b, name=f'{vn}_hyb_rt{rtp}_sp{sp}', stack=st,
                                    patch=dict(rt=rtp*b['ri'], s_end=sp*w, l=0.5*w),
                                    hoop_end=0.0, hoop_taper=0.5*w,
                                    meta=dict(vessel=vn, study='hybrid', rtp=rtp, sp=sp,
                                              e_w=0.0, l_w=0.5, order='ref')))
    return out


def cases_series():
    out = []
    for vn in SERIES:
        b = base(vn)
        w = wj(b)
        for e in E_GRID:
            out.append(dict(b, name=f'{vn}_e{e:+.3f}_l0.50', hoop_end=e*w, hoop_taper=0.5*w,
                            meta=dict(vessel=vn, study='hoop', e_w=e, l_w=0.5, order='ref')))
        for e in (0.0, 0.25):
            for l in (0.25, 1.0):
                out.append(dict(b, name=f'{vn}_e{e:+.3f}_l{l:.2f}', hoop_end=e*w, hoop_taper=l*w,
                                meta=dict(vessel=vn, study='hoop', e_w=e, l_w=l, order='ref')))
        for o in ('H block outside', 'H block inside'):
            bo = base(vn, o)
            out.append(dict(bo, name=f'{vn}_stack_{o.replace(" ", "_")}', hoop_end=0.0, hoop_taper=0.5*w,
                            meta=dict(vessel=vn, study='stack', order=o, e_w=0.0, l_w=0.5)))
        for mu in (0.1, 0.2, 0.3):
            ec = e_crit(b, mu)
            for o in (None, 'H block outside'):
                bo = base(vn, o)
                tag = 'ref' if o is None else 'out'
                out.append(dict(bo, name=f'{vn}_combo_mu{mu}_{tag}', hoop_end=ec*w, hoop_taper=0.5*w,
                                meta=dict(vessel=vn, study='combo', mu=mu, e_w=ec, l_w=0.5, order=o or 'ref')))
    return out


def one(c):
    f = f'{OUT}/{c["name"]}.json'
    if os.path.exists(f):
        return json.load(open(f))
    meta = c.pop('meta')
    o = T.run_case(c, tag='pf')
    o['meta'] = meta
    os.makedirs(OUT, exist_ok=True)
    json.dump(o, open(f, 'w'), default=float)
    return o


if __name__ == '__main__':
    import sys
    cs = cases_series() if 'series' in sys.argv else cases()
    print(len(cs), 'cases', flush=True)
    with Pool(2) as pool:
        for o in pool.imap_unordered(one, cs):
            print(o['name'], o.get('ok'), round(o.get('pf', 0), 2), flush=True)
