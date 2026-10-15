"""Paper 1: 3D (axisymmetric layer-wise) interlaminar study of the transition."""
import json
import os
import sys
import time
import numpy as np
import axi
import interlam as IL
import p1_study as PS
import core
import vessel_tr as vt

OUT = 'results_il'


def rep_stack(vn, n_rep, order='ref'):
    """netting vessel vn with its hoop and helical thickness split into n_rep
    interleaved repeats (ref: H-F0-F1 per repeat; out: hoop block outside;
    in: hoop block inside)"""
    b = PS.base(vn)
    th, tf = b['t_h']/n_rep, b['t_hel']/(4*n_rep)
    h = ('H', th)
    f = [(0, +1, tf), (0, -1, tf), (1, +1, tf), (1, -1, tf)]
    if order == 'ref':
        st = ([h] + f)*n_rep
    elif order == 'out':
        st = f*n_rep + [h]*n_rep
    elif order == 'in':
        st = [h]*n_rep + f*n_rep
    b = dict(b, stack=st)
    return b


T_LAYER = 0.34      # cured hoop winding-layer thickness (mm), fixed across n_rep


def case(vn, n_rep=2, order='ref', e_w=0.0, l_w=0.5, n_sub=None, ds_fine=0.5, n_el_band=2,
         tag=None, deltas=(0.0, 0.25, 0.5, 1.0), Lp_fac=4.0, hyb=None):
    b = rep_stack(vn, n_rep, order)
    if n_sub is None:
        n_sub = max(1, int(round(b['t_h']/n_rep/T_LAYER)))
    if hyb is not None:
        rtp, sp = hyb
        w0 = PS.wj(b)
        t0 = b['t_hel']/2
        b = dict(b, stack=list(b['stack']) + [('P', +1, t0/4), ('P', -1, t0/4)],
                 patch=dict(rt=rtp*b['ri'], s_end=sp*w0, l=0.5*w0))
    w = PS.wj(b)
    c = dict(b, name='il', hoop_end=e_w*w, hoop_taper=l_w*w)
    tag = tag or (f'{vn}_r{n_rep}_{order}_e{e_w:+.3f}_l{l_w:.2f}_n{n_sub}_d{ds_fine}_b{n_el_band}'
                  + (f'_Lp{Lp_fac:g}' if Lp_fac != 4.0 else '')
                  + (f'_hyb{hyb[0]:g}_{hyb[1]:g}' if hyb else ''))
    f = f'{OUT}/{tag}.json'
    if os.path.exists(f):
        return json.load(open(f))
    path = f'runs/il_{tag}'
    t0 = time.time()
    meta = axi.build(c, path, n_sub=n_sub, ds_fine=ds_fine, n_el_band=n_el_band, Lp_fac=Lp_fac)
    ok, log = axi.run(path, ncpu=1)
    if not ok:
        return dict(tag=tag, ok=False, log=log)
    g = meta['g']
    win = (min(g['he'] - g['lh'], 0.0) - w, max(g['he'], 0.0) + w)
    res, pk, prof = IL.evaluate(meta, path, win, deltas)
    Sg, Eg = axi.read(path)
    F = axi.ply_fibre(meta, Eg)
    Xt = core.MAT['Xt']
    rho = {}
    for e in meta['elems']:
        if e['id'] in F:
            rho[e['ie']] = max(rho.get(e['ie'], 0.0), float(F[e['id']].mean())/Xt)
    sc = {e['ie']: e['sc'] for e in meta['elems']}
    rprof = sorted((sc[k], v) for k, v in rho.items())
    s_arr = np.array([a for a, _ in rprof]); r_arr = np.array([v for _, v in rprof])
    mid = r_arr[np.argmin(abs(s_arr - s_arr.min()))]
    mtr = (s_arr >= win[0]) & (s_arr <= win[1])
    # interface profiles kept for figures (window +- 1.5 w only, every interface)
    keep = {}
    for k, p in prof.items():
        m = (p['s'] >= win[0] - 0.5*w) & (p['s'] <= win[1] + 0.5*w)
        keep[k] = dict(below=p['below'], above=p['above'], s=p['s'][m].tolist(),
                       snn=p['snn'][m].tolist(), tsn=p['tsn'][m].tolist())
    out = dict(tag=tag, ok=True, vessel=vn, n_rep=n_rep, order=order, e_w=e_w, l_w=l_w,
               Lp_fac=Lp_fac, hyb=hyb, mass=vt.composite_mass(meta) if False else None,
               n_sub=n_sub, ds_fine=ds_fine, n_el_band=n_el_band, p=b['p'], w=w,
               lh=g['lh'], he=g['he'], window=win, nel=len(meta['elems']),
               t_sub=b['t_h']/n_rep/n_sub,
               fi={str(k): v for k, v in res.items()}, peaks=pk,
               rho_mid=float(mid), rho_tr=float(r_arr[mtr].max()),
               RT=float(r_arr[mtr].max()/mid), rho_prof=rprof, iface=keep,
               cpu=time.time() - t0)
    # pressure at delamination onset (quadratic criterion scales with p^2)
    out['pd'] = {d: b['p']/np.sqrt(v['fi']) for d, v in out['fi'].items()}
    out['pf'] = b['p']/float(r_arr.max())
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f, 'w'), default=float)
    import shutil
    shutil.rmtree(path, ignore_errors=True)
    return out


def cases():
    out = []
    P = dict(n_rep=4, ds_fine=0.25)
    for vn in ('H5', 'H1', 'H7', 'H3'):
        b = rep_stack(vn, 4)
        for e in (-0.5, -0.25, 0.0, 0.125, 0.25, 0.375, 0.5):
            out.append(dict(vn=vn, e_w=e, l_w=0.5, **P))
        for e in (0.0, 0.25):
            for l in (0.25, 1.0):
                out.append(dict(vn=vn, e_w=e, l_w=l, **P))
        for mu in (0.1, 0.2, 0.3):
            ec = round(PS.e_crit(b, mu), 4)
            out.append(dict(vn=vn, e_w=ec, l_w=0.5, **P))
            out.append(dict(vn=vn, e_w=ec, l_w=1.0, **P))
            out.append(dict(vn=vn, e_w=ec, l_w=0.5, order='out', **P))
        for o in ('out', 'in'):
            out.append(dict(vn=vn, e_w=0.0, l_w=0.5, order=o, **P))
        for r in (2, 8):
            out.append(dict(vn=vn, e_w=0.0, l_w=0.5, n_rep=r, ds_fine=0.25))
        for hyb in ((0.55, -1.0), (0.4, -1.0)):
            out.append(dict(vn=vn, e_w=0.0, l_w=0.5, hyb=hyb, **P))
    for Lp in (2.0, 8.0):
        out.append(dict(vn='H5', e_w=0.0, l_w=0.5, Lp_fac=Lp, **P))
    return out


def one(kw):
    kw = dict(kw)
    vn = kw.pop('vn')
    try:
        o = case(vn, **kw)
    except Exception as ex:                    # keep the batch going
        return dict(tag=str(kw), ok=False, log=repr(ex))
    return o


if __name__ == '__main__':
    cs = cases()
    print(len(cs), 'cases', flush=True)
    for kw in cs:
        o = one(kw)
        v = o.get('fi', {}).get('1.0', {})
        print(o.get('tag'), o.get('ok'), round(v.get('fi', -1), 3), round(o.get('cpu', 0), 1),
              '' if o.get('ok') else o.get('log', '')[-200:], flush=True)
