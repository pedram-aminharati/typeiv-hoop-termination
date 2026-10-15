"""Run and evaluate one transition-design case."""
import json
import os
import shutil
import numpy as np
import core
import vessel_tr as vt
import bench_agne as ba


def matdict(m):
    d = dict(core.MAT)
    d.update(m)
    d['S23'] = d['Yt']
    d['p_psi'] = d['S12']/d['Yc']
    return d


def evaluate(path, meta, c):
    S, rf, U = vt.read_dat(path)
    m = matdict(c['mat'])
    prof = []
    zmax = {}
    for e, er, ej in meta['elems']:
        info = meta['rowinfo'][er]
        if ej != 0:
            continue                       # axisymmetric: one meridian suffices
        s = S[e].reshape(len(info['plies']), 8, 6)
        s1, s2, t12 = s[:, :, 0], s[:, :, 1], -s[:, :, 3]
        cr = core.criteria(s1, s2, t12, m)
        rho = np.where(s1 >= 0, s1/m['Xt'], -s1/m['Xc']).max(axis=1)
        rec = dict(s=info['s'], rc=info['rc'], zc=info['zc'], zone=info['zone'],
                   t=info['ttot'], n=len(info['plies']))
        for k in ('tri', 'f', 'm'):
            v = cr[k].max(axis=1)
            rec[k] = float(v.max())
            rec[k + '_ply'] = info['plies'][int(v.argmax())][2]
        rec['rho'] = float(rho.max())
        rec['rho_ply'] = info['plies'][int(rho.argmax())][2]
        prof.append(rec)
    right = [p for p in prof if p['zc'] >= 0]
    mid = min(right, key=lambda p: abs(p['zc']))
    out = dict(name=c['name'], prof=right, mid=mid,
               rf=float(np.linalg.norm(rf)) if rf else None,
               mass=vt.composite_mass(meta))
    for k in ('tri', 'f', 'm', 'rho'):
        z = {}
        for p in right:
            if p['zone'] == 'boss':
                continue
            z[p['zone']] = max(z.get(p['zone'], 0.0), p[k])
        loc = max(z, key=z.get)
        best = max((p for p in right if p['zone'] != 'boss'), key=lambda p: p[k])
        out[k] = dict(zones=z, max=z[loc], loc=loc, s=best['s'],
                      ply=best[k + '_ply'], mid=mid[k])
    # transition-to-barrel ratios (> 1: the transition governs)
    for k in ('tri', 'f', 'rho'):
        tr = max(out[k]['zones'].get('junction', 0.0), out[k]['zones'].get('dome', 0.0))
        out[k]['RJ'] = tr/mid[k]
    out['pf'] = c['p']/out['rho']['max']
    out['pf_barrel'] = c['p']/mid['rho']
    return out


def run_case(c, tag='tr', keep=False, **mesh):
    path = f'runs/{tag}_{c["name"]}'
    meta = vt.build(c, path, **mesh)
    ok, log = vt.run(path, ncpu=1)
    if not ok:
        return dict(name=c['name'], ok=False, log=log[-400:])
    out = evaluate(path, meta, c)
    out['ok'] = True
    out['design'] = {k: v for k, v in c.items() if k not in ('mat',)}
    if not keep:
        shutil.rmtree(path, ignore_errors=True)
    return out


def agne_case(name, order='hoop_middle', **design):
    fam, st = ba.stack(name, order)
    c = dict(name=f'{name}_{order}', ri=160.0, L=522.5, hd=126.0, fam=fam,
             stack=st, mat=ba.material(name), hoop=88.0, p=10.0)
    c.update(design)
    return c
