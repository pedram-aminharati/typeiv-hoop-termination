"""Agne et al. (2025) CNG Type IV benchmark with the rebuilt pipeline.

Data: Agne, Almeida Jr., Maruyama, de Souza, Amico, Tonatto, Compos. Struct.
370 (2025) 119424, Tables 2-5 (open version: LUME UFRGS 10183/294225).
Each listed helical layer is a +/- pair (two plies), which reproduces the
measured cylinder thickness (GFRP 49 x 0.37 = 18.1 mm vs 18.00 mm;
CFRP 28 x 0.45 = 12.6 mm vs 12.55 mm).
"""
import json
import sys
import numpy as np
import vessel_gen as vg
import core

MATS = {
    'GFRP': dict(E1=43490., E2=9720., nu12=0.267, G12=3106.,
                 Xt=625.3, Xc=360.1, Yt=7., Yc=74.3, S12=40.42),
    'CFRP': dict(E1=123200., E2=8200., nu12=0.291, G12=2927.,
                 Xt=830.7, Xc=532.7, Yt=13.7, Yc=86.1, S12=41.01),
}
LAYUP = {   # angle: number of listed layers (+/- pairs); 88 = hoop plies
    'GFRP': ({12: 4, 18: 3, 23: 2, 30: 2, 35: 2, 42: 2}, 19, 0.37, 29.39),
    'CFRP': ({12: 3, 23: 1, 30: 1, 35: 1, 42: 1, 55: 2}, 10, 0.45, 27.04),
}
NU23 = 0.4   # assumed, not reported
NC = 32      # circumferential elements, as in the DOE


def material(name):
    m = dict(MATS[name])
    m.update(E3=m['E2'], nu13=m['nu12'], nu23=NU23, G13=m['G12'],
             G23=m['E2']/(2*(1+NU23)))
    return m


def stack(name, order):
    """Sublaminate grouping: each family -> one +a and one -a layer with the
    summed thickness (CalculiX composite shells fail above ~20 layers)."""
    hel, nh, tp, _ = LAYUP[name]
    fam = sorted(hel)
    grp = [[(k, +1, hel[a]*tp), (k, -1, hel[a]*tp)] for k, a in enumerate(fam)]
    H = nh*tp
    half = len(grp)//2
    if order == 'hoop_middle':      # CalculiX 2.21 accepts at most 13 layers
        st = sum(grp[:half], []) + [('H', H)] + sum(grp[half:], [])
    elif order == 'hoop_inside':
        st = [('H', H)] + sum(grp, [])
    elif order == 'hoop_outside':
        st = sum(grp, []) + [('H', H)]
    return fam, st


def run(name, order='hoop_middle', pref=10.0, variant='V1', mesh=(16, 50, None)):
    fam, st = stack(name, order)
    m = material(name)
    c = dict(name=f'agne_{name}_{order}', ri=160.0, L=522.5, hd=126.0,
             fam=fam, stack=st, mat=m, hoop=88.0, p=pref)
    path = f'runs/{c["name"]}'
    meta = vg.build_inp(c, path, variant=variant, nb=mesh[0], nd=mesh[1], nc=mesh[2] or NC)
    ok, log = vg.run(path, ncpu=2)
    assert ok, log
    S, rf, U = vg.read_dat(path)
    rho_max, where = 0.0, None
    zmax = {}
    prof = []
    for e, er, ej in meta['elems']:
        info = meta['rowinfo'][er]
        if info['zone'] == 'boss':
            continue
        s = S[e].reshape(len(info['plies']), 8, 6)
        s1 = s[:, :, 0]
        rho = np.where(s1 >= 0, s1/m['Xt'], np.abs(s1)/m['Xc']).max(axis=1)
        k = int(rho.argmax())
        if ej == 0:
            prof.append(dict(rc=info['rc'], zc=info['zc'], zone=info['zone'],
                             rho=float(rho[k]), kappa=info['kappa'],
                             plies=[(float(a), float(t), n) for a, t, n in info['plies']]))
        zmax[info['zone']] = max(zmax.get(info['zone'], 0), float(rho[k]))
        if rho[k] > rho_max:
            rho_max = float(rho[k])
            where = (info['zone'], round(info['rc'], 1), round(info['zc'], 1),
                     info['plies'][k][2], round(info['plies'][k][0], 1))
    pf = pref/rho_max
    pexp = LAYUP[name][3]
    t_bar = sum(e[-1] for e in st)
    # Lekhnitskii barrel fibre proxy for the same stack
    pl = [(88.0 if e[0] == 'H' else e[1]*fam[e[0]], e[-1]) for e in st]
    mm = dict(core.MAT)
    mm.update(m)
    res, _ = core.lekhnitskii(160.0, pl, pref, m=mm)
    rl = 0.0
    for ply in res:
        s1, _, _ = core.to_ply(ply['sz'], ply['sth'], ply['tzth'], ply['theta'])
        rl = max(rl, float(np.max(np.where(s1 >= 0, s1/m['Xt'], -s1/m['Xc']))))
    out = dict(variant=variant, material=name, order=order, t_barrel=t_bar, n_layers=len(st),
               pf=pf, pexp=pexp, err=(pf - pexp)/pexp, where=where,
               zone_rho={z: pref/v for z, v in zmax.items()},
               pf_lekhnitskii=pref/rl, rf=float(np.linalg.norm(rf)), prof=prof,
               rt=meta['g']['rt'], a0=meta['g']['a0'], rb=meta['g']['rb'])
    return out


if __name__ == '__main__':
    allr = []
    for name in ('GFRP', 'CFRP'):
        for order in ('hoop_middle', 'hoop_inside', 'hoop_outside'):
            allr.append(run(name, order))
        for v in ('V0', 'V2', 'V3'):
            allr.append(run(name, 'hoop_middle', variant=v))
        for mesh in ((8, 25, 16), (24, 80, 40)):
            r = run(name, 'hoop_middle', mesh=mesh)
            r['mesh'] = mesh
            allr.append(r)
    for r in allr:
        print(r['material'], r['order'], r['variant'], r.get('mesh'), round(r['pf'], 2),
              round(100*r['err'], 1), r['where'], flush=True)
    json.dump(allr, open('results/agne.json', 'w'), indent=1)
