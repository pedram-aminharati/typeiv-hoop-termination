"""Deterministic first-fibre-failure efficiency eta = p_f / p_cyl with and without the
cure cool-down residual stress (Paper 1 definition: tensile fibre failure, boss excluded)."""
import json, os, shutil, sys
import numpy as np
sys.path.insert(0, '.')
import vessel_tr as vt
import p1_study as PS
from rel import study_full as F, thermal as TH

NAMES = ('H1-ref', 'H1-e0125', 'H5-ref', 'H5-e0125', 'H3-ref', 'H3-out20', 'H7-ref', 'H7-out20',
         'S70-ref', 'S70-out20')
XT = 2000.0


def pf(d, mid, th=True):
    s0 = d['sig_th'] if th else 0*d['sig_th']
    sh = d['sig']
    ok = (sh > 1e-9) & (d['zone'][:, None] != 3)
    p = np.where(ok, (XT - s0)/np.where(sh > 1e-9, sh, 1), np.inf)
    pm = np.where(ok & mid[:, None], p, np.inf)
    return float(p.min()), float(pm.min())


def main():
    fn = f'{F.OUT}/thermal_p1.json'
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    for nm in NAMES:
        if nm in res: continue
        spec = F.designs()[nm]
        c = F.nominal(spec, nm)
        path = f'./runs/thp1_{nm}'
        meta = TH.build(c, path)
        ok, log = vt.run(path, ncpu=1)
        d = TH.extract(path, meta, c)
        rows = meta['rowinfo']
        zc = []
        for e, er, ej in meta['elems']:
            if ej != 0: continue
            zc += [rows[er]['zc']]*len(rows[er]['plies'])
        zc = np.array(zc)
        mid = np.abs(zc) <= np.sort(np.abs(zc))[0] + 1e-6
        p0, pc0 = pf(d, mid, th=False)
        p1, pc1 = pf(d, mid, th=True)
        res[nm] = dict(pf=p0, pcyl=pc0, eta=p0/pc0, pf_th=p1, pcyl_th=pc1, eta_th=p1/pc1,
                       sth_min=float(d['sig_th'].min()), sth_max=float(d['sig_th'].max()))
        shutil.rmtree(path, ignore_errors=True)
        json.dump(res, open(fn, 'w'), indent=1)
        print(nm, {k: round(v, 3) for k, v in res[nm].items()}, flush=True)

if __name__ == '__main__':
    main()
