"""Resumable, memory-aware job runner with automatic verification.

Every case is identified by a hash of its design dictionary.  A finished case
leaves  <store>/<hash>.npz (weakest-link data) and <store>/<hash>.json
(design, checks, timing); a re-run skips it.  Jobs are launched only when
enough memory is free, and the CalculiX directory is deleted after extraction.
"""
import hashlib
import json
import os
import shutil
import sys
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, '.')
import core
import vessel_tr as vt
from rel import weakest as W

STORE = './rel_store'
MIN_FREE_MB = 900


def key(c):
    d = {k: v for k, v in c.items() if k not in ('name', 'meta')}
    s = json.dumps(d, sort_keys=True, default=lambda o: float(o) if np.isscalar(o) else str(o))
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def free_mb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable'):
            return int(line.split()[1])/1024
    return 1e9


def checks(c, d, meta):
    """automatic verification of every analysis"""
    out = {}
    R, p = c['ri'], c['p']
    out['rf_rel'] = float(d['rf']/(p*np.pi*R**2)) if np.isfinite(d['rf']) else None
    # left/right symmetry of the fibre-stress field
    L = d['sig'][d['side'] < 0].max()
    Rr = d['sig'][d['side'] > 0].max()
    out['sym'] = float(abs(L - Rr)/Rr)
    # barrel hoop/helical fibre stress against the Lekhnitskii solution
    try:
        mid = min(range(len(meta['rowinfo'])), key=lambda i: abs(meta['rowinfo'][i]['zc']))
        info = meta['rowinfo'][mid]
        plies = [(a, t) for a, t, _ in info['plies']]
        m = dict(core.MAT); m.update(c['mat'])
        res, _ = core.lekhnitskii(R, plies, p, m=m)
        lek = []
        for q in res:
            th = np.radians(q['theta'])
            s1 = q['sz']*np.cos(th)**2 + q['sth']*np.sin(th)**2 + 2*q['tzth']*np.sin(th)*np.cos(th)
            lek.append(np.mean(s1))
        sel = d['row'] == mid
        fe = d['sig'][sel].mean(axis=1)*p
        out['lek_dev'] = float(np.max(np.abs(fe - np.array(lek))/np.max(np.abs(lek))))
    except Exception as ex:                                  # noqa: BLE001
        out['lek_dev'] = None
        out['lek_err'] = str(ex)[:120]
    out['ok'] = (out['rf_rel'] is None or out['rf_rel'] < 1e-4) and out['sym'] < 1e-3 and \
        (out['lek_dev'] is None or out['lek_dev'] < 0.08)
    return out


def run_one(c, mesh=None, keep=False):
    h = key(c)
    fn = os.path.join(STORE, h)
    if os.path.exists(fn + '.json'):
        return json.load(open(fn + '.json'))
    path = f'./runs/rel_{h}'
    t0 = time.time()
    meta = vt.build(c, path, variant=c.get('variant', 'V1'), **(c.get('mesh') or mesh or {}))
    ok, log = vt.run(path, ncpu=1)
    if not ok:
        rec = dict(hash=h, name=c.get('name'), ok=False, log=log[-300:])
    else:
        d = W.extract(path, meta, c, vt.read_dat)
        W.save(d, fn + '.npz')
        rec = dict(hash=h, name=c.get('name'), ok=True, checks=checks(c, d, meta),
                   mass=vt.composite_mass(meta), time=time.time() - t0,
                   design={k: v for k, v in c.items() if k != 'mat'}, meta=c.get('meta', {}))
    os.makedirs(STORE, exist_ok=True)
    json.dump(rec, open(fn + '.json', 'w'), default=lambda o: float(o) if np.isscalar(o) else str(o))
    if not keep:
        shutil.rmtree(path, ignore_errors=True)
    return rec


def run_many(cases, workers=2, mesh=None):
    todo = [c for c in cases if not os.path.exists(os.path.join(STORE, key(c) + '.json'))]
    done = [json.load(open(os.path.join(STORE, key(c) + '.json'))) for c in cases if c not in todo]
    print(f'{len(done)} cached, {len(todo)} to run', flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for c in todo:
            while free_mb() < MIN_FREE_MB:
                time.sleep(5)
            futs[ex.submit(run_one, c, mesh)] = c
            time.sleep(1.0)
        for i, f in enumerate(as_completed(futs)):
            try:
                r = f.result()
            except Exception as e:                           # noqa: BLE001
                r = dict(name=futs[f].get('name'), ok=False, log=str(e)[:300])
            done.append(r)
            if i % 10 == 0:
                print(f'  {i+1}/{len(todo)} {r.get("name")} ok={r.get("ok")}', flush=True)
    return done


def data(c):
    return W.load(os.path.join(STORE, key(c) + '.npz'))
