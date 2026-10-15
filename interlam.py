"""Interlaminar stresses and Brewer-Lagace delamination index from the
axisymmetric layer-wise model.

sigma_nn and tau_sn are continuous across ply interfaces, so they are taken
from the extrapolated, nodal-averaged stresses (.frd, global r-z-theta) and
rotated to the local meridian frame (s, n). The quadratic delamination
criterion of Brewer and Lagace (1988),
    FI_d = (<sigma_nn>/Z_t)^2 + (tau_sn/S_13)^2,
is evaluated on every ply interface with the average-stress approach: both
components are averaged over a characteristic length delta along the
interface before the criterion is applied.
"""
import os
import numpy as np
import axi

ZT = 50.0      # transverse (through-thickness) tensile strength, = Y_t of the ply
S13 = 80.0     # interlaminar shear strength, = S_12 (transversely isotropic ply)


def frd_stress(path):
    out = {}
    blk = False
    for line in open(os.path.join(path, 'a.frd')):
        if line.startswith(' -4  STRESS'):
            blk = True; continue
        if blk:
            if line.startswith(' -3'):
                break
            if line.startswith(' -1'):
                nid = int(line[3:13])
                v = [float(line[13 + 12*k:25 + 12*k]) for k in range(6)]
                out[nid] = v      # sxx(r) syy(z) szz(th) sxy(rz) syz szx
    return out


def interface_profiles(meta, N):
    """per interface k (between band k-1 and k): s, r, sigma_nn, tau_sn"""
    g, S, nodes = meta['g'], meta['S'], meta['nodes']
    nb, ne = len(meta['bands']), meta['n_el_band']
    prof = {}
    for k in range(1, nb):
        j = 2*ne*k
        rows = []
        X = meta['coords']
        ns = len(S)
        for i, s in enumerate(S):
            n = nodes[(i, j)]
            sxx, syy, szz, sxy = N[n][:4]
            # interface's own tangent from its neighbouring nodes on the same row
            a = np.array(X[nodes[(max(i - 1, 0), j)] - 1])
            b = np.array(X[nodes[(min(i + 1, ns - 1), j)] - 1])
            t = (b - a)/np.linalg.norm(b - a)
            nv = np.array([t[1], -t[0]])
            sig = np.array([[sxx, sxy], [sxy, syy]])
            rows.append((s, nv @ sig @ nv, t @ sig @ nv))
        a = np.array(rows)
        prof[k] = dict(s=a[:, 0], snn=a[:, 1], tsn=a[:, 2],
                       below=meta['bands'][k - 1]['id'], above=meta['bands'][k]['id'])
    return prof


def avg(s, y, delta):
    """centred moving average of y(s) over a length delta (trapezoidal)"""
    if delta <= 0:
        return y.copy()
    out = np.empty_like(y)
    for q, s0 in enumerate(s):
        lo, hi = s0 - delta/2, s0 + delta/2
        ss = np.linspace(max(lo, s[0]), min(hi, s[-1]), 41)
        out[q] = np.trapezoid(np.interp(ss, s, y), ss)/(ss[-1] - ss[0])
    return out


def fi(snn, tsn):
    return (np.maximum(snn, 0)/ZT)**2 + (tsn/S13)**2


def evaluate(meta, path, window, deltas=(0.0, 0.25, 0.5, 1.0)):
    N = frd_stress(path)
    prof = interface_profiles(meta, N)
    res = {}
    for d in deltas:
        best = dict(fi=-1)
        for k, p in prof.items():
            m = (p['s'] >= window[0]) & (p['s'] <= window[1])
            sn, ts = avg(p['s'], p['snn'], d), avg(p['s'], p['tsn'], d)
            f = fi(sn, ts)
            q = np.argmax(np.where(m, f, -1))
            if f[q] > best['fi']:
                best = dict(fi=float(f[q]), s=float(p['s'][q]), k=k, below=p['below'],
                            above=p['above'], snn=float(sn[q]), tsn=float(ts[q]))
        res[d] = best
    # peak magnitudes of the unaveraged components in the window
    pk = dict(snn_max=-1e9, snn_min=1e9, tsn_abs=0.0)
    for p in prof.values():
        m = (p['s'] >= window[0]) & (p['s'] <= window[1])
        pk['snn_max'] = max(pk['snn_max'], float(p['snn'][m].max()))
        pk['snn_min'] = min(pk['snn_min'], float(p['snn'][m].min()))
        pk['tsn_abs'] = max(pk['tsn_abs'], float(np.abs(p['tsn'][m]).max()))
    return res, pk, prof
