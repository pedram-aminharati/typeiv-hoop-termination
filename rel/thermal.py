"""Cure cool-down residual stresses (item 8 of the code plan).

Step 1 cools the laminate from the stress-free temperature to room
temperature (no pressure); step 2 adds the internal pressure.  The fibre
stress at pressure p is then sigma_1 = sigma_th + p * s_hat, which is no longer
proportional to p, so the weakest-link failure probability is evaluated
directly at each pressure.
"""
import os
import re
import numpy as np
import vessel_tr as vt

# carbon/epoxy expansion coefficients [1/K] (fibre, transverse, transverse)
ALPHA = (-0.3e-6, 30e-6, 30e-6)


def build(c, path, dT=-100.0, T0=120.0, **mesh):
    meta = vt.build(c, path, **mesh)
    fn = os.path.join(path, 'v.inp')
    s = open(fn).read()
    # orthotropic expansion right after the elastic constants
    s = re.sub(r'(\*ELASTIC, TYPE=ENGINEERING CONSTANTS\n[^\n]*\n[^\n]*\n)',
               lambda m: m.group(1) + '*EXPANSION, TYPE=ORTHO\n' + ','.join(f'{a:.3e}' for a in ALPHA) + '\n', s)
    T1 = T0 + dT
    head, step = s.split('*STEP', 1)
    head += f'*INITIAL CONDITIONS, TYPE=TEMPERATURE\nNALL,{T0}\n'
    cool = (f'*STEP\n*STATIC\n*TEMPERATURE\nNALL,{T1}\n'
            '*EL PRINT, ELSET=EALL\nS\n*END STEP\n')
    step = '*STEP' + step.replace('*STATIC\n', f'*STATIC\n*TEMPERATURE\nNALL,{T1}\n', 1)
    open(fn, 'w').write(head + cool + step)
    return meta


def read_steps(path):
    """stress blocks of every step (list of dicts element -> array)"""
    blocks, cur = [], None
    with open(os.path.join(path, 'v.dat')) as f:
        for line in f:
            if 'stresses (elem' in line:
                cur = {}
                blocks.append(cur)
                continue
            if 'total force' in line or 'displacements' in line:
                cur = None
                continue
            parts = line.split()
            if cur is not None and len(parts) >= 8:
                try:
                    e = int(parts[0])
                except ValueError:
                    continue
                cur.setdefault(e, []).append([float(x) for x in parts[2:8]])
    return [{e: np.array(v) for e, v in b.items()} for b in blocks]


def extract(path, meta, c):
    """fibre stress: thermal part and part per unit pressure, with volumes"""
    B = read_steps(path)
    Sth, Sp = B[0], B[-1]
    p = c['p']
    th, sh, vol, zone = [], [], [], []
    from rel.weakest import ZC
    for e, er, ej in meta['elems']:
        if ej != 0:
            continue
        info = meta['rowinfo'][er]
        n = len(info['plies'])
        a = Sth[e].reshape(n, 8, 6)[:, :, 0]
        b = Sp[e].reshape(n, 8, 6)[:, :, 0]
        for k, (ang, t, nm) in enumerate(info['plies']):
            th.append(a[k]); sh.append((b[k] - a[k])/p)
            vol.append(2*np.pi*info['rc']*info['seglen']*t); zone.append(ZC[info['zone']])
    return dict(sig_th=np.array(th), sig=np.array(sh), vol=np.array(vol), zone=np.array(zone))


def pf(d, wb, p, exclude_boss=True):
    s = np.clip(d['sig_th'] + p*d['sig'], 0, None)
    term = d['vol']/wb.V0*np.mean((s/wb.s0)**wb.m, axis=1)
    if exclude_boss:
        term = np.where(d['zone'] == 3, 0.0, term)
    return 1 - np.exp(-term.sum())
