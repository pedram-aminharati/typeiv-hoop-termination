import numpy as np
def metrics(r):
    d = r['design']
    st = d['stack']
    ttot = sum(e[-1] for e in st if e[0] != 'P')
    wj = np.sqrt(d['ri']*ttot)
    lh = d.get('hoop_taper', 0.5*wj); he = d.get('hoop_end', 0.0)
    s0 = he - lh
    if d.get('patch'):
        s0 = min(s0, d['patch']['s_end'])
    s0 -= wj
    pr = [p for p in r['prof'] if p['zone'] != 'boss']
    tr = [p for p in pr if p['s'] >= s0]
    out = {}
    for k in ('rho', 'tri', 'f'):
        best = max(tr, key=lambda p: p[k])
        out[k] = dict(RT=best[k]/r['mid'][k], s=best['s']/wj, ply=best[k+'_ply'], zone=best['zone'])
    out['pf'] = r['pf']; out['pf_mid'] = r['pf_barrel']; out['mass'] = r['mass']; out['wj'] = wj
    return out
