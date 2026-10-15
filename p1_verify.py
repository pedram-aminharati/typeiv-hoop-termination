"""Verification of the Paper 1 models on the final (dome-consistent) vessel H5."""
import json
import numpy as np
import axi
import core
import il_study as I
import p1_study as PS
import tr_eval as T

OUT = 'p1_verify.json'


def main():
    import os
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    vn = 'H5'
    if 'profile' in res:
        return tail(res, vn)
    # --- (1) layer-wise model with the shell stacking (n_rep = 2): equilibrium, Lekhnitskii
    b = I.rep_stack(vn, 2)
    n_sub = max(1, int(round(b['t_h']/2/I.T_LAYER)))
    w = PS.wj(b)
    c = dict(b, name='ver', hoop_end=0.0, hoop_taper=0.5*w)
    meta = axi.build(c, 'runs/ver', n_sub=n_sub, ds_fine=0.25)
    ok, _ = axi.run('runs/ver')
    Sg, Eg = axi.read('runs/ver')
    col = sorted([e for e in meta['elems'] if e['ie'] == 0], key=lambda e: e['je'])
    R = b['ri']; r = R; Fth = 0; Fz = 0
    for e in col:
        s = Sg[e['id']].mean(0); t = e['t']/2
        Fth += s[2]*t; Fz += s[0]*t*2*np.pi*(r + t/2); r += t
    res['equilibrium'] = dict(hoop=Fth/(b['p']*R), axial=Fz/(np.pi*R**2*b['p']))
    F = axi.ply_fibre(meta, Eg)
    grp = {}
    for e in col:
        if e['id'] in F:
            key = e['bid'].split('.')[0]
            grp.setdefault(key, []).append(float(F[e['id']].mean()))
    pl = [(90.0 if e[0] == 'H' else e[1]*b['fam'][e[0]], e[-1]) for e in b['stack']]
    lk, _ = core.lekhnitskii(R, pl, b['p'])
    lks = []
    for ply in lk:
        s1, _, _ = core.to_ply(ply['sz'], ply['sth'], ply['tzth'], ply['theta'])
        lks.append((float(ply['theta']), float(np.mean(s1))))
    # pair the layer groups in order (H0, F0, F1, H1, F0, F1) with the Lekhnitskii plies
    order = []
    for e in col:
        k = e['bid'].split('.')[0]
        if e['id'] in F and (not order or order[-1] != k):
            order.append(k)
    axv = []
    seen = {}
    for e in col:
        if e['id'] not in F:
            continue
    # group means in through-thickness order
    cur, vals = None, []
    for e in col:
        if e['id'] not in F:
            continue
        k = e['bid'].split('.')[0] + ('' if e['kind'] == 'H' else str(e['je']//(2*meta['n_el_band'])))
        if k != cur:
            if cur is not None:
                axv.append((cur, float(np.mean(vals))))
            cur, vals = k, []
        vals.append(float(F[e['id']].mean()))
    axv.append((cur, float(np.mean(vals))))
    lk_pairs = []
    i = 0
    for th, v in lks:
        if th < 0:
            continue                       # +/- pair share sigma_1
        lk_pairs.append((th, v))
    res['lekhnitskii'] = dict(axi=axv, lek=lk_pairs)
    json.dump(res, open(OUT, 'w'), indent=1, default=float)
    # --- (2) fibre-ratio profile: layer-wise versus shell
    Xt = core.MAT['Xt']
    colmax = {}
    for e in meta['elems']:
        if e['id'] in F:
            v = float(F[e['id']].mean())/Xt
            if v > colmax.get(e['ie'], (0,))[0]:
                colmax[e['ie']] = (v, float(e['sc']))
    ax_prof = sorted((s, v) for v, s in colmax.values())
    o = T.run_case(dict(PS.base(vn), name='ver_shell', hoop_end=0.0, hoop_taper=0.5*w), tag='ver')
    sh_prof = sorted((p['s'], p['rho']) for p in o['prof'] if p['zone'] != 'boss')
    res['profile'] = dict(axi=ax_prof, shell=sh_prof, w=w, shell_mid=o['mid']['rho'])
    json.dump(res, open(OUT, 'w'), indent=1, default=float)
    return tail(res, vn)


def tail(res, vn):
    b = PS.base(vn)
    w = PS.wj(b)
    # --- (3) shell mesh convergence (reference and two extreme designs)
    conv = []
    for (e_w, l_w) in ((0.0, 0.5), (0.25, 0.1), (-0.5, 1.0)):
        cc = dict(PS.base(vn), name=f'conv_{e_w}_{l_w}', hoop_end=e_w*w, hoop_taper=l_w*w)
        o1 = T.run_case(cc, tag='conv1')
        o2 = T.run_case(dict(cc, name=cc['name'] + '_f'), tag='conv2', nb=48, nd=100, nc=48)
        conv.append(dict(e=e_w, l=l_w, pf1=o1['pf'], pf2=o2['pf'],
                         RT1=o1['rho']['max']/o1['mid']['rho'], RT2=o2['rho']['max']/o2['mid']['rho']))
    res['shell_conv'] = conv
    json.dump(res, open(OUT, 'w'), indent=1, default=float)
    # --- (4) layer-wise mesh convergence of the delamination index (production stack n_rep = 4)
    il = []
    for kw in (dict(ds_fine=0.5), dict(ds_fine=0.25), dict(ds_fine=0.125), dict(ds_fine=0.25, n_el_band=3)):
        o = I.case(vn, n_rep=4, **kw)
        if not o.get('ok'):
            il.append(dict(kw, failed=True)); continue
        il.append(dict(kw, nel=o['nel'], fi={d: o['fi'][d]['fi'] for d in o['fi']}))
    res['il_conv'] = il
    json.dump(res, open(OUT, 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != 'profile'}, indent=1, default=float))


if __name__ == '__main__':
    main()
