"""Keep element fields of the drop-off region for contour figures."""
import numpy as np
import axi
import il_study as I
import p1_study as PS


def fields(vn, e_w, l_w, order='ref', n_rep=4, tag='fld', zoom=(-1.6, 0.9)):
    b = I.rep_stack(vn, n_rep, order)
    n_sub = max(1, int(round(b['t_h']/n_rep/I.T_LAYER)))
    w = PS.wj(b)
    c = dict(b, name='fld', hoop_end=e_w*w, hoop_taper=l_w*w)
    path = f'runs/{tag}'
    meta = axi.build(c, path, n_sub=n_sub, ds_fine=0.25, n_el_band=2)
    ok, _ = axi.run(path, ncpu=1)
    assert ok
    Sg, Eg = axi.read(path)
    X = meta['coords']
    polys, tsn, snn, kind = [], [], [], []
    for e in meta['elems']:
        if not (zoom[0]*w <= e['sc'] <= zoom[1]*w):
            continue
        pts = [X[n - 1] for n in e['conn'][:4]]
        polys.append(pts)
        s = Sg[e['id']].mean(0)                 # local frame: s, n, theta
        snn.append(s[1]); tsn.append(s[3])
        kind.append(2 if e['resin'] else (1 if e['kind'] == 'H' else 0))
    g = meta['g']
    np.savez(f'runs/{tag}.npz', polys=np.array(polys), tsn=np.array(tsn), snn=np.array(snn),
             kind=np.array(kind), w=w, R=g['R'], zt=g['L']/2, he=g['he'], lh=g['lh'], p=b['p'])
    import shutil
    shutil.rmtree(path, ignore_errors=True)


if __name__ == '__main__':
    import sys
    b = I.rep_stack('H5', 4)
    ec = PS.e_crit(b, 0.2)
    fields('H5', 0.0, 0.5, tag='fld_H5_ref')
    fields('H5', round(ec, 4), 1.0, tag='fld_H5_best')
    fields('H5', 0.0, 0.5, order='out', tag='fld_H5_out')
    print('done', ec)
