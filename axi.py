"""Axisymmetric layer-by-layer solid model of the half vessel (CalculiX CAX8).

Purpose: interlaminar stresses at the hoop drop-off that the shell model
cannot resolve. Each hoop group is split into n_sub sub-plies with staggered
ends; beyond its end a sub-ply collapses over a resin-pocket length L_p to a
thin resin band (t_min). Each helical +/- pair is a homogenised orthotropic
layer (the coupling terms of +alpha and -alpha cancel), with the Clairaut
angle and the thickness law of the shell model. Local axes per element:
1 = meridian tangent s, 2 = outward normal n, 3 = hoop theta.
Symmetry u_z = 0 at the mid-plane z = 0; internal pressure on the liner
surface; boss closure force as flange pressure (as in the shell model).
"""
import os
import subprocess
import numpy as np
import core
import vessel_gen as vg
import vessel_tr as vt

RESIN = dict(E=3500.0, nu=0.35, Xt=70.0)


def C3(theta_deg, m):
    """3D stiffness of a UD ply in the frame (s, n, theta) with the fibre at
    theta_deg from s toward theta; Voigt order (11,22,33,23,13,12) in that
    frame, i.e. 1 = s, 2 = n, 3 = theta."""
    # core.stiffness_cyl returns frame (x = z/s, y = theta, 3 = r/n) with
    # Voigt (xx, yy, 33, y3, x3, xy); permute to (s, n, theta)
    Cc = core.stiffness_cyl(theta_deg, m)
    # index map: s<-x(0), n<-3(2), th<-y(1); shear: n-th <- y3(3), s-th <- xy(5), s-n <- x3(4)
    perm = [0, 2, 1, 3, 5, 4]
    return Cc[np.ix_(perm, perm)]


def ortho_constants(C):
    """CalculiX ORTHO: D1111,D1122,D2222,D1133,D2233,D3333,D1212,D1313,D2323
    with Voigt (11,22,33,23,13,12): D1212 = C[5,5], D1313 = C[4,4], D2323 = C[3,3]."""
    return [C[0, 0], C[0, 1], C[1, 1], C[0, 2], C[1, 2], C[2, 2], C[5, 5], C[4, 4], C[3, 3]]


def pair_C(alpha, m):
    return 0.5*(C3(alpha, m) + C3(-alpha, m))


def bands_of(c, g, n_sub, Lp_fac, t_min):
    """ordered band list (inner -> outer); each band is a function of (s, r)
    returning (kind, angle, thickness, is_resin)."""
    bands = []
    R = g['R']
    gv = dict(R=R, rt=g['rt'], a0=g['a0'], tk0=g['tk0'])
    hcount = 0
    n_grp = sum(1 for ent in c['stack'] if ent[0] == 'H')
    n_tot = n_grp*n_sub
    for ent in c['stack']:
        if ent[0] == 'H':
            th = ent[1]/n_sub
            for i in range(n_sub):
                # every hoop layer ends at its own station: ranks interleave the
                # groups (layer i of group g -> rank i*n_grp + g), so each group
                # thins about linearly over l_h and no two drops coincide;
                # inner layers run furthest (rank 0 ends at e)
                k = i*n_grp + hcount
                s_end = g['he'] - g['lh']*k/max(n_tot - 1, 1) if g['lh'] > 0 else g['he']
                bands.append(dict(kind='H', t0=th, s_end=s_end, id=f'H{hcount}.{i}'))
            hcount += 1
        elif ent[0] == 'P':
            continue
        else:
            k, sg, tp = ent
            if sg < 0:
                continue                       # +/- pair merged into one band
            bands.append(dict(kind='F', fam=k, t0=2*tp, id=f'F{k}'))
    if 'patch' in g:
        bands.append(dict(kind='P', t0=sum(e[2] for e in c['stack'] if e[0] == 'P'), id='P'))

    s_turn = [vg._arc(R, g['hd'], np.arccos(min(rt/R, 1.0))) for rt in g['rt']]

    def evaluate(b, s, r, region):
        Lp = Lp_fac*b['t0']
        if b['kind'] == 'H':
            if s <= b['s_end']:
                return 90.0, b['t0'], False
            if s <= b['s_end'] + Lp:
                f = 1 - (s - b['s_end'])/Lp
                return 90.0, max(f*b['t0'], t_min), True
            return 90.0, t_min, True
        if b['kind'] == 'F':
            k = b['fam']
            if region == 'barrel':
                return g['a0'][k], b['t0'], False
            rt = g['rt'][k]
            if r > rt:
                tk = vg.family_thickness(gv, k, r, 'V1')*b['t0']/g['tk0'][k]
                return vg.family_angle(gv, k, r), max(tk, t_min), False
            # inside the turnaround: resin pocket from the capped thickness
            st_ = s_turn[k]
            tcap = vg.family_thickness(gv, k, rt*1.0000001, 'V1')*b['t0']/g['tk0'][k]
            Lp_t = Lp_fac*tcap
            if s - st_ < Lp_t:
                return 90.0, max(tcap*(1 - (s - st_)/Lp_t), t_min), True
            return 90.0, t_min, True
        if b['kind'] == 'P':
            p = g['patch']
            if region == 'barrel':
                f = np.clip((s - p['s_end'])/p['l'], 0, 1)
                if f*b['t0'] > t_min:
                    return p['a0'], f*b['t0'], False
                return p['a0'], t_min, True
            gp = dict(R=R, rt=[p['rt']], a0=[p['a0']], tk0=[b['t0']])
            tk = vg.family_thickness(gp, 0, r, 'V1')
            if tk > t_min:
                return vg.family_angle(gp, 0, r), tk, False
            return 90.0, t_min, True
    return bands, evaluate


def meridian_stations(c, g, ds_fine, nb=40, nd=80):
    """liner-surface stations from the mid-plane to the pole, refined in the
    drop-off window"""
    R, L, hd = g['R'], g['L'], g['hd']
    wj = g['wj']
    s_lo = g['he'] - g['lh'] - 0.6*wj
    s_hi = g['he'] + 0.8*wj
    if 'patch' in g:
        s_lo = min(s_lo, g['patch']['s_end'] - 0.3*wj)
    sl = list(np.arange(s_lo, s_hi + 1e-9, ds_fine))
    u = np.linspace(0, 1, nb + 1)
    zb = list((L/2 + s_lo)*(1 - (1 - u)**1.5))
    phib = np.arccos(g['rb']/R)
    v = np.linspace(0, 1, nd + 1)
    ph = list(phib*(1 - (1 - v)**1.3))
    pts = []
    for z in zb:
        if z < L/2 + s_lo - 1e-9:
            pts.append(('b', z - L/2))
    pts += [('s', s) for s in sl]
    smax_ph = vt.arc_to_phi(R, hd, s_hi) if s_hi > 0 else 0.0
    for p in ph:
        if p > smax_ph + 1e-6:
            pts.append(('p', p))
    # convert to (s, r, z, phi)
    out = []
    for kind, val in pts:
        if kind in ('b', 's') and val <= 0:
            s = val
            out.append((s, R, L/2 + s, 0.0))
        else:
            p = vt.arc_to_phi(R, hd, val) if kind == 's' else val
            s = vg._arc(R, hd, p)
            out.append((s, R*np.cos(p), L/2 + hd*np.sin(p), p))
    out.sort(key=lambda x: x[0])
    dedup = [out[0]]
    for o in out[1:]:
        if o[0] - dedup[-1][0] > 0.05:
            dedup.append(o)
    return dedup


def geom_point(g, s):
    R, L, hd = g['R'], g['L'], g['hd']
    if s <= 0:
        return R, L/2 + s, (0.0, 1.0), (1.0, 0.0)
    p = vt.arc_to_phi(R, hd, s)
    r, z = R*np.cos(p), L/2 + hd*np.sin(p)
    tr, tz = -R*np.sin(p), hd*np.cos(p)
    nrm = np.hypot(tr, tz)
    tr, tz = tr/nrm, tz/nrm
    return r, z, (tr, tz), (tz, -tr)          # outward normal = rotate tangent


def build(c, path, n_sub=4, Lp_fac=4.0, t_min=0.02, ds_fine=0.5, n_el_band=2):
    g = vt.geometry(c)
    m = dict(core.MAT)
    m.update(c['mat'])
    bands, evaluate = bands_of(c, g, n_sub, Lp_fac, t_min)
    st = meridian_stations(c, g, ds_fine)
    # corner stations -> add midside stations
    S = [st[0][0]]
    for a, b in zip(st[:-1], st[1:]):
        S += [0.5*(a[0] + b[0]), b[0]]
    S = np.array(S)
    rf = g['rf']
    nb = len(bands)
    nte = nb*n_el_band                         # elements through the thickness
    nodes = {}
    coords = []
    band_data = []                             # per s-index: list of (angle, t, resin)
    for i, s in enumerate(S):
        r0, z0, tg, nn = geom_point(g, s)
        region = 'barrel' if s <= 0 else 'dome'
        bd = [evaluate(b, s, r0, region) for b in bands]
        band_data.append(bd)
        # node offsets: each band split into 2*n_el_band node intervals
        offs = [0.0]
        for (_, t, _) in bd:
            for q in range(2*n_el_band):
                offs.append(offs[-1] + t/(2*n_el_band))
        for j, o in enumerate(offs):
            if i % 2 == 1 and j % 2 == 1:
                continue
            nodes[(i, j)] = len(coords) + 1
            coords.append((r0 + nn[0]*o, z0 + nn[1]*o))
    # straight-edged elements: midside nodes along s at the midpoint of the
    # corner nodes (robust for thin resin bands on steep ramps)
    for i in range(1, len(S), 2):
        for j in range(0, 2*nte + 1, 2):
            a, b_ = coords[nodes[(i - 1, j)] - 1], coords[nodes[(i + 1, j)] - 1]
            coords[nodes[(i, j)] - 1] = (0.5*(a[0] + b_[0]), 0.5*(a[1] + b_[1]))
    lines = ['*HEADING', 'axisymmetric layered vessel', '*NODE, NSET=NALL']
    lines += [f'{k},{x:.6f},{y:.6f},0.0' for k, (x, y) in enumerate(coords, 1)]
    elems = []
    for ie in range(0, len(S) - 1, 2):
        for je in range(0, 2*nte, 2):
            n1 = nodes[(ie, je)]; n2 = nodes[(ie, je + 2)]
            n3 = nodes[(ie + 2, je + 2)]; n4 = nodes[(ie + 2, je)]
            m12 = nodes[(ie, je + 1)]; m23 = nodes[(ie + 1, je + 2)]
            m34 = nodes[(ie + 2, je + 1)]; m41 = nodes[(ie + 1, je)]
            elems.append(dict(id=len(elems) + 1, conn=[n1, n2, n3, n4, m12, m23, m34, m41],
                              ie=ie//2, je=je//2, band=je//(2*n_el_band), sc=S[ie + 1]))
    lines.append('*ELEMENT, TYPE=CAX8, ELSET=EALL')
    lines += [f"{e['id']}," + ','.join(map(str, e['conn'])) for e in elems]
    # materials (binned) and orientations (per element column)
    mats = {}
    def matname(kind, angle, resin):
        if resin:
            return 'RESIN'
        key = f'{kind}_{round(angle*2)/2:.1f}'
        if key not in mats:
            if kind == 'H':
                C = C3(90.0, m)
            else:
                C = pair_C(round(angle*2)/2, m)
            mats[key] = ortho_constants(C)
        return key
    elset = {}
    orients = {}
    BIN = 0.2                                   # deg, material-axis binning
    for e in elems:
        i = 2*e['ie'] + 1
        ang, t, resin = band_data[i][e['band']]
        kind = bands[e['band']]['kind']
        kind = 'F' if kind in ('F', 'P') else kind
        mn = matname(kind, ang, resin)
        # element's own meridional direction from its corner nodes (follows ramps)
        c1, c2, c3, c4 = (np.array(coords[n - 1]) for n in e['conn'][:4])
        tv = (c4 - c1) + (c3 - c2)
        phi = np.degrees(np.arctan2(tv[1], tv[0]))
        key = int(round(phi/BIN))
        orients[key] = None
        e.update(mat=mn, angle=ang, resin=resin, kind=bands[e['band']]['kind'],
                 bid=bands[e['band']]['id'], t=t, phi=phi)
        elset.setdefault((mn, key), []).append(e['id'])
    for key, C in mats.items():
        lines += [f'*MATERIAL, NAME={key}', '*ELASTIC, TYPE=ORTHO',
                  ','.join(f'{v:.4f}' for v in C[:8]), f'{C[8]:.4f}']
    lines += ['*MATERIAL, NAME=RESIN', '*ELASTIC', f"{RESIN['E']},{RESIN['nu']}"]
    for key in orients:
        a = np.radians(key*BIN)
        tg = (np.cos(a), np.sin(a))
        nn = (tg[1], -tg[0]) if True else None
        # outward normal: liner tangent (0,1) in the barrel -> normal (1,0)
        lines += [f'*ORIENTATION, NAME=OR{key - min(orients)}, SYSTEM=RECTANGULAR',
                  f'{tg[0]:.8f},{tg[1]:.8f},0.0,{nn[0]:.8f},{nn[1]:.8f},0.0']
    omin = min(orients)
    for (mn, key), ids in elset.items():
        nm = f'S_{mn}_{key - omin}'.replace('.', 'p').replace('-', 'm')
        lines.append(f'*ELSET, ELSET={nm}')
        for k in range(0, len(ids), 12):
            lines.append(','.join(map(str, ids[k:k + 12])))
        lines.append(f'*SOLID SECTION, ELSET={nm}, MATERIAL={mn}, ORIENTATION=OR{key - omin}')
    # boundary: mid-plane nodes u_z = 0
    mid = [nodes[(0, j)] for j in range(2*nte + 1) if (0, j) in nodes]
    lines += ['*NSET, NSET=MID'] + [str(n) for n in mid]
    lines += ['*BOUNDARY', 'MID,2,2', '*STEP', '*STATIC', '*DLOAD']
    p = c['p']
    dp = p*g['rb']**2/(g['rf']**2 - g['rb']**2)
    for e in elems:
        if e['je'] == 0:
            r0 = geom_point(g, e['sc'])[0]
            pe = p + (dp if (e['sc'] > 0 and r0 <= rf) else 0.0)
            lines.append(f"{e['id']},P4,{pe:.6f}")
    lines += ['*EL PRINT, ELSET=EALL', 'S', '*EL PRINT, ELSET=EALL', 'E', '*EL FILE', 'S', '*END STEP']
    os.makedirs(path, exist_ok=True)
    open(os.path.join(path, 'a.inp'), 'w').write('\n'.join(lines) + '\n')
    return dict(g=g, elems=elems, bands=bands, S=S, m=m, nte=nte, nodes=nodes, coords=coords,
                band_data=band_data, n_el_band=n_el_band)


def run(path, ncpu=2):
    env = dict(os.environ, OMP_NUM_THREADS=str(ncpu))
    r = subprocess.run(['ccx', 'a'], cwd=path, capture_output=True, text=True, env=env)
    return 'Job finished' in r.stdout, r.stdout[-1500:]


def read(path):
    S, E = {}, {}
    mode = None
    for line in open(os.path.join(path, 'a.dat')):
        if 'stresses (elem' in line:
            mode = 'S'; continue
        if 'strains (elem' in line:
            mode = 'E'; continue
        p = line.split()
        if len(p) >= 8 and mode:
            try:
                e = int(p[0])
            except ValueError:
                continue
            (S if mode == 'S' else E).setdefault(e, []).append([float(x) for x in p[2:8]])
    return ({k: np.array(v) for k, v in S.items()}, {k: np.array(v) for k, v in E.items()})


def ply_fibre(meta, Eg):
    """fibre-direction stress of every real (non-resin) element from its strain;
    helical pairs: the +alpha and -alpha plies share the strain and, without
    torsion (eps_s-theta = 0), carry the same sigma_1."""
    m = meta['m']
    Cp = np.linalg.inv(core.compliance(m))
    out = {}
    for e in meta['elems']:
        if e['resin']:
            continue
        eps = Eg[e['id']]                       # (npt, 6): ss nn tt sn st nt (tensor)
        a = np.radians(e['angle'])
        c2, s2, sc = np.cos(a)**2, np.sin(a)**2, np.sin(a)*np.cos(a)
        e11 = c2*eps[:, 0] + s2*eps[:, 2] + 2*sc*eps[:, 4]
        e22 = s2*eps[:, 0] + c2*eps[:, 2] - 2*sc*eps[:, 4]
        s1 = Cp[0, 0]*e11 + Cp[0, 1]*e22 + Cp[0, 2]*eps[:, 1]
        out[e['id']] = s1
    return out
