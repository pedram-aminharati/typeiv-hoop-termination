"""Vessel generator with explicit design of the cylinder-dome transition.

Extends vessel_gen with
  * a signed meridional coordinate s measured from the tangent line
    (s < 0 in the cylinder, s > 0 on the dome);
  * hoop layers whose end position e and drop-off length l_h are design
    variables: thickness factor h(s) = clip((e - s)/l_h, 0, 1); e > 0 lets
    the hoop layers run onto the dome shoulder;
  * an optional hybrid reinforcement layer ('patch'): a +/- pair that turns
    around on the dome at r_tp (Clairaut angle, geodesic build-up with cap)
    and ends in the cylinder at s = s_p with a drop-off length l_p
    (cf. the RWTH 'hybrid layer' concept);
  * any stacking order of hoop, helical and patch layers (<= 13 layers).
"""
import numpy as np
import vessel_gen as vg
from vessel_gen import cap, _arc, run, read_dat  # noqa: F401

RF_RB = 1.5


def arc_to_phi(R, hd, s, n=400):
    phis = np.linspace(0, np.pi/2, n)
    arcs = np.array([_arc(R, hd, p) for p in phis])
    return float(np.interp(s, arcs, phis))


def geometry(c):
    R = c['ri']
    g = dict(R=R, L=c['L'], hd=c['hd'])
    g['a0'] = list(c['fam'])
    g['rt'] = [R*np.sin(np.radians(a)) for a in g['a0']]
    g['rb'] = min(g['rt'])
    g['rf'] = RF_RB*g['rb']
    g['tk0'] = [sum(e[2] for e in c['stack'] if e[0] == k) for k in range(len(g['a0']))]
    g['th'] = sum(e[1] for e in c['stack'] if e[0] == 'H')
    g['ttot'] = sum(e[-1] for e in c['stack'] if e[0] != 'P')
    g['wj'] = np.sqrt(R*g['ttot'])                      # junction half-width
    g['lh'] = c.get('hoop_taper', 0.5*g['wj'])          # hoop drop-off length
    g['he'] = c.get('hoop_end', 0.0)                    # hoop end position s
    p = c.get('patch')
    if p:
        p = dict(p)
        p['a0'] = np.degrees(np.arcsin(p['rt']/R))
        p['t0'] = sum(e[2] for e in c['stack'] if e[0] == 'P')
        g['patch'] = p
    return g


def stations(g, nb, nd, ref=1.0):
    """meridional stations of the right half (z >= 0), refined around the
    transition and aligned with every design breakpoint."""
    R, L, hd = g['R'], g['L'], g['hd']
    u = np.linspace(0, 1, nb + 1)
    zb = list((L/2)*(1 - (1 - u)**1.6))
    phib = np.arccos(g['rb']/R)
    v = np.linspace(0, 1, nd + 1)
    ph = list(phib*(1 - (1 - v)**1.4))
    # extra refinement: 12 stations per side within 1.5 w_j of the tangent line
    sref = np.linspace(-1.5*g['wj'], 1.5*g['wj'], int(round(24*ref)) + 1)
    # every drop-off is resolved by at least 8 elements
    breaks = list(np.linspace(g['he'] - g['lh'], g['he'], int(round(8*ref)) + 1))
    if 'patch' in g:
        p = g['patch']
        breaks += list(np.linspace(p['s_end'], p['s_end'] + p['l'], int(round(6*ref)) + 1))
        ph.append(np.arccos(min(p['rt']/R, 1.0)))
    for s in list(sref) + breaks:
        if s < 0:
            zb.append(L/2 + s)
        elif s > 0:
            ph.append(arc_to_phi(R, hd, s))
    ph += [np.arccos(min(rt/R, 1.0)) for rt in g['rt'][1:]] + [np.arccos(g['rf']/R)]
    zb = np.unique(np.clip(np.round(zb, 6), 0, L/2))
    ph = np.unique(np.clip(np.round(ph, 8), 0, phib))
    # remove stations closer than 0.3 mm
    def thin(vals, scale):
        out = [vals[0]]
        for x in vals[1:]:
            if (x - out[-1])*scale > 0.15 or x == vals[-1]:
                out.append(x)
        return out
    zb = thin(list(zb), 1.0)
    ph = thin(list(ph), R)
    pts = [('b', R, z) for z in zb]
    pts += [('d', R*np.cos(x), L/2 + hd*np.sin(x), x) for x in ph[1:]]
    return pts


def build(c, path, variant='V1', nb=24, nd=50, nc=32, ref=1.0):
    g = geometry(c)
    R, L, hd = g['R'], g['L'], g['hd']
    half = stations(g, nb, nd, ref)

    def mid(s0, s1):
        if s0[0] == 'b' and s1[0] == 'b':
            return ('b', R, 0.5*(s0[2] + s1[2]))
        p0 = s0[3] if s0[0] == 'd' else 0.0
        x = 0.5*(p0 + s1[3])
        return ('d', R*np.cos(x), L/2 + hd*np.sin(x), x)

    rrows = []
    for i in range(len(half) - 1):
        rrows.append(half[i])
        rrows.append(mid(half[i], half[i + 1]))
    rrows.append(half[-1])
    lrows = [(s[0], s[1], -s[2]) + tuple(s[3:]) for s in reversed(rrows[1:])]
    rows = lrows + rrows
    nrow = len(rows)
    nth = 2*nc
    psi = np.arange(nth)*2*np.pi/nth
    nid, nodes = {}, []
    for i, srow in enumerate(rows):
        r, z = srow[1], srow[2]
        for j in range(nth):
            if i % 2 == 1 and j % 2 == 1:
                continue
            nid[(i, j)] = len(nodes) + 1
            nodes.append((len(nodes) + 1, r*np.cos(psi[j]), r*np.sin(psi[j]), z))
    imid = [i for i, s in enumerate(rows) if abs(s[2]) < 1e-9][0]
    elems = []
    for e_i in range(0, nrow - 1, 2):
        for e_j in range(0, nth, 2):
            j0, j1, j2 = e_j, (e_j + 1) % nth, (e_j + 2) % nth
            i0, i1, i2 = e_i, e_i + 1, e_i + 2
            conn = [nid[(i0, j0)], nid[(i0, j2)], nid[(i2, j2)], nid[(i2, j0)],
                    nid[(i0, j1)], nid[(i1, j2)], nid[(i2, j1)], nid[(i1, j0)]]
            elems.append((len(elems) + 1, conn, e_i//2, e_j//2))

    gv = dict(R=R, rt=g['rt'], a0=g['a0'], tk0=g['tk0'])   # for vessel_gen laws
    rowinfo = []
    for er in range((nrow - 1)//2):
        s0, s1, sm = rows[2*er], rows[2*er + 2], rows[2*er + 1]
        rc, zc = sm[1], sm[2]
        dr, dz = s1[1] - s0[1], s1[2] - s0[2]
        ln = np.hypot(dr, dz)
        ts = (dr/ln, dz/ln)
        if abs(zc) <= L/2:
            s, region = abs(zc) - L/2, 'barrel'
        else:
            s, region = _arc(R, hd, sm[3]), 'dome'
        if region == 'dome' and rc <= g['rf']:
            zone = 'boss'
        elif abs(s) <= g['wj']:
            zone = 'junction'
        else:
            zone = region
        plies = []
        for ent in c['stack']:
            if ent[0] == 'H':
                hf = np.clip((g['he'] - s)/g['lh'], 0.0, 1.0) if g['lh'] > 0 else float(s <= g['he'])
                if hf*ent[1] > 1e-6:
                    plies.append((c.get('hoop', 90.0), hf*ent[1], 'H'))
            elif ent[0] == 'P':
                p = g['patch']
                sg, tp = ent[1], ent[2]
                if region == 'barrel':
                    pf = np.clip((s - p['s_end'])/p['l'], 0.0, 1.0)
                    if pf*tp > 1e-6:
                        plies.append((sg*p['a0'], pf*tp, 'P' + ('+' if sg > 0 else '-')))
                else:
                    if rc < p['rt']:
                        continue
                    gp = dict(R=R, rt=[p['rt']], a0=[p['a0']], tk0=[p['t0']])
                    tk = vg.family_thickness(gp, 0, rc, variant)*tp/p['t0']
                    if tk > 1e-6:
                        plies.append((sg*vg.family_angle(gp, 0, rc), tk, 'P' + ('+' if sg > 0 else '-')))
            else:
                k, sg, tp = ent
                nm = f'F{k}' + ('+' if sg > 0 else '-')
                if region == 'barrel':
                    plies.append((sg*g['a0'][k], tp, nm))
                else:
                    tk = vg.family_thickness(gv, k, rc, variant)*tp/g['tk0'][k]
                    if tk > 1e-6:
                        plies.append((sg*vg.family_angle(gv, k, rc), tk, nm))
        assert len(plies) <= 13, ('CalculiX layer limit', len(plies))
        rowinfo.append(dict(rc=rc, zc=zc, s=s, ts=ts, zone=zone, region=region,
                            side='R' if zc > 0 else 'L', plies=plies,
                            ttot=sum(t for _, t, _ in plies), seglen=ln))

    m = c['mat']
    lines = ['*HEADING', f"case {c.get('name', '')}", '*NODE, NSET=NALL']
    lines += [f'{n},{x:.6f},{y:.6f},{z:.6f}' for n, x, y, z in nodes]
    lines.append('*ELEMENT, TYPE=S8R, ELSET=EALL')
    lines += [f'{e},' + ','.join(map(str, cn)) for e, cn, _, _ in elems]
    lines += ['*MATERIAL, NAME=CF', '*ELASTIC, TYPE=ENGINEERING CONSTANTS',
              f"{m['E1']},{m['E2']},{m['E3']},{m['nu12']},{m['nu13']},{m['nu23']},{m['G12']},{m['G13']}",
              f"{m['G23']},0"]
    secl = []
    for e, cn, er, ej in elems:
        info = rowinfo[er]
        pc = (ej*2 + 1)*2*np.pi/nth
        eth = np.array([-np.sin(pc), np.cos(pc), 0.0])
        erad = np.array([np.cos(pc), np.sin(pc), 0.0])
        es = info['ts'][0]*erad + np.array([0, 0, info['ts'][1]])
        es /= np.linalg.norm(es)
        lines += [f'*ELSET, ELSET=E{e}', str(e)]
        secl.append(f'*SHELL SECTION, ELSET=E{e}, COMPOSITE, OFFSET=-0.5')
        for li, (a, tl, nm) in enumerate(info['plies']):
            ar = np.radians(a)
            av = np.cos(ar)*es + np.sin(ar)*eth
            bv = -np.sin(ar)*es + np.cos(ar)*eth
            lines += [f'*ORIENTATION, NAME=O{e}_{li}, SYSTEM=RECTANGULAR',
                      ','.join(f'{v:.7f}' for v in np.concatenate([av, bv]))]
            secl.append(f'{tl:.6f},,CF,O{e}_{li}')
    lines += secl
    ring = [nid[(imid, j)] for j in range(nth)]
    lines += ['*NSET, NSET=RING'] + [str(n) for n in ring]
    jx = [j for j in range(nth) if abs(np.cos(psi[j])) < 1e-9]
    jy = [j for j in range(nth) if abs(np.sin(psi[j])) < 1e-9]
    lines.append('*BOUNDARY')
    lines += [f'{n},3,3' for n in ring]
    lines += [f'{nid[(imid, j)]},1,1' for j in jx] + [f'{nid[(imid, j)]},2,2' for j in jy]
    lines += ['*STEP', '*STATIC', '*DLOAD']
    p = c['p']
    dp = p*g['rb']**2/(g['rf']**2 - g['rb']**2)
    for e, cn, er, ej in elems:
        info = rowinfo[er]
        pe = p + (dp if (info['region'] == 'dome' and info['rc'] <= g['rf']) else 0.0)
        lines.append(f'{e},P,{pe:.6f}')
    lines += ['*NODE PRINT, NSET=RING, TOTALS=ONLY', 'RF', '*NODE PRINT, NSET=NALL', 'U',
              '*EL PRINT, ELSET=EALL', 'S', '*END STEP']
    import os
    os.makedirs(path, exist_ok=True)
    open(os.path.join(path, 'v.inp'), 'w').write('\n'.join(lines) + '\n')
    return dict(g=g, rowinfo=rowinfo, elems=[(e, er, ej) for e, _, er, ej in elems])


def composite_mass(meta, rho=1.55e-6):
    """composite mass [kg] from the meridional rows (one full revolution)"""
    m = 0.0
    for info in meta['rowinfo']:
        m += 2*np.pi*info['rc']*info['seglen']*info['ttot']*rho
    return m


def slippage(R, hd, s):
    """slippage tendency of a hoop (parallel-circle) path on an ellipsoidal
    dome at meridional distance s from the tangent line: lambda = k_g/k_n =
    |dr/ds|/|dz/ds| = (R/h_d) tan(phi); a hoop path is stable while
    lambda <= mu (tow-on-laminate friction coefficient)."""
    if s <= 0:
        return 0.0
    phi = arc_to_phi(R, hd, s)
    return (R/hd)*np.tan(phi)
