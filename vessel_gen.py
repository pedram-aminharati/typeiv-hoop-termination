"""Parametric Type IV composite shell vessel for CalculiX.

Reference surface = inner (liner) surface, laminate built outward.
Meridian: cylinder of length L = 3R, ellipsoidal domes of height h_d,
polar opening r_b.  Families: hoop (barrel only, terminated at the
junction), helical family 1 (turnaround at r_b), helical family 2
(turnaround r_t2).  Each helical family is wound as a +/- pair.
"""
import numpy as np
import os
import subprocess

B_REF = 10.0      # band width for cap calibration, V1 [mm]
B_V2 = 4.0        # narrower averaging band -> larger cap, V2 [mm]
EPS_W = 0.05      # V3 coverage floor
RF_RB = 1.5       # boss flange radius ratio r_f / r_b
JUNC = 0.2        # junction half-band, as a fraction of R (arc length)
LFAC = 6.0        # cylinder length L = LFAC R (L/D = 3)
HOOP_TAPER = 0.5  # hoop drop-off length as a fraction of sqrt(R t) (staggered termination)
NC = 24           # circumferential elements


def case_geometry(c):
    """c: ri, L, hd, fam=[a0...], stack=[('H',t) or (k, sign, t)] inner->outer"""
    R = c['ri']
    g = dict(R=R, L=c['L'], hd=c['hd'])
    g['a0'] = list(c['fam'])
    g['rt'] = [R*np.sin(np.radians(a)) for a in g['a0']]
    g['rb'] = min(g['rt'])
    g['rf'] = RF_RB*g['rb']
    g['tk0'] = [sum(e[2] for e in c['stack'] if e[0] == k) for k in range(len(g['a0']))]
    g['th'] = sum(e[1] for e in c['stack'] if e[0] == 'H')
    return g


def cap(R, a0, rt, B):
    """band-averaged geodesic factor over [rt, rt+B] (closed form)."""
    return R*np.cos(np.radians(a0))*np.arccosh((rt + B)/rt)/B


def family_thickness(g, k, r, variant):
    """thickness of helical family k (both +/- plies together) at radius r"""
    R, rt, a0, t0 = g['R'], g['rt'][k], g['a0'][k], g['tk0'][k]
    geo = lambda rr: R*np.cos(np.radians(a0))/np.sqrt(max(rr*rr - rt*rt, 1e-12))
    if variant == 'V0':
        return t0 if r >= rt else 0.0
    if variant in ('V1', 'V2'):
        if r < rt:
            return 0.0
        B = B_REF if variant == 'V1' else B_V2
        return t0*min(geo(r), cap(R, a0, rt, B))
    if variant == 'V3':
        dr = B_REF/2
        w = EPS_W + (1-EPS_W)/(1 + np.exp(-(r - rt)/dr))
        cp = cap(R, a0, rt, B_REF)
        f = min(geo(r), cp) if r > rt else cp
        return t0*w*f
    raise ValueError(variant)


def family_angle(g, k, r):
    rt = g['rt'][k]
    return 90.0 if r <= rt else np.degrees(np.arcsin(rt/r))


# ------------------------------------------------------------------ mesh
def meridian(g, nb=10, nd=30):
    """Return list of meridional stations (s, r, z, tangent) for the right
    half (z >= 0); stations include the barrel and the dome."""
    R, L, hd = g['R'], g['L'], g['hd']
    # barrel: from z=0 to L/2, graded toward the junction
    u = np.linspace(0, 1, nb+1)
    zb = (L/2)*(1 - (1-u)**1.6)
    # dome: phi from 0 to phi_b; insert stations at rt2 and rf
    phib = np.arccos(g['rb']/R)
    v = np.linspace(0, 1, nd+1)
    ph = phib*(1 - (1-v)**1.4)
    extra = [np.arccos(min(rt/R, 1.0)) for rt in g['rt'][1:]] + [np.arccos(g['rf']/R)]
    ph = np.unique(np.concatenate([ph, extra]))
    # drop stations too close to the inserted ones
    keep = [ph[0]]
    for x in ph[1:]:
        if x - keep[-1] > 0.15*(phib/nd) or x in extra or x == ph[-1]:
            keep.append(x)
    ph = np.array(sorted(set(keep)))
    pts = []
    for z in zb:
        pts.append(('b', R, z))
    for x in ph[1:]:
        pts.append(('d', R*np.cos(x), L/2 + hd*np.sin(x), x))
    return pts


def build_inp(c, path, variant='V1', nlgeom=False, nb=10, nd=30, nc=NC):
    g = case_geometry(c)
    R, L, hd = g['R'], g['L'], g['hd']
    half = meridian(g, nb, nd)

    def rz_of(sta):
        return sta[1], sta[2]

    # dome station param for midpoints: barrel linear in z, dome in phi
    def mid(s0, s1):
        if s0[0] == 'b' and s1[0] == 'b':
            return ('b', R, 0.5*(s0[2]+s1[2]))
        p0 = s0[3] if s0[0] == 'd' else 0.0
        p1 = s1[3]
        x = 0.5*(p0+p1)
        return ('d', R*np.cos(x), L/2 + hd*np.sin(x), x)

    # right half rows (corner, mid, corner, ...) from z=0 to the pole
    rrows = []
    for i in range(len(half)-1):
        rrows.append(half[i])
        rrows.append(mid(half[i], half[i+1]))
    rrows.append(half[-1])
    lrows = [(s[0], s[1], -s[2]) + tuple(s[3:]) for s in reversed(rrows[1:])]
    rows = lrows + rrows
    nrow = len(rows)
    nth = 2*nc
    psi = np.arange(nth)*2*np.pi/nth
    nid = {}
    nodes = []
    for i, srow in enumerate(rows):
        r, z = rz_of(srow)
        for j in range(nth):
            if i % 2 == 1 and j % 2 == 1:
                continue
            n = len(nodes) + 1
            nid[(i, j)] = n
            nodes.append((n, r*np.cos(psi[j]), r*np.sin(psi[j]), z))
    imid = [i for i, s in enumerate(rows) if abs(s[2]) < 1e-9]
    assert len(imid) == 1
    imid = imid[0]

    elems = []
    for e_i in range(0, nrow-1, 2):
        for e_j in range(0, nth, 2):
            j0, j1, j2 = e_j, (e_j+1) % nth, (e_j+2) % nth
            i0, i1, i2 = e_i, e_i+1, e_i+2
            # order -> outward normal: theta x z = r
            conn = [nid[(i0, j0)], nid[(i0, j2)], nid[(i2, j2)], nid[(i2, j0)],
                    nid[(i0, j1)], nid[(i1, j2)], nid[(i2, j1)], nid[(i1, j0)]]
            elems.append((len(elems)+1, conn, e_i//2, e_j//2))

    # per element-row properties
    nerow = (nrow-1)//2
    rowinfo = []
    zt = L/2
    for er in range(nerow):
        s0, s1 = rows[2*er], rows[2*er+2]
        sm = rows[2*er+1]
        rc, zc = sm[1], sm[2]
        dr, dz = s1[1]-s0[1], s1[2]-s0[2]
        ln = np.hypot(dr, dz)
        ts = (dr/ln, dz/ln)   # meridian tangent in (r, z)
        # arc distance from the nearest tangent line
        az = abs(zc)
        if az <= zt:
            dj = zt - az
            region = 'barrel'
        else:
            x = sm[3]
            dj = hd*0 + _arc(R, hd, x)
            region = 'dome'
        zone = region
        if dj <= JUNC*R:
            zone = 'junction'
        if region == 'dome' and rc <= g['rf']:
            zone = 'boss'
        side = 'R' if zc > 0 else 'L'
        plies = []
        ttot = sum(e[-1] for e in c['stack'])
        lh = HOOP_TAPER*np.sqrt(R*ttot)
        hf = min(1.0, (L/2 - abs(zc))/lh) if lh > 0 else 1.0
        for ent in c['stack']:
            if ent[0] == 'H':
                if region == 'barrel' and hf*ent[1] > 1e-6:
                    plies.append((c.get('hoop', 90.0), hf*ent[1], 'H'))
                continue
            k, sg, tp = ent
            if region == 'barrel':
                plies.append((sg*g['a0'][k], tp, f'F{k}{"+" if sg > 0 else "-"}'))
            else:
                tk = family_thickness(g, k, rc, variant)*tp/g['tk0'][k]
                if tk <= 1e-6:
                    continue
                plies.append((sg*family_angle(g, k, rc), tk, f'F{k}{"+" if sg > 0 else "-"}'))
        helical0 = sum(g['tk0'])
        kappa = (sum(t for a, t, n in plies if n != 'H')/helical0)
        rowinfo.append(dict(rc=rc, zc=zc, ts=ts, zone=zone, side=side,
                            region=region, plies=plies, kappa=kappa))

    lines = ['*HEADING', f"case {c.get('name','')} variant {variant}",
             '*NODE, NSET=NALL']
    lines += [f'{n},{x:.6f},{y:.6f},{z:.6f}' for n, x, y, z in nodes]
    lines.append('*ELEMENT, TYPE=S8R, ELSET=EALL')
    lines += [f'{e},' + ','.join(map(str, conn)) for e, conn, _, _ in elems]
    m = c['mat']
    lines += ['*MATERIAL, NAME=CF', '*ELASTIC, TYPE=ENGINEERING CONSTANTS',
              f"{m['E1']},{m['E2']},{m['E3']},{m['nu12']},{m['nu13']},{m['nu23']},{m['G12']},{m['G13']}",
              f"{m['G23']},0"]
    # sections and orientations
    secl = []
    for e, conn, er, ej in elems:
        info = rowinfo[er]
        pc = (ej*2+1)*2*np.pi/nth          # element centre angle
        etheta = np.array([-np.sin(pc), np.cos(pc), 0.0])
        er_ = np.array([np.cos(pc), np.sin(pc), 0.0])
        es = info['ts'][0]*er_ + np.array([0, 0, info['ts'][1]])
        es /= np.linalg.norm(es)
        lines.append(f'*ELSET, ELSET=E{e}')
        lines.append(str(e))
        secl.append(f'*SHELL SECTION, ELSET=E{e}, COMPOSITE, OFFSET=-0.5')
        for li, (a, tl, nm) in enumerate(info['plies']):
            ar = np.radians(a)
            av = np.cos(ar)*es + np.sin(ar)*etheta
            bv = -np.sin(ar)*es + np.cos(ar)*etheta
            on = f'O{e}_{li}'
            lines.append(f'*ORIENTATION, NAME={on}, SYSTEM=RECTANGULAR')
            lines.append(','.join(f'{v:.7f}' for v in np.concatenate([av, bv])))
            secl.append(f'{tl:.6f},,CF,{on}')
    lines += secl
    # boundary conditions on the mid-barrel ring
    ring = [nid[(imid, j)] for j in range(nth)]
    lines.append('*NSET, NSET=RING')
    lines += [str(n) for n in ring]
    jx = [j for j in range(nth) if abs(np.cos(psi[j])) < 1e-9]   # on y axis
    jy = [j for j in range(nth) if abs(np.sin(psi[j])) < 1e-9]   # on x axis
    lines.append('*BOUNDARY')
    lines += [f'{n},3,3' for n in ring]
    lines += [f'{nid[(imid, j)]},1,1' for j in jx]
    lines += [f'{nid[(imid, j)]},2,2' for j in jy]
    lines.append('*STEP' + (', NLGEOM, INC=100' if nlgeom else ''))
    lines.append('*STATIC' + ('\n0.1,1.0,1e-5,0.25' if nlgeom else ''))
    p = c['p']
    dp = p*g['rb']**2/(g['rf']**2 - g['rb']**2)
    lines.append('*DLOAD')
    for e, conn, er, ej in elems:
        info = rowinfo[er]
        pe = p + (dp if (info['region'] == 'dome' and info['rc'] <= g['rf']) else 0.0)
        lines.append(f'{e},P,{pe:.6f}')
    lines += ['*NODE PRINT, NSET=RING, TOTALS=ONLY', 'RF',
              '*NODE PRINT, NSET=NALL', 'U',
              '*EL PRINT, ELSET=EALL', 'S', '*END STEP']
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'v.inp'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    meta = dict(g=g, rowinfo=rowinfo, elems=[(e, er, ej) for e, _, er, ej in elems],
                dp=dp, nnodes=len(nodes))
    return meta


def _arc(R, hd, x, n=50):
    t = np.linspace(0, x, n)
    return np.trapezoid(np.hypot(R*np.sin(t), hd*np.cos(t)), t)


def run(path, ncpu=1, timeout=1800):
    env = dict(os.environ, OMP_NUM_THREADS=str(ncpu))
    r = subprocess.run(['ccx', 'v'], cwd=path, capture_output=True, text=True,
                       env=env, timeout=timeout)
    ok = os.path.exists(os.path.join(path, 'v.dat')) and 'Job finished' in r.stdout
    return ok, r.stdout[-2000:]


def read_dat(path):
    """parse stresses and RF totals from v.dat"""
    S = {}
    rf = None
    U = {}
    mode = None
    with open(os.path.join(path, 'v.dat')) as f:
        for line in f:
            if 'stresses (elem' in line:
                mode = 'S'
                S = {}          # keep only the last increment (NLGEOM)
                continue
            if 'total force' in line:
                mode = 'RF'
                continue
            if 'displacements' in line:
                mode = 'U'
                continue
            parts = line.split()
            if not parts:
                continue
            if mode == 'S' and len(parts) >= 8:
                e, ip = int(parts[0]), int(parts[1])
                S.setdefault(e, []).append([float(x) for x in parts[2:8]])
            elif mode == 'RF' and len(parts) >= 3:
                rf = [float(x) for x in parts[-3:]]
                mode = None
            elif mode == 'U' and len(parts) == 4:
                U[int(parts[0])] = [float(x) for x in parts[1:]]
    return {e: np.array(v) for e, v in S.items()}, rf, U
