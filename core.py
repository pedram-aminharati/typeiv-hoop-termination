"""Material, failure criteria and Lekhnitskii thick-wall solver.

Material and criteria follow the companion study (Aminharati et al.,
Composites Part C 21 (2026) 100796, Table 1 and Eqs. 10-19).
Angles are measured from the cylinder axis z toward the hoop direction.
"""
import numpy as np

# Companion study ply (carbon/epoxy), MPa
MAT = dict(E1=135e3, E2=9e3, E3=9e3, G12=5e3, G13=5e3, G23=5e3,
           nu12=0.3, nu13=0.3, nu23=0.3,
           Xt=2000., Xc=1500., Yt=50., Yc=200., S12=80.)
MAT['S23'] = MAT['Yt']          # companion substitution S23 = Yt
MAT['p_psi'] = MAT['S12'] / MAT['Yc']   # reduced Puck inclination (0.4)


def compliance(m=MAT):
    E1, E2, E3 = m['E1'], m['E2'], m['E3']
    S = np.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1/E1, 1/E2, 1/E3
    S[0, 1] = S[1, 0] = -m['nu12']/E1
    S[0, 2] = S[2, 0] = -m['nu13']/E1
    S[1, 2] = S[2, 1] = -m['nu23']/E2
    S[3, 3], S[4, 4], S[5, 5] = 1/m['G23'], 1/m['G13'], 1/m['G12']
    return S   # Voigt order 11,22,33,23,13,12


def _tensor_from_voigt(C):
    idx = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    T = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(idx):
        for J, (k, l) in enumerate(idx):
            for a, b in ((i, j), (j, i)):
                for c, d in ((k, l), (l, k)):
                    T[a, b, c, d] = C[I, J]
    return T, idx


def stiffness_cyl(theta_deg, m=MAT):
    """Ply stiffness in the frame (x=z axial, y=theta hoop, 3=r radial).
    Fibre axis 1 lies at theta from x toward y."""
    C = np.linalg.inv(compliance(m))
    T, idx = _tensor_from_voigt(C)
    t = np.radians(theta_deg)
    Q = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    Tg = np.einsum('ai,bj,ck,dl,ijkl->abcd', Q, Q, Q, Q, T)
    Cg = np.zeros((6, 6))
    for I, (i, j) in enumerate(idx):
        for J, (k, l) in enumerate(idx):
            Cg[I, J] = Tg[i, j, k, l]
    return Cg   # Voigt order zz, thth, rr, (th,r), (z,r), (z,th)


# ---------------------------------------------------------------- criteria
def criteria(s1, s2, t12, m=MAT):
    """Return dict of indices (arrays) for plane-stress ply states."""
    s1, s2, t12 = map(np.asarray, (s1, s2, t12))
    Xt, Xc, Yt, Yc, S = m['Xt'], m['Xc'], m['Yt'], m['Yc'], m['S12']
    F1, F2 = 1/Xt - 1/Xc, 1/Yt - 1/Yc
    F11, F22, F66 = 1/(Xt*Xc), 1/(Yt*Yc), 1/S**2
    F12 = -0.5*np.sqrt(F11*F22)
    tw = F1*s1 + F2*s2 + F11*s1**2 + F22*s2**2 + F66*t12**2 + 2*F12*s1*s2
    tw = np.maximum(tw, 0.0)
    ft = np.where(s1 >= 0, (s1/Xt)**2 + (t12/S)**2, 0.0)
    fc = np.where(s1 < 0, (s1/Xc)**2, 0.0)
    S23 = m['S23']
    mt = np.where(s2 >= 0, (s2/Yt)**2 + (t12/S)**2, 0.0)
    mc = np.where(s2 < 0, (s2/(2*S23))**2 + ((Yc/(2*S23))**2 - 1)*s2/Yc
                  + (t12/S)**2, 0.0)
    mc = np.maximum(mc, 0.0)
    pff = np.where(s1 >= 0, s1/Xt, np.abs(s1)/Xc)
    pp = m['p_psi']
    iff_t = np.sqrt((s2/Yt)**2 + (t12/S)**2)
    den = np.maximum(S - pp*s2, 1e-9)        # s2<0 -> den > S
    iff_c = np.sqrt((t12/den)**2 + (s2/Yc)**2)
    iff = np.where(s2 >= 0, iff_t, iff_c)
    hashin = np.maximum.reduce([ft, fc, mt, mc])
    puck = np.maximum(pff, iff)
    tri = np.maximum.reduce([tw, hashin, puck])
    fib = np.maximum.reduce([ft, fc, pff])
    mat = np.maximum.reduce([mt, mc, iff])
    return dict(TW=tw, H=hashin, P=puck, tri=tri, f=fib, m=mat,
                Hft=ft, Hfc=fc, Hmt=mt, Hmc=mc, PFF=pff, PIFF=iff)


def to_ply(sz, sth, tzth, theta_deg):
    """Rotate in-plane stresses (z, theta) to ply axes."""
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    s1 = c*c*sz + s*s*sth + 2*c*s*tzth
    s2 = s*s*sz + c*c*sth - 2*c*s*tzth
    t12 = -c*s*sz + c*s*sth + (c*c - s*s)*tzth
    return s1, s2, t12


# ---------------------------------------------------------------- Lekhnitskii
def lekhnitskii(ri, plies, p, npts=9, m=MAT):
    """plies: list of (theta_deg, thickness). Closed-end, generalised plane
    strain, no twist. Returns per-ply dict with r, sr, sth, sz, tzth."""
    N = len(plies)
    rs = [ri]
    for _, tk in plies:
        rs.append(rs[-1] + tk)
    Cs = [stiffness_cyl(th, m) for th, _ in plies]
    # indices in Voigt (zz=0, thth=1, rr=2, thr=3, zr=4, zth=5)
    pars = []
    for C in Cs:
        Crr, Ctt, Czz = C[2, 2], C[1, 1], C[0, 0]
        Crt, Crz, Ctz = C[2, 1], C[2, 0], C[1, 0]
        b = np.sqrt(Ctt/Crr)
        if abs(b-1) < 1e-6:
            b = 1 + 1e-6
        a1 = (Crz - Ctz)/(Crr*(b*b - 1))
        pars.append((b, a1))

    def comps(k, r):
        """stress = coefA*A + coefB*B + coefE*eps0  for sr, sth, sz, tzth"""
        C = Cs[k]
        b, a1 = pars[k]
        # strains for A term: er = b r^(b-1), eth = r^(b-1)
        out = {}
        one = np.ones_like(np.asarray(r, dtype=float))
        for key, (er, et, ez) in {
                'A': (b*r**(b-1), r**(b-1), 0.0),
                'B': (-b*r**(-b-1), r**(-b-1), 0.0),
                'E': (a1*one, a1*one, one)}.items():
            sr = C[2, 2]*er + C[2, 1]*et + C[2, 0]*ez
            st = C[1, 2]*er + C[1, 1]*et + C[1, 0]*ez
            sz = C[0, 2]*er + C[0, 1]*et + C[0, 0]*ez
            tz = C[5, 2]*er + C[5, 1]*et + C[5, 0]*ez
            out[key] = np.array([sr, st, sz, tz])
        return out

    n = 2*N + 1
    K = np.zeros((n, n))
    F = np.zeros(n)
    row = 0
    c0 = comps(0, rs[0])
    K[row, 0], K[row, 1], K[row, -1] = c0['A'][0], c0['B'][0], c0['E'][0]
    F[row] = -p
    row += 1
    cN = comps(N-1, rs[N])
    K[row, 2*(N-1)], K[row, 2*(N-1)+1], K[row, -1] = cN['A'][0], cN['B'][0], cN['E'][0]
    row += 1
    for k in range(N-1):
        r = rs[k+1]
        bk, a1k = pars[k]
        bl, a1l = pars[k+1]
        # u continuity
        K[row, 2*k], K[row, 2*k+1] = r**bk, r**(-bk)
        K[row, 2*k+2], K[row, 2*k+3] = -r**bl, -r**(-bl)
        K[row, -1] = (a1k - a1l)*r
        row += 1
        ck, cl = comps(k, r), comps(k+1, r)
        K[row, 2*k], K[row, 2*k+1] = ck['A'][0], ck['B'][0]
        K[row, 2*k+2], K[row, 2*k+3] = -cl['A'][0], -cl['B'][0]
        K[row, -1] = ck['E'][0] - cl['E'][0]
        row += 1
    # axial force: int sz 2 pi r dr = pi ri^2 p  (Gauss quadrature per ply)
    gx, gw = np.polynomial.legendre.leggauss(12)
    for k in range(N):
        a, bnd = rs[k], rs[k+1]
        rr = 0.5*(a+bnd) + 0.5*(bnd-a)*gx
        w = 0.5*(bnd-a)*gw*2*np.pi*rr
        for j, key in enumerate(('A', 'B')):
            K[row, 2*k+j] += sum(wi*comps(k, ri_)[key][2] for wi, ri_ in zip(w, rr))
        K[row, -1] += sum(wi*comps(k, ri_)['E'][2] for wi, ri_ in zip(w, rr))
    F[row] = np.pi*ri**2*p
    sol = np.linalg.solve(K, F)
    res = []
    for k in range(N):
        r = np.linspace(rs[k], rs[k+1], npts)
        c = comps(k, r)
        st = c['A']*sol[2*k] + c['B']*sol[2*k+1] + c['E']*sol[-1]
        res.append(dict(theta=plies[k][0], r=r, sr=st[0], sth=st[1], sz=st[2],
                        tzth=st[3]))
    return res, sol[-1]


def fast_fi(ri, plies, p, m=MAT):
    """Max indices over plies of the barrel (fast model)."""
    res, _ = lekhnitskii(ri, plies, p, m=m)
    best = {}
    for ply in res:
        s1, s2, t12 = to_ply(ply['sz'], ply['sth'], ply['tzth'], ply['theta'])
        c = criteria(s1, s2, t12, m)
        for k, v in c.items():
            best[k] = max(best.get(k, 0.0), float(np.max(v)))
    return best, res
