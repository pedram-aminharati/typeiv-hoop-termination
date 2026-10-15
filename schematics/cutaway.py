"""3D cutaway of the Type IV vessel H5 (paper geometry and thickness laws)."""
import json
import numpy as np
import pyvista as pv
import sys
sys.path.insert(0, '.')
import p1_study as PS
import vessel_tr as vt
import vessel_gen as vg

pv.OFF_SCREEN = True
X = 1.6                       # thickness exaggeration
TL = 6.0                      # liner thickness (mm), schematic
CAPF = 2.2                    # schematic cap of the polar build-up
b = PS.base('H5')
w = PS.wj(b)
c = dict(b, name='cut', hoop_end=0.125*w, hoop_taper=0.5*w)
g = vt.geometry(c)
R, L, hd, rb = g['R'], g['L'], g['hd'], g['rb']
gv = dict(R=R, rt=g['rt'], a0=g['a0'], tk0=g['tk0'])
phib = np.arccos(rb/R)

# ---------- meridian stations over the full vessel (left pole -> right pole)
xc = np.concatenate([np.linspace(0, L/2 - 1.6*w, 60), np.linspace(L/2 - 1.6*w, L/2, 80)[1:]])
ph = np.linspace(0, phib, 140)[1:]
right = [(x, R, 0.0, 1.0, x - L/2) for x in xc]
for p in ph:
    x = L/2 + hd*np.sin(p); r = R*np.cos(p)
    tx, tr = hd*np.cos(p), -R*np.sin(p); n = np.hypot(tx, tr); tx, tr = tx/n, tr/n
    s = vg._arc(R, hd, p)
    right.append((x, r, -tr, tx, s))     # outward normal (nx, nr) = (-t_r, t_x)
left = [(-x, r, -nx, nr, s) for (x, r, nx, nr, s) in right[::-1][:-1]]
ST = left + right
xs = np.array([q[0] for q in ST]); rs = np.array([q[1] for q in ST])
nxs = np.array([q[2] for q in ST]); nrs = np.array([q[3] for q in ST]); ss = np.array([q[4] for q in ST])

# ---------- layer thicknesses (inner -> outer): liner, H_a, F0_a, F1_a, H_b, F0_b, F1_b
th = b['t_h']/2
def t_layers(s, r):
    fh = np.clip((g['he'] - s)/g['lh'], 0, 1) if s > g['he'] - g['lh'] else 1.0
    f0 = 0.5*min(vg.family_thickness(gv, 0, r*1.0000001, 'V1'), CAPF*g['tk0'][0])
    f1 = 0.5*min(vg.family_thickness(gv, 1, r*1.0000001, 'V1'), CAPF*g['tk0'][1])
    return [TL/X, th*fh, f0, f1, th*fh, f0, f1]
T = np.array([t_layers(s, r) for s, r in zip(ss, rs)])*X
OFF = np.concatenate([-T[:, :1], np.zeros((len(ST), 1)), np.cumsum(T[:, 1:], axis=1)], axis=1)  # boundaries
NAMES = ['liner', 'hoop (inner group)', r'helical $\pm\alpha_1$', r'helical $\pm 30^\circ$', 'hoop (outer group)',
         r'helical $\pm\alpha_1$', r'helical $\pm 30^\circ$']
COL = ['#ece6d6', '#2f6db8', '#6f7680', '#9aa1a9', '#2f6db8', '#6f7680', '#9aa1a9']
LCOL = ['#ece6d6', '#8fb3e3', '#9ea4ad', '#c8ccd1', '#8fb3e3', '#9ea4ad', '#c8ccd1']

# peel steps on the left cylinder: layer i covers x >= XP[i]
XP = [None] + list(-L/2 + 70 + 52*np.arange(6))
XCUT = L/2 - 1.45*w                    # quarter cut for x > XCUT
TH_CAM = np.radians(140.0)
WA, WB = np.radians(90.0), np.radians(180.0)
NT = 220
TH = np.linspace(0, 2*np.pi, NT)


def in_wedge(t):
    t = np.mod(t - WA, 2*np.pi)
    return t <= (WB - WA)


def surface(j, x0=None, cut=True, eps=0.02):
    """boundary surface j (0 = liner inner); visible where x >= x0 and outside the wedge"""
    P = []
    for i in range(len(ST)):
        rr = rs[i] + nrs[i]*OFF[i, j]; xx = xs[i] + nxs[i]*OFF[i, j]
        P.append([(xx, rr*np.cos(t), rr*np.sin(t)) for t in TH])
    P = np.array(P)                     # (ns, nt, 3)
    grid = pv.StructuredGrid(P[:, :, 0], P[:, :, 1], P[:, :, 2])
    # cell mask
    ns = len(ST)
    keep = np.ones((ns - 1, NT - 1), bool)
    xm = 0.5*(xs[:-1] + xs[1:])
    tm = 0.5*(TH[:-1] + TH[1:])
    if x0 is not None:
        keep &= (xm >= x0)[:, None]
    if cut:
        keep &= ~(((xm > XCUT)[:, None]) & in_wedge(tm)[None, :])
    if j >= 2:                           # layer must exist (thickness > eps)
        tt = 0.5*(T[:-1, j - 1] + T[1:, j - 1])
        keep &= (tt > eps)[:, None]
    ids = np.where(keep.T.ravel())[0]   # structured-grid cell order: i fastest
    return grid.extract_cells(ids).extract_surface()


def cut_faces(j):
    """longitudinal faces of layer between boundaries j-1 and j at the wedge planes,
    and the transverse face at XCUT"""
    meshes = []
    sel = np.where(xs >= XCUT)[0]
    for ang in (WA, WB):
        pts, faces = [], []
        for k, i in enumerate(sel):
            for jj in (j - 1, j):
                rr = rs[i] + nrs[i]*OFF[i, jj]; xx = xs[i] + nxs[i]*OFF[i, jj]
                pts.append((xx, rr*np.cos(ang), rr*np.sin(ang)))
        for k in range(len(sel) - 1):
            a, b_, c_, d = 2*k, 2*k + 1, 2*k + 3, 2*k + 2
            faces += [4, a, b_, c_, d]
        m = pv.PolyData(np.array(pts), np.array(faces))
        meshes.append(m)
    # transverse face at XCUT (annular sector)
    i0 = np.argmin(abs(xs - XCUT))
    r0 = rs[i0] + OFF[i0, j - 1]; r1 = rs[i0] + OFF[i0, j]
    if r1 - r0 > 1e-3:
        ang = np.linspace(WA, WB, 80)
        pts = [(XCUT, r*np.cos(a), r*np.sin(a)) for a in ang for r in (r0, r1)]
        faces = []
        for k in range(len(ang) - 1):
            faces += [4, 2*k, 2*k + 1, 2*k + 3, 2*k + 2]
        meshes.append(pv.PolyData(np.array(pts), np.array(faces)))
    return meshes


def step_face(j, x0):
    """annulus at the peel step of layer j (between boundaries j-1 and j)"""
    i0 = np.argmin(abs(xs - x0))
    r0 = rs[i0] + OFF[i0, j - 1]; r1 = rs[i0] + OFF[i0, j]
    return pv.Disc(center=(x0, 0, 0), inner=r0, outer=r1, normal=(-1, 0, 0), r_res=1, c_res=180)


def end_faces(j, eps=0.02):
    """close layer j (between boundaries j-1 and j) where it ends towards each pole"""
    tt = T[:, j - 1]
    out = []
    mid = len(ST)//2
    for side in (1, -1):
        rng = range(mid, len(ST)) if side == 1 else range(mid, -1, -1)
        last = None
        for i in rng:
            if tt[i] > eps:
                last = i
            else:
                break
        if last is None or tt[last] < 0.3:
            continue
        if side == -1 and XP[j - 1] is not None and xs[last] < XP[j - 1]:
            continue                       # peeled away on this side
        i = last
        th_ = TH if (side == -1 or xs[i] <= XCUT) else TH[~in_wedge(TH)]
        pts = []
        for t in th_:
            for jj in (j - 1, j):
                rr = rs[i] + nrs[i]*OFF[i, jj]; xx = xs[i] + nxs[i]*OFF[i, jj]
                pts.append((xx, rr*np.cos(t), rr*np.sin(t)))
        faces = []
        for k in range(len(th_) - 1):
            if side == 1 and xs[i] > XCUT and abs(th_[k + 1] - th_[k]) > 2*(TH[1] - TH[0]):
                continue
            faces += [4, 2*k, 2*k + 1, 2*k + 3, 2*k + 2]
        out.append(pv.PolyData(np.array(pts), np.array(faces)))
    return out


def fibre_lines(j, x0, x1, alpha_family=None, hoop=False, n=40):
    """fibre paths on the outer boundary of layer (boundary j) restricted to x0<=x<=x1, outside wedge"""
    lines = []
    lift = 0.25
    if hoop:
        for xq in np.arange(x0 + 3, x1 - 1, 5.0):
            i = np.argmin(abs(xs - xq))
            rr = rs[i] + OFF[i, j] + lift
            t = TH[~(((xq > XCUT)) & in_wedge(TH))] if xq > XCUT else TH
            lines.append(np.c_[np.full(len(t), xq), rr*np.cos(t), rr*np.sin(t)])
        return lines
    k = alpha_family
    rt = g['rt'][k]
    for sign in (1, -1):
        for t0 in np.linspace(0, 2*np.pi, n, endpoint=False):
            # integrate along stations from left pole to right: dtheta = tan(alpha)/r ds
            th_ = [t0]
            for i in range(1, len(ST)):
                rm = 0.5*(rs[i] + rs[i - 1])
                sa = min(rt/rm, 0.999)
                ds = np.hypot(xs[i] - xs[i - 1], rs[i] - rs[i - 1])
                th_.append(th_[-1] + sign*np.tan(np.arcsin(sa))/rm*ds)
            th_ = np.array(th_)
            m = (xs >= x0) & (xs <= x1)
            if not m.any():
                continue
            rr = rs + nrs*(OFF[:, j] + lift); xx = xs + nxs*(OFF[:, j] + lift)
            P = np.c_[xx, rr*np.cos(th_), rr*np.sin(th_)]
            vis = m & ~((xs > XCUT) & in_wedge(th_)) & (T[:, j - 1] > 0.05)
            # split into contiguous runs
            idx = np.where(vis)[0]
            if len(idx) < 2:
                continue
            br = np.where(np.diff(idx) > 1)[0]
            for seg in np.split(idx, br + 1):
                if len(seg) > 1:
                    lines.append(P[seg])
    return lines


def polyline(arrs):
    pts, cells = [], []
    off = 0
    for a in arrs:
        pts.append(a); cells += [len(a)] + list(range(off, off + len(a))); off += len(a)
    if not pts:
        return None
    m = pv.PolyData(np.vstack(pts)); m.lines = np.array(cells)
    return m


def boss_meshes():
    ms = []
    for sgn in (-1, 1):
        xa = sgn*(L/2 + hd)
        neck = pv.Cylinder(center=(xa + sgn*30, 0, 0), direction=(1, 0, 0), radius=rb*0.97, height=70, resolution=120)
        flange = pv.Cylinder(center=(xa - sgn*4, 0, 0), direction=(1, 0, 0), radius=g['rf'], height=6, resolution=120)
        bore = pv.Cylinder(center=(xa + sgn*65.5, 0, 0), direction=(1, 0, 0), radius=9, height=1.2, resolution=60)
        ms += [neck, bore]
    return ms


def build(fn='cutaway.png', size=(3200, 1700), view='full'):
    p = pv.Plotter(off_screen=True, window_size=size, lighting='three lights')
    p.set_background('white')
    kw = dict(smooth_shading=True, specular=0.25, specular_power=18, ambient=0.18, diffuse=0.8)
    # liner outer surface (visible where no layer covers it, i.e. left part) and inner surface in the cut
    p.add_mesh(surface(1, x0=None), color=COL[0], **kw)
    inner = surface(0, x0=XCUT - 1, cut=False)
    p.add_mesh(inner, color='#d9d2bf', **kw)
    for j in range(2, 8):
        p.add_mesh(surface(j, x0=XP[j - 1]), color=COL[j - 1], **kw)
        p.add_mesh(step_face(j, XP[j - 1]), color=COL[j - 1], ambient=0.35, diffuse=0.7)
    for j in range(2, 8):
        for m in end_faces(j):
            p.add_mesh(m, color=COL[j - 1], ambient=0.35, diffuse=0.7)
    for j in range(1, 8):
        for m in cut_faces(j):
            p.add_mesh(m, color=COL[j - 1], ambient=0.45, diffuse=0.6, specular=0.0)
    # fibre texture
    fam = {2: 0, 3: 1, 5: 0, 6: 1}
    for j in range(2, 8):
        x0 = XP[j - 1]
        x1 = XP[j] if j < 7 else L/2 + hd + 1
        if j - 1 in (1, 4):
            ls = fibre_lines(j, x0, x1, hoop=True)
        else:
            ls = fibre_lines(j, x0, x1, alpha_family=fam[j - 1], n=(90 if fam[j - 1] == 0 else 50) if j < 7 else 64)
        m = polyline(ls)
        if m is not None:
            p.add_mesh(m, color=LCOL[j - 1], line_width=1.6)
    for m in boss_meshes():
        p.add_mesh(m, color='#b9bec5', smooth_shading=True, specular=0.7, specular_power=40, ambient=0.2)
    try:
        p.enable_ssao(radius=12, bias=0.5, kernel_size=128)
    except Exception as ex:
        print('no ssao', ex)
    p.enable_anti_aliasing('ssaa')
    cam_dist = 2.1*L if view == 'full' else 0.75*L
    focal = (40, 0, 0) if view == 'full' else (L/2 + 0.2*hd, 0, 0)
    p.camera.focal_point = focal
    p.camera.position = (focal[0] - 0.42*L, cam_dist*np.cos(TH_CAM), cam_dist*np.sin(TH_CAM))
    p.camera.up = (0, 0, 1)
    p.camera.view_angle = 24
    if view != 'full':
        focal = (L/2 - 0.05*w, 0, R - 18)
        p.camera.focal_point = focal
        D = 0.62*L
        a = np.radians(165)
        p.camera.position = (focal[0] - 0.18*D, D*np.cos(a), focal[2] + D*np.sin(a) + 0.12*D)
        p.camera.view_angle = 15
    p.render()
    # anchor points for labels (world -> display), image coordinates (0..1, origin top-left)
    def proj(pt):
        ren = p.renderer
        ren.SetWorldPoint(*pt, 1.0); ren.WorldToDisplay()
        x, y, _ = ren.GetDisplayPoint()
        return (x/size[0], 1 - y/size[1])
    def onb(sq, jb, ang, side=1):
        cand = np.where((xs > 0) if side == 1 else (xs < 0))[0]
        i = cand[np.argmin(abs(ss[cand] - sq))]
        rr = rs[i] + nrs[i]*OFF[i, jb]; xx = xs[i] + nxs[i]*OFF[i, jb]
        return (xx*side if side == 1 else xx, rr*np.cos(ang), rr*np.sin(ang))
    anchors = {}
    ta = np.radians(108)
    for j in range(1, 8):
        if j == 1:
            xq = -L/2 + 20
        else:
            xq = 0.5*(XP[j - 1] + (XP[j] if j < 7 else XP[6] + 160))
        i = np.argmin(abs(xs - xq))
        rr = rs[i] + OFF[i, j]
        anchors[f'L{j}'] = proj((xs[i], rr*np.cos(ta), rr*np.sin(ta)))
    anchors['boss'] = proj((-(L/2 + hd) - 45, 0, 0))
    anchors['section'] = proj(onb(0.0, 7, np.radians(135)))
    pl = np.radians(90)
    anchors['tl'] = proj((L/2, 0, R - 2))
    anchors['tl_top'] = proj(onb(0.0, 7, pl))
    anchors['ramp0'] = proj(onb(g['he'] - g['lh'], 5, pl))
    anchors['hend'] = proj(onb(g['he'], 4, pl))
    anchors['Ha'] = proj(onb(-0.9*w, 1.5, pl)) if False else proj(((XCUT + 25), 0, R + 0.5*(OFF[np.argmin(abs(xs - XCUT - 25)), 1] + OFF[np.argmin(abs(xs - XCUT - 25)), 2])))
    for nm, jb in (('Hb', 4.5), ('F0a', 2.5), ('F1a', 3.5), ('F0b', 5.5), ('F1b', 6.5), ('liner', 0.5)):
        ii = np.argmin(abs(xs - XCUT - 25))
        o = 0.5*(OFF[ii, int(jb - 0.5)] + OFF[ii, int(jb + 0.5)])
        anchors[nm] = proj((XCUT + 25, 0, R + o))
    anchors['outer'] = proj(onb(-0.9*w, 7, np.radians(150)))
    anchors['dome'] = proj(onb(1.6*w, 7, np.radians(95)))
    p.screenshot(fn, transparent_background=False)
    json.dump(dict(anchors=anchors, size=size, X=X, he=g['he'], lh=g['lh'], w=w), open(fn.replace('.png', '.json'), 'w'), indent=1)
    p.close()


if __name__ == '__main__':
    build()
    build('cutaway_zoom.png', size=(2400, 1800), view='zoom')
    print('done')
