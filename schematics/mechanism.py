"""3D render of the hoop-tow slippage mechanism on the dome shoulder (vessel H7)."""
import json, sys
import numpy as np
import pyvista as pv
sys.path.insert(0, '.')
import p1_study as PS
import vessel_tr as vt
import vessel_gen as vg

pv.OFF_SCREEN = True
b = PS.base('H7'); w = PS.wj(b)
R, L, hd = b['ri'], b['L'], b['hd']
rb = R*np.sin(np.radians(b['fam'][0]))
e = PS.e_crit(b, 0.3)*w                  # hoop end at the friction limit for mu = 0.3
phi_e = vt.arc_to_phi(R, hd, e)
T0 = 14.0                                # schematic wall thickness for the render


def prof(n_c=40, n_d=120):
    pts = [(x, R, 0.0, 1.0) for x in np.linspace(L/2 - 1.2*R, L/2, n_c)]
    for p in np.linspace(0, np.arccos(rb/R), n_d)[1:]:
        x = L/2 + hd*np.sin(p); r = R*np.cos(p)
        tx, tr = hd*np.cos(p), -R*np.sin(p); nn = np.hypot(tx, tr)
        pts.append((x, r, -tr/nn, tx/nn))
    return np.array(pts)


def revolve(P, off, th):
    X = P[:, 0] + P[:, 2]*off; Rr = P[:, 1] + P[:, 3]*off
    G = np.array([[(x, r*np.cos(t), r*np.sin(t)) for t in th] for x, r in zip(X, Rr)])
    return pv.StructuredGrid(G[:, :, 0], G[:, :, 1], G[:, :, 2]).extract_surface()


def main(fn='mechanism.png', size=(2400, 1800)):
    P = prof()
    th = np.linspace(0, 2*np.pi, 260)
    p = pv.Plotter(off_screen=True, window_size=size, lighting='three lights')
    p.set_background('white')
    kw = dict(smooth_shading=True, specular=0.25, specular_power=18, ambient=0.2, diffuse=0.8)
    p.add_mesh(revolve(P, 0.0, th), color='#9aa1a9', **kw)                 # helical wall (outer surface)
    # hoop band on the cylinder ending at the tangent line region (context)
    Pc = P[P[:, 0] <= L/2 - 0.02*R]
    p.add_mesh(revolve(Pc, 3.0, th), color='#2f6db8', **kw)
    # the hoop tow (parallel circle) on the shoulder at s = e
    xe = L/2 + hd*np.sin(phi_e); re_ = R*np.cos(phi_e)
    tx, tr = hd*np.cos(phi_e), -R*np.sin(phi_e); nn = np.hypot(tx, tr); tx, tr = tx/nn, tr/nn
    nx, nr = -tr, tx
    tow_r = 4.0
    ring = pv.Spline(np.array([(xe + nx*tow_r, (re_ + nr*tow_r)*np.cos(t), (re_ + nr*tow_r)*np.sin(t))
                               for t in np.linspace(0, 2*np.pi, 400)]), 800)
    p.add_mesh(ring.tube(radius=tow_r, n_sides=30), color='#eb6834', smooth_shading=True, specular=0.4)
    neck = pv.Cylinder(center=(L/2 + hd + 22, 0, 0), direction=(1, 0, 0), radius=rb*0.97, height=60, resolution=120)
    p.add_mesh(neck, color='#c3c7cc', smooth_shading=True, specular=0.7, specular_power=40)
    # vectors at the tow point theta0
    t0 = np.radians(95)
    P0 = np.array([xe + nx*tow_r, (re_ + nr*tow_r)*np.cos(t0), (re_ + nr*tow_r)*np.sin(t0)])
    e_r = np.array([0, np.cos(t0), np.sin(t0)])
    n3 = np.array([nx, nr*np.cos(t0), nr*np.sin(t0)])
    t3 = np.array([tx, tr*np.cos(t0), tr*np.sin(t0)])       # meridian tangent towards the pole
    sc = 95.0
    kap = -e_r                                               # curvature vector of the circle (towards the axis)
    kn = (kap @ n3)*n3; kg = (kap @ t3)*t3
    p.enable_anti_aliasing('ssaa')
    try:
        p.enable_ssao(radius=10, bias=0.5, kernel_size=128)
    except Exception:
        pass
    focal = np.array([L/2 - 0.15*R, 0, 0.0])
    p.camera.focal_point = tuple(focal)
    d = 6.0*R
    p.camera.position = (focal[0] + 0.45*d, focal[1] - 0.85*d, focal[2] + 0.38*d)
    p.camera.up = (0, 0, 1)
    p.camera.view_angle = 26
    p.render()
    def proj(pt):
        ren = p.renderer; ren.SetWorldPoint(*pt, 1.0); ren.WorldToDisplay()
        x, y, _ = ren.GetDisplayPoint(); return (x/size[0], 1 - y/size[1])
    anc = dict(ring=proj((xe + nx*tow_r, (re_ + nr*tow_r)*np.cos(np.radians(250)),
                                                   (re_ + nr*tow_r)*np.sin(np.radians(250)))),
               hoop=proj((L/2 - 0.7*R, (R + 3)*np.cos(np.radians(250)), (R + 3)*np.sin(np.radians(250)))),
               tl=proj((L/2, R*np.cos(np.radians(255)), R*np.sin(np.radians(255)))),
               pole=proj((L/2 + hd, 0, 0)), dome=proj((L/2 + hd*np.sin(1.0), R*np.cos(1.0)*np.cos(np.radians(240)),
                                                       R*np.cos(1.0)*np.sin(np.radians(240)))))
    json.dump(dict(anchors=anc, size=size, e_w=e/w, phi=np.degrees(phi_e), lam=(R/hd)*np.tan(phi_e)),
              open(fn.replace('.png', '.json'), 'w'), indent=1)
    p.screenshot(fn)
    p.close()


if __name__ == '__main__':
    main()
    print(json.load(open('mechanism.json'))['lam'])
