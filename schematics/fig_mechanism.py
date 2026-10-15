import json, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Polygon, Arc
from PIL import Image
sys.path.insert(0, '.')
import pubstyle as P
import p1_study as PS
import vessel_tr as vt
from compose_cutaway import crop
P.use()


def arrow(ax, p0, v, col, lw=1.1, ms=8, z=6):
    a = FancyArrowPatch(p0, (p0[0] + v[0], p0[1] + v[1]), arrowstyle='-|>', mutation_scale=ms, lw=lw,
                        color=col, shrinkA=0, shrinkB=0, zorder=z)
    ax.add_patch(a)


def main():
    fig = plt.figure(figsize=(P.W2, 0.40*P.W2))
    axa = fig.add_axes([0.0, 0.06, 0.27, 0.86])
    axb = fig.add_axes([0.30, 0.10, 0.36, 0.84])
    axc = fig.add_axes([0.745, 0.17, 0.245, 0.75])
    # ---------------- (a) 3D context
    im, cb = crop('mechanism.png')
    axa.imshow(im); axa.axis('off')
    A = json.load(open('mechanism.json'))
    def to_px(a):
        H, W = im.shape[:2]
        return ((a[0] - cb[0])/(cb[2] - cb[0])*W, (a[1] - cb[1])/(cb[3] - cb[1])*H)
    H, W = im.shape[:2]
    lk = dict(arrowstyle='-', lw=0.5, color=P.INK, shrinkA=0, shrinkB=1)
    axa.annotate('hoop tow on the\ndome shoulder', xy=to_px(A['anchors']['ring']), xytext=(0.62*W, 1.06*H),
                 ha='center', va='top', fontsize=6.5, arrowprops=lk, annotation_clip=False)
    axa.annotate('hoop layers\n(cylinder)', xy=to_px(A['anchors']['hoop']), xytext=(0.12*W, 1.06*H), ha='center',
                 va='top', fontsize=6.5, arrowprops=lk, annotation_clip=False)
    P.label(axa, 'a', x=0.02, y=1.0)
    # ---------------- (b) meridian section with force decomposition (vessel H7)
    b = PS.base('H7'); w = PS.wj(b)
    R, L, hd = b['ri'], b['L'], b['hd']
    e = PS.e_crit(b, 0.3)*w
    ph = vt.arc_to_phi(R, hd, e)
    x0 = L/2
    xs_c = np.linspace(x0 - 0.55*R, x0, 50)
    pp = np.linspace(0, np.radians(62), 200)
    X = np.r_[xs_c, x0 + hd*np.sin(pp)] - x0
    Y = np.r_[np.full(50, R), R*np.cos(pp)]
    t = 9.0
    # wall band (schematic thickness) below the surface
    nx = np.r_[np.zeros(50), (R*np.sin(pp))/np.hypot(hd*np.cos(pp), R*np.sin(pp))]
    ny = np.r_[np.ones(50), (hd*np.cos(pp))/np.hypot(hd*np.cos(pp), R*np.sin(pp))]
    inner = np.c_[X - nx*t, Y - ny*t]
    axb.add_patch(Polygon(np.vstack([np.c_[X, Y], inner[::-1]]), closed=True, fc='#c8ccd1', ec='none'))
    axb.plot(X, Y, color=P.INK, lw=0.9)
    axb.plot(inner[:, 0], inner[:, 1], color=P.INK2, lw=0.5)
    # hoop band on the cylinder up to the tangent line (schematic)
    hb = (X <= -0.5) & (X >= X.min())
    axb.fill_between(X[hb], Y[hb], Y[hb] + 5, color=P.C['blue'], lw=0)
    axb.axvline(0, color=P.INK2, lw=0.5, ls=(0, (4, 2)))
    axb.text(1.5, R + 0.165*R, 'tangent line', fontsize=6.3, color=P.INK2, va='top')
    # point P and tow
    xp, yp = hd*np.sin(ph), R*np.cos(ph)
    tx, ty = hd*np.cos(ph), -R*np.sin(ph); nrm = np.hypot(tx, ty); tx, ty = tx/nrm, ty/nrm
    nxp, nyp = -ty, tx
    rt = 3.2
    cx, cy = xp + nxp*rt, yp + nyp*rt
    axb.add_patch(Circle((cx, cy), rt, fc=P.C['orange'], ec=P.INK, lw=0.5, zorder=7))
    S = 58.0
    beta = np.arctan2(-ty, tx)            # meridian slope
    kap = np.array([0, -1.0])             # tension resultant per unit length, towards the axis
    nvec = np.array([nxp, nyp]); tvec = np.array([tx, ty])
    kn = (kap @ nvec)*nvec; kg = (kap @ tvec)*tvec
    pc = (cx, cy)
    arrow(axb, pc, kap*S, P.INK)
    arrow(axb, pc, kn*S, P.C['aqua'])
    arrow(axb, pc, kg*S, P.C['magenta'])
    arrow(axb, (cx - 0.0, cy), -kg*S*0.9, P.C['yellow'])
    axb.plot([pc[0] + kn[0]*S, pc[0] + kap[0]*S], [pc[1] + kn[1]*S, pc[1] + kap[1]*S], color=P.MUTED, lw=0.5, ls=':')
    axb.plot([pc[0] + kg[0]*S, pc[0] + kap[0]*S], [pc[1] + kg[1]*S, pc[1] + kap[1]*S], color=P.MUTED, lw=0.5, ls=':')
    fs = 6.8
    axb.text(pc[0] + kap[0]*S + 1.5, pc[1] + kap[1]*S - 1, r'$T/r$', fontsize=fs, va='top')
    axb.text(pc[0] + kn[0]*S - 2.0, pc[1] + kn[1]*S - 1.5, r'$T\kappa_n$' + '\n(presses tow on)', fontsize=fs,
             color=P.C['aqua'], ha='right', va='top', bbox=dict(fc='white', ec='none', pad=0.5))
    axb.text(pc[0] + kg[0]*S + 2.0, pc[1] + kg[1]*S + 5, r'$T\kappa_g$ (slip)', fontsize=fs, color='#c2427a',
             ha='left', va='bottom')
    axb.text(pc[0] - kg[0]*S*0.9 - 1, pc[1] - kg[1]*S*0.9 + 6, r'friction $\leq\mu T\kappa_n$', fontsize=fs,
             color='#b07a00', ha='center', va='bottom')
    # angle beta between the resultant T/r (towards the axis) and the inward normal (T kappa_n)
    ang_r = -90.0                                           # direction of T/r
    ang_n = np.degrees(np.arctan2(kn[1], kn[0]))            # direction of T kappa_n (left of vertical)
    rA = 0.42*S
    axb.add_patch(Arc(pc, 2*rA, 2*rA, theta1=ang_n, theta2=ang_r, lw=0.8, color=P.INK, zorder=8))
    am = np.radians(0.5*(ang_n + ang_r))
    axb.text(pc[0] + 1.22*rA*np.cos(am), pc[1] + 1.22*rA*np.sin(am), r'$\beta$', fontsize=7.5, ha='center',
             va='center', zorder=9, bbox=dict(fc='white', ec='none', pad=0.3))
    axb.text(0.62, 0.03, r'$\lambda=\kappa_g/\kappa_n=\tan\beta=\dfrac{R}{h_d}\tan\phi\;\leq\;\mu$' + '\n' + r'$\beta$: angle between $T/r$ and the normal (= meridian slope)',
             transform=axb.transAxes, ha='center', va='bottom', fontsize=7.2, multialignment='center',
             bbox=dict(fc='white', ec=P.LIGHT, pad=2.5))
    axb.annotate('hoop end $e$', xy=(xp, yp), xytext=(xp + 32, yp + 22), fontsize=fs, ha='left',
                 arrowprops=dict(arrowstyle='-', lw=0.5, color=P.INK), zorder=8)
    axb.set_aspect('equal')
    axb.set_xlim(-0.45*R, 0.52*R); axb.set_ylim(R - 0.62*R, R + 0.18*R)
    axb.axis('off')
    axb.text(-0.44*R, R + 0.165*R, 'meridian section, vessel H7,\nhoop end at the $\\mu=0.3$ limit', fontsize=6.0,
             color=P.INK2, va='top')
    P.label(axb, 'b', x=0.02, y=1.0)
    # ---------------- (c) slippage tendency vs hoop end
    ax = axc
    hds = [('H5', 0.6), ('S65', 0.65), ('S70', 0.7), ('S75', 0.75), ('H7', 0.8)]
    ax.axhspan(0.1, 0.3, color=P.LIGHT, lw=0, zorder=0)
    ax.text(0.97, 0.285, r'$\mu=0.1$–0.3', fontsize=6.0, color=P.INK2, va='top', ha='right', transform=ax.get_yaxis_transform())
    for i, (vn, hdr) in enumerate(hds):
        bb = PS.base(vn); ww = PS.wj(bb)
        es = np.linspace(0, 0.9, 120)
        lam = [vt.slippage(bb['ri'], bb['hd'], x*ww) for x in es]
        ax.plot(es, lam, color=P.SERIES[i], ls=P.LINES[i], lw=1.0, label=fr'$h_d/R={hdr:g}$')
        ec = PS.e_crit(bb, 0.2)
        ax.plot([ec], [0.2], 'o', ms=3, color=P.SERIES[i], mec='white', mew=0.4, zorder=5)
    ax.set_xlim(0, 0.9); ax.set_ylim(0, 0.5)
    ax.set_xlabel(r'Hoop end $e/\sqrt{R\,t}$')
    ax.set_ylabel(r'Slippage tendency $\lambda$')
    ax.legend(fontsize=5.6, loc='upper left', handlelength=1.6)
    ax.text(0.97, 0.04, r'dots: $e_{\mathrm{crit}}(\mu{=}0.2)$', transform=ax.transAxes, ha='right', fontsize=5.8,
            color=P.INK2)
    P.label(ax, 'c')
    P.save(fig, 'fig_mechanism')


if __name__ == '__main__':
    main()
