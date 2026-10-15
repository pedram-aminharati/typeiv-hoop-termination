"""Methodology workflow (Paper 1), vector layout in millimetres."""
import json, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.collections import PolyCollection
from matplotlib.colors import TwoSlopeNorm
from PIL import Image
sys.path.insert(0, '.')
import pubstyle as P
import p1_study as PS
P.use()

FW, FH = 190.0, 150.0
COLW, GAPX = 55.0, 10.5
X0 = [2.5, 2.5 + COLW + GAPX, 2.5 + 2*(COLW + GAPX)]
ROWH, GAPY = 63.0, 13.0
TOP = 141.0
Y0 = [TOP - ROWH, TOP - 2*ROWH - GAPY]          # bottom of row 1 and row 2
HEAD = 7.5


def mm_axes(fig, x, y, w, h):
    return fig.add_axes([x/FW, y/FH, w/FW, h/FH])


def box(ax, x, y, w, h, title, text, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=2.2', fc='white', ec=col,
                                lw=0.9, zorder=1))
    ax.add_patch(FancyBboxPatch((x, y + h - HEAD), w, HEAD, boxstyle='round,pad=0,rounding_size=2.2', fc=col,
                                ec=col, lw=0.9, zorder=2))
    ax.add_patch(Rectangle((x, y + h - HEAD), w, 3.0, fc=col, ec=col, lw=0, zorder=2))
    ax.text(x + 3, y + h - HEAD/2, title, color='white', fontsize=7.4, fontweight='bold', va='center', zorder=3)
    ax.text(x + 3, y + h - HEAD - 2.2, text, fontsize=6.4, va='top', ha='left', linespacing=1.3, zorder=3, color=P.INK)


def arrow(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=11, lw=1.3, color=P.INK2,
                                 shrinkA=1.5, shrinkB=1.5, zorder=4))


def picture(fig, fn, x, y, w, h, crop=None):
    im = Image.open(fn).convert('RGB')
    if crop:
        W, H = im.size
        im = im.crop((int(crop[0]*W), int(crop[1]*H), int(crop[2]*W), int(crop[3]*H)))
    iw, ih = im.size
    if w/h > iw/ih:
        w2 = h*iw/ih; x += (w - w2)/2; w = w2
    else:
        h2 = w*ih/iw; y += (h - h2)/2; h = h2
    a = mm_axes(fig, x, y, w, h)
    a.imshow(np.asarray(im)); a.axis('off')


def mini(a, fs=5.8):
    a.tick_params(labelsize=fs, length=2, pad=1.5)
    for sp in a.spines.values():
        sp.set_linewidth(0.5)


def main():
    fig = plt.figure(figsize=(FW/25.4, FH/25.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis('off')
    cA, cB, cC = P.C['blue'], '#4d4d4d', P.C['orange']
    for x, t in zip(X0, ('Vessels and design space', 'Models', 'Results and guidance')):
        ax.text(x + 1, 146.0, t, fontsize=8.2, fontweight='bold', va='center', color=P.INK)
    tx = [('1  Vessels', '11 carbon/epoxy H$_2$ vessels (35, 70 MPa)\n+ 2 benchmarks; dome sized = cylinder', cA),
          ('3  Shell model (S8R)', '542 analyses, resolved drop-offs\n$\\eta=p_f/p_{\\mathrm{cyl}}$; verified vs Lekhnitskii', cB),
          ('5  Fibre vs delamination', 'best windable design per $\\mu$;\nthe criteria disagree on $\\ell_h$', cC),
          ('2  Design space', '$e$, $\\ell_h$, stacking, hybrid layers;\nfriction bound $\\lambda(e)\\leq\\mu$', cA),
          ('4  Layer-wise model (CAX8)', '106 analyses, every hoop layer;\nrank correlation with shell 0.96', cB),
          ('6  Design rule', 'hoop-strain mismatch $\\chi$;\n$\\mu_{\\mathrm{req}}\\approx0.22+0.03\\chi$', cC)]
    k = 0
    for row in range(2):
        for col in range(3):
            t, body, c = tx[k]
            box(ax, X0[col], Y0[row], COLW, ROWH, t, body, c)
            k += 1
    # picture area inside each box
    def area(col, row):
        return X0[col] + 3, Y0[row] + 3, COLW - 6, ROWH - HEAD - 14
    # 1: 3D vessel render
    picture(fig, 'cutaway.png', *area(0, 0), crop=(0.03, 0.17, 0.98, 0.80))
    # 2: transition section render
    picture(fig, 'cutaway_zoom.png', *area(0, 1), crop=(0.10, 0.14, 0.98, 0.78))
    # 3: design map of H5
    x, y, w, h = area(1, 0)
    a = mm_axes(fig, x + 9, y + 7, w - 11, h - 8)
    rs = [json.load(open(f)) for f in __import__('glob').glob('./results_p1full/H5_e*_l*.json')]
    E = [-1.0, -0.5, -0.25, 0.0, 0.125, 0.25, 0.375, 0.5]; Lw = [0.1, 0.25, 0.5, 1.0]
    Z = np.full((4, 8), np.nan)
    for r in rs:
        m = r['meta']
        Z[Lw.index(m['l_w']), E.index(m['e_w'])] = 100*r['pf']/r['pf_barrel']
    Xg, Yg = np.meshgrid(E, Lw)
    a.contourf(Xg, Yg, np.clip(Z, 20.1, 100), levels=np.arange(20, 101, 5), cmap='viridis')
    a.contour(Xg, Yg, Z, levels=[98], colors='white', linewidths=0.8)
    a.set_yscale('log'); a.set_yticks(Lw); a.set_yticklabels(['0.1', '0.25', '0.5', '1']); a.minorticks_off(); a.set_xlim(-1.04, 0.5); a.set_ylim(0.093, 1.0)
    a.set_xticks([-1, -0.5, 0, 0.5]); a.set_xticklabels(['$-$1', '$-$0.5', '0', '0.5']); a.tick_params(axis='x', pad=2.5)
    a.set_xlabel(r'$e/\sqrt{Rt}$', fontsize=6, labelpad=1); a.set_ylabel(r'$\ell_h/\sqrt{Rt}$', fontsize=6, labelpad=1)
    a.text(0.04, 0.95, r'$\eta$, vessel H5', transform=a.transAxes, color='white', fontsize=5.8, va='top')
    mini(a)
    # 4: interlaminar shear field
    x, y, w, h = area(1, 1)
    d = np.load('./runs/fld_H5_ref.npz')
    pol = d['polys'][:, :, ::-1].copy(); pol[:, :, 0] -= float(d['zt'])
    a = mm_axes(fig, x, y + 5, w, h - 5)
    pc = PolyCollection(pol, array=np.clip(d['tsn'], -100, 100), cmap='RdBu_r', norm=TwoSlopeNorm(0, -100, 100),
                        edgecolors='face', linewidths=0.05)
    a.add_collection(pc); a.set_xlim(-34, 14); a.set_ylim(146, float(pol[:, :, 1].max()) + 0.5)
    a.set_aspect('equal'); a.axis('off')
    ax.text(x + w/2, y + 1.5, r'interlaminar shear $\tau_{sn}$ at the hoop drop-off (H5, NWP)', fontsize=5.8,
            ha='center', va='bottom', color=P.INK2)
    # 5: fibre vs delamination
    import fig_paper1 as FP
    il = FP.load_il('./results_il')
    x, y, w, h = area(2, 0)
    a = mm_axes(fig, x + 9, y + 7, w - 11, h - 8)
    for j, vn in enumerate(('H1', 'H3', 'H5', 'H7')):
        pts = np.array([(100*o['eta_f'], o['pd1']/o['p']) for o in il if o['vessel'] == vn and FP.il_cat(o)])
        a.scatter(pts[:, 0], pts[:, 1], s=5, color=P.SERIES[j], lw=0, label=vn)
    a.set_yscale('log'); a.set_yticks([0.3, 1, 3]); a.set_yticklabels(['0.3', '1', '3']); a.minorticks_off()
    a.set_xlabel(r'fibre efficiency $\eta_f$ (%)', fontsize=6, labelpad=1)
    a.set_ylabel(r'$p_d/\mathrm{NWP}$', fontsize=6, labelpad=1)
    a.legend(fontsize=5.2, loc='upper left', handletextpad=0.1, borderaxespad=0.2, ncol=2, columnspacing=0.5)
    mini(a)
    # 6: design rule
    D = json.load(open('./p1_rule_data.json'))
    x, y, w, h = area(2, 1)
    a = mm_axes(fig, x + 9, y + 7, w - 11, h - 8)
    pts = np.array([(v['chi'], v['mu_req']) for k_, v in D.items() if k_ not in ('GFRP', 'CFRP') and v['mu_req'] is not None])
    a.axhspan(0.1, 0.3, color=P.LIGHT, lw=0)
    a.scatter(pts[:, 0], pts[:, 1], s=7, color=P.C['blue'], lw=0, zorder=3)
    xx = np.linspace(-6.5, 2.2, 5); a.plot(xx, np.clip(0.22 + 0.03*xx, 0, None), color=P.INK, lw=0.8, ls='--')
    a.set_xlabel(r'hoop-strain mismatch $\chi$', fontsize=6, labelpad=1)
    a.set_ylabel(r'$\mu$ for $\eta\geq95\%$', fontsize=6, labelpad=1)
    a.set_ylim(-0.01, 0.34)
    mini(a)
    # arrows
    for row in range(2):
        yc = Y0[row] + ROWH/2
        arrow(ax, (X0[0] + COLW, yc), (X0[1], yc))
        arrow(ax, (X0[1] + COLW, yc), (X0[2], yc))
    for col in (0, 2):
        xc = X0[col] + COLW/2
        arrow(ax, (xc, Y0[0]), (xc, Y0[1] + ROWH))
    P.save(fig, 'fig_workflow')


if __name__ == '__main__':
    main()
