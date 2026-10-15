import json, sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, '.')
import pubstyle as P
P.use()
import matplotlib as mpl
mpl.rcParams.update({'font.size': 8.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5})
MM = 1/25.4


def main():
    D = json.load(open('./p1_rule_data.json'))
    fig = plt.figure(figsize=(133*MM, 60*MM))
    fig.text(0.01, 0.965, 'Hoop termination decides whether a Type IV vessel bursts where it was sized to burst',
             fontsize=8.6, fontweight='bold', va='top')
    # (1) render
    a1 = fig.add_axes([0.0, 0.02, 0.34, 0.76])
    im = np.asarray(Image.open('cutaway_zoom.png').convert('RGB'))
    H, W = im.shape[:2]
    im = im[int(0.13*H):int(0.80*H), int(0.12*W):int(0.97*W)]
    a1.imshow(im); a1.axis('off')
    a1.text(0.03, 0.03, 'hoop layers (blue) end\non the dome shoulder', transform=a1.transAxes, fontsize=7.0, va='bottom',
            color=P.INK, bbox=dict(fc='white', ec='none', alpha=0.85, pad=1))
    a1.text(0.97, 0.30, 'tow slips if\n' + r'$\lambda=\kappa_g/\kappa_n>\mu$', transform=a1.transAxes, fontsize=7.0,
            ha='right', va='bottom', bbox=dict(fc='white', ec='none', alpha=0.85, pad=1))
    # (2) efficiency bars
    a2 = fig.add_axes([0.435, 0.16, 0.255, 0.50])
    groups = [('H5', r'$h_d/R=0.6$' + '\n' + r'($\chi=-6$)'), ('H7', r'$h_d/R=0.8$' + '\n' + r'($\chi=+1.8$)')]
    cols = [P.MUTED, P.C['blue'], P.C['aqua'], P.C['orange']]
    labs = ['reference', r'$\mu=0.1$', r'$\mu=0.2$', r'$\mu=0.3$']
    wbar = 0.19
    for gi, (vn, gl) in enumerate(groups):
        d = D[vn]
        vals = [d['eta_ref'], d['best']['0.1'], d['best']['0.2'], d['best']['0.3']]
        for k, v in enumerate(vals):
            a2.bar(gi + (k - 1.5)*wbar, 100*v, wbar*0.92, color=cols[k], label=labs[k] if gi == 0 else None)
    a2.axhline(100, color=P.INK2, lw=0.6)
    a2.set_xticks([0, 1]); a2.set_xticklabels([g for _, g in groups], fontsize=7.2)
    a2.tick_params(axis='x', which='both', bottom=False, top=False)
    a2.set_ylim(0, 112); a2.set_ylabel(r'$p_f/p_{\mathrm{cyl}}$ (%)')
    a2.legend(fontsize=6.2, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0), handlelength=1.0,
              columnspacing=0.8, borderaxespad=0.2)
    # (3) rule
    a3 = fig.add_axes([0.785, 0.16, 0.205, 0.58])
    pts = [(d['chi'], d['mu_req']) for k, d in D.items() if k not in ('GFRP', 'CFRP') and d['mu_req'] is not None]
    x, y = np.array(pts).T
    a3.axhspan(0.1, 0.3, color=P.LIGHT, lw=0)
    a3.scatter(x, y, s=12, color=P.C['blue'], zorder=3)
    xx = np.linspace(-6.5, 2.2, 10)
    a3.plot(xx, np.clip(0.22 + 0.030*xx, 0, None), color=P.INK, lw=0.9, ls='--')
    a3.set_xlabel(r'hoop-strain mismatch $\chi$')
    a3.set_ylabel(r'$\mu$ needed for 95%')
    a3.set_xlim(-6.8, 2.4); a3.set_ylim(-0.01, 0.34)
    a3.text(-6.4, 0.31, r'$\mu_{\mathrm{req}}\approx0.22+0.03\chi$', fontsize=6.8)
    for x0, t in ((0.01, '1  Problem: transition'), (0.37, '2  Recovery by design'), (0.72, '3  Design rule')):
        fig.text(x0, 0.84, t, fontsize=7.6, fontweight='bold', color=P.C['blue'], va='center')
    P.save(fig, 'graphical_abstract')
    fig2 = plt.figure()
    plt.close(fig2)


if __name__ == '__main__':
    main()
