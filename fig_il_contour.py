"""Interlaminar stress fields at the hoop drop-off (axisymmetric layer-wise model)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import TwoSlopeNorm
import pubstyle as P

P.use()
OUT = 'figs_paper1'


def panel(ax, d, key, vmax, cmap):
    pol = d['polys'][:, :, ::-1].copy()        # (z, r)
    zt = float(d['zt'])
    pol[:, :, 0] -= zt
    v = d[key]
    pc = PolyCollection(pol, array=np.clip(v, -vmax, vmax), cmap=cmap,
                        norm=TwoSlopeNorm(0, -vmax, vmax), edgecolors='face', linewidths=0.05)
    ax.add_collection(pc)
    # resin-pocket elements outlined lightly to show the ply terminations
    rs = d['kind'] == 2
    big = rs & (np.ptp(d['polys'][:, :, 0], axis=1) + np.ptp(d['polys'][:, :, 1], axis=1) > 0)
    he, lh = float(d['he']), float(d['lh'])
    for x, ls in ((he - lh, ':'), (he, ':')):
        ax.axvline(x, color=P.INK2, lw=0.5, ls=ls)
    ax.axvline(0, color=P.INK, lw=0.6, ls=(0, (4, 2)))
    ax.set_aspect('equal')
    ax.set_xlim(-72, 40)
    ax.set_ylim(140, float(d['polys'][:, :, 0].max()) + 1.5)
    return pc


def main():
    cases = [('ref', r'Reference: $e=0$, $\ell_h=0.5\sqrt{Rt}$, H|F|F interleaved'),
             ('out', r'Hoop block outside: $e=0$, $\ell_h=0.5\sqrt{Rt}$'),
             ('best', r'Windable at $\mu=0.2$: $e=0.19\sqrt{Rt}$, $\ell_h=1.0\sqrt{Rt}$')]
    fig, axs = plt.subplots(3, 2, figsize=(P.W2, 0.62*P.W2), sharex=True, sharey=True,
                            gridspec_kw=dict(hspace=0.12, wspace=0.06))
    for i, (tag, title) in enumerate(cases):
        d = np.load(f'runs/fld_H5_{tag}.npz')
        pa = panel(axs[i, 0], d, 'tsn', 100, 'RdBu_r')
        pb = panel(axs[i, 1], d, 'snn', 80, 'PuOr_r')
        axs[i, 0].text(0.01, 0.04, title, transform=axs[i, 0].transAxes, fontsize=6.5,
                       va='bottom', bbox=dict(fc='white', ec='none', pad=0.6, alpha=0.85))
        axs[i, 0].set_ylabel('Radius $r$ (mm)')
        P.label(axs[i, 0], 'abc'[i]); P.label(axs[i, 1], 'def'[i])
        if i == 0:
            for ax, nm in ((axs[0, 0], 'liner'),):
                pass
            axs[0, 0].text(0.8, float(d['polys'][:, :, 0].max()), 'tangent line', fontsize=6, va='top', color=P.INK)
            he, lh = float(d['he']), float(d['lh'])
            axs[0, 1].text(he - lh + 0.8, float(d['polys'][:, :, 0].max()), r'hoop drop-off $\ell_h$', fontsize=6, va='top', color=P.INK2)
    for j in range(2):
        axs[2, j].set_xlabel(r'Axial distance from tangent line, $z-z_t$ (mm)')
    c1 = fig.colorbar(pa, ax=axs[:, 0], orientation='horizontal', fraction=0.035, pad=0.09, aspect=40, extend='both')
    c1.set_label(r'Interlaminar shear stress $\tau_{sn}$ (MPa) at NWP')
    c2 = fig.colorbar(pb, ax=axs[:, 1], orientation='horizontal', fraction=0.035, pad=0.09, aspect=40, extend='both')
    c2.set_label(r'Through-thickness stress $\sigma_{nn}$ (MPa) at NWP')
    for c in (c1, c2):
        c.ax.tick_params(direction='in', which='both')
    P.save(fig, f'{OUT}/fig_il_fields')


if __name__ == '__main__':
    main()
