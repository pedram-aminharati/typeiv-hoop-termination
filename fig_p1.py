"""Figures for Paper 1 (cylinder-dome transition design)."""
import glob
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
import pubstyle as P
import vessel_tr as vt
import scope_p1 as S
import p1_metrics as M
from scipy.optimize import brentq

P.use()
OUT = 'figs_p1'
os.makedirs(OUT, exist_ok=True)


def load(d='results_p1'):
    rs = {}
    for f in glob.glob(f'{d}/*.json'):
        r = json.load(open(f))
        if r.get('ok'):
            rs[r['name']] = r
    return rs


# ---------------------------------------------------------------- schematic
def _bands(rows, exag):
    z = np.array([r['zc'] for r in rows]); rr = np.array([r['rc'] for r in rows])
    tz = np.array([r['ts'][1] for r in rows]); tr = np.array([r['ts'][0] for r in rows])
    nz, nr = -tr, tz
    sgn = np.sign(nr + 1e-12); nz, nr = nz*sgn, nr*sgn
    out, base = [], np.zeros(len(rows))
    for kind in ('F', 'H', 'P'):
        th = np.array([sum(t for a, t, n in r['plies'] if n[0] == kind) for r in rows])*exag
        x0, y0 = z + nz*base, rr + nr*base
        x1, y1 = z + nz*(base + th), rr + nr*(base + th)
        out.append((kind, np.vstack([np.c_[x0, y0], np.c_[x1, y1][::-1]])))
        base = base + th
    return z, rr, out, (z + nz*base, rr + nr*base)


def fig_schematic():
    from matplotlib.patches import Patch, Rectangle
    b = S.doe_case('c001')
    wj = np.sqrt(b['ri']*sum(e[-1] for e in b['stack']))
    tpair = sum(e[-1] for e in b['stack'] if e[0] == 0)/2
    c = dict(b, name='schem', hoop_end=0.25*wj, hoop_taper=0.5*wj,
             stack=list(b['stack']) + [('P', +1, tpair/2), ('P', -1, tpair/2)],
             patch=dict(rt=0.6*b['ri'], s_end=-1.5*wj, l=0.5*wj))
    meta = vt.build(c, 'runs/schem')
    g = meta['g']; R = g['R']; zt = g['L']/2
    rows = sorted([r for r in meta['rowinfo'] if r['zc'] >= 0], key=lambda r: r['s'])
    rows = [r for r in rows if r['s'] > -3.2*wj]
    exag = 2.0
    col = {'F': '#c8c8c8', 'H': P.C['blue'], 'P': P.C['orange']}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(P.W2, 0.47*P.W2),
                                 gridspec_kw=dict(width_ratios=[1.0, 1.15], wspace=0.28))
    for ax in (a1, a2):
        z, rr, bands, outer = _bands(rows, exag)
        for kind, poly in bands:
            ax.add_patch(Polygon(poly, closed=True, fc=col[kind], ec='none'))
        ax.plot(z, rr, color=P.INK, lw=0.9)
        ax.plot(*outer, color=P.INK, lw=0.5)
        ax.axvline(zt, color=P.INK2, lw=0.6, ls=(0, (4, 2)))
        ax.set_aspect('equal')
        ax.set_xlabel(r'Axial position $z$ (mm)')
    a1.set_ylabel(r'Radius $r$ (mm)')
    ttot = sum(e[-1] for e in b['stack'])
    tmax = max(r['ttot'] for r in rows if abs(r['s']) < 2*wj)
    # (a) overview
    a1.set_xlim(zt - 3.2*wj, z.max() + 4)
    ox, oy = outer
    sel = (ox > zt - 1.9*wj) & (ox < zt + 1.1*wj)
    ymax = float(oy[sel].max())
    zz = (zt - 1.9*wj, zt + 1.1*wj, R - 22, ymax + 15)
    a1.set_ylim(0, ymax + 22)
    a1.add_patch(Rectangle((zz[0], zz[2]), zz[1] - zz[0], zz[3] - zz[2], fill=False, lw=0.6, ec=P.INK))
    a1.text(zz[0] + 1.5, zz[2] + 1.5, '(b)', fontsize=7, va='bottom', fontweight='bold')
    a1.text(zt + 3, 3, 'tangent line', rotation=90, fontsize=6.5, va='bottom', color=P.INK2)
    rtp = g['patch']['rt']
    i = int(np.argmin([abs(r['rc'] - rtp) if r['zc'] > zt else 1e9 for r in rows]))
    a1.annotate(r'hybrid-layer turnaround $r_{t,p}$', xy=(rows[i]['zc'] + 7, rows[i]['rc'] + 2),
                xytext=(zt - 3.0*wj, 42), fontsize=6.5,
                arrowprops=dict(arrowstyle='-', lw=0.5, color=P.INK2))
    a1.annotate('polar opening $r_b$', xy=(z.max(), rows[-1]['rc']), xytext=(zt - 3.0*wj, 24),
                fontsize=6.5, arrowprops=dict(arrowstyle='-', lw=0.5, color=P.INK2))
    a1.text(zt - 3.0*wj, 6, f'laminate thickness ×{exag:.0f}', fontsize=6.5, color=P.INK2)
    # (b) zoom with dimensions
    a2.set_xlim(zz[0], zz[1]); a2.set_ylim(zz[2], zz[3])
    def dim(ax, x0, x1, y, txt, side='above'):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='<->', mutation_scale=6,
                                     lw=0.6, color=P.INK, shrinkA=0, shrinkB=0))
        if side == 'above':
            ax.text(0.5*(x0 + x1), y + 0.8, txt, ha='center', va='bottom', fontsize=7)
        else:
            ax.text(min(x0, x1) - 1.0, y, txt, ha='right', va='center', fontsize=7)
    he, lh, sp = g['he'], g['lh'], g['patch']['s_end']
    ytop = ymax + 3.5
    dim(a2, zt + he - lh, zt + he, ytop, r'drop-off $\ell_h$')
    dim(a2, zt, zt + he, ytop + 7, '')
    a2.text(zt + he + 1.0, ytop + 7, r'hoop end $e$', ha='left', va='center', fontsize=7)
    dim(a2, zt - wj, zt, R - 7, r'$\sqrt{R\,t}$', side='left')
    dim(a2, zt + sp, zt, R - 15, r'$|s_p|$', side='left')
    for x in (zt + he - lh, zt + he):
        a2.plot([x, x], [R + 1, ytop], color=P.INK2, lw=0.4, ls=':')
    a2.plot([zt, zt], [ytop + 9, ytop + 9], lw=0)
    a2.plot([zt + sp, zt + sp], [R - 16, R + exag*ttot + 2], color=P.INK2, lw=0.4, ls=':')
    a2.plot([zt - wj, zt - wj], [R - 8, R], color=P.INK2, lw=0.4, ls=':')
    a2.text(zt + 1.0, zz[2] + 1.0, 'tangent line', rotation=90, fontsize=6.5, va='bottom', color=P.INK2)
    a2.text(zz[0] + 1.5, R + 0.8, 'liner surface', fontsize=6.5, va='bottom', color=P.INK2)
    a2.set_ylabel(r'Radius $r$ (mm)')
    P.label(a1, 'a'); P.label(a2, 'b')
    fig.legend(handles=[Patch(fc=col['F'], label='helical families'),
                        Patch(fc=col['H'], label='hoop layers'),
                        Patch(fc=col['P'], label='hybrid reinforcement layer')],
               loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.04))
    P.save(fig, f'{OUT}/fig1_transition_variables')


# ---------------------------------------------------------------- profiles
def fig_profiles(rs):
    fig, axs = plt.subplots(1, 2, figsize=(P.W2, 0.36*P.W2))
    for ax, base, title, lab in zip(axs, ('GFRP', 'H2-c001'), ('Agne et al. GFRP', 'H$_2$ vessel c001'), 'ab'):
        for i, e in enumerate((-0.5, 0.0, 0.25, 0.5)):
            r = rs.get(f'{base}_e{e:+.2f}_l0.50')
            if not r:
                continue
            wj = M.metrics(r)['wj']
            pr = sorted([p for p in r['prof'] if p['zone'] != 'boss'], key=lambda p: p['s'])
            s = np.array([p['s'] for p in pr])/wj
            y = np.array([p['rho'] for p in pr])/r['mid']['rho']
            ax.plot(s, y, ls=P.LINES[i], color=P.SERIES[i], label=fr'$e={e:+.2f}\sqrt{{Rt}}$')
        ax.axhline(1.0, color=P.MUTED, lw=0.6)
        ax.axvspan(-1, 1, color=P.LIGHT, zorder=0, lw=0)
        ax.axvline(0, color=P.INK2, lw=0.5, ls=(0, (4, 2)))
        ax.text(0.04, 0.035, 'junction band', fontsize=6, color=P.INK2, transform=ax.transAxes)
        ax.set_xlim(-3, 3)
        ax.set_ylim(0.0, 2.9 if base != 'GFRP' else 2.0)
        ax.set_xlabel(r'Meridional distance from tangent line, $s/\sqrt{R\,t}$')
        ax.set_ylabel(r'Fibre-stress ratio $\rho(s)/\rho_{\mathrm{mid}}$')
        ax.set_title(title, fontsize=7.5)
        P.label(ax, lab)
    axs[0].legend(loc='upper left', fontsize=6.3)
    fig.tight_layout(w_pad=2.0)
    P.save(fig, f'{OUT}/fig_profiles_hoop_end')


# ---------------------------------------------------------------- design map
def e_crit(b, mu):
    t = sum(e[-1] for e in b['stack']); wj = np.sqrt(b['ri']*t)
    f = lambda e: vt.slippage(b['ri'], b['hd'], e*wj) - mu
    return brentq(f, 1e-4, 2.0)


def fig_map(rs):
    import tr_eval as T
    bases = [('GFRP', T.agne_case('GFRP'), 'Agne et al. GFRP'),
             ('H2-c001', S.doe_case('c001'), 'H$_2$ vessel c001'),
             ('H2-c000', S.doe_case('c000'), 'H$_2$ vessel c000')]
    fig, axs = plt.subplots(1, 3, figsize=(P.W2, 0.36*P.W2), sharey=True)
    E = [-1.0, -0.5, 0.0, 0.25, 0.5]
    Lw = [0.1, 0.25, 0.5, 1.0]
    lv = np.linspace(0.6, 2.2, 17)
    for ax, (bn, b, title), lab in zip(axs, bases, 'abc'):
        pts = []
        for r in rs.values():
            if r['meta']['base'] == bn and r['meta']['study'] == 'hoop':
                pts.append((r['meta']['e_w'], r['meta']['l_w'], M.metrics(r)['rho']['RT']))
        pts = np.array(pts)
        if bn == 'GFRP':
            Z = np.full((len(Lw), len(E)), np.nan)
            for e, l, v in pts:
                Z[Lw.index(l), E.index(e)] = v
            X, Y = np.meshgrid(E, Lw)
            cf = ax.contourf(X, Y, Z, levels=lv, cmap='RdBu_r', extend='both')
            cs = ax.contour(X, Y, Z, levels=[1.0], colors=P.INK, linewidths=1.0)
            ax.clabel(cs, fmt={1.0: r'$R_T=1$'}, fontsize=6.5)
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], cmap='RdBu_r', vmin=lv[0], vmax=lv[-1],
                        s=22, edgecolors=P.INK, linewidths=0.5, zorder=4)
        for e, l, v in pts:
            ax.text(e + 0.035, l*1.07, f'{v:.2f}', ha='left', va='bottom', fontsize=5.3, zorder=5,
                    bbox=dict(fc='white', ec='none', pad=0.3, alpha=0.8))
        for mu, ls in ((0.1, ':'), (0.2, '--'), (0.3, '-.')):
            ec = e_crit(b, mu)
            ax.axvline(ec, color=P.INK2, lw=0.7, ls=ls)
            ax.text(ec - 0.012, 0.082, fr'$\mu={mu}$', rotation=90, fontsize=5.5, va='bottom', ha='right')
        ax.set_xlabel(r'Hoop end position $e/\sqrt{R\,t}$')
        ax.set_yscale('log')
        ax.set_ylim(0.075, 1.4)
        ax.set_xlim(-1.12, 0.62)
        ax.set_yticks(Lw); ax.set_yticklabels([str(v) for v in Lw]); ax.minorticks_off()
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.125))
        ax.set_title(title, fontsize=7.5)
        P.label(ax, lab)
    axs[0].set_ylabel(r'Drop-off length $\ell_h/\sqrt{R\,t}$')
    cb = fig.colorbar(sc, ax=axs, shrink=0.92, pad=0.015)
    cb.set_label(r'$R_T=\rho_{\max,\mathrm{transition}}/\rho_{\mathrm{mid}}$')
    cb.ax.tick_params(direction='in', which='both')
    P.save(fig, f'{OUT}/fig_design_map')


def fig_levers(rs):
    """effect of each lever on the fibre-proxy pressure, relative to the
    reference transition, against the change in composite mass"""
    bases = [('GFRP', 'Agne GFRP'), ('H2-c001', 'H$_2$ c001'), ('H2-c000', 'H$_2$ c000')]
    fig, axs = plt.subplots(1, 2, figsize=(P.W2, 0.38*P.W2))
    cats = {'hoop end $e>0$': ('hoop', lambda m: m['e_w'] > 0 and m['l_w'] == 0.5),
            'hoop end $e<0$': ('hoop', lambda m: m['e_w'] < 0 and m['l_w'] == 0.5),
            'drop-off $\\ell_h$': ('hoop', lambda m: m['e_w'] == 0 and m['l_w'] != 0.5),
            'stacking order': ('stack', lambda m: True),
            'hybrid layer': ('patch', lambda m: True)}
    mk = dict(zip(cats, P.MARKERS)); cl = dict(zip(cats, P.SERIES))
    for bi, (bn, bl) in enumerate(bases):
        ref = rs[f'{bn}_e+0.00_l0.50']
        mref = M.metrics(ref)
        for cn, (study, sel) in cats.items():
            xs, ys = [], []
            for r in rs.values():
                md = r['meta']
                if md['base'] != bn or md['study'] != study or not sel(md):
                    continue
                m = M.metrics(r)
                xs.append(100*(m['mass']/mref['mass'] - 1)); ys.append(100*(m['pf']/mref['pf'] - 1))
            if xs:
                axs[0].scatter(xs, ys, marker=mk[cn], s=18, facecolors=cl[cn] if bi == 0 else 'white',
                               edgecolors=cl[cn], linewidths=0.8, zorder=3)
    for cn in cats:
        axs[0].scatter([], [], marker=mk[cn], s=18, facecolors=cl[cn], edgecolors=cl[cn], label=cn)
    axs[0].axhline(0, color=P.MUTED, lw=0.6); axs[0].axvline(0, color=P.MUTED, lw=0.6)
    axs[0].set_xlabel('Change in composite mass (%)')
    axs[0].set_ylabel(r'Change in fibre-proxy pressure $p_f$ (%)')
    axs[0].legend(loc='lower right', fontsize=6)
    axs[0].text(0.02, 0.97, 'filled: Agne GFRP; open: H$_2$ vessels', transform=axs[0].transAxes,
                va='top', fontsize=6, color=P.INK2)
    P.label(axs[0], 'a')
    # (b) stacking order
    orders = ['H-F-F x2 (ref.)', 'F-F-H x2', 'H block inside', 'H block middle', 'H block outside']
    w = 0.38
    for bi, bn in enumerate(('H2-c001', 'H2-c000')):
        vals = []
        for o in orders:
            if o.endswith('(ref.)'):
                r = rs[f'{bn}_e+0.00_l0.50']
            else:
                r = rs[f'{bn}_{o.replace(" ", "_")}']
            vals.append(M.metrics(r)['rho']['RT'])
        x = np.arange(len(orders)) + (bi - 0.5)*w
        axs[1].bar(x, vals, w*0.92, color=P.SERIES[bi], label=['H$_2$ c001', 'H$_2$ c000'][bi],
                   hatch=None if bi == 0 else '////', edgecolor='white', linewidth=0.4)
    axs[1].axhline(1.0, color=P.INK2, lw=0.7)
    axs[1].set_xticks(np.arange(len(orders)))
    axs[1].set_xticklabels(['H|F|F ×2\n(ref.)', 'F|F|H ×2', 'hoop\ninside', 'hoop\nmiddle', 'hoop\noutside'], fontsize=6.5)
    axs[1].tick_params(axis='x', which='minor', bottom=False, top=False)
    axs[1].set_ylabel(r'$R_T$ (fibre mode)')
    axs[1].set_ylim(0, 2.2)
    axs[1].legend(loc='upper left', fontsize=6.5)
    P.label(axs[1], 'b')
    fig.tight_layout(w_pad=2.0)
    P.save(fig, f'{OUT}/fig_levers')


if __name__ == '__main__':
    rs = load()
    print(len(rs), 'results')
    fig_schematic()
    if rs:
        fig_profiles(rs)
        fig_map(rs)
        fig_levers(rs)
