"""Publication figures for Paper 1 (Composites Part B)."""
import glob
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Patch, Rectangle
import pubstyle as P
import vessel_tr as vt
import p1_study as PS
import fig_p1 as F1

P.use()
OUT = 'figs_paper1'
os.makedirs(OUT, exist_ok=True)


def fig_schematic(vn='H5'):
    b = PS.base(vn)
    wj = PS.wj(b)
    t0 = sum(e[-1] for e in b['stack'] if e[0] == 0)
    c = dict(b, name='schem', hoop_end=0.25*wj, hoop_taper=0.5*wj,
             stack=list(b['stack']) + [('P', +1, t0/4), ('P', -1, t0/4)],
             patch=dict(rt=0.55*b['ri'], s_end=-1.0*wj, l=0.5*wj))
    meta = vt.build(c, 'runs/schem')
    g = meta['g']; R = g['R']; zt = g['L']/2
    rows = sorted([r for r in meta['rowinfo'] if r['zc'] >= 0], key=lambda r: r['s'])
    rows = [r for r in rows if r['s'] > -2.6*wj]
    exag = 1.5
    col = {'F': '#c8c8c8', 'H': P.C['blue'], 'P': P.C['orange']}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(P.W2, 0.47*P.W2),
                                 gridspec_kw=dict(width_ratios=[1.0, 1.15], wspace=0.28))
    for ax in (a1, a2):
        z, rr, bands, outer = F1._bands(rows, exag)
        for kind, poly in bands:
            ax.add_patch(Polygon(poly, closed=True, fc=col[kind], ec='none'))
        ax.plot(z, rr, color=P.INK, lw=0.9)
        ax.plot(*outer, color=P.INK, lw=0.5)
        ax.axvline(zt, color=P.INK2, lw=0.6, ls=(0, (4, 2)))
        ax.set_aspect('equal')
        ax.set_xlabel(r'Axial position $z$ (mm)')
    a1.set_ylabel(r'Radius $r$ (mm)')
    ttot = sum(e[-1] for e in b['stack'])
    a1.set_xlim(zt - 2.6*wj, z.max() + 6)
    ox, oy = outer
    sel = (ox > zt - 1.7*wj) & (ox < zt + 1.0*wj)
    ymax = float(oy[sel].max())
    zz = (zt - 1.7*wj, zt + 1.0*wj, R - 26, ymax + 18)
    a1.set_ylim(0, ymax + 25)
    a1.add_patch(Rectangle((zz[0], zz[2]), zz[1] - zz[0], zz[3] - zz[2], fill=False, lw=0.6, ec=P.INK))
    a1.text(zz[0] + 2, zz[2] + 2, '(b)', fontsize=7, va='bottom', fontweight='bold')
    a1.text(zt + 4, 4, 'tangent line', rotation=90, fontsize=6.5, va='bottom', color=P.INK2)
    rtp = g['patch']['rt']
    i = int(np.argmin([abs(r['rc'] - rtp) if r['zc'] > zt else 1e9 for r in rows]))
    a1.annotate(r'hybrid-layer turnaround $r_{t,p}$', xy=(rows[i]['zc'] + 8, rows[i]['rc'] + 3),
                xytext=(zt - 2.5*wj, 62), fontsize=6.5,
                arrowprops=dict(arrowstyle='-', lw=0.5, color=P.INK2))
    a1.annotate('polar opening $r_b$', xy=(z.max(), rows[-1]['rc']), xytext=(zt - 2.5*wj, 36),
                fontsize=6.5, arrowprops=dict(arrowstyle='-', lw=0.5, color=P.INK2))
    a1.text(zt - 2.5*wj, 8, f'laminate thickness ×{exag:g}', fontsize=6.5, color=P.INK2)

    def dim(ax, x0, x1, y, txt, side='above'):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='<->', mutation_scale=6,
                                     lw=0.6, color=P.INK, shrinkA=0, shrinkB=0))
        if side == 'above':
            ax.text(0.5*(x0 + x1), y + 1.0, txt, ha='center', va='bottom', fontsize=7)
        else:
            ax.text(min(x0, x1) - 1.5, y, txt, ha='right', va='center', fontsize=7)
    a2.set_xlim(zz[0], zz[1]); a2.set_ylim(zz[2], zz[3])
    he, lh, sp = g['he'], g['lh'], g['patch']['s_end']
    ytop = ymax + 4
    dim(a2, zt + he - lh, zt + he, ytop, r'drop-off $\ell_h$')
    dim(a2, zt, zt + he, ytop + 8, '')
    a2.text(zt + he + 1.5, ytop + 8, r'hoop end $e$', ha='left', va='center', fontsize=7)
    dim(a2, zt - wj, zt, R - 9, r'$\sqrt{R\,t}$', side='left')
    dim(a2, zt + sp, zt, R - 19, r'$|s_p|$', side='left')
    for x in (zt + he - lh, zt + he):
        a2.plot([x, x], [R + 1, ytop], color=P.INK2, lw=0.4, ls=':')
    a2.plot([zt + sp, zt + sp], [R - 20, R + exag*ttot + 2], color=P.INK2, lw=0.4, ls=':')
    a2.plot([zt - wj, zt - wj], [R - 10, R], color=P.INK2, lw=0.4, ls=':')
    a2.text(zt + 1.2, zz[2] + 1.2, 'tangent line', rotation=90, fontsize=6.5, va='bottom', color=P.INK2)
    a2.text(zz[0] + 2, R + 1.0, 'liner surface', fontsize=6.5, va='bottom', color=P.INK2)
    a2.set_ylabel(r'Radius $r$ (mm)')
    P.label(a1, 'a'); P.label(a2, 'b')
    fig.legend(handles=[Patch(fc=col['F'], label='helical families'),
                        Patch(fc=col['H'], label='hoop layers'),
                        Patch(fc=col['P'], label='hybrid reinforcement layer')],
               loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.04))
    P.save(fig, f'{OUT}/fig1_transition_variables')


if __name__ == '__main__':
    import sys
    for name in sys.argv[1:]:
        globals()[name]()


# ------------------------------------------------------------------ data
def load(d='results_p1full'):
    import p1_metrics as M
    rs = {}
    for f in glob.glob(f'{d}/*.json'):
        r = json.load(open(f))
        if not r.get('ok'):
            continue
        m = M.metrics(r)
        r['_m'] = m
        r['eta'] = r['pf']/r['pf_barrel']
        rs[r['name']] = r
    return rs


VORDER = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8']


def vlabel(vn, two=True):
    v = PS.VESSELS[vn]
    sep = '\n' if two else ', '
    return f"{vn} ({v['P']:.0f} MPa){sep}$h_d/R$ = {v['hd_R']:g}, $r_b/R$ = {v['rb_R']:g}"


def fig_maps(rs, tag=''):
    E = [-1.0, -0.5, -0.25, 0.0, 0.125, 0.25, 0.375, 0.5]
    Lw = [0.1, 0.25, 0.5, 1.0]
    fig, axs = plt.subplots(2, 4, figsize=(P.W2, 0.56*P.W2), sharex=True, sharey=True,
                            gridspec_kw=dict(hspace=0.32, wspace=0.16))
    allv = [100*r['eta'] for r in rs.values() if r['meta']['study'] == 'hoop']
    lo = 5*np.floor(min(allv)/5)
    lv = np.arange(lo, 100.01, 5)
    cf = None
    for k, vn in enumerate(VORDER):
        ax = axs.flat[k]
        Z = np.full((len(Lw), len(E)), np.nan)
        for r in rs.values():
            md = r['meta']
            if md['vessel'] == vn and md['study'] == 'hoop':
                Z[Lw.index(md['l_w']), E.index(md['e_w'])] = 100*r['eta']
        if np.isnan(Z).all():
            ax.set_visible(False); continue
        X, Y = np.meshgrid(E, Lw)
        cf = ax.contourf(X, Y, np.clip(Z, lv[0], 100), levels=lv, cmap='viridis')
        cs = ax.contour(X, Y, Z, levels=[98], colors=['white'], linewidths=[0.9], linestyles=['-'])
        ax.scatter(X, Y, s=2.5, c='white', lw=0, zorder=3, clip_on=False)
        b = PS.base(vn)
        for mu, ls in ((0.1, ':'), (0.2, '--'), (0.3, '-.')):
            ec = PS.e_crit(b, mu)
            if ec < 0.62:
                ax.axvline(ec, color='white', lw=0.9, ls=ls)
                ax.text(ec - 0.012, 0.106, fr'$\mu$ = {mu}', rotation=90, fontsize=5.2, va='bottom',
                        ha='right', color='white')
        ax.set_yscale('log'); ax.set_ylim(0.1, 1.0)
        ax.set_yticks(Lw); ax.set_yticklabels([str(v) for v in Lw]); ax.minorticks_off()
        ax.set_xlim(-1.03, 0.53)
        ax.set_xticks([-1, -0.5, 0, 0.5]); ax.set_xticklabels(['$-$1', '$-$0.5', '0', '0.5'])
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.125))
        ax.set_title(vlabel(vn), fontsize=6.3, linespacing=1.15)
        P.label(ax, 'abcdefgh'[k])
    for ax in axs[1]:
        ax.set_xlabel(r'Hoop end $e/\sqrt{R\,t}$')
    for ax in axs[:, 0]:
        ax.set_ylabel(r'Drop-off $\ell_h/\sqrt{R\,t}$')
    cb = fig.colorbar(cf, ax=axs, shrink=0.9, pad=0.015, ticks=np.arange(lo, 100.1, 10))
    cb.set_label(r'Transition efficiency $\eta=p_f/p_{\mathrm{cyl}}$ (%)')
    cb.ax.tick_params(direction='in', which='both')
    P.save(fig, f'{OUT}/fig_design_maps{tag}')


def _prof(r):
    w = r['_m']['wj']
    pr = sorted([p for p in r['prof'] if p['zone'] != 'boss'], key=lambda p: p['s'])
    return np.array([p['s'] for p in pr])/w, np.array([p['rho'] for p in pr])/r['mid']['rho']


def fig_profiles(rs, vessels=('H1', 'H3', 'H5', 'H7'), tag=''):
    fig, axs = plt.subplots(2, 2, figsize=(P.W2, 0.55*P.W2), sharex=True,
                            gridspec_kw=dict(hspace=0.3, wspace=0.18))
    es = (-0.5, 0.0, 0.125, 0.25, 0.5)
    for k, (ax, vn) in enumerate(zip(axs.flat, vessels)):
        ymax = 1.0
        for i, e in enumerate(es):
            r = rs.get(f'{vn}_e{e:+.3f}_l0.50')
            if not r:
                continue
            s, y = _prof(r)
            ax.plot(s, y, ls=P.LINES[i], color=P.SERIES[i], lw=1.0,
                    label=(r'$e=0$ (ref.)' if e == 0 else fr'$e={e:+g}\sqrt{{Rt}}$'))
            ymax = max(ymax, y[(s > -3) & (s < 4)].max())
        ax.axhline(1.0, color=P.MUTED, lw=0.6)
        ax.axvspan(-1, 1, color=P.LIGHT, zorder=0, lw=0)
        ax.axvline(0, color=P.INK2, lw=0.5, ls=(0, (4, 2)))
        b = PS.base(vn)
        ec = PS.e_crit(b, 0.2)
        ax.axvline(ec, color=P.INK, lw=0.5, ls=':')
        ax.text(ec + 0.05, 0.06, r'$e_{\mathrm{crit}}(\mu{=}0.2)$', fontsize=5.6, color=P.INK, transform=ax.get_xaxis_transform())
        ax.text(0.015, 0.035, 'junction band', fontsize=5.8, color=P.INK2, transform=ax.transAxes)
        ax.set_xlim(-3, 4)
        ax.set_ylim(0, 1.08*ymax)
        ax.set_title(vlabel(vn, two=False), fontsize=6.6)
        P.label(ax, 'abcd'[k])
    for ax in axs[1]:
        ax.set_xlabel(r'Meridional distance from tangent line $s/\sqrt{R\,t}$')
    for ax in axs[:, 0]:
        ax.set_ylabel(r'Fibre-stress ratio $\rho(s)/\rho_{\mathrm{mid}}$')
    axs[0, 0].legend(loc='upper left', fontsize=6, ncol=1)
    P.save(fig, f'{OUT}/fig_profiles{tag}')


def fig_stack_hybrid(rs, tag=''):
    """(a) stacking order: eta for all H vessels; (b) hybrid layer: change in p_f vs change in mass"""
    orders = ['ref', 'F-F-H x2', 'H block inside', 'H block middle', 'H block outside']
    lab = ['H|F|F×2\n(ref.)', 'F|F|H×2', 'hoop block\ninside', 'hoop block\nmiddle', 'hoop block\noutside']
    fig, axs = plt.subplots(1, 2, figsize=(P.W2, 0.36*P.W2), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.25))
    ax = axs[0]
    n = len(VORDER); wbar = 0.8/n
    for j, vn in enumerate(VORDER):
        vals = []
        for o in orders:
            nm = f'{vn}_e+0.000_l0.50' if o == 'ref' else f'{vn}_stack_{o.replace(" ", "_")}'
            r = rs.get(nm)
            vals.append(100*r['eta'] if r else np.nan)
        x = np.arange(len(orders)) + (j - (n - 1)/2)*wbar
        ax.bar(x, vals, wbar*0.9, color=P.SERIES[j % 4], alpha=1.0 if j < 4 else 0.55,
               hatch=None if j < 4 else '////', edgecolor='white', linewidth=0.3, label=vn)
    ax.axhline(100, color=P.INK2, lw=0.6)
    ax.set_xticks(range(len(orders))); ax.set_xticklabels(lab, fontsize=6.3)
    ax.tick_params(axis='x', which='minor', bottom=False, top=False)
    ax.set_ylabel(r'Transition efficiency $\eta$ (%)')
    ax.set_ylim(0, 108)
    ax.legend(ncol=4, fontsize=5.8, loc='upper left', columnspacing=0.8, handlelength=1.2)
    P.label(ax, 'a')
    ax = axs[1]
    for j, vn in enumerate(VORDER):
        ref = rs.get(f'{vn}_e+0.000_l0.50')
        if not ref:
            continue
        xs, ys = [], []
        for r in rs.values():
            if r['meta']['vessel'] == vn and r['meta']['study'] == 'hybrid':
                xs.append(100*(r['mass']/ref['mass'] - 1)); ys.append(100*(r['pf']/ref['pf'] - 1))
        ax.scatter(xs, ys, marker=P.MARKERS[j % 6], s=14, facecolors=P.SERIES[j % 4] if j < 4 else 'white',
                   edgecolors=P.SERIES[j % 4], linewidths=0.7, label=vn)
    ax.axhline(0, color=P.MUTED, lw=0.6)
    ax.set_xlabel('Change in composite mass (%)')
    ax.set_ylabel(r'Change in $p_f$ (%)')
    ax.legend(ncol=4, fontsize=5.8, loc='lower center', bbox_to_anchor=(0.5, 1.0), handletextpad=0.2,
              columnspacing=0.6, borderaxespad=0.2)
    P.label(ax, 'b', y=1.13)
    P.save(fig, f'{OUT}/fig_stack_hybrid{tag}')


def load_il(d='results_il'):
    out = []
    for f in glob.glob(f'{d}/*.json'):
        o = json.load(open(f))
        if o.get('ok') and o['ds_fine'] == 0.25 and o['n_el_band'] == 2:
            o['FI'] = o['fi']['1.0']['fi']
            o['pd1'] = o['p']/np.sqrt(o['FI'])
            o['eta_f'] = 1.0/max(o['RT'], 1.0)
            out.append(o)
    return out


def il_cat(o):
    if o.get('hyb'):
        return 'hybrid layer'
    if o['order'] != 'ref':
        return 'stacking order'
    if o['n_rep'] != 4 or o.get('Lp_fac', 4.0) != 4.0:
        return None
    if o['l_w'] != 0.5:
        return r'drop-off $\ell_h$'
    return r'hoop end $e$'


def _logy(ax, ticks=(0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5)):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_locator(NullLocator())


def fig_il_summary(il, vessels=('H1', 'H3', 'H5', 'H7'), tag=''):
    fig, axs = plt.subplots(1, 2, figsize=(P.W2, 0.38*P.W2), gridspec_kw=dict(wspace=0.26))
    ax = axs[0]
    for j, vn in enumerate(vessels):
        for l, ls, mf in ((0.5, '-', True), (1.0, '--', False)):
            pts = sorted((o['e_w'], o['pd1']/o['p']) for o in il
                         if o['vessel'] == vn and o['order'] == 'ref' and o['n_rep'] == 4 and not o.get('hyb')
                         and o['l_w'] == l and o.get('Lp_fac', 4.0) == 4.0)
            if not pts:
                continue
            x, y = np.array(pts).T
            ax.plot(x, y, ls=ls, color=P.SERIES[j], marker=P.MARKERS[j], ms=3.2,
                    mfc=P.SERIES[j] if mf else 'white', mew=0.7,
                    label=f'{vn}' if l == 0.5 else None)
    ax.axvline(0, color=P.INK2, lw=0.5, ls=(0, (4, 2)))
    ax.set_xlabel(r'Hoop end $e/\sqrt{R\,t}$')
    ax.set_ylabel(r'Delamination onset $p_d/\mathrm{NWP}$')
    ax.set_yscale('log'); _logy(ax)
    ax.legend(loc='lower right', fontsize=6, ncol=2, title=r'solid: $\ell_h=0.5\sqrt{Rt}$; open: $1.0\sqrt{Rt}$',
              title_fontsize=5.8)
    P.label(ax, 'a')
    ax = axs[1]
    cats = [r'hoop end $e$', r'drop-off $\ell_h$', 'stacking order', 'hybrid layer']
    for ci, cn in enumerate(cats):
        for j, vn in enumerate(vessels):
            pts = [(100*o['eta_f'], o['pd1']/o['p']) for o in il if o['vessel'] == vn and il_cat(o) == cn]
            if pts:
                x, y = np.array(pts).T
                ax.scatter(x, y, marker=P.MARKERS[ci], s=13, facecolors=P.SERIES[j] if ci % 2 == 0 else 'white',
                           edgecolors=P.SERIES[j], linewidths=0.7, zorder=3)
    for ci, cn in enumerate(cats):
        ax.scatter([], [], marker=P.MARKERS[ci], s=13, facecolors=P.MUTED if ci % 2 == 0 else 'white',
                   edgecolors=P.MUTED, label=cn)
    ax.set_xlabel(r'Fibre transition efficiency $\eta_f$ (%)')
    ax.set_ylabel(r'Delamination onset $p_d/\mathrm{NWP}$')
    ax.set_yscale('log'); _logy(ax)
    ax.legend(loc='upper left', fontsize=6)
    ax.text(0.98, 0.03, 'colour: vessel as in (a)', transform=ax.transAxes, ha='right', fontsize=5.8, color=P.INK2)
    P.label(ax, 'b')
    P.save(fig, f'{OUT}/fig_il_summary{tag}')


def fig_verif(tag=''):
    V = json.load(open('p1_verify.json'))
    fig, axs = plt.subplots(1, 3, figsize=(P.W2, 0.33*P.W2),
                            gridspec_kw=dict(width_ratios=[0.8, 1.3, 1.0], wspace=0.34))
    # (a) Lekhnitskii vs layer-wise fibre stress per layer group (through thickness)
    ax = axs[0]
    lab = []
    ax_v = [v for _, v in V['lekhnitskii']['axi']]
    lk_v = [v for _, v in V['lekhnitskii']['lek']]
    names = []
    for (th, _), (nm, _) in zip(V['lekhnitskii']['lek'], V['lekhnitskii']['axi']):
        names.append('hoop' if th == 90 else fr'$\pm{th:.1f}^\circ$')
    y = np.arange(len(ax_v))
    ax.barh(y + 0.18, lk_v, 0.34, color=P.MUTED, label='Lekhnitskii')
    ax.barh(y - 0.18, ax_v, 0.34, color=P.C['blue'], label='layer-wise FE')
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6.3)
    ax.tick_params(axis='y', which='minor', left=False, right=False)
    ax.set_ylim(len(y) - 0.5, -0.5)
    ax.set_xlabel(r'Fibre stress $\sigma_1$ (MPa)')
    ax.text(0.97, 0.03, 'liner side at top', transform=ax.transAxes, ha='right', fontsize=5.6, color=P.INK2)
    ax.legend(loc='lower left', fontsize=5.6, bbox_to_anchor=(0.0, 1.0), ncol=1, borderaxespad=0.2)
    P.label(ax, 'a', y=1.14)
    # (b) fibre-stress ratio along the meridian: shell vs layer-wise
    ax = axs[1]
    w = V['profile']['w']
    sh = np.array(V['profile']['shell']); axp = np.array(V['profile']['axi'])
    mid = sh[np.argmin(sh[:, 0]), 1]
    mid_a = axp[np.argmin(axp[:, 0]), 1]
    axp = axp[axp[:, 0] <= sh[:, 0].max()]          # boss region excluded, as in the shell
    ax.plot(sh[:, 0]/w, sh[:, 1]/mid, color=P.INK, lw=1.0, label='shell (S8R)')
    ax.plot(axp[:, 0]/w, axp[:, 1]/mid_a, color=P.C['orange'], lw=0.9, ls='--', label='layer-wise (CAX8)')
    ax.axvline(0, color=P.INK2, lw=0.5, ls=(0, (4, 2)))
    ax.set_xlim(-3, 4)
    ax.set_xlabel(r'$s/\sqrt{R\,t}$')
    ax.set_ylabel(r'$\rho(s)/\rho_{\mathrm{mid}}$')
    ax.legend(loc='upper left', fontsize=6)
    P.label(ax, 'b')
    # (c) delamination index vs averaging length for three meshes
    ax = axs[2]
    for i, (ds, nb) in enumerate(((0.5, 2), (0.25, 2), (0.125, 2), (0.25, 3))):
        f = glob.glob(f'results_il/H5_r4_ref_e+0.000_l0.50_n*_d{ds}_b{nb}.json')
        if not f:
            continue
        o = json.load(open(f[0]))
        d = sorted((float(k), v['fi']) for k, v in o['fi'].items())
        x, yv = np.array(d).T
        x = np.where(x == 0, 0.06, x)
        ax.plot(x, yv, marker=P.MARKERS[i], ls=P.LINES[i], color=P.SERIES[i], ms=3.2,
                label=(f'{ds} mm' if nb == 2 else f'{ds} mm, 3 el./band'))
    ax.set_xscale('log')
    ax.set_xticks([0.06, 0.25, 0.5, 1.0]); ax.set_xticklabels(['0', '0.25', '0.5', '1'])
    ax.minorticks_off()
    ax.axvline(1.0, color=P.MUTED, lw=0.6)
    ax.set_xlabel(r'Averaging length $\delta$ (mm)')
    ax.set_ylabel(r'Delamination index $FI_d$')
    ax.legend(loc='upper right', fontsize=5.8, title='element size', title_fontsize=5.8)
    P.label(ax, 'c')
    P.save(fig, f'{OUT}/fig_verification{tag}')


def rule_data():
    import p1_rule as PR
    import p1_analysis as A
    rs = A.load()
    S = A.summary(rs)
    out = {}
    for vn, v in S.items():
        c = PR.chi(vn)
        pc = v['ref']['pf_mid']
        best = {f'{float(mu):g}': b['eta'] for mu, b in v['best'].items()}
        # mu required for eta >= 0.95: linear interpolation of the best windable eta in mu,
        # with eta(mu = 0) = best design at e <= 0
        e0 = max([g for g in v['grid'] if g['e'] <= 0], key=lambda g: g['pf'])['eta']
        mus = [0.0, 0.1, 0.2, 0.3]
        etas = [e0] + [best[m] for m in ('0.1', '0.2', '0.3')]
        etas = list(np.maximum.accumulate(etas))
        mreq = None
        for k in range(1, 4):
            if etas[k] >= 0.95:
                mreq = mus[k - 1] + (0.95 - etas[k - 1])*(mus[k] - mus[k - 1])/(etas[k] - etas[k - 1]) if etas[k - 1] < 0.95 else mus[k - 1]
                break
        out[vn] = dict(chi=c['chi'], hd_R=c['hd_R'], eta_ref=v['ref']['pf']/pc, dpf={f'{float(mu):g}': b['dpf'] for mu, b in v['best'].items()}, best=best, e0=e0, mu_req=mreq,
                       P=PS.VESSELS[vn]['P'] if vn in PS.VESSELS else None)
    return out


def fig_rule(tag=''):
    D = rule_data()
    json.dump(D, open('p1_rule_data.json', 'w'), indent=1, default=float)
    fig, axs = plt.subplots(1, 2, figsize=(P.W2, 0.38*P.W2), gridspec_kw=dict(wspace=0.28))
    ax = axs[0]
    ser = sorted([vn for vn in D if vn in ('H5', 'S65', 'S70', 'S75', 'H7')], key=lambda v: D[v]['chi'])
    xs = [D[v]['chi'] for v in ser]
    curves = [('reference ($e=0$)', [D[v]['eta_ref'] for v in ser], P.INK, 'o', '-')]
    for i, mu in enumerate(('0.1', '0.2', '0.3')):
        curves.append((fr'best windable, $\mu={mu}$', [D[v]['best'][mu] for v in ser], P.SERIES[i], P.MARKERS[i + 1], P.LINES[i + 1]))
    for lab, ys, cl, mk, ls in curves:
        ax.plot(xs, np.array(ys)*100, color=cl, marker=mk, ls=ls, ms=3.5, label=lab)
    others = [vn for vn in D if vn not in ser]
    for vn in others:
        d = D[vn]
        ax.scatter([d['chi']], [100*d['eta_ref']], marker='o', s=12, facecolors='white', edgecolors=P.INK, lw=0.6, zorder=3)
        ax.scatter([d['chi']], [100*d['best']['0.2']], marker='s', s=12, facecolors='white', edgecolors=P.SERIES[1], lw=0.6, zorder=3)
        if vn in ('GFRP', 'CFRP'):
            ax.annotate(vn, (d['chi'], 100*d['eta_ref']), xytext=(4, -9), textcoords='offset points', fontsize=5.6, color=P.INK2)
    ax.axvline(1.0, color=P.MUTED, lw=0.6, ls=(0, (4, 2)))
    ax.set_xlim(-7, 2.6)
    ax.set_xlabel(r'Hoop-strain mismatch $\chi=\varepsilon_{\theta,\mathrm{dome}}/\varepsilon_{\theta,\mathrm{cyl}}$')
    ax.set_ylabel(r'Transition efficiency $\eta$ (%)')
    ax.set_ylim(35, 104)
    ax.legend(loc='lower left', fontsize=5.8)
    P.label(ax, 'a')
    ax = axs[1]
    for vn, d in D.items():
        if d['mu_req'] is None:
            y, mk = 0.33, '^'
        else:
            y, mk = d['mu_req'], 'o'
        filled = vn in ser
        ax.scatter([d['chi']], [y], marker=mk, s=16, facecolors=P.C['blue'] if filled else 'white',
                   edgecolors=P.C['blue'], lw=0.7, zorder=3)
        off = {'H1': (3, 5), 'H2': (-11, 5), 'H6': (-4, -11), 'H5': (3, 3), 'H8': (-13, 2), 'H3': (4, -6), 'H4': (4, -2),
               'H7': (4, 3), 'GFRP': (4, 4), 'CFRP': (4, 2)}.get(vn, (3, 3))
        ax.annotate(vn, (d['chi'], y), xytext=off, textcoords='offset points', fontsize=5.4, color=P.INK2)
    ax.axhspan(0.1, 0.3, color=P.LIGHT, zorder=0, lw=0)
    ax.text(0.02, 0.2, r'studied range of $\mu$', transform=ax.get_yaxis_transform(), fontsize=5.8, color=P.INK2, va='center')
    ax.axvline(1.0, color=P.MUTED, lw=0.6, ls=(0, (4, 2)))
    ax.set_xlim(-7, 2.6)
    ax.set_ylim(-0.01, 0.36)
    ax.set_xlabel(r'Hoop-strain mismatch $\chi$')
    ax.set_ylabel(r'Slippage coefficient needed for $\eta\geq95\%$')
    ax.text(0.03, 0.97, r'$\triangle$: not reached at $\mu=0.3$', transform=ax.transAxes, ha='left', va='top', fontsize=5.6, color=P.INK2)
    P.label(ax, 'b')
    P.save(fig, f'{OUT}/fig_design_rule{tag}')
    return D
