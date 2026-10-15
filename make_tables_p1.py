"""Generate the LaTeX tables of Paper 1 from the result files."""
import json
import numpy as np
import fig_paper1 as F
import p1_study as PS
import p1_rule as PR

S = json.load(open('p1_summary.json'))
Z = json.load(open('sizing.json'))
D = json.load(open('p1_rule_data.json'))
VES = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'S65', 'S70', 'S75']
out = []

# ---- Table 1: vessels
L = [r'\begin{table*}[t]\centering\scriptsize\setlength{\tabcolsep}{3.2pt}',
     r'\caption{Vessels. Dome-consistent walls: helical wall scaled by $k_{\mathrm{hel}}$ from netting theory so that $R_D=1$; hoop wall from netting hoop equilibrium. $\chi$: hoop-strain mismatch, Eq.~(\ref{eq:chi}). $e_{\mathrm{crit}}$: largest windable hoop end. $p_{\mathrm{cyl}}$: fibre-critical pressure of the cylinder; $\eta_{\mathrm{ref}}$: efficiency of the reference transition ($e=0$, $\ell_h=0.5\sqrt{Rt}$).}\label{tab:vessels}',
     r'\begin{tabular}{lccccccccccccc}', r'\toprule',
     r'Vessel & NWP & $h_d/R$ & $r_b/R$ & $\alpha_1$ & $k_{\mathrm{hel}}$ & $t_h$ & $t_{\mathrm{hel}}$ & $R/t$ & $\chi$ & \multicolumn{3}{c}{$e_{\mathrm{crit}}/\sqrt{Rt}$ at $\mu=$} & $\eta_{\mathrm{ref}}$\\',
     r' & (MPa) & & & ($^\circ$) & & (mm) & (mm) & & & 0.1 & 0.2 & 0.3 & ($p_{\mathrm{cyl}}$, MPa)\\', r'\midrule']
for vn in VES:
    v = S[vn]; b = PS.base(vn); z = Z[vn]
    ec = [PS.e_crit(b, m) for m in (0.1, 0.2, 0.3)]
    t = sum(e[-1] for e in b['stack'])
    L.append(f"{vn} & {b['P']:.0f} & {b['hd_R']:.2f} & {b['rb_R']:.2f} & {b['fam'][0]:.1f} & {z['k']:.2f} & {b['t_h']:.2f} & {b['t_hel']:.2f} & {b['ri']/t:.1f} & ${D[vn]['chi']:+.2f}$ & "
             + ' & '.join(f'{x:.2f}' for x in ec) + f" & {v['ref']['pf']/v['ref']['pf_mid']:.2f} ({v['ref']['pf_mid']:.0f})\\\\")
L.append(r'\midrule')
for vn in ('GFRP', 'CFRP'):
    v = S[vn]; b = PS.base(vn)
    ec = [PS.e_crit(b, m) for m in (0.1, 0.2, 0.3)]
    t = sum(e[-1] for e in b['stack'])
    L.append(f"Agne {vn} & 10$^a$ & {b['hd']/b['ri']:.2f} & -- & -- & -- & -- & -- & {b['ri']/t:.1f} & ${D[vn]['chi']:+.2f}$ & "
             + ' & '.join(f'{x:.2f}' for x in ec) + f" & {v['ref']['pf']/v['ref']['pf_mid']:.2f} ({v['ref']['pf_mid']:.0f})\\\\")
L += [r'\bottomrule', r'\end{tabular}', r'\par\smallskip\raggedright $^a$ reference pressure of the analysis; layup and properties from \cite{agne2025}.', r'\end{table*}']
out += L

# ---- Table 2: best windable designs
L = [r'\begin{table*}[t]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}',
     r'\caption{Best windable transition design per vessel and friction coefficient (largest $p_f$ among all analysed designs with $e\le e_{\mathrm{crit}}(\mu)$). Design: $e/\sqrt{Rt}$, $\ell_h/\sqrt{Rt}$, order (ref = H|F|F$\times$2, out = hoop block outside). $\Delta p_f$ and $\Delta m$ in \% relative to the reference transition.}\label{tab:best}',
     r'\begin{tabular}{l' + 'lcc'*3 + '}', r'\toprule',
     r'Vessel & \multicolumn{3}{c}{$\mu=0.1$} & \multicolumn{3}{c}{$\mu=0.2$} & \multicolumn{3}{c}{$\mu=0.3$}\\',
     r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}',
     r' & design & $\eta$ & $\Delta p_f$/$\Delta m$ & design & $\eta$ & $\Delta p_f$/$\Delta m$ & design & $\eta$ & $\Delta p_f$/$\Delta m$\\',
     r'\midrule']
for vn in VES + ['GFRP', 'CFRP']:
    v = S[vn]; row = [vn if vn in PS.VESSELS else f'Agne {vn}']
    for mu in ('0.1', '0.2', '0.3'):
        bb = v['best'][mu]
        o = bb.get('order', 'ref'); o = 'out' if 'outside' in o else 'ref'
        ev = bb['e']
        grid = abs(ev*8 - round(ev*8)) < 1e-6
        es = '0' if ev == 0 else (f'{ev:.3f}'.rstrip('0') if grid else f'{ev:.3f}')
        ls = '1.0' if bb['l'] == 1 else f"{bb['l']:g}"
        dm = 100*bb['dm']; dm = 0.0 if abs(dm) < 0.05 else dm
        row += [f'{es}, {ls}, {o}', f"{bb['eta']:.2f}", f"${100*bb['dpf']:+.0f}$/" + (f"${dm:+.1f}$" if dm != 0 else '0.0')]
    L.append(' & '.join(row) + r'\\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
out += L

# ---- Table 3: interlaminar
il = F.load_il()
def find(vn, **kw):
    for o in il:
        if o['vessel'] != vn:
            continue
        ok = True
        for k, val in kw.items():
            ov = o.get(k)
            if k == 'e_w':
                ok &= abs(ov - val) < 2e-3
            elif k == 'hyb':
                ok &= (ov is None and val is None) or (ov is not None and val is not None and abs(ov[0] - val[0]) < 1e-6)
            else:
                ok &= ov == val
        if ok and (kw.get('Lp_fac') or o.get('Lp_fac', 4.0) == 4.0):
            return o
    return None
rows = [('Reference ($e=0$, $\\ell_h=0.5$)', dict(e_w=0.0, l_w=0.5, order='ref', n_rep=4, hyb=None)),
        ('$e=-0.5\\sqrt{Rt}$', dict(e_w=-0.5, l_w=0.5, order='ref', n_rep=4, hyb=None)),
        ('$e=+0.125\\sqrt{Rt}$', dict(e_w=0.125, l_w=0.5, order='ref', n_rep=4, hyb=None)),
        ('$e=+0.5\\sqrt{Rt}$', dict(e_w=0.5, l_w=0.5, order='ref', n_rep=4, hyb=None)),
        ('$\\ell_h=0.25\\sqrt{Rt}$', dict(e_w=0.0, l_w=0.25, order='ref', n_rep=4, hyb=None)),
        ('$\\ell_h=1.0\\sqrt{Rt}$', dict(e_w=0.0, l_w=1.0, order='ref', n_rep=4, hyb=None)),
        ('Hoop block inside', dict(e_w=0.0, l_w=0.5, order='in', n_rep=4, hyb=None)),
        ('Hoop block outside', dict(e_w=0.0, l_w=0.5, order='out', n_rep=4, hyb=None)),
        ('Hoop block outside, $e=e_{\\mathrm{crit}}(0.3)$', 'outcrit'),
        ('Hybrid layer ($r_{t,p}=0.55R$)', dict(e_w=0.0, l_w=0.5, order='ref', n_rep=4, hyb=(0.55, -1.0))),
        ('2 interleaved repeats', dict(e_w=0.0, l_w=0.5, order='ref', n_rep=2, hyb=None)),
        ('8 interleaved repeats', dict(e_w=0.0, l_w=0.5, order='ref', n_rep=8, hyb=None))]
L = [r'\begin{table}[t]\centering\footnotesize',
     r'\caption{Layer-wise model: delamination index $FI_d$ at NWP ($\delta=1$~mm) and onset pressure $p_d/\mathrm{NWP}$. Four interleaved repeats unless stated.}\label{tab:il}',
     r'\begin{tabular}{lcccc}', r'\toprule',
     r'Design & H1 & H3 & H5 & H7\\',
     r' & \multicolumn{4}{c}{$FI_d$ ($p_d/\mathrm{NWP}$)}\\', r'\midrule']
for lab, kw in rows:
    cells = []
    for vn in ('H1', 'H3', 'H5', 'H7'):
        if kw == 'outcrit':
            ec = round(PS.e_crit(PS.base(vn), 0.3), 4)
            o = find(vn, e_w=ec, l_w=0.5, order='out', n_rep=4, hyb=None)
        else:
            o = find(vn, **kw)
        cells.append('--' if o is None else f"{o['FI']:.2f} ({o['pd1']/o['p']:.2f})")
    L.append(lab + ' & ' + ' & '.join(cells) + r'\\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
out += L
open('paper1/tables.tex', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
