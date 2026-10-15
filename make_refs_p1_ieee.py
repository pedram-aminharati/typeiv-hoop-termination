"""IEEE-style bibliography of Paper 1 (IJHE two-column template), order of first citation."""
import re, os, sys
sys.path.insert(0, '.')
from make_refs_p2_ieee import REF as R2

J = lambda s: r'\textit{' + s + '}'
REF = dict(R2)
REF.update({
'rwth2026': r"CompositesWorld, ``RWTH Aachen University uses filament winding alone for pressure vessel dome reinforcement,'' 20 May 2026. [Online]. Available: \url{https://www.compositesworld.com/news/rwth-aachen-university-uses-filament-winding-alone-for-pressure-vessel-dome-reinforcement}",
'belardi2024': r"V.~G. Belardi, M.~Ottaviano, and F.~Vivio, ``Bending theory of composite pressure vessels: A closed-form analytical approach,'' " + J('Compos. Struct.') + r", vol.~329, p.~117799, 2024.",
'nebe2020': r"M.~Nebe, T.~Asijee, C.~Braun, J.~van~Campen, and F.~Walther, ``Experimental and analytical analysis on the stacking sequence of composite pressure vessels,'' " + J('Compos. Struct.') + r", vol.~247, p.~112429, 2020.",
'kang2021': r"H.~Kang \textit{et al.}, ``Stress--strain and burst failure analysis of fiber wound composite material high-pressure vessel,'' " + J('Polym. Polym. Compos.') + r", vol.~29, 2021.",
'valilis2026': r"S.~E. Valilis and G.~N. Lampeas, ``Thermomechanical analysis and progressive damage modelling of composite cryogenic hydrogen tanks,'' " + J('Aeronaut. J.') + r", vol.~130, pp.~1173--1197, 2026.",
'lij2026': r"J.~Li \textit{et al.}, ``Study on structural design and stress simulation of carbon fiber composite Type IV hydrogen storage cylinder,'' " + J('Sci. Rep.') + r", vol.~16, p.~22960, 2026.",
'liy2026': r"Y.~Li \textit{et al.}, ``Progressive damage analysis and experimental validation of failure-path transition in filament-wound composite solid rocket motor casings,'' " + J('Appl. Compos. Mater.') + r", vol.~33, p.~124, 2026.",
'wangy2026': r"Y.~Wang \textit{et al.}, ``Lay-up design and simulation for type IV hydrogen storage vessels,'' " + J('J. Tsinghua Univ. (Sci. Technol.)') + r", vol.~66, pp.~1633--1643, 2026.",
'wangz2025': r"Z.~Wang \textit{et al.}, ``Cross-scale modeling of CFRP stacking sequence in filament-wound composite pressure vessels: In-plane and inter-layer homogenization analysis,'' " + J('Materials') + r", vol.~18, p.~4612, 2025.",
'zhou2025': r"Y.~Zhou \textit{et al.}, ``Simulation analysis and optimization design of dome structure in filament wound composite shells,'' " + J('Polymers') + r", vol.~17, p.~1421, 2025.",
'rozova2024': r"L.~Rozova, B.~Meemary, S.~Chaki, M.~Del\'{e}glise-Lagard\`{e}re, and D.~Vasiukov, ``Multi-objective optimization for a composite pressure vessel with unequal polar openings,'' arXiv:2410.03708, 2024.",
'kim2025': r"G.~Kim \textit{et al.}, ``Finite element analysis framework for structural safety evaluation of Type IV hydrogen storage vessel,'' " + J('Hydrogen') + r", vol.~6, p.~44, 2025.",
'paczkowska2026': r"K.~Paczkowska \textit{et al.}, ``Multiscale experimental and numerical assessment of filament-wound composite pressure vessel for hydrogen storage,'' " + J('Energies') + r", vol.~19, p.~4202, 2026.",
'mukherjee2001': r"A.~Mukherjee and B.~Varughese, ``Design guidelines for ply drop-off in laminated composite structures,'' " + J('Compos. Part B Eng.') + r", vol.~32, pp.~153--164, 2001.",
'gordon2021': r"T.~Gordon, X.~Xu, L.~Kawashita, M.~R. Wisnom, S.~R. Hallett, and B.~C. Kim, ``Delamination suppression in tapered unidirectional laminates with multiple ply drops using a tape scarfing technique,'' " + J('Compos. Part A Appl. Sci. Manuf.') + r", vol.~150, p.~106627, 2021.",
'brewer1988': r"J.~C. Brewer and P.~A. Lagace, ``Quadratic stress criterion for initiation of delamination,'' " + J('J. Compos. Mater.') + r", vol.~22, pp.~1141--1155, 1988.",
'whitney1974': r"J.~M. Whitney and R.~J. Nuismer, ``Stress fracture criteria for laminated composites containing stress concentrations,'' " + J('J. Compos. Mater.') + r", vol.~8, pp.~253--265, 1974.",
})


def build(texdir='./paper1_ieee', main='manuscript.tex'):
    order = []
    def scan(fn):
        s = open(os.path.join(texdir, fn)).read()
        for m in re.finditer(r'\\input\{(\w+)\}|\\cite\{([^}]*)\}', s):
            if m.group(1):
                if m.group(1) != 'references':
                    scan(m.group(1) + '.tex')
            else:
                for k in m.group(2).split(','):
                    k = k.strip()
                    if k not in order:
                        order.append(k)
    scan(main)
    miss = [k for k in order if k not in REF]
    L = ['\\begin{thebibliography}{99}', '']
    for k in order:
        if k in REF:
            L += [f'\\bibitem{{{k}}}', REF[k], '']
    L.append('\\end{thebibliography}')
    open(os.path.join(texdir, 'references.tex'), 'w').write('\n'.join(L) + '\n')
    return order, miss


if __name__ == '__main__':
    o, m = build()
    print(len(o), 'cited; missing:', m)
