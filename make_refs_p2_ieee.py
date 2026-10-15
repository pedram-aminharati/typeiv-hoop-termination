"""IEEE-style bibliography of Paper 2 (IJHE two-column template), order of first citation."""
import re, os, sys
sys.path.insert(0, '.')
from make_refs_p2 import build as _build_els  # noqa: F401  (same citation scan)

J = lambda s: r'\textit{' + s + '}'
D = lambda d: ''   # the IJHE template lists no DOIs
REF = {
'piraino2025': r"F.~Piraino, L.~Pagnotta, O.~Corigliano, M.~Genovese, and P.~Fragiacomo, ``Advances in Type IV tanks for safe hydrogen storage: Materials, technologies and challenges,'' " + J('Hydrogen') + r", vol.~6, p.~80, 2025." + D('10.3390/hydrogen6040080'),
'bouhala2024': r"L.~Bouhala \textit{et al.}, ``Advancement in the modeling and design of composite pressure vessels for hydrogen storage: A comprehensive review,'' " + J('J. Compos. Sci.') + r", vol.~8, p.~339, 2024." + D('10.3390/jcs8090339'),
'aminharati2025review': r"P.~Aminharati, M.~Shirinbayan, K.~Benfriha, F.~Meraghni, and J.~Fitoussi, ``AI-driven advances in composite materials for hydrogen storage vessels: A review,'' " + J('Int. J. Hydrogen Energy') + r", vol.~171, p.~151288, 2025." + D('10.1016/j.ijhydene.2025.151288'),
'gtr13': r"United Nations Economic Commission for Europe, ``Global Technical Regulation No.~13: Hydrogen and fuel cell vehicles,'' ECE/TRANS/180/Add.13, Geneva, 2013.",
'eu406': r"European Commission, ``Commission Regulation (EU) No 406/2010 implementing Regulation (EC) No 79/2009 on type-approval of hydrogen-powered motor vehicles,'' Annex~IV, Part~2, Table~IV.3.6, 2010.",
'vasiliev2009': r"V.~V. Vasiliev, \textit{Composite Pressure Vessels: Analysis, Design, and Manufacturing}. Blacksburg, VA, USA: Bull Ridge Publishing, 2009.",
'ozaslan2021': r"E.~\"{O}zaslan \textit{et al.}, ``Experimental and numerical investigation of the cylinder-dome transition region of a pressure vessel,'' " + J('Compos. Part C Open Access') + r", vol.~4, p.~100119, 2021." + D('10.1016/j.jcomc.2021.100119'),
'cho2018': r"S.-M. Cho \textit{et al.}, ``Effect of dome curvature on failure mode of Type4 composite pressure vessel,'' " + J('Int. J. Precis. Eng. Manuf.') + r", vol.~19, pp.~405--410, 2018." + D('10.1007/s12541-018-0048-5'),
'liu2025ce': r"C.~Liu, Z.~Ouyang, J.~Yu, and L.~Ma, ``Investigation of crack initiation during the burst test of hydrogen storage composite overwrapped pressure vessels,'' " + J('Clean Energy') + r", vol.~9, no.~1, pp.~239--249, 2025." + D('10.1093/ce/zkae123'),
'leng2026': r"X.~Leng \textit{et al.}, ``Damage evolution and critical failure regions in dry-towpreg-wound 70~MPa Type IV hydrogen vessels,'' Research Square preprint, 2026." + D('10.21203/rs.3.rs-10520024/v1'),
'companion': r"P.~Aminharati, M.~Shirinbayan, K.~Benfriha, and J.~Fitoussi, ``A domain-bounded multi-fidelity correction framework for dome- and junction-critical failure onset in Type IV filament-wound composite pressure vessels,'' " + J('Int. J. Hydrogen Energy') + r", under review, 2026.",
'p1': r"P.~Aminharati, M.~Shirinbayan, K.~Benfriha, and J.~Fitoussi, ``Friction-bounded design of the cylinder--dome transition of Type~IV composite pressure vessels: Hoop termination, stacking order and hybrid reinforcement layers,'' " + J('Int. J. Hydrogen Energy') + r", submitted, 2026.",
'hwang2003': r"T.-K. Hwang, C.-S. Hong, and C.-G. Kim, ``Size effect on the fiber strength of composite pressure vessels,'' " + J('Compos. Struct.') + r", vol.~59, no.~4, pp.~489--498, 2003." + D('10.1016/S0263-8223(02)00250-7'),
'wisnom2017': r"M.~R. Wisnom, M.~A. Cantera, G.~Cz\'{e}l, and M.~Jalalvand, ``New approaches to quantifying tensile strength variability and size effects in unidirectional composites,'' in \textit{Proc. Am. Soc. Compos. 32nd Tech. Conf.}, Lancaster, PA, USA: DEStech, 2017, pp.~2902--2909.",
'swolfs2015': r"Y.~Swolfs \textit{et al.}, ``Synchrotron radiation computed tomography for experimental validation of a tensile strength model for unidirectional fibre-reinforced composites,'' " + J('Compos. Part A Appl. Sci. Manuf.') + r", vol.~77, pp.~106--113, 2015." + D('10.1016/j.compositesa.2015.06.018'),
'camara2011': r"S.~Camara, A.~R. Bunsell, A.~Thionnet, and D.~H. Allen, ``Determination of lifetime probabilities of carbon fibre composite plates and pressure vessels for hydrogen storage,'' " + J('Int. J. Hydrogen Energy') + r", vol.~36, no.~10, pp.~6031--6038, 2011.",
'chou2013': r"H.-Y. Chou, A.~R. Bunsell, G.~Mair, and A.~Thionnet, ``Effect of the loading rate on failure of composite pressure vessel,'' in \textit{Proc. ASME 2013 Pressure Vessels and Piping Conf.}, vol.~5. New York, NY, USA: ASME, 2014.",
'sidamer2023': r"Y.~Sid Amer, S.~Benammar, K.~F. Tee, M.~Wadi, and M.~S. Jouda, ``A contribution to structural reliability analysis of composite high-pressure hydrogen storage tanks,'' " + J('J. KONBiN') + r", vol.~53, no.~2, pp.~33--44, 2023." + D('10.5604/01.3001.0053.7104'),
'sidamer2025': r"Y.~Sid Amer, S.~Benammar, K.~F. Tee, and Z.~Iourzikene, ``A study on the small failure probabilities of cylindrical composite hydrogen storage tanks using subset simulation,'' " + J('Proc. Inst. Mech. Eng. O, J. Risk Reliab.') + r", vol.~239, no.~4, pp.~826--838, 2025." + D('10.1177/1748006X241270735'),
'azizian2023': r"M.~Azizian, J.~H.~S. Almeida~Jr, and M.~Azizian, ``Efficient strategies for reliability analysis and uncertainty quantification for filament-wound cylinders under internal pressure,'' " + J('J. Compos. Mater.') + r", vol.~57, no.~11, 2023." + D('10.1177/00219983231164221'),
'li2025jrpc': r"W.~Li, H.~Lv, L.~Zhang, P.~He, and C.~Zhang, ``Reliability analysis and optimization design of hydrogen storage composite pressure vessel with hybrid random-fuzzy uncertainties,'' " + J('J. Reinf. Plast. Compos.') + r", vol.~44, no.~7--8, pp.~358--374, 2025." + D('10.1177/07316844231220707'),
'koutsawa2025': r"Y.~Koutsawa and L.~Bouhala, ``Uncertainty analysis in the design of Type-IV composite pressure vessels for hydrogen storage,'' " + J('Compos. Part C Open Access') + r", vol.~16, p.~100544, 2025." + D('10.1016/j.jcomc.2024.100544'),
'beckmann2026': r"C.~Beckmann \textit{et al.}, ``Probabilistic integrity assessment of filament wound CFRP pressure vessels,'' " + J('J. Compos. Mater.') + r", vol.~60, no.~5, pp.~461--479, 2026." + D('10.1177/00219983251363845'),
'vanbavel2024': r"B.~Van~Bavel, D.~Vandepitte, and D.~Moens, ``Composite pressure vessel failure simulation considering spatial variability,'' " + J('AIAA J.') + r", 2024." + D('10.2514/1.J064163'),
'schlegel2024': r"D.~Schlegel \textit{et al.}, ``Multi scale systematisation of damage and failure modes of high-pressure hydrogen composite vessels in aviation, Part 1: Methodology,'' " + J('Int. J. Hydrogen Energy') + r", vol.~95, pp.~796--805, 2024." + D('10.1016/j.ijhydene.2024.11.115'),
'schlegel2025': r"D.~Schlegel, M.~Vater, S.~Spitzer, M.~Gude, and A.~Hurtado, ``Multi scale systematisation of damage and failure modes of high-pressure hydrogen composite vessels in aviation, Part 2: Analysis,'' " + J('Int. J. Hydrogen Energy') + r", vol.~98, pp.~52--66, 2025." + D('10.1016/j.ijhydene.2024.11.457'),
'schlegel2026': r"D.~Schlegel, M.~Gude, A.~Hurtado, and S.~Spitzer, ``Multi scale reliability analysis for high-pressure hydrogen composite vessels in aviation, Part 2: Systemic failure propagation and safety factor optimisation,'' " + J('Int. J. Hydrogen Energy') + r", vol.~239, p.~155321, 2026." + D('10.1016/j.ijhydene.2026.155321'),
'echard2011': r"B.~Echard, N.~Gayton, and M.~Lemaire, ``AK-MCS: An active learning reliability method combining Kriging and Monte Carlo simulation,'' " + J('Struct. Saf.') + r", vol.~33, no.~2, pp.~145--154, 2011." + D('10.1016/j.strusafe.2011.01.002'),
'du2026': r"Z.~Du, D.~Jiang, T.~Fan, and H.~Zhu, ``A dynamic-switching active-learning Kriging considering epistemic uncertainty for structural reliability analysis,'' " + J('Proc. Inst. Mech. Eng. O, J. Risk Reliab.') + r", vol.~240, no.~4, pp.~1292--1309, 2026." + D('10.1177/1748006X261455467'),
'aminharati2026ga': r"P.~Aminharati, M.~Shirinbayan, K.~Benfriha, F.~Meraghni, and J.~Fitoussi, ``Optimization of composite overwrapped pressure vessel stacking sequences via genetic algorithms: A triadic evaluation of Tsai-Wu, Hashin, and Puck failure criteria,'' " + J('Compos. Part C Open Access') + r", vol.~21, p.~100796, 2026." + D('10.1016/j.jcomc.2026.100796'),
'bouhalaperbal2026': r"L.~Bouhala and S.~Perbal, ``AI in composite overwrapped pressure vessels: A review and advanced roadmap from materials design to predictive maintenance,'' " + J('J. Compos. Sci.') + r", vol.~10, p.~171, 2026." + D('10.3390/jcs10030171'),
'jiao2012': r"W.~Jiao, R.~Wang, W.~Liu, F.~Yang, and X.~He, ``Measurement of slippage coefficient between fiber and mandrel surface for non-geodesic filament winding,'' " + J('Acta Mater. Compos. Sin.') + r", vol.~29, no.~3, pp.~191--196, 2012.",
'jois2021': r"K.~C. Jois, M.~Welsh, T.~Gries, and J.~Sackmann, ``Numerical analysis of filament wound cylindrical composite pressure vessels accounting for variable dome contour,'' " + J('J. Compos. Sci.') + r", vol.~5, p.~56, 2021." + D('10.3390/jcs5020056'),
'jois2023': r"K.~C. Jois, M.~Persiyanov-Dubrov, T.~M\"{o}lling, and T.~Gries, ``A digital shadow of filament wound lightweight components,'' in \textit{Proc. 23rd Int. Conf. Compos. Mater. (ICCM23)}, Belfast, U.K., 2023.",
'agne2025': r"L.~L. Agne, J.~H.~S. Almeida~Jr, M.~H.~M. Maruyama, D.~T. de~Souza, S.~C. Amico, and M.~L.~P. Tonatto, ``Progressive failure modelling of Type IV composite overwrapped pressure vessels for compressed natural gas (CNG) storage,'' " + J('Compos. Struct.') + r", vol.~370, p.~119424, 2025." + D('10.1016/j.compstruct.2025.119424'),
'dhondt2004': r"G.~Dhondt, \textit{The Finite Element Method for Three-Dimensional Thermomechanical Applications}. Chichester, U.K.: Wiley, 2004.",
'lekhnitskii1981': r"S.~G. Lekhnitskii, \textit{Theory of Elasticity of an Anisotropic Body}. Moscow, Russia: Mir Publishers, 1981.",
'iso2394': r"\textit{General Principles on Reliability for Structures}, ISO 2394:2015, International Organization for Standardization, Geneva, 2015.",
'jcss2001': r"Joint Committee on Structural Safety, \textit{Probabilistic Model Code, Part 1: Basis of Design}. JCSS, 2001.",
'rasmussen2006': r"C.~E. Rasmussen and C.~K.~I. Williams, \textit{Gaussian Processes for Machine Learning}. Cambridge, MA, USA: MIT Press, 2006.",
'hasofer1974': r"A.~M. Hasofer and N.~C. Lind, ``Exact and invariant second-moment code format,'' " + J('J. Eng. Mech. Div.') + r", vol.~100, no.~1, pp.~111--121, 1974.",
'rackwitz1978': r"R.~Rackwitz and B.~Fiessler, ``Structural reliability under combined random load sequences,'' " + J('Comput. Struct.') + r", vol.~9, no.~5, pp.~489--494, 1978.",
'saltelli2010': r"A.~Saltelli, P.~Annoni, I.~Azzini, F.~Campolongo, M.~Ratto, and S.~Tarantola, ``Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index,'' " + J('Comput. Phys. Commun.') + r", vol.~181, no.~2, pp.~259--270, 2010.",
}


def build(texdir='./paper2_ieee', main='manuscript.tex'):
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
    L = ['\\begin{thebibliography}{99}', '\\providecommand{\\doi}[1]{doi:~\\href{https://doi.org/#1}{#1}.}', '']
    for k in order:
        if k in REF:
            L += [f'\\bibitem{{{k}}}', REF[k], '']
    L.append('\\end{thebibliography}')
    open(os.path.join(texdir, 'references.tex'), 'w').write('\n'.join(L) + '\n')
    return order, miss


if __name__ == '__main__':
    o, m = build()
    print(len(o), 'cited; missing:', m)
