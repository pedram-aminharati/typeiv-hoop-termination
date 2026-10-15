"""Mechanistic design-rule parameter: membrane hoop-strain mismatch at the tangent line.

chi = eps_theta(dome, helical-only laminate at r = R, equatorial membrane forces of
the ellipsoid) / eps_theta(cylinder, full laminate, N_z = pR/2, N_theta = pR).
Equatorial membrane forces of an ellipsoidal head (semi-axes R, h_d):
N_phi = pR/2, N_theta = pR (1 - R^2/(2 h_d^2)).
chi > 1: the bare shoulder wants to expand more than the cylinder, so the hoop
layers must carry load on the shoulder; chi <= 1: they can end at the tangent line.
"""
import json
import numpy as np
import core
import p1_study as PS


def Qbar(theta, m):
    C = core.stiffness_cyl(theta, m)          # 3D, frame (z, theta, r), Voigt (zz, tt, rr, tr, zr, zt)
    S = np.linalg.inv(C)
    idx = [0, 1, 5]
    return np.linalg.inv(S[np.ix_(idx, idx)])  # plane stress, engineering shear


def A_matrix(stack, fam, m, hoop=True, angle_of=None):
    A = np.zeros((3, 3))
    for e in stack:
        if e[0] == 'H':
            if hoop:
                A += Qbar(90.0, m)*e[-1]
        elif e[0] == 'P':
            continue
        else:
            k, sg, t = e[0], e[1], e[-1]
            A += Qbar(sg*fam[k], m)*t
    return A


def chi(vn):
    b = PS.base(vn)
    m = dict(core.MAT); m.update(b['mat'])
    R, hd = b['ri'], b['hd']
    p = 1.0
    Acyl = A_matrix(b['stack'], b['fam'], m, hoop=True)
    Adome = A_matrix(b['stack'], b['fam'], m, hoop=False)
    ec = np.linalg.solve(Acyl, [p*R/2, p*R, 0.0])
    ed = np.linalg.solve(Adome, [p*R/2, p*R*(1 - R**2/(2*hd**2)), 0.0])
    return dict(chi=ed[1]/ec[1], n_eq=1 - R**2/(2*hd**2), eps_cyl=ec[1], eps_dome=ed[1], hd_R=hd/R)


if __name__ == '__main__':
    S = json.load(open('p1_summary.json'))
    out = {}
    for vn in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'GFRP', 'CFRP']:
        c = chi(vn)
        v = S[vn]
        c['eta_ref'] = v['ref']['pf']/v['ref']['pf_mid']
        c['eta_best'] = {mu: b['pf']/v['ref']['pf_mid'] for mu, b in v['best'].items()}
        g = max(v['grid'], key=lambda g: g['pf'])
        c['eta_grid'] = g['pf']/v['ref']['pf_mid']
        out[vn] = c
        print(vn, 'hd/R %.3f n_eq %+.3f chi %.3f eta_ref %.3f eta_grid %.3f best' % (c['hd_R'], c['n_eq'], c['chi'], c['eta_ref'], c['eta_grid']),
              {k: round(x, 3) for k, x in c['eta_best'].items()})
    json.dump(out, open('p1_rule.json', 'w'), indent=1)
