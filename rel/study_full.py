"""Paper 2 full study.

Scatter (literature-based): layer-thickness factors CoV 5% and winding-angle
CoV 2% (Azizian et al., J. Compos. Mater. 57 (2023)), fibre modulus CoV 3%,
slippage coefficient lognormal (median 0.20, CoV 25%).  Fibre strength:
Weibull m = 25, anchored per vessel on the barrel of its reference design at
P_f(2.25 NWP) = 1e-4.

Designs: H1..H8 and S65..S75, reference transition and Paper-1 transition;
programmed hoop-end sweep for H3, H7 and S70.
"""
import json
import os
import sys
import numpy as np
from scipy.stats import norm
sys.path.insert(0, '.')
import p1_study as PS
from rel import runner as RN, weakest as W, scatter as SC, surrogate as SG

OUT = './rel_results'
M, V0, PF_T = 25.0, 2000.0, 1e-4
N0, NAD = 20, 8
COMPACT = ('H3', 'H4', 'H7', 'H8', 'S65', 'S70', 'S75')


def designs():
    D = {}
    for vn in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'S65', 'S70', 'S75']:
        D[f'{vn}-ref'] = dict(vessel=vn, order=None, e_w=0.0, l_w=0.5)
        if vn in COMPACT:
            D[f'{vn}-out20'] = dict(vessel=vn, order='H block outside', mu_d=0.20, l_w=0.5)
        else:
            D[f'{vn}-e0125'] = dict(vessel=vn, order=None, e_w=0.125, l_w=0.5)
    for vn in ('H3', 'H7', 'S70'):
        for mu in (0.10, 0.15, 0.25, 0.30):
            D[f'{vn}-out{int(round(100*mu))}'] = dict(vessel=vn, order='H block outside', mu_d=mu, l_w=0.5)
    return D


class Scatter(SC.Scatter):
    """literature-based scatter; the dome-family angle error is 2% of its angle"""

    def __init__(self, alpha2):
        super().__init__(cv_t=0.05, sd_a=0.02*alpha2, mu_med=0.20, cv_mu=0.25, cv_E=0.03)


def scaled_base(spec, k=1.0):
    b = PS.base(spec['vessel'], spec['order'])
    st = [('H', e[1]*k) if e[0] == 'H' else (e[0], e[1], e[2]*k) for e in b['stack']]
    return dict(b, stack=st, k=k)


def design(spec, k=1.0):
    c0 = scaled_base(spec, k)
    w = PS.wj(c0)
    e_d = PS.e_crit(c0, spec['mu_d'])*w if 'mu_d' in spec else spec['e_w']*w
    return dict(c=c0, e_d=e_d, l_h=spec['l_w']*w)


def nominal(spec, tag, k=1.0):
    return SC.realize(design(spec, k), [1, 1, 0, 10.0, 1], f'{tag}_k{k:.3f}_nom')


_WB = {}


def weibull(vessel, m=M, v0=V0, pf_t=PF_T):
    key = (vessel, m, v0, pf_t)
    if key not in _WB:
        c = nominal(dict(vessel=vessel, order=None, e_w=0.0, l_w=0.5), f'{vessel}-anchor')
        RN.run_many([c])
        P = PS.VESSELS[vessel]['P']
        _WB[key] = W.Weibull.anchored(RN.data(c), 2.25*P, pf_t, m, v0)
    return _WB[key]


def evaluate(c, vessel, m=M):
    return weibull(vessel, m).burst(RN.data(c))


def phys_fmap(D, sc):
    """physical features (k_h, k_hel, da2, e/sqrt(Rt), E1f); valid for any scatter model"""
    c0 = D['c']; w = PS.wj(c0); e_d = D['e_d']
    mus = np.geomspace(0.02, 2.0, 400)
    ec = []
    for mu in mus:
        try:
            ec.append(PS.e_crit(c0, mu)*w)
        except ValueError:
            ec.append(10*w)
    ec = np.array(ec)

    def f(U):
        X = sc.to_x(np.atleast_2d(U))
        e = np.full(len(X), e_d) if e_d <= 0 else np.minimum(e_d, np.interp(X[:, 3], mus, ec))
        return np.column_stack([X[:, 0], X[:, 1], X[:, 2], e/w, X[:, 4]])
    return f


def train(name, spec):
    D = design(spec)
    sc = Scatter(D['c']['fam'][1])
    vn = spec['vessel']
    U = sc.lhs(N0, seed=sum(map(ord, name)))
    cs = [SC.realize(D, x, f'{name}_lhs{i}') for i, x in enumerate(sc.to_x(U))]
    RN.run_many(cs)
    bs = [evaluate(c, vn) for c in cs]
    S = SG.Surrogate(U, [np.log(b['p0']) for b in bs],
                     [b['shares']['junction'] + b['shares']['dome'] for b in bs], fmap=phys_fmap(D, sc))
    P = PS.VESSELS[vn]['P']
    for j in range(NAD):
        u, crit = SG.next_point(S, M, 2.25*P, seed=10 + j)
        c = SC.realize(D, sc.to_x(u)[0], f'{name}_ad{j}')
        RN.run_many([c])
        b = evaluate(c, vn)
        S.add(u, np.log(b['p0']), b['shares']['junction'] + b['shares']['dome'])
    return S, sc, D


def k_required(S, gam, pr, pf_t=PF_T, n=100000):
    from scipy.optimize import brentq
    U = np.random.default_rng(11).standard_normal((n, 5))
    lp = S.predict(U)
    f = lambda k: np.mean(-np.expm1(-(pr/(k**gam*np.exp(lp)))**M)) - pf_t
    return float(brentq(f, 0.5, 3.0))


def run_design(name, spec):
    vn = spec['vessel']
    P = PS.VESSELS[vn]['P']
    pr = 2.25*P
    cn = [nominal(spec, name, k) for k in (1.0, 1.15)]
    RN.run_many(cn)
    bn = [evaluate(c, vn) for c in cn]
    gam = float(np.log(bn[1]['p0']/bn[0]['p0'])/np.log(1.15))
    S, sc, D = train(name, spec)
    cur = SG.pf_curve(S, M, [pr, 1.5*P, 1.25*P], gp_samples=20)
    fo = SG.form(S, M, pr, 5)
    s1, st, var = SG.sobol(S, n=10000)
    loo = S.loo()
    U = np.random.default_rng(7).standard_normal((50000, 5))
    rec = dict(vessel=vn, spec=spec, P=P, gamma=gam, nominal=bn[0],
               pf_nom=float(-np.expm1(-(pr/bn[0]['p0'])**M)),
               pf_req=float(cur['pf'][0]), pf_band=[float(np.percentile(cur['band'][:, 0], q)) for q in (5, 95)],
               pf_15=float(cur['pf'][1]), pf_125=float(cur['pf'][2]), beta=float(SG.beta(cur['pf'][0])),
               beta_125=float(SG.beta(cur['pf'][2])),
               form=dict(beta=fo['beta'], alpha=fo['alpha'].tolist(), u_star=fo['u_star'].tolist()),
               sobol_S1=s1.tolist(), sobol_ST=st.tolist(), var_lnp0=float(var),
               loo_rms=float(np.sqrt(np.mean(loo**2))), loo_max=float(np.max(np.abs(loo))),
               share_transition=float(np.mean(S.share(U))), k_req=k_required(S, gam, pr),
               mass=json.load(open(os.path.join(RN.STORE, RN.key(cn[0]) + '.json')))['mass'],
               U=S.U.tolist(), y=S.y.tolist(), sh=S.sh.tolist(), e_d=D['e_d'], wj=PS.wj(D['c']))
    return rec


def main(only=None):
    fn = f'{OUT}/full.json'
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    for name, spec in designs().items():
        if only and name not in only:
            continue
        if name in res:
            continue
        r = run_design(name, spec)
        res[name] = r
        json.dump(res, open(fn, 'w'), indent=1)
        print(f"{name}: Pf_nom={r['pf_nom']:.2e} Pf={r['pf_req']:.2e} beta={r['beta']:.2f} "
              f"FORM={r['form']['beta']:.2f} trans={r['share_transition']:.3f} k_req={r['k_req']:.3f} "
              f"beta125={r['beta_125']:.2f} LOO={r['loo_rms']:.4f}", flush=True)


if __name__ == '__main__':
    main(sys.argv[1:] or None)
