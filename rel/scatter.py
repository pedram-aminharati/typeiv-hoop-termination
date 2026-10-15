"""Manufacturing and material scatter of a wound vessel, and its mapping onto
the CalculiX design dictionary.

Random inputs (independent; standard-normal coordinates u):
  k_h   hoop-layer thickness factor            normal,    mean 1, CoV cv_t
  k_hel helical-layer thickness factor         normal,    mean 1, CoV cv_t
  da2   winding-angle error of the dome family normal,    mean 0, sd sd_a [deg]
  mu    tow-on-surface slippage coefficient    lognormal, median mu_med, CoV cv_mu
  E1f   fibre-direction modulus factor         normal,    mean 1, CoV cv_E
The fibre strength is not a sampled input: it enters analytically through the
weakest-link model (weakest.py).

The hoop end is limited by friction: a hoop layer programmed to end at e_d
on the dome shoulder stays at e_d only if the actual slippage coefficient
allows it; otherwise it slips back to the stable position e_crit(mu).
"""
import numpy as np
from scipy.stats import norm
import p1_study as PS

NAMES = ('k_h', 'k_hel', 'da2', 'mu', 'E1f')


class Scatter:
    def __init__(self, cv_t=0.03, sd_a=0.5, mu_med=0.2, cv_mu=0.25, cv_E=0.02):
        self.cv_t, self.sd_a, self.mu_med, self.cv_mu, self.cv_E = cv_t, sd_a, mu_med, cv_mu, cv_E
        self.s_mu = np.sqrt(np.log(1 + cv_mu**2))

    def to_x(self, u):
        u = np.atleast_2d(u)
        x = np.empty_like(u, dtype=float)
        x[:, 0] = 1 + self.cv_t*u[:, 0]
        x[:, 1] = 1 + self.cv_t*u[:, 1]
        x[:, 2] = self.sd_a*u[:, 2]
        x[:, 3] = self.mu_med*np.exp(self.s_mu*u[:, 3])
        x[:, 4] = 1 + self.cv_E*u[:, 4]
        return x

    def to_u(self, x):
        x = np.atleast_2d(x)
        u = np.empty_like(x, dtype=float)
        u[:, 0] = (x[:, 0] - 1)/self.cv_t
        u[:, 1] = (x[:, 1] - 1)/self.cv_t
        u[:, 2] = x[:, 2]/self.sd_a
        u[:, 3] = np.log(x[:, 3]/self.mu_med)/self.s_mu
        u[:, 4] = (x[:, 4] - 1)/self.cv_E
        return u

    def lhs(self, n, seed=0, span=3.0):
        """Latin hypercube in u-space, truncated at +/- span (training design)"""
        rng = np.random.default_rng(seed)
        d = len(NAMES)
        q = (rng.permuted(np.tile(np.arange(n), (d, 1)), axis=1).T + rng.random((n, d)))/n
        lo, hi = norm.cdf(-span), norm.cdf(span)
        return norm.ppf(lo + q*(hi - lo))


def realize(design, x, name):
    """design: dict with the base case 'c' (p1_study format), the programmed
    hoop end e_d [mm] and drop-off l_h [mm]; x: one row of physical inputs."""
    c0 = design['c']
    k_h, k_hel, da2, mu, E1f = [float(v) for v in x]
    st = []
    for ent in c0['stack']:
        if ent[0] == 'H':
            st.append(('H', ent[1]*k_h))
        elif ent[0] == 'P':
            st.append((ent[0], ent[1], ent[2]*k_hel))
        else:
            st.append((ent[0], ent[1], ent[2]*k_hel))
    fam = list(c0['fam'])
    fam[1] = fam[1] + da2
    mat = dict(c0['mat']); mat['E1'] = mat['E1']*E1f
    c = dict(c0, name=name, stack=st, fam=fam, mat=mat)
    w = PS.wj(c0)
    e_d = design['e_d']
    if e_d > 0:
        try:
            e_c = PS.e_crit(c0, mu)*w
        except ValueError:                     # friction never limits within 3 sqrt(Rt)
            e_c = np.inf
        e = min(e_d, e_c)
    else:
        e = e_d
    c['hoop_end'] = float(e)
    c['hoop_taper'] = float(design['l_h'])
    return c


def make_fmap(design, sc):
    """features seen by the CalculiX model: (u_kh, u_khel, u_da2, e/w, u_E1); the
    friction-limited hoop end e = min(e_d, e_crit(mu)) replaces mu"""
    c0 = design['c']
    w = PS.wj(c0)
    e_d = design['e_d']
    mus = np.geomspace(0.02, 2.0, 400)
    ec = []
    for m in mus:
        try:
            ec.append(PS.e_crit(c0, m)*w)
        except ValueError:
            ec.append(np.inf)
    ec = np.array(ec)
    ec = np.where(np.isfinite(ec), ec, 10*w)

    def f(U):
        U = np.atleast_2d(U)
        mu = sc.to_x(U)[:, 3]
        e = np.full(len(U), e_d) if e_d <= 0 else np.minimum(e_d, np.interp(mu, mus, ec))
        return np.column_stack([U[:, 0], U[:, 1], U[:, 2], e/w, U[:, 4]])
    return f
