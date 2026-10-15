"""Gaussian-process surrogate of the burst scale p0(u) and of the transition
share of the risk integral, with strength-integrated active learning, and
the reliability measures built on it.

Limit state (strength integrated analytically):
    P_f(p | u) = 1 - exp(-(p/p0(u))^m)
    P_f(p)     = E_u[ P_f(p | u) ]
Active learning adds the training point that maximises
    sigma_GP(u) * f(u),   f(u) = d P_f(p_req | u) / d ln p0
i.e. where the surrogate is uncertain *and* the uncertainty matters for the
failure probability at the requirement p_req.
"""
import numpy as np
from scipy.stats import norm
from scipy.special import gamma
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel


def _gp(d):
    k = ConstantKernel(1.0, (1e-3, 1e3))*Matern(length_scale=np.ones(d)*2.0,
                                                 length_scale_bounds=(0.05, 1e3), nu=2.5) \
        + WhiteKernel(1e-6, (1e-10, 1e-2))
    return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=4, random_state=0)


class Surrogate:
    """GP on features f(u); fmap maps standard-normal inputs to the physical
    features seen by the model (e.g. mu -> friction-limited hoop end), which
    removes the kink of e = min(e_d, e_crit(mu)) from the regression."""

    def __init__(self, U, lnp0, share, fmap=None):
        self.fmap = fmap or (lambda U: np.atleast_2d(U))
        self.U = np.asarray(U, float)
        self.y = np.asarray(lnp0, float)
        self.sh = np.clip(np.asarray(share, float), 1e-6, 1 - 1e-6)
        self.fit()

    def fit(self):
        F = self.fmap(self.U)
        self.Fmu, self.Fsd = F.mean(0), np.where(F.std(0) > 1e-12, F.std(0), 1.0)
        Z = (F - self.Fmu)/self.Fsd
        d = Z.shape[1]
        self.g = _gp(d).fit(Z, self.y)
        self.gs = _gp(d).fit(Z, np.log(self.sh/(1 - self.sh)))

    def z(self, U):
        return (self.fmap(np.atleast_2d(U)) - self.Fmu)/self.Fsd

    def add(self, u, lnp0, share):
        self.U = np.vstack([self.U, u])
        self.y = np.append(self.y, lnp0)
        self.sh = np.append(self.sh, np.clip(share, 1e-6, 1 - 1e-6))
        self.fit()

    def predict(self, U, std=False):
        return self.g.predict(self.z(U), return_std=std)

    def share(self, U):
        z = self.gs.predict(self.z(U))
        return 1/(1 + np.exp(-z))

    def loo(self):
        """leave-one-out error of ln p0 (in %, i.e. relative error of p0)"""
        err = []
        for i in range(len(self.y)):
            m = np.ones(len(self.y), bool); m[i] = False
            Z = self.z(self.U)
            g = _gp(Z.shape[1]).fit(Z[m], self.y[m])
            err.append(np.exp(g.predict(Z[i:i + 1])[0] - self.y[i]) - 1)
        return np.array(err)


def next_point(S, m, p_req, n_cand=20000, seed=1):
    rng = np.random.default_rng(seed)
    C = rng.standard_normal((n_cand, S.U.shape[1]))
    mu, sd = S.predict(C, std=True)
    a = (p_req/np.exp(mu))**m
    f = m*a*np.exp(-a)                         # |dPf/dlnp0|
    i = int(np.argmax(sd*f))
    return C[i], float(sd[i]*f[i])


def pf_curve(S, m, p, n=200000, seed=2, gp_samples=0):
    """P_f(p) = E_u[1 - exp(-(p/p0(u))^m)] by Monte Carlo on the surrogate"""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n, S.U.shape[1]))
    mu = S.predict(U)
    p = np.atleast_1d(p)
    pf = np.array([np.mean(-np.expm1(-(pp/np.exp(mu))**m)) for pp in p])
    out = dict(p=p, pf=pf)
    if gp_samples:
        Ys = S.g.sample_y(S.z(U[:4000]), gp_samples, random_state=3)
        out['band'] = np.array([[np.mean(-np.expm1(-(pp/np.exp(Ys[:, j]))**m)) for pp in p]
                                for j in range(gp_samples)])
    return out


def beta(pf):
    return -norm.ppf(np.clip(pf, 1e-300, 1))


def form(S, m, p_req, d):
    """FORM in (u, u_W): burst = p0(u) * W, W ~ Weibull(1, m) standard.
    g(u, u_w) = ln p0(u) + ln W(u_w) - ln p_req"""
    def lnW(uw):
        return np.log(-np.log(norm.sf(uw)))/m

    def g(v):
        return S.predict(v[:d])[0] + lnW(v[d]) - np.log(p_req)

    def grad(v, h=1e-3):
        g0 = g(v)
        return np.array([(g(v + h*np.eye(d + 1)[i]) - g0)/h for i in range(d + 1)]), g0

    v = np.zeros(d + 1)
    for _ in range(60):
        gr, g0 = grad(v)
        vn = (gr @ v - g0)/(gr @ gr)*gr
        if np.linalg.norm(vn - v) < 1e-6:
            v = vn
            break
        v = vn
    b = float(np.linalg.norm(v))
    alpha = -grad(v)[0]/np.linalg.norm(grad(v)[0])
    return dict(beta=b if g(np.zeros(d + 1)) > 0 else -b, u_star=v, alpha=alpha,
                pf=float(norm.cdf(-b)))


def sobol(S, n=20000, seed=4):
    """first-order and total Sobol indices of ln p0 (Saltelli/Jansen estimators)"""
    rng = np.random.default_rng(seed)
    d = S.U.shape[1]
    A, B = rng.standard_normal((n, d)), rng.standard_normal((n, d))
    fA, fB = S.predict(A), S.predict(B)
    V = np.var(np.concatenate([fA, fB]))
    S1, ST = [], []
    for i in range(d):
        ABi = A.copy(); ABi[:, i] = B[:, i]
        fAB = S.predict(ABi)
        S1.append(np.mean(fB*(fAB - fA))/V)
        ST.append(0.5*np.mean((fA - fAB)**2)/V)
    return np.array(S1), np.array(ST), V


def weibull_cov(m):
    return np.sqrt(gamma(1 + 2/m) - gamma(1 + 1/m)**2)/gamma(1 + 1/m)
