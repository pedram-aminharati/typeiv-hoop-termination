"""Weakest-link (Weibull) burst model evaluated on the CalculiX shell stresses.

For a linear analysis at reference pressure p_ref, the fibre stress in every
ply integration point scales with p: sigma_1 = p * s_hat.  With a two-parameter
Weibull strength of the fibre direction (modulus m, scale sigma_0 for the
reference volume V_0), the probability that the vessel has failed by fibre
rupture at pressure p is

    P_f(p) = 1 - exp( - sum_i V_i/V_0 <p s_hat_i / sigma_0>^m ) ,

so the burst pressure is itself Weibull distributed with the same modulus m
and the scale

    p_0 = sigma_0 * ( sum_i V_i/V_0 <s_hat_i>^m )^(-1/m) .

The share of the risk integral contributed by each zone is the probability
that burst originates in that zone (competing risks).
"""
import os
import numpy as np
from scipy.special import gamma

ZONES = ('barrel', 'junction', 'dome', 'boss')
ZC = {z: i for i, z in enumerate(ZONES)}


def extract(path, meta, c, read_dat):
    """per-ply-point fibre stress per unit pressure and ply volumes (full ring)."""
    S, rf, U = read_dat(path)
    sig, vol, zone, row, fam, s_coord, r_coord, side = [], [], [], [], [], [], [], []
    p = c['p']
    for e, er, ej in meta['elems']:
        if ej != 0:
            continue                           # axisymmetric field: one meridian
        info = meta['rowinfo'][er]
        n = len(info['plies'])
        s = S[e].reshape(n, 8, 6)
        for k, (a, t, nm) in enumerate(info['plies']):
            sig.append(s[k, :, 0]/p)
            vol.append(2*np.pi*info['rc']*info['seglen']*t)
            zone.append(ZC[info['zone']])
            row.append(er)
            fam.append(nm[0] if nm[0] in 'HP' else nm[:2])
            s_coord.append(info['s'])
            r_coord.append(info['rc'])
            side.append(1 if info['zc'] > 0 else -1)
    return dict(sig=np.array(sig), vol=np.array(vol), zone=np.array(zone), row=np.array(row),
                fam=np.array(fam), s=np.array(s_coord), r=np.array(r_coord), side=np.array(side),
                rf=np.linalg.norm(rf) if rf else np.nan)


def save(d, fn):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    np.savez_compressed(fn, **d)


def load(fn):
    z = np.load(fn, allow_pickle=False)
    return {k: z[k] for k in z.files}


class Weibull:
    """fibre-direction tensile strength: mean X_t on the reference volume V_0."""

    def __init__(self, Xt, m=25.0, V0=2000.0, s0=None):
        self.m, self.V0 = m, V0
        self.s0 = Xt/gamma(1 + 1/m) if s0 is None else s0
        self.Xt = self.s0*gamma(1 + 1/m)

    @classmethod
    def anchored(cls, d_ref, p_req, pf_t, m=25.0, V0=2000.0, zone='barrel'):
        """scale sigma_0 such that the given zone of a reference vessel alone has
        P_f(p_req) = pf_t: the netting strength is then 'correct' for an ideal,
        cylinder-governed vessel at the target reliability"""
        sp = np.clip(d_ref['sig'], 0, None)
        Bz = np.sum(np.where(d_ref['zone'] == ZC[zone],
                             d_ref['vol']/V0*np.mean(sp**m, axis=1), 0.0))
        s0 = p_req*(Bz/(-np.log1p(-pf_t)))**(1/m)
        return cls(None, m, V0, s0=s0)

    def risk_terms(self, d, exclude_boss=True):
        """risk-integral contribution of every ply point per unit pressure^m"""
        sp = np.clip(d['sig'], 0, None)
        term = d['vol']/self.V0*np.mean((sp/self.s0)**self.m, axis=1)
        if exclude_boss:
            term = np.where(d['zone'] == ZC['boss'], 0.0, term)
        return term

    def burst(self, d, exclude_boss=True):
        term = self.risk_terms(d, exclude_boss)
        B = term.sum()
        p0 = B**(-1/self.m)
        shares = {z: float(term[d['zone'] == i].sum()/B) for z, i in ZC.items()}
        m = self.m
        return dict(p0=float(p0), p50=float(p0*np.log(2)**(1/m)),
                    mean=float(p0*gamma(1 + 1/m)),
                    cov=float(np.sqrt(gamma(1 + 2/m) - gamma(1 + 1/m)**2)/gamma(1 + 1/m)),
                    shares=shares, B=float(B))

    def pf(self, p, p0):
        return 1 - np.exp(-(np.asarray(p)/p0)**self.m)

    def first_failure(self, d, exclude_boss=True):
        """deterministic first-fibre-failure pressure with the mean coupon strength"""
        sp = np.clip(d['sig'], 0, None).max(axis=1)
        if exclude_boss:
            sp = np.where(d['zone'] == ZC['boss'], 0.0, sp)
        i = int(sp.argmax())
        return dict(p_ff=float(self.Xt/sp[i]), zone=ZONES[int(d['zone'][i])], s=float(d['s'][i]))
