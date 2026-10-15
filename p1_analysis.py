"""Paper 1: analysis of the full shell matrix (results_p1full)."""
import glob
import json
import numpy as np
import p1_metrics as M
import p1_study as PS

D = 'results_p1full'


def load():
    rs = []
    for f in glob.glob(f'{D}/*.json'):
        r = json.load(open(f))
        if r.get('ok'):
            m = M.metrics(r)
            rs.append(dict(name=r['name'], meta=r['meta'], RT=m['rho']['RT'], s_crit=m['rho']['s'],
                           zone=m['rho']['zone'], ply=m['rho']['ply'], pf=m['pf'], pf_mid=m['pf_mid'],
                           mass=m['mass'], wj=m['wj'], RT_tri=m['tri']['RT'], RT_f=m['f']['RT']))
    return rs


def vessel_info(vn):
    b = PS.base(vn)
    w = PS.wj(b)
    t = sum(e[-1] for e in b['stack'])
    info = dict(vessel=vn, R=b['ri'], t=t, R_t=b['ri']/t, wj=w, hd_R=b['hd']/b['ri'],
                p=b['p'], ec={mu: PS.e_crit(b, mu) for mu in (0.1, 0.2, 0.3)})
    if vn in PS.VESSELS:
        info.update(rb_R=b['rb_R'], t_h=b['t_h'], t_hel=b['t_hel'], a1=b['fam'][0], a2=b['fam'][1],
                    hoop_frac=b['t_h']/t)
    return info


def summary(rs):
    out = {}
    vessels = sorted({r['meta']['vessel'] for r in rs})
    for vn in vessels:
        R = [r for r in rs if r['meta']['vessel'] == vn]
        hoop = [r for r in R if r['meta']['study'] == 'hoop']
        ref = next((r for r in hoop if r['meta']['e_w'] == 0 and r['meta']['l_w'] == 0.5), None)
        if ref is None:
            continue
        v = vessel_info(vn)
        v['ref'] = dict(pf=ref['pf'], RT=ref['RT'], mass=ref['mass'], pf_mid=ref['pf_mid'], zone=ref['zone'])
        v['grid'] = [dict(e=r['meta']['e_w'], l=r['meta']['l_w'], RT=r['RT'], pf=r['pf'], eta=r['pf']/r['pf_mid'],
                          dpf=r['pf']/ref['pf'] - 1, dm=r['mass']/ref['mass'] - 1, zone=r['zone'])
                     for r in hoop]
        v['stack'] = [dict(order=r['meta']['order'], RT=r['RT'], pf=r['pf'], dpf=r['pf']/ref['pf'] - 1,
                           eta=r['pf']/r['pf_mid'])
                      for r in R if r['meta']['study'] == 'stack']
        v['hybrid'] = [dict(rtp=r['meta']['rtp'], sp=r['meta']['sp'], RT=r['RT'], pf=r['pf'], eta=r['pf']/r['pf_mid'],
                            dpf=r['pf']/ref['pf'] - 1, dm=r['mass']/ref['mass'] - 1)
                       for r in R if r['meta']['study'] == 'hybrid']
        v['combo'] = [dict(mu=r['meta']['mu'], order=r['meta']['order'], e=r['meta']['e_w'], RT=r['RT'], eta=r['pf']/r['pf_mid'],
                           pf=r['pf'], dpf=r['pf']/ref['pf'] - 1, dm=r['mass']/ref['mass'] - 1)
                      for r in R if r['meta']['study'] == 'combo']
        # best windable hoop-only design per mu (l_h free)
        v['best'] = {}
        for mu, ec in v['ec'].items():
            cand = [g for g in v['grid'] if g['e'] <= ec + 1e-9]
            cand += [dict(e=c['e'], l=0.5, RT=c['RT'], pf=c['pf'], eta=c['eta'], dpf=c['dpf'], dm=c['dm'], zone='',
                          order=c['order']) for c in v['combo'] if c['mu'] == mu]
            if cand:
                v['best'][mu] = max(cand, key=lambda g: g['pf'])
        out[vn] = v
    return out


if __name__ == '__main__':
    rs = load()
    print(len(rs), 'results')
    S = summary(rs)
    json.dump(S, open('p1_summary.json', 'w'), indent=1, default=float)
    for vn, v in S.items():
        print(vn, 'R/t %.1f hd/R %.1f p %g' % (v['R_t'], v['hd_R'], v['p']), 'ref pf %.1f RT %.2f pf_mid %.1f' % (v['ref']['pf'], v['ref']['RT'], v['ref']['pf_mid']),
              'n grid', len(v['grid']), 'best', {mu: (round(b['e'], 3), b['l'], round(100*b['dpf'], 1)) for mu, b in v['best'].items()})
