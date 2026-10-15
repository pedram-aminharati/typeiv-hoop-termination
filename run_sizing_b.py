import json, sizing as Z
out = {}
for vn in ('H2', 'H4', 'H6', 'H8'):
    r = Z.size(vn); out[vn] = r
    json.dump(out, open('sizing_b.json', 'w'), indent=1, default=float)
    print(vn, round(r['k'], 3), round(r['RD'], 3), round(r['hoop_frac'], 3), round(r['dome_peak_s'], 2),
          r['dome_peak_ply'], round(r['pf_ref'], 1), round(r['pf_mid'], 1), flush=True)
