import json, sys, sizing as Z
vn = sys.argv[1]
r = Z.size(vn)
json.dump({vn: r}, open(f'sizing_{vn}.json', 'w'), indent=1, default=float)
print(vn, round(r['k'], 3), round(r['RD'], 3), round(r['hoop_frac'], 3), round(r['pf_ref'], 1), round(r['pf_mid'], 1), flush=True)
