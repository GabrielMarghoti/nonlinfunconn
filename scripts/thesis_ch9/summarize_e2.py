import numpy as np, json
import ch9_analysis as AN
res = {}
for name in ['gap-gap', 'gap-chem', 'chem-chem']:
    d = np.load(f'out/e2_hh_{name}.npz'); dtau = d['tau'][1]-d['tau'][0]
    f, tot = AN.telescoping_factors(d['G'], d['GS'], dtau)
    sh = AN.covariance_shares(f, tot)
    rng = {k: float(np.ptp(v)) for k, v in f.items()}
    res[name] = dict(shares=sh, node_share=sum(v for k,v in sh.items() if k[0]=='N'),
                     link_share=sum(v for k,v in sh.items() if k[0]=='T'),
                     range_log=rng, range_total=float(np.ptp(tot)),
                     max_gain_rel_rest=float(np.exp(tot.max()-tot[-1])), min_gain_rel_rest=float(np.exp(tot.min()-tot[-1])))
    print(f"{name:10s} node share {res[name]['node_share']:6.3f}  link share {res[name]['link_share']:6.3f}   "
          f"range(log total)={np.ptp(tot):5.2f}   " + "  ".join(f"{k}:{v:+.3f}" for k, v in sh.items()))
    print("            ptp log-factor:", {k: round(v,2) for k, v in rng.items()})
json.dump(res, open('out/e2_hh_decomposition.json','w'), indent=1)
