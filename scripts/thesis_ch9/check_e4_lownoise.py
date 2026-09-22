"""Evoked spike-count distribution of node 3 at weak and at working noise (HH gap->chem motif),
M = 100 onset states x K = 8 repeats; supports the weak-noise statement of Sec. 9.4."""
import numpy as np, json
import run_e4_conditioning as E
import ch9_circuits as CC
E.N_CHAINS, E.N_SEG, E.K = 20, 5, 8
c = CC.chain('hh', [('gap', 1, 0), ('chem', 2, 1)]); c.find_equilibrium()
out = {}
for sg in (0.5, 1.0):
    X, Eta, H = E.sample_states(c, sg, 555)
    R, dN = E.evoked(c, X, Eta, sg, 777)
    vals, cnt = np.unique(dN, return_counts=True)
    out[sg] = dict(dist={int(v): float(c_ / dN.size) for v, c_ in zip(vals, cnt)}, R_sd=float(R.std()),
                   meanR_by_dN={int(v): float(R[dN == v].mean()) for v in vals})
    print(sg, out[sg])
json.dump(out, open('out/e4_hh_lownoise_check.json', 'w'), indent=1)
