"""E4b -- dependence of the reducible fraction on the background-noise level."""
import numpy as np, json, sys
import run_e4_conditioning as E
import ch9_circuits as CC
E.N_CHAINS, E.N_SEG, E.K = 40, 10, 8          # M = 400 states, K = 8 repeats
SIG = {'hh': [0.5, 0.75, 1.0, 1.5, 2.0], 'izh': [0.25, 0.375, 0.5, 0.75, 1.0]}
out = {}
for m in sys.argv[1:] or ['hh', 'izh']:
    c = CC.chain(m, [('gap', 1, 0), ('chem', 2, 1)], **E.CFG[m]['kw']); c.find_equilibrium()
    nv = (4 if m == 'hh' else 2) * c.N
    out[m] = {}
    for sg in SIG[m]:
        X, Ee, H = E.sample_states(c, sg, E.SEED0 + 7)
        R, dN = E.evoked(c, X, Ee, sg, E.SEED0 + 7 + 10 ** 6)
        truth, irr = E.rho_truth(R)
        rV3, _ = E.rho_cv(H[:, 0, 2:3], R, n_boot=50)
        rVh, _ = E.rho_cv(H.reshape(len(H), -1), R, n_boot=50)
        rX, _ = E.rho_cv(X[:, np.r_[0:nv, nv + 2 * c.N + 1]], R, n_boot=50)
        out[m][str(sg)] = dict(rho_truth=truth, rho_V3=rV3, rho_Vhist=rVh, rho_x=rX,
                               R_mean=float(R.mean()), R_sd=float(R.std()))
        print(f"[{m}] sigma={sg:5.3f}  truth={truth:.3f}  V3={rV3:.3f}  Vhist={rVh:.3f}  x={rX:.3f}  R={R.mean():.1f}+-{R.std():.1f}", flush=True)
    json.dump(out, open(f'out/e4b_sigma_sweep_{"_".join(sys.argv[1:]) or "all"}.json', 'w'), indent=1)
