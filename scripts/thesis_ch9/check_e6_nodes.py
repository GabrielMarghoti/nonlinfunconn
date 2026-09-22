"""Sharp validation of the covariance estimator on nodes 1 and 2, whose responses to a probe into
node 1 are nearly deterministic (they rarely spike), so the pump-probe average is precise."""
import numpy as np, json, sys
import ch9_circuits as CC, ch9_modelfree as MF
out = {}
for MODEL in ('hh', 'izh'):
    M = 3000
    kw = {} if MODEL == 'hh' else dict(g_gap=0.5, g_chem=0.075); sig = 1.0 if MODEL == 'hh' else 0.5
    c = CC.chain(MODEL, [('gap', 1, 0), ('chem', 2, 1)], **kw); xeq = c.find_equilibrium()
    SIGP2, DB, AMP = 0.5, 0.1, 0.2; VARXI = SIGP2 / DB
    rng = np.random.default_rng(4242 + (MODEL == 'izh'))
    nchain = 100; nseg = M // nchain
    x = np.tile(xeq, (nchain, 1)); eta = sig * rng.standard_normal((nchain, 3))
    r = MF.run_xi(c, x, rng.normal(0, np.sqrt(VARXI), (nchain, 5001)), np.zeros((nchain, 1, 4)), rng.integers(1, 2**31, nchain), 500.0, eta, sigma=sig)
    x, eta = r['x_end'], r['eta_end']; X, E = [], []
    for s in range(nseg):
        r = MF.run_xi(c, x, rng.normal(0, np.sqrt(VARXI), (nchain, 1001)), np.zeros((nchain, 1, 4)), rng.integers(1, 2**31, nchain), 100.0, eta, sigma=sig)
        x, eta = r['x_end'], r['eta_end']; X.append(x.copy()); E.append(eta.copy())
    X = np.concatenate(X); E = np.concatenate(E)
    xi = rng.normal(0, np.sqrt(VARXI), (M, 201)); seeds = rng.integers(1, 2**31, M)
    pul = np.zeros((M, 1, 4)); pul[:, 0] = [0, 0.0, 0.095, AMP]
    Vp = MF.run_xi(c, X, xi, pul, seeds, 20.0, E, sigma=sig)['V']
    pul[:, 0, 3] = -AMP
    Vm = MF.run_xi(c, X, xi, pul, seeds, 20.0, E, sigma=sig)['V']
    G = (Vp - Vm) / (2 * AMP * DB)                                   # (M, 201, 3), tau = 0..20 ms
    d = np.load(f'out/e6_{MODEL}.npz'); tau = d['tau']; sel = (tau >= 0) & (tau <= 20.0 + 1e-9)
    res = {}
    for k in (0, 1):
        g = G[:, :, k].mean(0); gse = G[:, :, k].std(0, ddof=1) / np.sqrt(M)
        K = d['K_all'][sel, k]; Kse = d['K_se'][sel, k]
        w = slice(1, 201)
        rel = float(np.linalg.norm(g[w] - K[w]) / np.linalg.norm(g[w]))
        z = (g[w] - K[w]) / np.sqrt(gse[w] ** 2 + Kse[w] ** 2)
        res[f'node{k+1}'] = dict(rel_L2=rel, z_rms=float(np.sqrt(np.mean(z ** 2))), peak_pp=float(g.max()), peak_cov=float(K.max()),
                                 pp_rel_se=float(np.linalg.norm(gse[w]) / np.linalg.norm(g[w])), cov_rel_se=float(np.linalg.norm(Kse[w]) / np.linalg.norm(g[w])))
    out[MODEL] = res; print(MODEL, json.dumps(res), flush=True)
    np.savez_compressed(f'out/e6_nodes_{MODEL}.npz', G_mean=G.mean(0), G_se=G.std(0, ddof=1) / np.sqrt(M))
json.dump(out, open('out/e6_nodes_check.json', 'w'), indent=1)
