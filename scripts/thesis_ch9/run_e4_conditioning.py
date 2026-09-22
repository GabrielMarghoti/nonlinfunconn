"""
E4 -- trial-by-trial variability of the evoked response and how much of it is
explained by the state at stimulus onset, as a function of what is observed.

Protocol (per model, Fig. 7.12 motif gap 1->2, chem 2->3):
  * the circuit runs under OU background noise; M pre-stimulus states are sampled
    (full neural state x, noise state eta, and the recent voltage history)
  * from every sampled state, K repeats of the identical Chapter 7 stimulus are
    delivered with independent post-stimulus noise (OU continued from eta)
  * response R = causal effect of the stimulus on node 3: the integral over 40 ms
    of V_3(stimulated) - V_3(unstimulated), both arms sharing the state and every
    random number (CRN), so ongoing activity cancels and only what the stimulus
    changed remains
Ground truth: with K repeats per state, the fraction of Var(R) fixed by the full
Markov state (x, eta) is  rho_full = [Var_m(mean_k R) - E_m(var_k R)/K] / Var(R).
Partial observations: rho(s) = out-of-sample R^2 of a gradient-boosted regressor
predicting R from s, cross-validated with folds grouped by state (no leakage).
"""
import numpy as np, json, time, os, sys
import ch9_circuits as CC
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

OUT = 'out'; os.makedirs(OUT, exist_ok=True)
DT = 0.01
CFG = {
    'hh':  dict(kw={}, sigma=1.0),
    'izh': dict(kw=dict(g_gap=0.5, g_chem=0.075), sigma=0.5),
}
N_CHAINS, N_SEG, SEG_T, BURN = 50, 20, 100.0, 300.0     # M = 1000 states
K = 12
T_RESP, W_RESP = 45.0, 40.0
LAGS = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]                  # ms, voltage history
SEED0 = 20260921

def sample_states(c, sigma, rng_seed):
    x = np.tile(c.xeq, (N_CHAINS, 1)); eta = None
    seeds = rng_seed + np.arange(N_CHAINS)
    r = c.run(x, np.zeros((N_CHAINS, 1, 4)), seeds, BURN, DT, sigma=sigma, save_every=10)
    x, eta = r['x_end'], r['eta_end']
    X, E, H = [], [], []
    se = 10; lag_idx = [int(round(l / (DT * se))) for l in LAGS]
    for s in range(N_SEG):
        seeds = rng_seed + 1000 * (s + 1) + np.arange(N_CHAINS)
        r = c.run(x, np.zeros((N_CHAINS, 1, 4)), seeds, SEG_T, DT, sigma=sigma, save_every=se, eta0=eta)
        x, eta = r['x_end'], r['eta_end']
        V = r['V']                                           # (chains, nt, N)
        H.append(np.stack([V[:, -1 - li, :] for li in lag_idx], axis=1))   # (chains, nlag, N)
        X.append(x.copy()); E.append(eta.copy())
    return np.concatenate(X), np.concatenate(E), np.concatenate(H)

def evoked(c, X, E, sigma, rng_seed):
    M = X.shape[0]
    x0 = np.repeat(X, K, axis=0); e0 = np.repeat(E, K, axis=0)
    seeds = rng_seed + np.arange(M * K)
    pul = np.tile(CC.pulse_table([(0, 0.0, 3.0, 10.0)])[None], (M * K, 1, 1))
    ctl = np.zeros_like(pul)
    rs = c.run(x0, pul, seeds, T_RESP, DT, sigma=sigma, save_every=5, eta0=e0, max_spikes=16)
    rc = c.run(x0, ctl, seeds, T_RESP, DT, sigma=sigma, save_every=5, eta0=e0, max_spikes=16)
    t = np.arange(rs['V'].shape[1]) * DT * 5; w = t < W_RESP
    dV3 = rs['V'][:, w, 2] - rc['V'][:, w, 2]
    R = np.sum(dV3, axis=1) * DT * 5
    cnt = lambda r: np.sum((r['spikes'][:, 2, :] >= 0) & (r['spikes'][:, 2, :] < W_RESP), axis=1)
    dN = cnt(rs) - cnt(rc)
    return R.reshape(M, K), dN.reshape(M, K)

def rho_truth(R):
    M, K_ = R.shape
    var_tot = R.var()
    between = R.mean(1).var() - R.var(1, ddof=1).mean() / K_
    return float(between / var_tot), float(R.var(1, ddof=1).mean() / var_tot)

def rho_cv(S, R, n_boot=200, seed=0):
    M, K_ = R.shape
    y = R.reshape(-1); Xf = np.repeat(S, K_, axis=0); g = np.repeat(np.arange(M), K_)
    pred = np.empty_like(y)
    for tr, te in GroupKFold(5).split(Xf, y, g):
        mdl = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=15,
                                            min_samples_leaf=40, early_stopping=True,
                                            validation_fraction=0.15, n_iter_no_change=20,
                                            random_state=0)
        mdl.fit(Xf[tr], y[tr]); pred[te] = mdl.predict(Xf[te])
    r2 = 1 - np.mean((y - pred) ** 2) / np.var(y)
    rng = np.random.default_rng(seed); bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, M, M); ii = (idx[:, None] * K_ + np.arange(K_)).reshape(-1)
        bs.append(1 - np.mean((y[ii] - pred[ii]) ** 2) / np.var(y[ii]))
    return float(r2), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

if __name__ == '__main__':
    models = sys.argv[1:] or ['hh', 'izh']
    allres = {}
    for m in models:
        t0 = time.time()
        cfg = CFG[m]
        c = CC.chain(m, [('gap', 1, 0), ('chem', 2, 1)], **cfg['kw']); c.find_equilibrium()
        X, E, H = sample_states(c, cfg['sigma'], SEED0)
        R, NSP = evoked(c, X, E, cfg['sigma'], SEED0 + 10 ** 6)
        N = c.N; nv = (4 if m == 'hh' else 2) * N
        sets = {
            'V1(t0)':              H[:, 0, 0:1],
            'V3(t0)':              H[:, 0, 2:3],
            'V1..V3(t0)':          H[:, 0, :],
            'V history (6 lags)':  H.reshape(len(H), -1),
            'neural state x':      X[:, np.r_[0:nv, nv + 2 * N + 1]],     # all node vars + the one active synapse s_32
            'x + noise state':     np.hstack([X[:, np.r_[0:nv, nv + 2 * N + 1]], E]),
        }
        truth, irr = rho_truth(R)
        res = dict(M=int(R.shape[0]), K=K, sigma=cfg['sigma'], R_mean=float(R.mean()), R_sd=float(R.std()),
                   rho_truth_full=truth, irreducible_truth=irr,
                   evoked_spikes_node3_mean=float(NSP.mean()),
                   evoked_spike_count_distribution={int(k): float(np.mean(NSP == k)) for k in np.unique(NSP)})
        res['rho'] = {}
        for name, S in sets.items():
            r2, ci = rho_cv(S, R)
            res['rho'][name] = dict(r2=r2, ci95=ci)
            print(f"  [{m}] rho({name:20s}) = {r2:6.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]")
        print(f"  [{m}] ground-truth rho(x, eta) = {truth:.3f}   irreducible = {irr:.3f}   "
              f"R = {R.mean():.1f} +- {R.std():.1f} mV ms   ({time.time()-t0:.0f}s)")
        allres[m] = res
        np.savez_compressed(f'{OUT}/e4_{m}.npz', X=X, E=E, H=H, R=R, NSP=NSP, lags=np.array(LAGS))
    json.dump(allres, open(f'{OUT}/e4_conditioning.json', 'w'), indent=1)
