"""
E5c -- identical stimulation delivered to the Chapter 5 network while it sits in
the chimera-like state or in the incoherent state (same network, same gamma).

Branching from checkpoints of the unperturbed trajectory: three arms per trial
(+eps, -eps, and +2eps for the linearity check on a subset), all starting from the
identical state, so the difference is the causal effect of the stimulus alone.
Observable: synaptically filtered mean field z (tau_s = 5 ms), integrated in
continuous time. Response function per unit charge:
    y(t) = [z(t; +eps) - z(t; -eps)] / (2 eps dur)
"""
import numpy as np, json, time, sys
import ch9_network as NW
N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 3
KMAX = int(sys.argv[3]) if len(sys.argv) > 3 else 400
EPS = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0125
DUR, T, TAU_S = 1.0, 80.0, 5.0
b, c, d, Ib, g = 0.2, -50.0, 2.0, 10.0, 0.03
out = {}
t0 = time.time()
for name in ('increasing', 'decreasing'):
    s = np.load(f'out/e5_state_{name}_N{N}_s{seed}.npz')
    a = s['a']; ck_t, ck_v, ck_u = s['ck_t'], s['ck_v'], s['ck_u']
    # checkpoints every 50 ms; use one every 250 ms after a 10 s settling period
    idx = np.where(ck_t >= 10000.0)[0][::5][:KMAX]
    V0, U0 = ck_v[idx], ck_u[idx]
    Z0 = V0.mean(axis=1)                       # z starts at the instantaneous mean field
    K = len(idx)
    mask = np.ones(N)
    amps = np.concatenate([np.full(K, EPS), np.full(K, -EPS), np.full(K, 2 * EPS), np.full(K, -2 * EPS)])
    Vr, Ur, Zr = [np.concatenate([x] * 4) for x in (V0, U0, Z0)]
    Z, nsp = NW.run_branches_z(Vr, Ur, Zr, a, b, c, d, Ib, g, amps, DUR, mask, T, 0.01, 10, TAU_S)
    Zp, Zm, Zp2, Zm2 = Z[:K], Z[K:2 * K], Z[2 * K:3 * K], Z[3 * K:]
    y1 = (Zp - Zm) / (2 * EPS * DUR); y2 = (Zp2 - Zm2) / (4 * EPS * DUR)
    # causal state label: R2 over the second preceding each probe
    tg, R2 = s['tg'], s['R2']
    R2pre = np.array([np.nanmean(R2[(tg >= t - 1000) & (tg < t - 100)]) for t in ck_t[idx]])
    np.savez_compressed(f'out/e5_probe_{name}_N{N}_s{seed}.npz', y1=y1, y2=y2, R2pre=R2pre,
                        t=np.arange(Z.shape[1]) * 0.1, t_probe=ck_t[idx])
    tt = np.arange(Z.shape[1]) * 0.1; w = tt <= 60
    lin = np.linalg.norm(y1[:, w] - y2[:, w], axis=1) / np.linalg.norm(y1[:, w], axis=1)
    out[name] = dict(K=int(K), R2pre_median=float(np.nanmedian(R2pre)),
                     linearity_median=float(np.median(lin)), linearity_p90=float(np.percentile(lin, 90)))
    print(f"{name}: K={K}  median causal R2={np.nanmedian(R2pre):.3f}  linearity eps vs 2eps: "
          f"median {np.median(lin):.3f}, 90th pct {np.percentile(lin,90):.3f}  ({time.time()-t0:.0f}s)", flush=True)
json.dump(out, open(f'out/e5_probe_N{N}_s{seed}.json', 'w'), indent=1)
