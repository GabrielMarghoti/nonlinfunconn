"""Scaling of the eps-vs-2eps deviation with eps in both network states (N=120, seed 3)."""
import numpy as np, json, time
import ch9_network as NW
b, c, d, Ib, g, DUR, T, TAU_S = 0.2, -50.0, 2.0, 10.0, 0.03, 1.0, 80.0, 5.0
N, seed, K = 120, 3, 100
out = {}
for name in ('increasing', 'decreasing'):
    s = np.load(f'out/e5_state_{name}_N{N}_s{seed}.npz'); a = s['a']
    idx = np.where(s['ck_t'] >= 10000.0)[0][::5][:K]
    V0, U0 = s['ck_v'][idx], s['ck_u'][idx]; Z0 = V0.mean(1)
    epss = [0.00625, 0.0125, 0.025, 0.05, 0.1]
    amps = np.concatenate([np.full(K, e * sg) for e in epss for sg in (1, -1)])
    n = 2 * len(epss)
    Z, _ = NW.run_branches_z(*[np.concatenate([x] * n) for x in (V0, U0, Z0)], a, b, c, d, Ib, g, amps,
                             DUR, np.ones(N), T, 0.01, 10, TAU_S)
    Y = [(Z[(2*i)*K:(2*i+1)*K] - Z[(2*i+1)*K:(2*i+2)*K]) / (2 * e * DUR) for i, e in enumerate(epss)]
    tt = np.arange(Z.shape[1]) * 0.1
    res = {}
    for W in (30.0, 60.0):
        w = tt <= W; r = []
        for i in range(len(epss) - 1):
            dev = np.linalg.norm(Y[i][:, w] - Y[i + 1][:, w], axis=1) / np.linalg.norm(Y[i][:, w], axis=1)
            r.append([float(np.median(dev)), float(np.percentile(dev, 90)), float(np.mean(dev > 0.2))])
        res[f'W{int(W)}'] = dict(zip([f'{e}' for e in epss[:-1]], r))
    out[name] = res
    print(name, json.dumps(res), flush=True)
json.dump(out, open('out/e5_linearity_scaling.json', 'w'), indent=1)
