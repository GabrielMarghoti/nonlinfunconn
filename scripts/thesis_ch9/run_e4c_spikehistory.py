"""
E4c -- can an observable of spike history replace the hidden variable?
Re-generates the onset states of E4 (same seeds, so the states and responses are
identical to out/e4_{model}.npz), records the time since the last spike of node 3
at each onset, and adds it to the conditioning sets.
"""
import numpy as np, json, sys
import ch9_circuits as CC
import run_e4_conditioning as E

out = {}
for m in (sys.argv[1:] or ['hh', 'izh']):
    cfg = E.CFG[m]
    c = CC.chain(m, [('gap', 1, 0), ('chem', 2, 1)], **cfg['kw']); c.find_equilibrium()
    sig = cfg['sigma']; rs = E.SEED0
    x = np.tile(c.xeq, (E.N_CHAINS, 1))
    r = c.run(x, np.zeros((E.N_CHAINS, 1, 4)), rs + np.arange(E.N_CHAINS), E.BURN, E.DT, sigma=sig, save_every=10)
    x, eta = r['x_end'], r['eta_end']
    last = np.full(E.N_CHAINS, -1e4)
    def upd(last, spk, T):
        for b in range(E.N_CHAINS):
            s = spk[b, 2]; s = s[~np.isnan(s)]
            if len(s): last[b] = s[-1] - T
            else: last[b] -= T
        return last
    last = upd(last, r['spikes'], E.BURN)
    X, TSS = [], []
    for s in range(E.N_SEG):
        seeds = rs + 1000 * (s + 1) + np.arange(E.N_CHAINS)
        r = c.run(x, np.zeros((E.N_CHAINS, 1, 4)), seeds, E.SEG_T, E.DT, sigma=sig, save_every=10, eta0=eta)
        x, eta = r['x_end'], r['eta_end']
        last = upd(last, r['spikes'], E.SEG_T)
        X.append(x.copy()); TSS.append(-last.copy())
    X = np.concatenate(X); TSS = np.concatenate(TSS)
    d = np.load(f'out/e4_{m}.npz')
    assert np.allclose(X, d['X']), 'states differ from E4'
    R, H = d['R'], d['H']
    N = 3; nv = (4 if m == 'hh' else 2) * N
    hid = X[:, 3 * N + 2] if m == 'hh' else X[:, N + 2]
    from scipy.stats import spearmanr
    sets = {'tss3': TSS[:, None], 'V3 + tss3': np.c_[H[:, 0, 2], TSS],
            'V history + tss3': np.c_[H.reshape(len(H), -1), TSS],
            'V3 + hidden3': np.c_[H[:, 0, 2], hid]}
    res = dict(spearman_tss_hidden=float(spearmanr(TSS, hid)[0]), rho={})
    for k, S in sets.items():
        r2, ci = E.rho_cv(S, R); res['rho'][k] = dict(r2=r2, ci95=ci)
        print(f'[{m}] rho({k}) = {r2:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]', flush=True)
    out[m] = res
json.dump(out, open('out/e4c_spikehistory.json', 'w'), indent=1)
