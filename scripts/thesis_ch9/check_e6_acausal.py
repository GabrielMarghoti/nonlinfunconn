"""Null check for E6: acausal part of the Stein kernel, and the same statistic with the
probe noise paired to a different trial (true null), for several seeds."""
import numpy as np, json, sys
import ch9_circuits as CC, ch9_modelfree as MF
MODEL = sys.argv[1]
cfgk = {} if MODEL == 'hh' else dict(g_gap=0.5, g_chem=0.075); sig = 1.0 if MODEL == 'hh' else 0.5
c = CC.chain(MODEL, [('gap', 1, 0), ('chem', 2, 1)], **cfgk); xeq = c.find_equilibrium()
SIGP2, DB, NTR, CH = 0.5, 0.1, 64, 5000.0
nb = int(CH / DB); LMIN, LMAX = -100, 600; lags = np.arange(LMIN, LMAX + 1)
nfft = 1 << int(np.ceil(np.log2(nb + 710)))
valid = np.zeros(nb, bool); valid[100:nb - 601] = True
def xcorr(w, V):
    C = np.fft.irfft(np.conj(np.fft.rfft(w, nfft, axis=1)) * np.fft.rfft(V, nfft, axis=1), nfft, axis=1)
    return np.concatenate([C[:, nfft + LMIN:], C[:, :LMAX + 1]], axis=1)
out = []
for seed in range(3):
    rng = np.random.default_rng(1000 + seed)
    x = np.tile(xeq, (NTR, 1)); eta = sig * rng.standard_normal((NTR, 3))
    r = MF.run_xi(c, x, rng.normal(0, np.sqrt(SIGP2 / DB), (NTR, 5001)), np.zeros((NTR, 1, 4)), rng.integers(1, 2**31, NTR), 500.0, eta, sigma=sig)
    x, eta = r['x_end'], r['eta_end']
    per, null = [], []
    for ch in range(2):
        xi = rng.normal(0, np.sqrt(SIGP2 / DB), (NTR, nb + 1))
        r = MF.run_xi(c, x, xi, np.zeros((NTR, 1, 4)), rng.integers(1, 2**31, NTR), CH, eta, sigma=sig)
        x, eta = r['x_end'], r['eta_end']
        V = r['V'][:, :nb, 2]; V = V - V[:, valid].mean(1, keepdims=True)
        w = xi[:, :nb] * valid
        per.append(xcorr(w, V) / (valid.sum() * SIGP2)); null.append(xcorr(np.roll(w, 1, axis=0), V) / (valid.sum() * SIGP2))
    for name, P in (('real', np.concatenate(per)), ('null', np.concatenate(null))):
        K = P.mean(0); se = P.std(0, ddof=1) / np.sqrt(len(P)); neg = lags < -5
        out.append(dict(seed=seed, kind=name, acausal_rms_z=float(np.sqrt(np.mean((K[neg] / se[neg]) ** 2))),
                        acausal_mean_z=float(np.mean(K[neg] / se[neg]))))
        print(out[-1], flush=True)
json.dump(out, open(f'out/e6_acausal_check_{MODEL}.json', 'w'), indent=1)
