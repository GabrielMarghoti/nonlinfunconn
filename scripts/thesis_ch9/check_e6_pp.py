"""Second, independent pump-probe sample for the E6 validation (different seed, 2x trials),
compared with the covariance estimate stored in out/e6_{model}.npz."""
import numpy as np, json, sys
import ch9_circuits as CC, ch9_modelfree as MF
MODEL = sys.argv[1]; M = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
kw = {} if MODEL == 'hh' else dict(g_gap=0.5, g_chem=0.075); sig = 1.0 if MODEL == 'hh' else 0.5
c = CC.chain(MODEL, [('gap', 1, 0), ('chem', 2, 1)], **kw); xeq = c.find_equilibrium()
SIGP2, DB, AMP = 0.5, 0.1, 0.2; VARXI = SIGP2 / DB
rng = np.random.default_rng(777 + (MODEL == 'izh'))
nchain = 100; nseg = M // nchain
x = np.tile(xeq, (nchain, 1)); eta = sig * rng.standard_normal((nchain, 3))
r = MF.run_xi(c, x, rng.normal(0, np.sqrt(VARXI), (nchain, 5001)), np.zeros((nchain, 1, 4)), rng.integers(1, 2**31, nchain), 500.0, eta, sigma=sig)
x, eta = r['x_end'], r['eta_end']; X, E = [], []
for s in range(nseg):
    r = MF.run_xi(c, x, rng.normal(0, np.sqrt(VARXI), (nchain, 1001)), np.zeros((nchain, 1, 4)), rng.integers(1, 2**31, nchain), 100.0, eta, sigma=sig)
    x, eta = r['x_end'], r['eta_end']; X.append(x.copy()); E.append(eta.copy())
X = np.concatenate(X); E = np.concatenate(E)
xi = rng.normal(0, np.sqrt(VARXI), (M, 601)); seeds = rng.integers(1, 2**31, M)
pul = np.zeros((M, 1, 4)); pul[:, 0] = [0, 0.0, 0.095, AMP]
Vp = MF.run_xi(c, X, xi, pul, seeds, 60.0, E, sigma=sig)['V'][:, :, 2]
pul[:, 0, 3] = -AMP
Vm = MF.run_xi(c, X, xi, pul, seeds, 60.0, E, sigma=sig)['V'][:, :, 2]
G = (Vp - Vm) / (2 * AMP * DB)
I = G[:, 1:601].reshape(M, 24, 25).sum(-1) * DB
d = np.load(f'out/e6_{MODEL}.npz')
m, se = I.mean(0), I.std(0, ddof=1) / np.sqrt(M)
bs = np.array([I[rng.integers(0, M, M)].mean(0) for _ in range(400)]); se_b = bs.std(0)
z = (m - d['Ist_mean']) / np.sqrt(se ** 2 + d['Ist_se'] ** 2)
zb = (m - d['Ist_mean']) / np.sqrt(se_b ** 2 + d['Ist_se'] ** 2)
kurt = ((I - I.mean(0)) ** 4).mean(0) / I.var(0) ** 2
res = dict(model=MODEL, M=M, z_rms=float(np.sqrt(np.mean(z ** 2))), z_max=float(np.abs(z).max()),
           zb_rms=float(np.sqrt(np.mean(zb ** 2))), zb_max=float(np.abs(zb).max()),
           frac_abs_z_gt2=float(np.mean(np.abs(z) > 2)), window_kurtosis_median=float(np.median(kurt)),
           window_kurtosis_max=float(kurt.max()))
print(json.dumps(res)); json.dump(res, open(f'out/e6_pp_check_{MODEL}.json', 'w'), indent=1)
np.savez_compressed(f'out/e6_pp_check_{MODEL}.npz', Ipp=m, Ipp_se=se)
