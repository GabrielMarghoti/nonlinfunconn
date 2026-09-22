"""
E7b -- trial-by-trial variability of the evoked response in the noisy Hodgkin-Huxley
gap -> chemical motif, as a function of the stimulus amplitude and of the noise.

Onset states are sampled as in E4 (independent runs under OU noise, one state every
100 ms). From each state the stimulus (3 ms into node 1, amplitude A) is delivered K
times with independent noise after onset. The response is the causal effect on node 3,
R = int_0^40 [V3(stim) - V3(no stim)] dt, with both arms sharing state and noise, and
is reported per unit charge q = 3 A. In a linear system R would be identical on every
trial; its spread measures what the nonlinearity does.
Statistics: mean, SD, coefficient of variation, and the intraclass correlation
ICC = [Var_m(mean_k R) - E_m(var_k R)/K] / Var(R) (fraction of the variance between onset states).
"""
import numpy as np, json, time, sys
import ch9_circuits as CC
import run_e4_conditioning as E

DT, T_RESP, W_RESP = 0.01, 45.0, 40.0

def evoked(c, X, Eta, sigma, A, K, seed):
    M = X.shape[0]
    x0 = np.repeat(X, K, axis=0); e0 = np.repeat(Eta, K, axis=0)
    seeds = seed + np.arange(M * K)
    pul = np.tile(CC.pulse_table([(0, 0.0, 3.0, A)])[None], (M * K, 1, 1)); ctl = np.zeros_like(pul)
    rs = c.run(x0, pul, seeds, T_RESP, DT, sigma=sigma, save_every=5, eta0=e0, max_spikes=16)
    rc = c.run(x0, ctl, seeds, T_RESP, DT, sigma=sigma, save_every=5, eta0=e0, max_spikes=16)
    t = np.arange(rs['V'].shape[1]) * DT * 5; w = t < W_RESP
    R = np.sum(rs['V'][:, w, 2] - rc['V'][:, w, 2], axis=1) * DT * 5
    cnt = lambda r: np.sum((r['spikes'][:, 2, :] >= 0) & (r['spikes'][:, 2, :] < W_RESP), axis=1)
    return (R / (3.0 * A)).reshape(M, K), (cnt(rs) - cnt(rc)).reshape(M, K), rc['V'][:, 0, 2].reshape(M, K)[:, 0]

def stats(Rq, dN):
    M, K = Rq.shape
    icc = (Rq.mean(1).var() - Rq.var(1, ddof=1).mean() / K) / Rq.var()
    return dict(mean=float(Rq.mean()), sd=float(Rq.std()), cv=float(Rq.std() / abs(Rq.mean())), icc=float(icc),
                p_spike_change=float(np.mean(dN != 0)),
                dN_dist={int(k): float(np.mean(dN == k)) for k in np.unique(dN)})

c = CC.chain('hh', [('gap', 1, 0), ('chem', 2, 1)]); c.find_equilibrium()
out = {'amplitude_sweep': {}, 'noise_sweep': {}}; t0 = time.time()
# (1) amplitude sweep at sigma = 1 on the E4 onset states (M = 1000)
X, Eta, H = E.sample_states(c, 1.0, E.SEED0)
AMPS = [0.1, 0.3, 1.0, 2.0, 3.0, 5.0, 10.0]
keep = {}
for A in AMPS:
    Rq, dN, V3 = evoked(c, X, Eta, 1.0, A, 8, E.SEED0 + 2 * 10 ** 6)
    out['amplitude_sweep'][str(A)] = stats(Rq, dN); keep[f'Rq_{A}'] = Rq.astype(np.float32); keep[f'dN_{A}'] = dN.astype(np.int8)
    print(f'A={A:5.1f}: ' + json.dumps({k: v for k, v in out['amplitude_sweep'][str(A)].items() if k != 'dN_dist'}) + f' ({time.time()-t0:.0f}s)', flush=True)
keep['V3_onset'] = X[:, 2]; keep['n3_onset'] = X[:, 3 * 3 + 2]
np.savez_compressed('out/e7_amplitude.npz', **keep)
# (2) noise sweep for a weak and a strong stimulus (M = 400, K = 8)
E.N_CHAINS, E.N_SEG = 40, 10
for sg in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
    X, Eta, H = E.sample_states(c, sg, E.SEED0 + 7)
    # firing rate of node 3 without stimulus, from a 2 s run per chain
    rr = c.run(X[:40], np.zeros((40, 1, 4)), E.SEED0 + 9 + np.arange(40), 2000.0, DT, sigma=sg, save_every=100, eta0=Eta[:40], max_spikes=400)
    rate3 = float(np.sum(~np.isnan(rr['spikes'][:, 2, :])) / (40 * 2.0))
    res = {'rate3_Hz': rate3}
    for A in (0.1, 10.0):
        Rq, dN, _ = evoked(c, X, Eta, sg, A, 8, E.SEED0 + 3 * 10 ** 6)
        res[str(A)] = stats(Rq, dN)
    out['noise_sweep'][str(sg)] = res
    print(f'sigma={sg}: rate3={rate3:.2f} Hz  weak cv={res["0.1"]["cv"]:.3f} icc={res["0.1"]["icc"]:.3f}  '
          f'strong cv={res["10.0"]["cv"]:.3f} icc={res["10.0"]["icc"]:.3f} ({time.time()-t0:.0f}s)', flush=True)
json.dump(out, open('out/e7_variability.json', 'w'), indent=1)
