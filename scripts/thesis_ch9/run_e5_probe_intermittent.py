"""
E5e -- the intermittent Chapter 5 network (N=40, seed 2, gamma=0.03, 1500 s):
identical weak global pulses delivered from states sampled inside the
partially synchronized (chimera-like) epochs and inside the incoherent episodes.

The network is deterministic, so the response is a deterministic functional of
the microscopic state at stimulus onset; the question is how much of it is fixed
by macroscopic descriptors (order parameters, mean-field phase).

Arms per probe: +eps, -eps, +2eps, -2eps from the identical checkpoint (CRN is
trivial: no noise). Observable: synaptically filtered mean field z (tau_s = 5 ms).
    y(t) = [z(t;+eps) - z(t;-eps)] / (2 eps dur)       (response per unit charge)
State labels (causal): R_{1,2}^pre = mean order parameter over [t-1000, t-100] ms.
Instantaneous macroscopic state: phase and amplitude of each cluster's mean
(v,u) around its long-run centre, computed from the checkpoint itself.
"""
import numpy as np, json, time, sys
import ch9_network as NW
K_PER = int(sys.argv[1]) if len(sys.argv) > 1 else 700
EPS = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0125
DUR, T, TAU_S = 1.0, 80.0, 5.0
b, c, d, Ib, g = 0.2, -50.0, 2.0, 10.0, 0.03
N, seed = 40, 2
s = np.load(f'out/e5_state_intermittent_N{N}_s{seed}.npz')
a = s['a']; tg, R1, R2 = s['tg'], s['R1'], s['R2']
ck_t, ck_v, ck_u = s['ck_t'], s['ck_v'], s['ck_u']
order = np.argsort(a); c1, c2 = order[:N // 2], order[N // 2:]

def pre(R, t):
    return np.nanmean(R[(tg >= t - 1000) & (tg < t - 100)])

ok = np.where(ck_t >= 10000.0)[0]
R2pre_all = np.array([pre(R2, t) for t in ck_t[ok]])
rng = np.random.default_rng(7)
inc = rng.choice(ok[R2pre_all < 0.5], K_PER, replace=False)
coh = rng.choice(ok[R2pre_all > 0.75], K_PER, replace=False)
idx = np.concatenate([coh, inc]); lab = np.r_[np.zeros(K_PER), np.ones(K_PER)].astype(int)
frac_time = dict(incoherent=float(np.mean(R2pre_all < 0.5)), coherent=float(np.mean(R2pre_all > 0.75)))

V0, U0 = ck_v[idx], ck_u[idx]; Z0 = V0.mean(axis=1); K = len(idx)
amps = np.concatenate([np.full(K, EPS), np.full(K, -EPS), np.full(K, 2 * EPS), np.full(K, -2 * EPS)])
Vr, Ur, Zr = [np.concatenate([x] * 4) for x in (V0, U0, Z0)]
t0 = time.time()
Z, nsp = NW.run_branches_z(Vr, Ur, Zr, a, b, c, d, Ib, g, amps, DUR, np.ones(N), T, 0.01, 10, TAU_S)
Zp, Zm, Zp2, Zm2 = Z[:K], Z[K:2 * K], Z[2 * K:3 * K], Z[3 * K:]
y1 = (Zp - Zm) / (2 * EPS * DUR); y2 = (Zp2 - Zm2) / (4 * EPS * DUR)
dN = (nsp[:K] - nsp[K:2 * K]).astype(int)

# macroscopic descriptors at stimulus onset
cv, cu = ck_v[ok].mean(0), ck_u[ok].mean(0)
def phase_amp(cl):
    vm = V0[:, cl].mean(1); um = U0[:, cl].mean(1)
    v0 = ck_v[ok][:, cl].mean(); u0 = ck_u[ok][:, cl].mean()
    sv = ck_v[ok][:, cl].mean(1).std(); su = ck_u[ok][:, cl].mean(1).std()
    x, y = (vm - v0) / sv, (um - u0) / su
    return np.arctan2(y, x), np.hypot(x, y)
th1, A1 = phase_amp(c1); th2, A2 = phase_amp(c2)
R1pre = np.array([pre(R1, t) for t in ck_t[idx]]); R2pre = np.array([pre(R2, t) for t in ck_t[idx]])
tt = np.arange(Z.shape[1]) * 0.1; w = tt <= 60
lin = np.linalg.norm(y1[:, w] - y2[:, w], axis=1) / np.linalg.norm(y1[:, w], axis=1)
np.savez_compressed(f'out/e5_probe_intermittent_N{N}_s{seed}.npz', y1=y1.astype(np.float32),
                    y2=y2.astype(np.float32), t=tt, t_probe=ck_t[idx], label=lab, R1pre=R1pre, R2pre=R2pre,
                    th1=th1, A1=A1, th2=th2, A2=A2, V0=V0.astype(np.float32), U0=U0.astype(np.float32),
                    Z0=Z0, dN=dN, lin=lin)
out = dict(K_per_state=K_PER, frac_time=frac_time, linearity_median=float(np.median(lin)),
           linearity_p90=float(np.percentile(lin, 90)),
           linearity_median_by_state=[float(np.median(lin[lab == k])) for k in (0, 1)])
json.dump(out, open(f'out/e5_probe_intermittent_N{N}_s{seed}.json', 'w'), indent=1)
print(json.dumps(out), f'({time.time()-t0:.0f} s)')
