"""
E5b -- prepare the two coexisting states of Chapter 5 by the continuation protocol
(Sec. 5.2): gamma increased from 0 in steps of 0.001 reaches the chimera-like state
at gamma = 0.03; gamma decreased from 0.1 reaches the incoherent state there.
Then long runs in each state, with checkpoints for the stimulation experiments.
"""
import numpy as np, time, sys
import ch9_network as NW
N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 3
T_state = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0e5
G_TARGET, STEP_T = 0.03, 2000.0
rng = np.random.default_rng(seed)
a = rng.uniform(0.013, 0.024, N)
order = np.argsort(a); c1, c2 = order[:N // 2], order[N // 2:]
b, c, d, Ib = 0.2, -50.0, 2.0, 10.0

def advance(v, u, g, T):
    o = NW.run_free(v, u, a, b, c, d, Ib, g, T, 0.01, 10 ** 9, 10 ** 9, 5 * 10 ** 6)
    return o

def R_of(o, T):
    ons = NW.burst_onsets(o[0], o[1], N)
    tg = np.arange(0.3 * T, 0.9 * T, 5.0)
    return np.nanmean(NW.order_parameter(ons, c1, tg)), np.nanmean(NW.order_parameter(ons, c2, tg))

t0 = time.time()
v = rng.uniform(-70, -50, N); u = 0.2 * v
branches = {}
# increasing branch
for g in np.round(np.arange(0.0, G_TARGET + 1e-9, 0.001), 4):
    o = advance(v, u, g, STEP_T); v, u = o[6], o[7]
R1, R2 = R_of(o, STEP_T); branches['increasing'] = (v.copy(), u.copy(), R1, R2)
print(f"increasing branch at gamma=0.03: <R1>={R1:.3f} <R2>={R2:.3f}  ({time.time()-t0:.0f}s)", flush=True)
# decreasing branch: start synchronised at 0.1
v = rng.uniform(-70, -50, N); u = 0.2 * v
o = advance(v, u, 0.1, 4000.0); v, u = o[6], o[7]
for g in np.round(np.arange(0.1, G_TARGET - 1e-9, -0.001), 4):
    o = advance(v, u, g, STEP_T); v, u = o[6], o[7]
R1, R2 = R_of(o, STEP_T); branches['decreasing'] = (v.copy(), u.copy(), R1, R2)
print(f"decreasing branch at gamma=0.03: <R1>={R1:.3f} <R2>={R2:.3f}  ({time.time()-t0:.0f}s)", flush=True)
# long runs in each state with checkpoints every 50 ms
for name, (v0, u0, _, _) in branches.items():
    o = NW.run_free(v0, u0, a, b, c, d, Ib, G_TARGET, T_state, 0.01, 100, 5000, 2 * 10 ** 8)
    spk_t, spk_i, mf, ck_t, ck_v, ck_u, v, u = o
    ons = NW.burst_onsets(spk_t, spk_i, N)
    tg = np.arange(0.0, T_state, 10.0)
    R1 = NW.order_parameter(ons, c1, tg); R2 = NW.order_parameter(ons, c2, tg)
    np.savez_compressed(f'out/e5_state_{name}_N{N}_s{seed}.npz', a=a, tg=tg, R1=R1.astype(np.float32),
                        R2=R2.astype(np.float32), mf=mf.astype(np.float32), ck_t=ck_t, ck_v=ck_v, ck_u=ck_u)
    print(f"  {name}: {T_state/1000:.0f} s run; R2 quantiles {np.round(np.nanpercentile(R2,[1,25,50,75,99]),3)}; "
          f"R1 median {np.nanmedian(R1):.3f}  ({time.time()-t0:.0f}s)", flush=True)
