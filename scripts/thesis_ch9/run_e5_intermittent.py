"""E5d -- long run of the intermittent N=40 network with checkpoints every 50 ms."""
import numpy as np, time, sys
import ch9_network as NW

N, seed = 40, 2
T = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5e6
rng = np.random.default_rng(seed)
a = rng.uniform(0.013, 0.024, N)
v0 = rng.uniform(-70, -50, N); u0 = 0.2 * v0
o = NW.run_free(v0, u0, a, 0.2, -50.0, 2.0, 10.0, 0.03, 500.0, 0.01, 10 ** 9, 10 ** 9, 10 ** 6)
t0 = time.time()
spk_t, spk_i, mf, ck_t, ck_v, ck_u, v, u = NW.run_free(o[6], o[7], a, 0.2, -50.0, 2.0, 10.0, 0.03, T, 0.01,
                                                        100, 5000, 3 * 10 ** 8)
ons = NW.burst_onsets(spk_t, spk_i, N)
tg = np.arange(0.0, T, 10.0)
order = np.argsort(a); c1, c2 = order[:N // 2], order[N // 2:]
R1 = NW.order_parameter(ons, c1, tg); R2 = NW.order_parameter(ons, c2, tg)
np.savez_compressed(f'out/e5_state_intermittent_N{N}_s{seed}.npz', a=a, tg=tg, R1=R1.astype(np.float32),
                    R2=R2.astype(np.float32), mf=mf.astype(np.float32), ck_t=ck_t, ck_v=ck_v, ck_u=ck_u)
print(f"N={N} seed={seed}: {T/1000:.0f} s in {time.time()-t0:.0f} s; R2 quantiles "
      f"{np.round(np.nanpercentile(R2, [1, 5, 25, 50]), 3)}")
