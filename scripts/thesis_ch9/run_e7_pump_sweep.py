"""
E7a -- how the state dependence of the pump-probe response grows with the pump.
For each Hodgkin-Huxley chain and pump amplitude (3 ms into node 1 at t = 5 ms),
a probe of charge q = 0.02 into node 1 at Delta = t_p - t_pump scans the state.
Stored: L2 norms over 50 ms of the probe responses of V_k and of the propagated
signal S_k (Eq. ch9_Q), and the spike counts of the pumped reference trajectory.
"""
import numpy as np, json, time
import ch9_circuits as CC, ch9_protocol as PP
DT, SAVE, DUR, AMP, W, T_PUMP = 0.01, 5, 0.2, 0.1, 50.0, 5.0
deltas = np.round(np.arange(-3.0, 60.0 + 1e-9, 0.25), 3)
PUMPS = [0.0, 1.0, 2.0, 3.0, 3.25, 3.5, 3.75, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0]
MOTIFS = {'gap-gap': [('gap', 1, 0), ('gap', 2, 1)], 'gap-chem': [('gap', 1, 0), ('chem', 2, 1)],
          'chem-chem': [('chem', 1, 0), ('chem', 2, 1)]}
out = {}; arrs = {}
t0 = time.time()
for name, links in MOTIFS.items():
    c = CC.chain('hh', links); c.find_equilibrium()
    tp = T_PUMP + deltas; T = tp.max() + W + 1.0
    r0 = PP.probe_arms(c, c.xeq, [], 0, np.array([T_PUMP]), AMP, DUR, T, dt=DT, save_every=SAVE)
    n0V = PP.response_norms(r0['G'], r0['t'], [T_PUMP], W)[0]; n0S = PP.response_norms(r0['GS'], r0['t'], [T_PUMP], W)[0]
    res = {'rest_norm_V': n0V.tolist(), 'rest_norm_S': n0S.tolist(), 'pumps': {}}
    NV, NS = [], []
    for A in PUMPS:
        rows = [(0, T_PUMP, 3.0, A)] if A > 0 else []
        r = PP.probe_arms(c, c.xeq, rows, 0, tp, AMP, DUR, T, dt=DT, save_every=SAVE)
        nV = PP.response_norms(r['G'], r['t'], tp, W) / n0V; nS = PP.response_norms(r['GS'], r['t'], tp, W) / n0S
        ref = c.run(c.xeq, CC.pulse_table(rows if rows else [(0, 0, 0, 0)])[None], [0], T, DT, save_every=SAVE)
        nspk = [int(np.sum(~np.isnan(ref['spikes'][0, k]))) for k in range(3)]
        res['pumps'][str(A)] = dict(spikes_ref=nspk,
                                    decades_V3=float(np.log10(nV[:, 2].max() / nV[:, 2].min())),
                                    decades_S3=float(np.log10(nS[:, 2].max() / nS[:, 2].min())),
                                    decades_V1=float(np.log10(nV[:, 0].max() / nV[:, 0].min())),
                                    max_V3=float(nV[:, 2].max()), min_V3=float(nV[:, 2].min()))
        NV.append(nV); NS.append(nS)
        print(f'{name:9s} A={A:5.1f} spikes={nspk} decades V3={res["pumps"][str(A)]["decades_V3"]:.2f} '
              f'S3={res["pumps"][str(A)]["decades_S3"]:.2f}  ({time.time()-t0:.0f}s)', flush=True)
    out[name] = res; arrs[name + '_NV'] = np.array(NV); arrs[name + '_NS'] = np.array(NS)
np.savez_compressed('out/e7_pump_sweep.npz', deltas=deltas, pumps=np.array(PUMPS), **arrs)
json.dump(out, open('out/e7_pump_sweep.json', 'w'), indent=1)
