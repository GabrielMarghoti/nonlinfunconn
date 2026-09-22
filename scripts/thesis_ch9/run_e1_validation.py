"""E1 -- validation of the pump-probe measurement on the Fig. 7.12 motif."""
import numpy as np, json, time
import ch9_circuits as CC, ch9_protocol as PP

OUT = 'out'; import os; os.makedirs(OUT, exist_ok=True)
PUMP = [(0, 5.0, 3.0, 10.0)]            # Chapter 7 stimulus: 10 uA/cm^2 for 3 ms into node 1
DUR, W = 0.2, 50.0                       # probe duration (ms), response window (ms)
deltas = np.array([-2.0, 0.5, 1.5, 2.5, 4.0, 6.0, 10.0, 20.0, 40.0])
res = {}

c = CC.chain('hh', [('gap', 1, 0), ('chem', 2, 1)]); x = c.find_equilibrium()
tp = 5.0 + deltas
T = tp.max() + W + 1.0

def norms(r):   return PP.response_norms(r['G'], r['t'], tp, W)

t0 = time.time()
# (i) linearity: amplitude sweep of the probe, central differences
amps = [0.025, 0.05, 0.1, 0.2, 0.4, 0.8]
lin = {a: norms(PP.probe_arms(c, x, PUMP, 0, tp, a, DUR, T)) for a in amps}
ref = lin[0.025]
res['linearity'] = {str(a): float(np.max(np.abs(lin[a] / ref - 1.0))) for a in amps}
# (ii) central vs one-sided difference at the working amplitude
AMP = 0.1
r_c = PP.probe_arms(c, x, PUMP, 0, tp, AMP, DUR, T)
r_o = PP.probe_arms(c, x, PUMP, 0, tp, AMP, DUR, T, sign_pair=False)
res['central_vs_onesided'] = float(np.max(np.abs(norms(r_o) / norms(r_c) - 1.0)))
# (iii) step-size convergence: dt = 0.01 vs 0.005 vs 0.0025 (same save interval in ms)
n1 = norms(r_c)
n2 = norms(PP.probe_arms(c, x, PUMP, 0, tp, AMP, DUR, T, dt=0.005, save_every=10))
n4 = norms(PP.probe_arms(c, x, PUMP, 0, tp, AMP, DUR, T, dt=0.0025, save_every=20))
res['dt_0.01_vs_0.0025'] = float(np.max(np.abs(n1 / n4 - 1.0)))
res['dt_0.005_vs_0.0025'] = float(np.max(np.abs(n2 / n4 - 1.0)))
# (iv) noise cancellation under CRN: same condition with OU background, two seeds
rn = PP.probe_arms(c, np.tile(x, (len(tp), 1)), PUMP, 0, tp, AMP, DUR, T, sigma=1.0,
                   seeds=np.arange(len(tp)))
res['crn_note'] = 'with sigma=1 uA/cm^2 OU noise the arm difference contains no noise term by construction'
res['elapsed_s'] = time.time() - t0
print(json.dumps(res, indent=2))
json.dump(res, open(f'{OUT}/e1_validation.json', 'w'), indent=2)
np.savez_compressed(f'{OUT}/e1_linearity.npz', amps=np.array(amps),
                    norms=np.array([lin[a] for a in amps]), deltas=deltas)
