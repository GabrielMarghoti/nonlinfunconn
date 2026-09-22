"""
E3 -- continuity of the state-conditioned response across a reset (Ch. 8 prediction).

(i)  Continuity test: probe just before / just after the first spike of node 1,
     at t* +- delta, delta -> 0.  For a continuous kernel the difference of any
     response functional vanishes linearly in delta; across a reset it tends to a
     finite jump.
(ii) Dirac atoms: the L2 norm of the voltage response per unit charge is finite
     for HH and diverges as q^(-1/2) for the resetting model.
"""
import numpy as np, json
import ch9_circuits as CC, ch9_protocol as PP
PUMP = [(0, 5.0, 3.0, 10.0)]
res = {}
for m, kw in (('hh', {}), ('izh', dict(g_gap=0.5, g_chem=0.075))):
    c = CC.chain(m, [('gap', 1, 0), ('chem', 2, 1)], **kw); x = c.find_equilibrium()
    ref = c.run(x, CC.pulse_table(PUMP)[None], [0], 60.0, 0.01, save_every=10)['spikes'][0]
    t_star = float(ref[0][0])                       # first spike of node 1
    # probe must END before the event to be 'before': probe of duration dur ending at t*-delta
    dur, amp = 0.02, 1.0
    deltas = np.array([3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3])
    tp = np.concatenate([t_star - deltas - dur, t_star + deltas])
    r = PP.probe_arms(c, x, PUMP, 0, tp, amp, dur, 60.0, dt=0.0005, save_every=40)
    s = (r['spikes'][:, 0] - r['spikes'][:, 1]) / (2 * r['q'])      # (2n, N, M)
    # functional: sensitivity of node-3 first spike (and node-1 second spike, if any)
    s3 = s[:, 2, 0]
    n = len(deltas)
    jump3 = np.abs(s3[:n] - s3[n:])
    res[m] = dict(t_star=t_star, deltas=deltas.tolist(), jump_node3_spike1=jump3.tolist())
    # (ii) norm of the node-1 voltage response vs probe amplitude, probe placed before t*
    tpn = np.array([t_star - 0.6])
    norms = []
    qs = [0.4, 0.2, 0.1, 0.05, 0.025]
    for a_ in qs:
        rr = PP.probe_arms(c, x, PUMP, 0, tpn, a_, 0.2, 60.0, dt=0.001, save_every=5)
        G = rr['G'][0, :, 0]; dtt = rr['t'][1] - rr['t'][0]
        norms.append(float(np.sqrt(np.sum(G ** 2) * dtt)))
    res[m]['amp'] = qs; res[m]['l2_norm_node1'] = norms
    print(m, 't*=%.3f' % t_star, 'jumps:', np.round(jump3, 5), '\n   L2 norms vs amp', np.round(norms, 3))
json.dump(res, open('out/e3_continuity.json', 'w'), indent=1)
