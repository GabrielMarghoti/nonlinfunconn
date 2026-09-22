"""
E2/E3 -- state-conditioned response functions along three-neuron chains.

For each motif, a pump (Ch. 7 stimulus) sets the state; a weak probe into node 1
at t_p = t_pump + Delta measures the linear response of every node. Delta scans
the state. Stored aligned to the probe onset: G[k, tau, node], GS[...] for S.
"""
import numpy as np, json, time, sys, os
import ch9_circuits as CC, ch9_protocol as PP

OUT = 'out'; os.makedirs(OUT, exist_ok=True)
DT, SAVE = 0.01, 5
DUR, AMP, W = 0.2, 0.1, 50.0
T_PUMP = 5.0
IZH_GAP, IZH_CHEM = 0.5, 0.075
deltas = np.round(np.arange(-3.0, 60.0 + 1e-9, 0.1), 3)

MOTIFS = {
    'gap-gap':   [('gap', 1, 0), ('gap', 2, 1)],
    'gap-chem':  [('gap', 1, 0), ('chem', 2, 1)],      # Fig. 7.12
    'chem-chem': [('chem', 1, 0), ('chem', 2, 1)],
}

def sweep(circ, pump_rows, tag, amp=AMP):
    x = circ.xeq
    tp = T_PUMP + deltas
    T = tp.max() + W + 1.0
    t0 = time.time()
    r = PP.probe_arms(circ, x, pump_rows, 0, tp, amp, DUR, T, dt=DT, save_every=SAVE)
    r0 = PP.probe_arms(circ, x, [], 0, np.array([T_PUMP]), amp, DUR, T, dt=DT, save_every=SAVE)
    t = r['t']; nW = int(round(W / (DT * SAVE)))
    G = np.stack([PP.window(r['G'][k], t, tp[k], W) for k in range(len(tp))])
    GS = np.stack([PP.window(r['GS'][k], t, tp[k], W) for k in range(len(tp))])
    G0 = PP.window(r0['G'][0], t, T_PUMP, W); GS0 = PP.window(r0['GS'][0], t, T_PUMP, W)
    # reference (pump-only) trajectory, from the latest probe condition (probe far away)
    Vref = r['Vref'][-1]
    spk = r['spikes']                                   # (n, 2 arms, N, maxspk)
    rr = circ.run(circ.xeq, CC.pulse_table(pump_rows)[None], [0], T, DT, save_every=SAVE)
    np.savez_compressed(f'{OUT}/e2_{tag}.npz', deltas=deltas, tau=np.arange(nW) * DT * SAVE,
                        G=G, GS=GS, G0=G0, GS0=GS0, Vref=Vref, t=t, tau0=circ.tau0,
                        xeq=circ.xeq, spikes=spk, spikes_ref=rr['spikes'][0],
                        G_abs=r['G'].astype(np.float32), q=r['q'], A=circ.A, B=circ.B)
    print(f"  {tag:28s} {time.time()-t0:5.1f}s  tau0={np.round(circ.tau0,3)}  eq_drift={circ.eq_drift:.1e}")

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'hh'
    if which == 'hh':
        for name, links in MOTIFS.items():
            c = CC.chain('hh', links); c.find_equilibrium()
            sweep(c, [(0, T_PUMP, 3.0, 10.0)], f'hh_{name}')
    if which == 'izh':
        # excitable Izhikevich nodes (I_b = 2 < 4); couplings calibrated in calib_izh.py
        for name, links in MOTIFS.items():
            c = CC.chain('izh', links, g_gap=IZH_GAP, g_chem=IZH_CHEM); c.find_equilibrium()
            sweep(c, [(0, T_PUMP, 3.0, 10.0)], f'izh_{name}')
