"""
ch9_protocol.py -- pump-probe measurement of state-conditioned response functions.

The linear response of node k to a probe of charge q into node p at time t_p,
conditioned on the reference trajectory set by a pump, is measured by a central
difference between two arms that share every random number:

    G_k(t; t_p) = [ V_k(t; pump + probe_{+q}) - V_k(t; pump + probe_{-q}) ] / (2 q)

This is the functional derivative delta V_k(t) / delta I_p(t_p) of the thesis
(errata Eq. sigma_def, Ch. 8 Eq. izh_deviation): the propagation kernel column,
measured instead of constructed. The same difference applied to the Route-D
propagated signal gives the response of S_k.
"""
import numpy as np


def probe_arms(circ, x0, pump_rows, probe_node, t_probe, amp, dur, T, dt=0.01,
               save_every=5, sigma=0.0, tau_noise=5.0, seeds=None, sign_pair=True):
    """
    t_probe : array of probe onset times (one condition per entry)
    returns dict with G (n, nt, N) response per unit charge, GS (same, for S),
            Vref (n, nt, N) reference trajectory, t (nt,)
    """
    t_probe = np.atleast_1d(np.asarray(t_probe, float))
    n = t_probe.size
    signs = (1.0, -1.0) if sign_pair else (1.0, 0.0)
    P = len(pump_rows) + 1
    pul = np.zeros((2 * n, P, 4))
    for a, sg in enumerate(signs):
        for k, tp in enumerate(t_probe):
            row = 2 * k + a
            for r, pr in enumerate(pump_rows):
                pul[row, r] = pr
            pul[row, P - 1] = (probe_node, tp, dur, sg * amp)
    if seeds is None:
        seeds = np.zeros(n, dtype=np.int64)
    seeds2 = np.repeat(np.asarray(seeds, dtype=np.int64), 2)       # CRN: both arms share a seed
    x0 = np.atleast_2d(x0)
    x0 = np.repeat(x0 if x0.shape[0] == n else np.tile(x0, (n, 1)), 2, axis=0)
    r = circ.run(x0, pul, seeds2, T, dt, sigma=sigma, tau_noise=tau_noise, save_every=save_every)
    V = r['V'].reshape(n, 2, -1, circ.N)
    Q = r['Q']
    S = circ.propagated_signal(Q, dt * save_every).reshape(n, 2, -1, circ.N)
    q = amp * dur
    denom = 2.0 * q if sign_pair else q
    G = (V[:, 0] - V[:, 1]) / denom
    GS = (S[:, 0] - S[:, 1]) / denom
    Vref = 0.5 * (V[:, 0] + V[:, 1]) if sign_pair else V[:, 1]
    t = np.arange(V.shape[2]) * dt * save_every
    spk = None
    if r['spikes'] is not None:
        spk = r['spikes'].reshape(n, 2, circ.N, -1)
    return dict(G=G, GS=GS, Vref=Vref, t=t, spikes=spk, q=q)


def window(arr, t, t0, W):
    """Slice arr[..., time, ...] over [t0, t0+W) for arr shaped (nt, N) or (nt,)."""
    i0 = int(round(t0 / (t[1] - t[0])))
    i1 = i0 + int(round(W / (t[1] - t[0])))
    return arr[i0:i1]


def response_norms(G, t, t_probe, W):
    """L2 norm over [t_p, t_p+W) for every condition and node: (n, N)."""
    dtt = t[1] - t[0]
    out = np.empty((G.shape[0], G.shape[2]))
    for k, tp in enumerate(np.atleast_1d(t_probe)):
        g = window(G[k], t, tp, W)
        out[k] = np.sqrt(np.sum(g ** 2, axis=0) * dtt)
    return out


def response_integrals(G, t, t_probe, W):
    dtt = t[1] - t[0]
    out = np.empty((G.shape[0], G.shape[2]))
    for k, tp in enumerate(np.atleast_1d(t_probe)):
        out[k] = np.sum(window(G[k], t, tp, W), axis=0) * dtt
    return out
