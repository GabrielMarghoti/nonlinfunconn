"""ch9_analysis.py -- scalar summaries, telescoping gain decomposition, variance shares."""
import numpy as np

def l2(G, dtau):
    return np.sqrt(np.sum(G ** 2, axis=1) * dtau)            # (n, N)

def telescoping_factors(G, GS, dtau):
    """
    Exact telescoping of the end-to-end gain along a chain 1 -> 2 -> ... -> N:
        ||dV_N|| / ||dS_1|| = prod_k (||dV_k||/||dS_k||) * prod_k (||dS_k||/||dV_{k-1}||)
    node factors N_k = ||dV_k||/||dS_k|| : what node k's own dynamics does to the
                                           signal it receives (filtering of chi_k)
    link factors T_k = ||dS_k||/||dV_{k-1}|| : what the coupling k-1 -> k transmits
    Returns dict of log-factors (n,) and the log end-to-end gain.
    """
    nV, nS = l2(G, dtau), l2(GS, dtau)
    N = G.shape[2]
    f = {}
    for k in range(N):
        f[f'N{k+1}'] = np.log(nV[:, k] / nS[:, k])
        if k > 0:
            f[f'T{k+1}{k}'] = np.log(nS[:, k] / nV[:, k - 1])
    total = np.log(nV[:, N - 1] / nS[:, 0])
    assert np.allclose(sum(f.values()), total), 'telescoping identity violated'
    return f, total

def covariance_shares(factors, total):
    """c_i = Cov(x_i, total)/Var(total); sums exactly to one."""
    v = np.var(total)
    return {k: float(np.cov(x, total, bias=True)[0, 1] / v) for k, x in factors.items()}

def spike_sensitivity(spikes, q):
    """spikes (n, 2 arms, N, M) -> dt_spike/dq (n, N, M), NaN where spike counts differ."""
    s = (spikes[:, 0] - spikes[:, 1]) / (2.0 * q)
    same = np.isnan(spikes[:, 0]) == np.isnan(spikes[:, 1])
    return np.where(same, s, np.nan)
