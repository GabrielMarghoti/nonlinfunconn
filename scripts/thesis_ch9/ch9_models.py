"""
ch9_models.py -- numerical core for the Chapter 9 stimulation experiments.

Two node models on the same coupling scaffold (thesis Eq. 6.1):

  * Hodgkin-Huxley, with the Chapter 7 parameters of
    scripts/HH_julia/HH_NEGF_DELTA_SYNAPSE_3neurons.jl
  * Izhikevich (thesis Eqs. 2.26-2.28), with an accurate reset

Coupling (current into node i, identical for both models):

    I_in,i = sum_j A_ij (V_j - V_i) - sum_j B_ij s_ij (V_i - E^s_ij) + I_ext,i
    ds_ij/dt = a_r sigma(beta (V_j - V_th,ij)) (1 - s_ij) - a_d s_ij

which is the synapse of the Chapter 7 script (g^s_ij = B_ij s_ij, 1/tau_rec = a_d).
A_ij need not be symmetric; the Chapter 7 chains use directed links.

Integration: classical RK4 at fixed step with piecewise-constant external input.
External input = rectangular pulses (deterministic) + Ornstein-Uhlenbeck
background drawn from a per-trial seed. Two arms that share a seed receive
bit-identical noise (common random numbers), so probe responses can be formed
as arm differences without any noise contamination.

Along with V the integrators return the Route-D input source (thesis Eq. 6.24)

    Q_i = sum_j A_ij dV_j - sum_j B_ij ds_ij (V_i - E^s_ij) + I_ext,i

whose filtering by exp(-t/tau_0,i) is the propagated signal S_i (Eq. 7.21),
computed without ever forming a kernel or dividing by dV.
"""
import numpy as np
from numba import njit, prange

# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
@njit(cache=True, inline='always')
def _linexp(c, x, s):
    # c * x / (1 - exp(-x/s)) with the removable singularity at x = 0
    if abs(x) < 1e-7:
        return c * s * (1.0 + x / (2.0 * s))
    return c * x / (1.0 - np.exp(-x / s))


@njit(cache=True, inline='always')
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


@njit(cache=True, inline='always')
def _pulse_current(pulses, node, t):
    # pulses: (P, 4) rows = (node, t_on, duration, amplitude)
    I = 0.0
    for p in range(pulses.shape[0]):
        if int(pulses[p, 0]) == node and t >= pulses[p, 1] and t < pulses[p, 1] + pulses[p, 2]:
            I += pulses[p, 3]
    return I


@njit(cache=True, inline='always')
def hh_rates(V):
    am = _linexp(0.1, V + 40.0, 10.0)
    bm = 4.0 * np.exp(-(V + 65.0) / 18.0)
    ah = 0.07 * np.exp(-(V + 65.0) / 20.0)
    bh = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    an = _linexp(0.01, V + 55.0, 10.0)
    bn = 0.125 * np.exp(-(V + 65.0) / 80.0)
    return am, bm, ah, bh, an, bn


# --------------------------------------------------------------------------
# Hodgkin-Huxley circuit
# state layout per trial: [V(N), m(N), h(N), n(N), s(N*N)]
# node params P = (C, gNa, gK, gL, ENa, EK, EL)
# --------------------------------------------------------------------------
@njit(cache=True)
def _hh_rhs(x, Iext, N, P, A, B, Es, ar, ad, beta, Vth, out):
    C, gNa, gK, gL, ENa, EK, EL = P[0], P[1], P[2], P[3], P[4], P[5], P[6]
    for i in range(N):
        V = x[i]; m = x[N + i]; h = x[2 * N + i]; n = x[3 * N + i]
        Iin = Iext[i]
        for j in range(N):
            if A[i, j] != 0.0:
                Iin += A[i, j] * (x[j] - V)
            if B[i, j] != 0.0:
                Iin -= B[i, j] * x[4 * N + i * N + j] * (V - Es[i, j])
        Iion = gNa * m * m * m * h * (V - ENa) + gK * n * n * n * n * (V - EK) + gL * (V - EL)
        out[i] = (Iin - Iion) / C
        am, bm, ah, bh, an, bn = hh_rates(V)
        out[N + i] = am * (1.0 - m) - bm * m
        out[2 * N + i] = ah * (1.0 - h) - bh * h
        out[3 * N + i] = an * (1.0 - n) - bn * n
    for i in range(N):
        for j in range(N):
            k = 4 * N + i * N + j
            if B[i, j] != 0.0:
                s = x[k]
                out[k] = ar[i, j] * _sig(beta[i, j] * (x[j] - Vth[i, j])) * (1.0 - s) - ad[i, j] * s
            else:
                out[k] = 0.0


@njit(cache=True, parallel=True)
def simulate_hh(x0, pulses, seeds, sigma, tau_noise, T, dt, P, A, B, Es, ar, ad,
                beta, Vth, xeq, save_every, max_spikes=64, V_spk=0.0, eta0=None):
    """
    x0      (Btr, 4N+N^2) initial states
    pulses  (Btr, Pmax, 4) pulse table per trial (rows with amplitude 0 are inert)
    seeds   (Btr,) int64 OU-noise seed per trial; sigma = 0 disables noise
    xeq     (4N+N^2,) reference (equilibrium) state used for the deviations in Q
    returns V (Btr, nsave, N), Q (Btr, nsave, N), x_end (Btr, 4N+N^2),
            spike times (Btr, N, max_spikes): upward crossings of V_spk, linearly
            interpolated within the step, NaN-padded
    """
    Btr, nx = x0.shape
    N = A.shape[0]
    nsteps = int(round(T / dt))
    nsave = nsteps // save_every + 1
    Vout = np.empty((Btr, nsave, N))
    Qout = np.empty((Btr, nsave, N))
    xend = np.empty((Btr, nx))
    spk = np.full((Btr, N, max_spikes), np.nan)
    etaend = np.zeros((Btr, N))
    dec = np.exp(-dt / tau_noise)
    kick = sigma * np.sqrt(1.0 - dec * dec)
    for b in prange(Btr):
        np.random.seed(seeds[b])
        x = x0[b].copy()
        k1 = np.empty(nx); k2 = np.empty(nx); k3 = np.empty(nx); k4 = np.empty(nx)
        xt = np.empty(nx)
        Iext = np.zeros(N)
        eta = np.zeros(N)
        Vprev = np.empty(N)
        nspk = np.zeros(N, dtype=np.int64)
        if sigma > 0.0:
            for i in range(N):
                eta[i] = sigma * np.random.randn()
            if eta0 is not None:
                for i in range(N):
                    eta[i] = eta0[b, i]
        isave = 0
        for step in range(nsteps + 1):
            t = step * dt
            for i in range(N):
                Iext[i] = _pulse_current(pulses[b], i, t) + eta[i]
            if step % save_every == 0:
                for i in range(N):
                    Vout[b, isave, i] = x[i]
                    q = Iext[i]
                    for j in range(N):
                        if A[i, j] != 0.0:
                            q += A[i, j] * (x[j] - xeq[j])
                        if B[i, j] != 0.0:
                            kk = 4 * N + i * N + j
                            q -= B[i, j] * (x[kk] - xeq[kk]) * (x[i] - Es[i, j])
                    Qout[b, isave, i] = q
                isave += 1
            if step == nsteps:
                break
            for i in range(N):
                Vprev[i] = x[i]
            _hh_rhs(x, Iext, N, P, A, B, Es, ar, ad, beta, Vth, k1)
            for q in range(nx): xt[q] = x[q] + 0.5 * dt * k1[q]
            _hh_rhs(xt, Iext, N, P, A, B, Es, ar, ad, beta, Vth, k2)
            for q in range(nx): xt[q] = x[q] + 0.5 * dt * k2[q]
            _hh_rhs(xt, Iext, N, P, A, B, Es, ar, ad, beta, Vth, k3)
            for q in range(nx): xt[q] = x[q] + dt * k3[q]
            _hh_rhs(xt, Iext, N, P, A, B, Es, ar, ad, beta, Vth, k4)
            for q in range(nx):
                x[q] += dt * (k1[q] + 2.0 * k2[q] + 2.0 * k3[q] + k4[q]) / 6.0
            for i in range(N):
                if Vprev[i] < V_spk and x[i] >= V_spk:
                    th = (V_spk - Vprev[i]) / (x[i] - Vprev[i])
                    if nspk[i] < max_spikes:
                        spk[b, i, nspk[i]] = t + th * dt
                    nspk[i] += 1
            if sigma > 0.0:
                for i in range(N):
                    eta[i] = dec * eta[i] + kick * np.random.randn()
        for q in range(nx):
            xend[b, q] = x[q]
        for i in range(N):
            etaend[b, i] = eta[i]
    return Vout, Qout, xend, spk, etaend


# --------------------------------------------------------------------------
# Izhikevich circuit
# state layout per trial: [v(N), u(N), s(N*N)]
# node params per node: a, b, c, d, Ib  (arrays of length N); v_peak = 30
# --------------------------------------------------------------------------
VPEAK = 30.0


@njit(cache=True)
def _izh_rhs(x, Iext, N, a, bb, Ib, A, B, Es, ar, ad, beta, Vth, out):
    for i in range(N):
        v = x[i]; u = x[N + i]
        Iin = Iext[i]
        for j in range(N):
            if A[i, j] != 0.0:
                Iin += A[i, j] * (x[j] - v)
            if B[i, j] != 0.0:
                Iin -= B[i, j] * x[2 * N + i * N + j] * (v - Es[i, j])
        out[i] = 0.04 * v * v + 5.0 * v + 140.0 - u + Ib[i] + Iin
        out[N + i] = a[i] * (bb[i] * v - u)
    for i in range(N):
        for j in range(N):
            k = 2 * N + i * N + j
            if B[i, j] != 0.0:
                s = x[k]
                out[k] = ar[i, j] * _sig(beta[i, j] * (x[j] - Vth[i, j])) * (1.0 - s) - ad[i, j] * s
            else:
                out[k] = 0.0


@njit(cache=True)
def _izh_rk4(x, h, Iext, N, a, bb, Ib, A, B, Es, ar, ad, beta, Vth, k1, k2, k3, k4, xt, out):
    nx = x.shape[0]
    _izh_rhs(x, Iext, N, a, bb, Ib, A, B, Es, ar, ad, beta, Vth, k1)
    for q in range(nx): xt[q] = x[q] + 0.5 * h * k1[q]
    _izh_rhs(xt, Iext, N, a, bb, Ib, A, B, Es, ar, ad, beta, Vth, k2)
    for q in range(nx): xt[q] = x[q] + 0.5 * h * k2[q]
    _izh_rhs(xt, Iext, N, a, bb, Ib, A, B, Es, ar, ad, beta, Vth, k3)
    for q in range(nx): xt[q] = x[q] + h * k3[q]
    _izh_rhs(xt, Iext, N, a, bb, Ib, A, B, Es, ar, ad, beta, Vth, k4)
    for q in range(nx):
        out[q] = x[q] + h * (k1[q] + 2.0 * k2[q] + 2.0 * k3[q] + k4[q]) / 6.0


@njit(cache=True, parallel=True)
def simulate_izh(x0, pulses, seeds, sigma, tau_noise, T, dt, a, bb, c, d, Ib,
                 A, B, Es, ar, ad, beta, Vth, xeq, save_every, max_spikes, eta0=None):
    """
    Same contract as simulate_hh. Resets are event-accurate: when a step carries
    some v_i past v_peak, the earliest crossing time within the step is located
    by bisection on a single RK4 substep of the WHOLE system, the reset map
    (v_i, u_i) -> (c_i, u_i + d_i) is applied there, and the remainder of the
    step is integrated from the reset state (repeating if another neuron
    crosses). No neuron ever sees another neuron's voltage beyond v_peak.
    Also returns spike times (Btr, N, max_spikes), padded with NaN.
    """
    Btr, nx = x0.shape
    N = A.shape[0]
    nsteps = int(round(T / dt))
    nsave = nsteps // save_every + 1
    Vout = np.empty((Btr, nsave, N))
    Qout = np.empty((Btr, nsave, N))
    xend = np.empty((Btr, nx))
    spk = np.full((Btr, N, max_spikes), np.nan)
    etaend = np.zeros((Btr, N))
    dec = np.exp(-dt / tau_noise)
    kick = sigma * np.sqrt(1.0 - dec * dec)
    for b in prange(Btr):
        np.random.seed(seeds[b])
        x = x0[b].copy()
        xn = np.empty(nx); xm = np.empty(nx)
        k1 = np.empty(nx); k2 = np.empty(nx); k3 = np.empty(nx); k4 = np.empty(nx)
        xt = np.empty(nx)
        Iext = np.zeros(N)
        eta = np.zeros(N)
        nspk = np.zeros(N, dtype=np.int64)
        if sigma > 0.0:
            for i in range(N):
                eta[i] = sigma * np.random.randn()
            if eta0 is not None:
                for i in range(N):
                    eta[i] = eta0[b, i]
        isave = 0
        for step in range(nsteps + 1):
            t = step * dt
            for i in range(N):
                Iext[i] = _pulse_current(pulses[b], i, t) + eta[i]
            if step % save_every == 0:
                for i in range(N):
                    Vout[b, isave, i] = x[i]
                    qq = Iext[i]
                    for j in range(N):
                        if A[i, j] != 0.0:
                            qq += A[i, j] * (x[j] - xeq[j])
                        if B[i, j] != 0.0:
                            kk = 2 * N + i * N + j
                            qq -= B[i, j] * (x[kk] - xeq[kk]) * (x[i] - Es[i, j])
                    Qout[b, isave, i] = qq
                isave += 1
            if step == nsteps:
                break
            h_left = dt
            t_now = t
            for _guard in range(4 * N + 4):
                _izh_rk4(x, h_left, Iext, N, a, bb, Ib, A, B, Es, ar, ad, beta, Vth, k1, k2, k3, k4, xt, xn)
                crossed = False
                for i in range(N):
                    if xn[i] >= VPEAK:
                        crossed = True
                if not crossed:
                    for q in range(nx): x[q] = xn[q]
                    break
                # earliest crossing: bisection on the substep length
                lo = 0.0; hi = h_left
                for _it in range(50):
                    mid = 0.5 * (lo + hi)
                    _izh_rk4(x, mid, Iext, N, a, bb, Ib, A, B, Es, ar, ad, beta, Vth, k1, k2, k3, k4, xt, xm)
                    anyc = False
                    for i in range(N):
                        if xm[i] >= VPEAK:
                            anyc = True
                    if anyc:
                        hi = mid
                    else:
                        lo = mid
                _izh_rk4(x, hi, Iext, N, a, bb, Ib, A, B, Es, ar, ad, beta, Vth, k1, k2, k3, k4, xt, xm)
                for q in range(nx): x[q] = xm[q]
                for i in range(N):
                    if x[i] >= VPEAK:
                        x[i] = c[i]
                        x[N + i] += d[i]
                        if nspk[i] < max_spikes:
                            spk[b, i, nspk[i]] = t_now + hi
                        nspk[i] += 1
                t_now += hi
                h_left -= hi
                if h_left <= 1e-12:
                    break
            if sigma > 0.0:
                for i in range(N):
                    eta[i] = dec * eta[i] + kick * np.random.randn()
        for q in range(nx):
            xend[b, q] = x[q]
        for i in range(N):
            etaend[b, i] = eta[i]
    return Vout, Qout, xend, spk, etaend


# --------------------------------------------------------------------------
# Route-D propagated signal:  S_i = phi_0,i * Q_i / C,   phi_0 = theta(t) e^{-t/tau_0}
# exact exponential filter for piecewise-linear Q (trapezoid in the kernel)
# --------------------------------------------------------------------------
@njit(cache=True, parallel=True)
def route_d_signal(Q, tau0, C, dt):
    Btr, T, N = Q.shape
    S = np.zeros_like(Q)
    for b in prange(Btr):
        for i in range(N):
            e = np.exp(-dt / tau0[i])
            for k in range(1, T):
                S[b, k, i] = e * S[b, k - 1, i] + 0.5 * dt * (e * Q[b, k - 1, i] + Q[b, k, i]) / C
    return S
