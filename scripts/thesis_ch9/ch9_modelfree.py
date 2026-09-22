"""
ch9_modelfree.py -- integrators for the model-free kernel experiment (E6).

Same circuits and numerics as ch9_models.py (RK4, dt = 0.01 ms, event-accurate
Izhikevich reset, OU background noise from a per-trial seed), plus one extra
input: a Gaussian white-noise probe current xi into node `xi_node`, piecewise
constant on bins of `xi_bin` steps and supplied by the caller, so the same
realization can be replayed in several arms.

Returns V (Btr, nsave, N) sampled at the bin starts, the hidden variable of
node `hid_node` (HH: potassium activation n; Izhikevich: recovery variable u),
the final state, spike times and the final OU state.
"""
import numpy as np
from numba import njit, prange
from ch9_models import _pulse_current, _hh_rhs, _izh_rk4, VPEAK


@njit(cache=True, parallel=True)
def simulate_hh_xi(x0, xi, xi_node, xi_bin, pulses, seeds, sigma, tau_noise, T, dt, P, A, B, Es,
                   ar, ad, beta, Vth, save_every, max_spikes, hid_node, eta0):
    Btr, nx = x0.shape
    N = A.shape[0]
    nsteps = int(round(T / dt)); nsave = nsteps // save_every + 1
    Vout = np.empty((Btr, nsave, N)); Hout = np.empty((Btr, nsave))
    xend = np.empty((Btr, nx)); etaend = np.zeros((Btr, N))
    spk = np.full((Btr, N, max_spikes), np.nan)
    dec = np.exp(-dt / tau_noise); kick = sigma * np.sqrt(1.0 - dec * dec)
    for b in prange(Btr):
        np.random.seed(seeds[b])
        x = x0[b].copy()
        k1 = np.empty(nx); k2 = np.empty(nx); k3 = np.empty(nx); k4 = np.empty(nx); xt = np.empty(nx)
        Iext = np.zeros(N); eta = np.zeros(N); Vprev = np.empty(N)
        nspk = np.zeros(N, dtype=np.int64)
        for i in range(N):
            eta[i] = eta0[b, i]
        isave = 0
        for step in range(nsteps + 1):
            t = step * dt
            for i in range(N):
                Iext[i] = _pulse_current(pulses[b], i, t) + eta[i]
            Iext[xi_node] += xi[b, step // xi_bin]
            if step % save_every == 0:
                for i in range(N):
                    Vout[b, isave, i] = x[i]
                Hout[b, isave] = x[3 * N + hid_node]
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
                if Vprev[i] < 0.0 and x[i] >= 0.0:
                    th = (0.0 - Vprev[i]) / (x[i] - Vprev[i])
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
    return Vout, Hout, xend, spk, etaend


@njit(cache=True, parallel=True)
def simulate_izh_xi(x0, xi, xi_node, xi_bin, pulses, seeds, sigma, tau_noise, T, dt, a, bb, c, d, Ib,
                    A, B, Es, ar, ad, beta, Vth, save_every, max_spikes, hid_node, eta0):
    Btr, nx = x0.shape
    N = A.shape[0]
    nsteps = int(round(T / dt)); nsave = nsteps // save_every + 1
    Vout = np.empty((Btr, nsave, N)); Hout = np.empty((Btr, nsave))
    xend = np.empty((Btr, nx)); etaend = np.zeros((Btr, N))
    spk = np.full((Btr, N, max_spikes), np.nan)
    dec = np.exp(-dt / tau_noise); kick = sigma * np.sqrt(1.0 - dec * dec)
    for b in prange(Btr):
        np.random.seed(seeds[b])
        x = x0[b].copy()
        xn = np.empty(nx); xm = np.empty(nx)
        k1 = np.empty(nx); k2 = np.empty(nx); k3 = np.empty(nx); k4 = np.empty(nx); xt = np.empty(nx)
        Iext = np.zeros(N); eta = np.zeros(N)
        nspk = np.zeros(N, dtype=np.int64)
        for i in range(N):
            eta[i] = eta0[b, i]
        isave = 0
        for step in range(nsteps + 1):
            t = step * dt
            for i in range(N):
                Iext[i] = _pulse_current(pulses[b], i, t) + eta[i]
            Iext[xi_node] += xi[b, step // xi_bin]
            if step % save_every == 0:
                for i in range(N):
                    Vout[b, isave, i] = x[i]
                Hout[b, isave] = x[N + hid_node]
                isave += 1
            if step == nsteps:
                break
            h_left = dt; t_now = t
            for _guard in range(4 * N + 4):
                _izh_rk4(x, h_left, Iext, N, a, bb, Ib, A, B, Es, ar, ad, beta, Vth, k1, k2, k3, k4, xt, xn)
                crossed = False
                for i in range(N):
                    if xn[i] >= VPEAK:
                        crossed = True
                if not crossed:
                    for q in range(nx): x[q] = xn[q]
                    break
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
                        x[i] = c[i]; x[N + i] += d[i]
                        if nspk[i] < max_spikes:
                            spk[b, i, nspk[i]] = t_now + hi
                        nspk[i] += 1
                t_now += hi; h_left -= hi
                if h_left <= 1e-12:
                    break
            if sigma > 0.0:
                for i in range(N):
                    eta[i] = dec * eta[i] + kick * np.random.randn()
        for q in range(nx):
            xend[b, q] = x[q]
        for i in range(N):
            etaend[b, i] = eta[i]
    return Vout, Hout, xend, spk, etaend


def run_xi(circ, x0, xi, pulses, seeds, T, eta0, dt=0.01, xi_node=0, xi_bin=10, sigma=0.0,
           tau_noise=5.0, save_every=10, max_spikes=512, hid_node=2):
    f = np.ascontiguousarray
    x0 = f(np.atleast_2d(x0), dtype=np.float64); xi = f(xi, dtype=np.float64)
    pulses = f(pulses, dtype=np.float64); seeds = f(seeds, dtype=np.int64); eta0 = f(eta0, dtype=np.float64)
    if circ.model == 'hh':
        out = simulate_hh_xi(x0, xi, xi_node, xi_bin, pulses, seeds, sigma, tau_noise, T, dt, circ.P,
                             circ.A, circ.B, circ.Es, circ.ar, circ.ad, circ.beta, circ.Vth, save_every,
                             max_spikes, hid_node, eta0)
    else:
        out = simulate_izh_xi(x0, xi, xi_node, xi_bin, pulses, seeds, sigma, tau_noise, T, dt, circ.a,
                              circ.b, circ.c, circ.d, circ.Ib, circ.A, circ.B, circ.Es, circ.ar, circ.ad,
                              circ.beta, circ.Vth, save_every, max_spikes, hid_node, eta0)
    return dict(zip(('V', 'H', 'x_end', 'spikes', 'eta_end'), out))
