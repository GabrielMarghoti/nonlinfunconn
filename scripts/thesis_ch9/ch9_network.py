"""
ch9_network.py -- globally coupled Izhikevich network of Chapter 5 (Eqs. 5.1-5.3)

    dv_i/dt = 0.04 v_i^2 + 5 v_i + 140 - u_i + I_b + gamma <v> + I_stim,i
    du_i/dt = a_i (b v_i - u_i),     reset v_i >= 30 -> (c, u_i + d)

integrated with RK4 and event-accurate resets (bisection on a whole-system
substep), so that the mean field never contains voltages beyond v_peak.
Returns burst onsets for the order parameter, the mean field, and full states
at requested checkpoint times, so that stimulation experiments can branch from
the unperturbed trajectory with identical initial conditions.
"""
import numpy as np
from numba import njit, prange

VPEAK = 30.0


@njit(cache=True, inline='always')
def _rhs(v, u, a, b, Ib, gamma, stim, dv, du):
    N = v.shape[0]
    mf = 0.0
    for i in range(N):
        mf += v[i]
    mf /= N
    for i in range(N):
        dv[i] = 0.04 * v[i] * v[i] + 5.0 * v[i] + 140.0 - u[i] + Ib + gamma * mf + stim[i]
        du[i] = a[i] * (b * v[i] - u[i])


@njit(cache=True)
def _rk4(v, u, h, a, b, Ib, gamma, stim, vo, uo, w):
    N = v.shape[0]
    k1v, k1u, k2v, k2u, k3v, k3u, k4v, k4u, vt, ut = w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9]
    _rhs(v, u, a, b, Ib, gamma, stim, k1v, k1u)
    for i in range(N):
        vt[i] = v[i] + 0.5 * h * k1v[i]; ut[i] = u[i] + 0.5 * h * k1u[i]
    _rhs(vt, ut, a, b, Ib, gamma, stim, k2v, k2u)
    for i in range(N):
        vt[i] = v[i] + 0.5 * h * k2v[i]; ut[i] = u[i] + 0.5 * h * k2u[i]
    _rhs(vt, ut, a, b, Ib, gamma, stim, k3v, k3u)
    for i in range(N):
        vt[i] = v[i] + h * k3v[i]; ut[i] = u[i] + h * k3u[i]
    _rhs(vt, ut, a, b, Ib, gamma, stim, k4v, k4u)
    for i in range(N):
        vo[i] = v[i] + h * (k1v[i] + 2 * k2v[i] + 2 * k3v[i] + k4v[i]) / 6.0
        uo[i] = u[i] + h * (k1u[i] + 2 * k2u[i] + 2 * k3u[i] + k4u[i]) / 6.0


@njit(cache=True)
def _step(v, u, t, dt, a, b, c, d, Ib, gamma, stim, w, vn, un, vm, um, spk_t, spk_i, nspk, cap):
    """advance (v,u) by dt with event-accurate resets; records spikes."""
    N = v.shape[0]
    h_left = dt; t_now = t
    for _g in range(4 * N + 4):
        _rk4(v, u, h_left, a, b, Ib, gamma, stim, vn, un, w)
        crossed = False
        for i in range(N):
            if vn[i] >= VPEAK:
                crossed = True
                break
        if not crossed:
            for i in range(N):
                v[i] = vn[i]; u[i] = un[i]
            return nspk
        lo = 0.0; hi = h_left
        for _it in range(40):
            mid = 0.5 * (lo + hi)
            _rk4(v, u, mid, a, b, Ib, gamma, stim, vm, um, w)
            anyc = False
            for i in range(N):
                if vm[i] >= VPEAK:
                    anyc = True
                    break
            if anyc:
                hi = mid
            else:
                lo = mid
        _rk4(v, u, hi, a, b, Ib, gamma, stim, vm, um, w)
        for i in range(N):
            v[i] = vm[i]; u[i] = um[i]
            if v[i] >= VPEAK:
                v[i] = c; u[i] += d
                if nspk < cap:
                    spk_t[nspk] = t_now + hi; spk_i[nspk] = i
                nspk += 1
        t_now += hi; h_left -= hi
        if h_left <= 1e-12:
            break
    return nspk


@njit(cache=True)
def run_free(v0, u0, a, b, c, d, Ib, gamma, T, dt, mf_every, ckpt_every, cap):
    """Long unperturbed run. Returns spikes, subsampled mean field, checkpoints."""
    N = v0.shape[0]
    v = v0.copy(); u = u0.copy()
    w = np.empty((10, N)); vn = np.empty(N); un = np.empty(N); vm = np.empty(N); um = np.empty(N)
    stim = np.zeros(N)
    nsteps = int(round(T / dt))
    spk_t = np.empty(cap); spk_i = np.empty(cap, dtype=np.int64); nspk = 0
    mf = np.empty(nsteps // mf_every + 1)
    nck = nsteps // ckpt_every + 1
    ck_v = np.empty((nck, N)); ck_u = np.empty((nck, N)); ck_t = np.empty(nck)
    imf = 0; ick = 0
    for s in range(nsteps + 1):
        t = s * dt
        if s % mf_every == 0:
            m = 0.0
            for i in range(N): m += v[i]
            mf[imf] = m / N; imf += 1
        if s % ckpt_every == 0:
            for i in range(N):
                ck_v[ick, i] = v[i]; ck_u[ick, i] = u[i]
            ck_t[ick] = t; ick += 1
        if s == nsteps:
            break
        nspk = _step(v, u, t, dt, a, b, c, d, Ib, gamma, stim, w, vn, un, vm, um, spk_t, spk_i, nspk, cap)
    n = min(nspk, cap)
    return spk_t[:n], spk_i[:n], mf, ck_t, ck_v, ck_u, v, u


@njit(cache=True, parallel=True)
def run_branches(V0, U0, a, b, c, d, Ib, gamma, stim_amp, stim_dur, stim_mask, T, dt, out_every):
    """
    Branch many trials from given initial states; trial k receives a pulse of
    amplitude stim_amp[k] and duration stim_dur at t=0 on neurons where
    stim_mask is 1. Returns the mean field (K, nt) sampled every out_every steps.
    """
    K, N = V0.shape
    nsteps = int(round(T / dt)); nt = nsteps // out_every + 1
    MF = np.empty((K, nt))
    for k in prange(K):
        v = V0[k].copy(); u = U0[k].copy()
        w = np.empty((10, N)); vn = np.empty(N); un = np.empty(N); vm = np.empty(N); um = np.empty(N)
        stim = np.zeros(N)
        dummy_t = np.empty(1); dummy_i = np.empty(1, dtype=np.int64)
        it = 0
        for s in range(nsteps + 1):
            t = s * dt
            if s % out_every == 0:
                m = 0.0
                for i in range(N): m += v[i]
                MF[k, it] = m / N; it += 1
            if s == nsteps:
                break
            on = 1.0 if t < stim_dur else 0.0
            for i in range(N):
                stim[i] = stim_amp[k] * stim_mask[i] * on
            _step(v, u, t, dt, a, b, c, d, Ib, gamma, stim, w, vn, un, vm, um, dummy_t, dummy_i, 0, 0)
    return MF


def burst_onsets(spk_t, spk_i, N, min_ibi=15.0):
    """a spike is a burst onset if the previous spike of that neuron is > min_ibi ms earlier"""
    ons = [[] for _ in range(N)]
    last = np.full(N, -1e9)
    for t, i in zip(spk_t, spk_i):
        if t - last[i] > min_ibi:
            ons[i].append(t)
        last[i] = t
    return [np.array(o) for o in ons]


def order_parameter(onsets, idx, t_grid):
    """Kuramoto R_g(t) (thesis Eqs. 4.1-4.2) for neuron subset idx on t_grid; NaN where undefined."""
    Z = np.zeros(len(t_grid), complex); ok = np.ones(len(t_grid), bool)
    for i in idx:
        o = onsets[i]
        k = np.searchsorted(o, t_grid, side='right')
        valid = (k > 0) & (k < len(o))
        kk = np.clip(k, 1, len(o) - 1)
        th = 2 * np.pi * (t_grid - o[kk - 1]) / (o[kk] - o[kk - 1])
        Z += np.where(valid, np.exp(1j * th), 0)
        ok &= valid
    R = np.abs(Z) / len(idx)
    R[~ok] = np.nan
    return R


# --------------------------------------------------------------------------
# Branch runs with a synaptically filtered mean field, integrated in continuous
# time alongside the network:  dz/dt = (<v> - z) / tau_s.  z is continuous across
# resets and is integrated up to the exact crossing time, so its response to a
# stimulus is regular even though <v> itself carries Dirac atoms at the shifted
# reset times.
# --------------------------------------------------------------------------
@njit(cache=True, inline='always')
def _rhs_z(v, u, z, a, b, Ib, gamma, stim, tau_s, dv, du):
    N = v.shape[0]
    mf = 0.0
    for i in range(N):
        mf += v[i]
    mf /= N
    for i in range(N):
        dv[i] = 0.04 * v[i] * v[i] + 5.0 * v[i] + 140.0 - u[i] + Ib + gamma * mf + stim[i]
        du[i] = a[i] * (b * v[i] - u[i])
    return (mf - z) / tau_s


@njit(cache=True)
def _rk4_z(v, u, z, h, a, b, Ib, gamma, stim, tau_s, vo, uo, w):
    N = v.shape[0]
    k1v, k1u, k2v, k2u, k3v, k3u, k4v, k4u, vt, ut = w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9]
    kz1 = _rhs_z(v, u, z, a, b, Ib, gamma, stim, tau_s, k1v, k1u)
    for i in range(N):
        vt[i] = v[i] + 0.5 * h * k1v[i]; ut[i] = u[i] + 0.5 * h * k1u[i]
    kz2 = _rhs_z(vt, ut, z + 0.5 * h * kz1, a, b, Ib, gamma, stim, tau_s, k2v, k2u)
    for i in range(N):
        vt[i] = v[i] + 0.5 * h * k2v[i]; ut[i] = u[i] + 0.5 * h * k2u[i]
    kz3 = _rhs_z(vt, ut, z + 0.5 * h * kz2, a, b, Ib, gamma, stim, tau_s, k3v, k3u)
    for i in range(N):
        vt[i] = v[i] + h * k3v[i]; ut[i] = u[i] + h * k3u[i]
    kz4 = _rhs_z(vt, ut, z + h * kz3, a, b, Ib, gamma, stim, tau_s, k4v, k4u)
    for i in range(N):
        vo[i] = v[i] + h * (k1v[i] + 2 * k2v[i] + 2 * k3v[i] + k4v[i]) / 6.0
        uo[i] = u[i] + h * (k1u[i] + 2 * k2u[i] + 2 * k3u[i] + k4u[i]) / 6.0
    return z + h * (kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6.0


@njit(cache=True, parallel=True)
def run_branches_z(V0, U0, Z0, a, b, c, d, Ib, gamma, stim_amp, stim_dur, stim_mask, T, dt, out_every, tau_s):
    """
    Trial k starts from (V0[k], U0[k], Z0[k]) and receives stim_amp[k] on the
    neurons flagged in stim_mask during [0, stim_dur). Returns z (K, nt) and the
    number of spikes fired by each trial in [0, T).
    """
    K, N = V0.shape
    nsteps = int(round(T / dt)); nt = nsteps // out_every + 1
    Zout = np.empty((K, nt)); nsp = np.zeros(K, dtype=np.int64)
    for k in prange(K):
        v = V0[k].copy(); u = U0[k].copy(); z = Z0[k]
        w = np.empty((10, N)); vn = np.empty(N); un = np.empty(N); vm = np.empty(N); um = np.empty(N)
        stim = np.zeros(N)
        it = 0
        for s in range(nsteps + 1):
            t = s * dt
            if s % out_every == 0:
                Zout[k, it] = z; it += 1
            if s == nsteps:
                break
            on = 1.0 if t < stim_dur else 0.0
            for i in range(N):
                stim[i] = stim_amp[k] * stim_mask[i] * on
            h_left = dt
            for _g in range(4 * N + 4):
                zn = _rk4_z(v, u, z, h_left, a, b, Ib, gamma, stim, tau_s, vn, un, w)
                crossed = False
                for i in range(N):
                    if vn[i] >= VPEAK:
                        crossed = True
                        break
                if not crossed:
                    for i in range(N):
                        v[i] = vn[i]; u[i] = un[i]
                    z = zn
                    break
                lo = 0.0; hi = h_left
                for _it in range(40):
                    mid = 0.5 * (lo + hi)
                    _rk4_z(v, u, z, mid, a, b, Ib, gamma, stim, tau_s, vm, um, w)
                    anyc = False
                    for i in range(N):
                        if vm[i] >= VPEAK:
                            anyc = True
                            break
                    if anyc:
                        hi = mid
                    else:
                        lo = mid
                z = _rk4_z(v, u, z, hi, a, b, Ib, gamma, stim, tau_s, vm, um, w)
                for i in range(N):
                    v[i] = vm[i]; u[i] = um[i]
                    if v[i] >= VPEAK:
                        v[i] = c; u[i] += d; nsp[k] += 1
                h_left -= hi
                if h_left <= 1e-12:
                    break
    return Zout, nsp
