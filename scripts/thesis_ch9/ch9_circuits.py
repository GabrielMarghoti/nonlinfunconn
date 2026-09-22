"""
ch9_circuits.py -- motif construction, equilibrium, tau_0, and a thin wrapper
that runs pump-probe arms with common random numbers.

Parameters default to the Chapter 7 script
scripts/HH_julia/HH_NEGF_DELTA_SYNAPSE_3neurons.jl :
  HH:  C=1, gNa=120, gK=36, gL=0.5, ENa=50, EK=-77, EL=-55
  gap junction 1->2 : A = 0.5 (directed)
  chemical 2->3     : B = 0.5, a_r = a_d = 5 /ms, beta = 0.125 /mV, E_s = 0,
                      V_th set self-consistently to the presynaptic rest
"""
import numpy as np
from dataclasses import dataclass, field
import ch9_models as M

HH_P = np.array([1.0, 120.0, 36.0, 0.5, 50.0, -77.0, -55.0])   # C,gNa,gK,gL,ENa,EK,EL


@dataclass
class Circuit:
    name: str
    model: str                      # 'hh' or 'izh'
    N: int
    A: np.ndarray                   # gap:   I_i += A_ij (V_j - V_i)
    B: np.ndarray                   # chem:  I_i -= B_ij s_ij (V_i - Es_ij)
    Es: np.ndarray
    ar: np.ndarray
    ad: np.ndarray
    beta: np.ndarray
    Vth: np.ndarray = None
    # izhikevich node parameters (arrays of length N)
    a: np.ndarray = None
    b: np.ndarray = None
    c: np.ndarray = None
    d: np.ndarray = None
    Ib: np.ndarray = None
    P: np.ndarray = field(default_factory=lambda: HH_P.copy())
    xeq: np.ndarray = None
    tau0: np.ndarray = None

    @property
    def nx(self):
        return (4 if self.model == 'hh' else 2) * self.N + self.N * self.N

    @property
    def C(self):
        return self.P[0] if self.model == 'hh' else 1.0

    # ------------------------------------------------------------------
    def run(self, x0, pulses, seeds, T, dt=0.01, sigma=0.0, tau_noise=5.0,
            save_every=1, max_spikes=64, eta0=None):
        x0 = np.ascontiguousarray(np.atleast_2d(x0), dtype=np.float64)
        pulses = np.ascontiguousarray(pulses, dtype=np.float64)
        seeds = np.ascontiguousarray(seeds, dtype=np.int64)
        xeq = self.xeq if self.xeq is not None else x0[0]
        if self.model == 'hh':
            V, Q, xe, spk, ee = M.simulate_hh(x0, pulses, seeds, sigma, tau_noise, T, dt, self.P,
                                              self.A, self.B, self.Es, self.ar, self.ad, self.beta,
                                              self.Vth, xeq, save_every, max_spikes, 0.0,
                                              None if eta0 is None else np.ascontiguousarray(eta0, dtype=np.float64))
            return dict(V=V, Q=Q, x_end=xe, spikes=spk, eta_end=ee)
        V, Q, xe, spk, ee = M.simulate_izh(x0, pulses, seeds, sigma, tau_noise, T, dt,
                                           self.a, self.b, self.c, self.d, self.Ib,
                                           self.A, self.B, self.Es, self.ar, self.ad, self.beta,
                                           self.Vth, xeq, save_every, max_spikes,
                                           None if eta0 is None else np.ascontiguousarray(eta0, dtype=np.float64))
        return dict(V=V, Q=Q, x_end=xe, spikes=spk, eta_end=ee)

    # ------------------------------------------------------------------
    def find_equilibrium(self, T=400.0, dt=0.01, iters=6, tol=1e-9):
        """Relax the unstimulated circuit; V_th <- presynaptic rest (Ch. 7 script)."""
        N = self.N
        if self.model == 'hh':
            V0 = -60.0
            am, bm, ah, bh, an, bn = M.hh_rates(V0)
            x = np.concatenate([np.full(N, V0), np.full(N, am / (am + bm)),
                                np.full(N, ah / (ah + bh)), np.full(N, an / (an + bn)),
                                np.full(N * N, 1.0 / 3.0)])
        else:
            vs = np.array([np.roots([0.04, 5.0 - self.b[i], 140.0 + self.Ib[i]]).real.min()
                           for i in range(N)])
            x = np.concatenate([vs, self.b * vs, np.full(N * N, 1.0 / 3.0)])
        if self.Vth is None:
            self.Vth = np.tile(x[:N], (N, 1))
        nopulse = np.zeros((1, 1, 4))
        for _ in range(iters):
            r = self.run(x, nopulse, [0], T, dt, save_every=int(round(T / dt)))
            x_new = r['x_end'][0]
            self.Vth = np.tile(x_new[:N], (N, 1))          # V_th[i,j] = V_eq[j]
            drift = np.max(np.abs(x_new - x))
            x = x_new
            if drift < tol:
                break
        # stationarity check: one more relaxation must not move the state
        r = self.run(x, nopulse, [0], 200.0, dt, save_every=int(round(200.0 / dt)))
        self.eq_drift = float(np.max(np.abs(r['x_end'][0] - x)))
        self.xeq = x
        self._set_tau0()
        return x

    def _set_tau0(self):
        N, x = self.N, self.xeq
        s = x[(4 if self.model == 'hh' else 2) * N:].reshape(N, N)
        cpl = self.A.sum(axis=1) + (self.B * s).sum(axis=1)
        if self.model == 'hh':
            C, gNa, gK, gL = self.P[:4]
            V, m, h, n = x[:N], x[N:2 * N], x[2 * N:3 * N], x[3 * N:4 * N]
            gtot = gL + gNa * m ** 3 * h + gK * n ** 4 + cpl
            self.tau0 = C / gtot
        else:
            mu = 0.08 * x[:N] + 5.0
            self.tau0 = 1.0 / (-mu + cpl)
        return self.tau0

    def propagated_signal(self, Q, dt):
        return M.route_d_signal(np.ascontiguousarray(Q), self.tau0, self.C, dt)


# ----------------------------------------------------------------------
# motif factories
# ----------------------------------------------------------------------
def _syn_arrays(N, ar=5.0, ad=5.0, beta=0.125, Es=0.0):
    return (np.full((N, N), Es), np.full((N, N), ar), np.full((N, N), ad),
            np.full((N, N), beta))


def chain(model, links, N=3, g_gap=0.5, g_chem=0.5, izh_params=None, Es=0.0, **syn):
    """
    links: list of ('gap'|'chem', post, pre) with 0-based node indices, e.g.
           [('gap', 1, 0), ('chem', 2, 1)]  is the motif of Fig. 7.12
    """
    A = np.zeros((N, N)); B = np.zeros((N, N))
    for kind, post, pre in links:
        if kind == 'gap':
            A[post, pre] = g_gap
        else:
            B[post, pre] = g_chem
    Esa, ar, ad, beta = _syn_arrays(N, Es=Es, **syn)
    name = model + ':' + '-'.join(k for k, _, _ in links)
    if model == 'hh':
        return Circuit(name, 'hh', N, A, B, Esa, ar, ad, beta)
    p = dict(a=0.017, b=0.20, c=-50.0, d=2.0, Ib=2.0)
    if izh_params:
        p.update(izh_params)
    arr = {k: np.full(N, float(v)) for k, v in p.items()}
    return Circuit(name, 'izh', N, A, B, Esa, ar, ad, beta, **arr)


def pulse_table(rows, Pmax=None):
    """rows: list of (node, t_on, duration, amplitude)."""
    Pmax = Pmax or max(1, len(rows))
    tab = np.zeros((Pmax, 4))
    for k, r in enumerate(rows):
        tab[k] = r
    return tab
