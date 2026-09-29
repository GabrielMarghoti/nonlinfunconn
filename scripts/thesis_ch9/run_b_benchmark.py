"""
Benchmark of the propagation kernels against functional derivatives computed from
a trajectory and a probed trajectory (thesis Ch. 6-7, Hodgkin-Huxley).

B1  gating kernel sigma_{x,V}(t,t') of Eq. (sigma_x_V): analytic formula vs the numerical
    functional derivative delta x(t)/delta V(t') of the gate driven by the recorded spike
    trajectory; and the constant-tau approximation (tilde sigma).
B2  rest self kernel of node 1: measured G_11(tau) vs the linearized propagator
    [exp(J tau)]_VV / C, and the eigenvalues of J recovered from G_11 by a matrix-pencil fit.
B3  short-time structure of cross kernels at rest in the three chains: G_k1(tau) ~ c tau^d,
    with d and c predicted by the Jacobian graph, [J^d]_{k,V1} / (C d!).
B4  instantaneous membrane conductance along the spike from the initial decay of the
    self kernel, g_eff(t') = -C d/dtau ln G_11(t'+tau, t') at tau -> 0+, vs the model
    g_tot = g_l + g_Na m^3 h + g_K n^4.
"""
import numpy as np, json, math
from scipy.linalg import expm
import ch9_circuits as CC, ch9_models as M

res = {}
MOT = {'gap-gap': [('gap', 1, 0), ('gap', 2, 1)], 'gap-chem': [('gap', 1, 0), ('chem', 2, 1)],
       'chem-chem': [('chem', 1, 0), ('chem', 2, 1)]}
c = CC.chain('hh', MOT['gap-chem']); xeq = c.find_equilibrium(); N = 3
C, gNa, gK, gL, ENa, EK, EL = c.P
T_PUMP, A_PUMP = 5.0, 10.0

def jac(circ, x, eps=1e-6):
    n = len(x); J = np.zeros((n, n)); f0 = np.zeros(n); f1 = np.zeros(n); I0 = np.zeros(circ.N)
    for k in range(n):
        xp = x.copy(); xm = x.copy(); xp[k] += eps; xm[k] -= eps
        M._hh_rhs(xp, I0, circ.N, circ.P, circ.A, circ.B, circ.Es, circ.ar, circ.ad, circ.beta, circ.Vth, f0)
        M._hh_rhs(xm, I0, circ.N, circ.P, circ.A, circ.B, circ.Es, circ.ar, circ.ad, circ.beta, circ.Vth, f1)
        J[:, k] = (f0 - f1) / (2 * eps)
    return J

# ---------------------------------------------------------------- reference spike (node 1)
DT = 0.001; T = 30.0
pul = CC.pulse_table([(0, T_PUMP, 3.0, A_PUMP)])[None]
r = c.run(xeq, pul, [0], T, DT, save_every=1)
V = r['V'][0, :, 0]; t = np.arange(len(V)) * DT
# gates of node 1 along the reference: re-integrate the same system storing full state
def full_traj():
    x = xeq.copy(); X = np.empty((len(V), len(x))); X[0] = x
    Iext = np.zeros(N); k1 = np.zeros(len(x)); k2 = np.zeros(len(x)); k3 = np.zeros(len(x)); k4 = np.zeros(len(x))
    for s in range(len(V) - 1):
        ts = s * DT; Iext[:] = 0; Iext[0] = A_PUMP if (T_PUMP <= ts < T_PUMP + 3.0) else 0.0
        args = (Iext, N, c.P, c.A, c.B, c.Es, c.ar, c.ad, c.beta, c.Vth)
        M._hh_rhs(x, *args, k1); M._hh_rhs(x + 0.5 * DT * k1, *args, k2)
        M._hh_rhs(x + 0.5 * DT * k2, *args, k3); M._hh_rhs(x + DT * k3, *args, k4)
        x = x + DT * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0; X[s + 1] = x
    return X
X = full_traj()
assert np.max(np.abs(X[:, 0] - V)) < 1e-9
m1, h1, n1 = X[:, N], X[:, 2 * N], X[:, 3 * N]

# ---------------------------------------------------------------- B1: gating kernel benchmark
def rates(v):
    return np.array(M.hh_rates(float(v)))
def gate_run(which, Vtraj, i0, amp):
    """integrate gate x driven by Vtraj (+amp on step i0) with RK4, V linear between samples."""
    k = {'m': 0, 'h': 2, 'n': 4}[which]
    x = {'m': m1, 'h': h1, 'n': n1}[which][0]
    out = np.empty(len(Vtraj)); out[0] = x
    Vp = Vtraj                                               # perturbation: +amp during the single step [t_i0, t_i0+dt]
    f = lambda v, x: (lambda a: a[k] * (1 - x) - a[k + 1] * x)(rates(v))
    for s in range(len(Vtraj) - 1):
        v0, v1 = Vp[s], Vp[s + 1]; vm = 0.5 * (v0 + v1)
        if s == i0:                                          # exact rectangle: constant V + amp over the step
            v0 = vm = v1 = Vtraj[s] + amp
        k1 = f(v0, x); k2 = f(vm, x + 0.5 * DT * k1); k3 = f(vm, x + 0.5 * DT * k2); k4 = f(v1, x + DT * k3)
        x = x + DT * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0; out[s + 1] = x
    return out
def sigma_analytic(which, i0, tt):
    k = {'m': 0, 'h': 2, 'n': 4}[which]; xg = {'m': m1, 'h': h1, 'n': n1}[which]
    ab = np.array([rates(v) for v in V])
    lam = ab[:, k] + ab[:, k + 1]
    ev = 1e-4; dp = (rates(V[i0] + ev) - rates(V[i0] - ev)) / (2 * ev)
    amp = dp[k] * (1 - xg[i0]) - dp[k + 1] * xg[i0]
    I = np.concatenate([[0], np.cumsum(0.5 * (lam[i0:-1] + lam[i0 + 1:]) * DT)])
    s = np.zeros(len(V)); s[i0:] = np.exp(-I) * amp
    tauv = 1.0 / lam[i0]; xinf = ab[:, k] / lam
    # constant-tau approximation: exp(-(t-t')/tau_x(V(t'))) x_inf'(V(t'))/tau_x(V(t'))
    a0, b0 = rates(V[i0])[k], rates(V[i0])[k + 1]; da, db = dp[k], dp[k + 1]
    dxinf = (da * (a0 + b0) - a0 * (da + db)) / (a0 + b0) ** 2
    st = np.zeros(len(V)); st[i0:] = np.exp(-(t[i0:] - t[i0]) / tauv) * dxinf / tauv
    return s, st
B1 = {}; b1_curves = {}
probe_times = [5.5, 7.0, 7.3, 8.0, 10.0]                     # before, upstroke, peak region, repolarization, AHP
for which in ('m', 'h', 'n'):
    errs, errs_t = [], []
    for tp in probe_times:
        i0 = int(round(tp / DT)); amp = 0.01
        xp = gate_run(which, V, i0, amp); xm = gate_run(which, V, i0, -amp)
        num = (xp - xm) / (2 * amp * DT)
        an, ap = sigma_analytic(which, i0, t)
        w = slice(i0 + 2, min(i0 + int(10 / DT), len(V)))
        errs.append(float(np.linalg.norm(num[w] - an[w]) / np.linalg.norm(an[w])))
        errs_t.append(float(np.linalg.norm(ap[w] - an[w]) / np.linalg.norm(an[w])))
        b1_curves[f'{which}_{tp}'] = np.stack([t[w] - tp, num[w], an[w], ap[w]])
    B1[which] = dict(probe_times=probe_times, rel_err_analytic=errs, rel_err_constant_tau=errs_t)
    print('B1', which, np.round(errs, 6), np.round(errs_t, 3), flush=True)
res['B1'] = B1

# ---------------------------------------------------------------- B2: rest spectrum of node 1
Jfull = jac(c, xeq); idx = [0, N, 2 * N, 3 * N]; J1 = Jfull[np.ix_(idx, idx)]
ev1 = np.linalg.eigvals(J1)
DT2 = 0.001; T2 = 40.0; q_amp = 20.0                          # impulse: amp * dt = q = 0.02
p1 = np.zeros((2, 1, 4)); p1[0, 0] = [0, 0.0, 0.5 * DT2, q_amp]; p1[1, 0] = [0, 0.0, 0.5 * DT2, -q_amp]
rr = c.run(np.tile(xeq, (2, 1)), p1, [0, 0], T2, DT2, save_every=10)
G11 = (rr['V'][0, :, 0] - rr['V'][1, :, 0]) / (2 * q_amp * DT2); tau2 = np.arange(len(G11)) * DT2 * 10
Gth = np.array([expm(J1 * tt)[0, 0] for tt in tau2]) / C
# the impulse acts during the first step: compare from tau >= 0.01 ms
w = tau2 >= 0.01
res['B2'] = dict(rel_err_linearized=float(np.linalg.norm(G11[w] - Gth[w]) / np.linalg.norm(Gth[w])),
                 eig_J=[[float(e.real), float(e.imag)] for e in ev1])
# matrix pencil on G11 (tau in [0.05, 30] ms, subsampled to 0.1 ms)
def pencil(y, dt, p):
    n = len(y); L = n // 2
    Y = np.array([y[i:i + L + 1] for i in range(n - L)])
    U, S, Vh = np.linalg.svd(Y, full_matrices=False); Vp = Vh[:p].conj().T
    V1, V2 = Vp[:-1], Vp[1:]
    z = np.linalg.eigvals(np.linalg.pinv(V1) @ V2)
    return np.log(z) / dt, S
sel = (tau2 >= 0.05) & (tau2 <= 30.0); ys = G11[sel][::10]
lam_fit, S = pencil(ys, 0.1, 4)
res['B2']['eig_fit'] = [[float(e.real), float(e.imag)] for e in lam_fit]
res['B2']['singular_values'] = [float(s) for s in S[:6] / S[0]]
print('B2', res['B2'], flush=True)

# ---------------------------------------------------------------- B3: short-time powers
B3 = {}; b3_curves = {}
DT3 = 1e-4; T3 = 0.05; q3 = 0.2
for name, links in MOT.items():
    cc = CC.chain('hh', links); x0 = cc.find_equilibrium(); J = jac(cc, x0)
    p = np.zeros((2, 1, 4)); p[0, 0] = [0, 0.0, 0.5 * DT3, q3 / DT3]; p[1, 0] = [0, 0.0, 0.5 * DT3, -q3 / DT3]
    rr = cc.run(np.tile(x0, (2, 1)), p, [0, 0], T3, DT3, save_every=1)
    G = (rr['V'][0] - rr['V'][1]) / (2 * q3); tt = np.arange(G.shape[0]) * DT3
    out = {}
    for k in (1, 2):
        # predicted order: first d with [J^d]_{k,0} != 0 (impulse into V1 gives 1/C in V1)
        Jd = np.eye(len(x0)); d = 0
        while True:
            d += 1; Jd = Jd @ J
            if abs(Jd[k, 0]) > 1e-12 or d > 8: break
        coef = Jd[k, 0] / C / math.factorial(d)
        te = tt - 0.5 * DT3                                    # time since the centre of the impulse
        good = (te > 0) & (np.abs(G[:, k]) > 1e-9) & (te <= 0.04)
        wfit = good & (te >= 2 * DT3)
        slope, icpt = np.polyfit(np.log(te[wfit][:40]), np.log(np.abs(G[wfit, k][:40])), 1)
        # leading coefficient: extrapolate G / te^d to te -> 0 with a quadratic in te
        cc2 = np.polyfit(te[wfit], G[wfit, k] / te[wfit] ** d, 2)
        cest = float(cc2[-1])
        out[f'G{k+1}1'] = dict(order_pred=d, slope_fit=float(slope), coef_pred=float(coef), coef_est=cest)
        b3_curves[f'{name}_G{k+1}1'] = np.stack([tt[1:], np.abs(G[1:, k])])
    B3[name] = out; print('B3', name, out, flush=True)
res['B3'] = B3

# ---------------------------------------------------------------- B4: conductance along the spike
DT4 = 0.001; tps = np.round(np.arange(4.5, 20.0 + 1e-9, 0.05), 3); n = len(tps); q4 = 0.02; T4 = tps.max() + 0.03
pp = np.zeros((2 * n, 2, 4))
for i, tp in enumerate(tps):
    for a, sg in enumerate((1, -1)):
        pp[2 * i + a, 0] = [0, T_PUMP, 3.0, A_PUMP]; pp[2 * i + a, 1] = [0, tp, 0.5 * DT4, sg * q4 / DT4]
rr = c.run(np.tile(xeq, (2 * n, 1)), pp, np.zeros(2 * n, int), T4, DT4, save_every=1)
Vv = rr['V'][:, :, 0].reshape(n, 2, -1)
geff = np.empty(n); gtot = np.empty(n)
for i, tp in enumerate(tps):
    i0 = int(round(tp / DT4)); G = (Vv[i, 0] - Vv[i, 1]) / (2 * q4)
    g = G[i0 + 1:i0 + 6]; tt = np.arange(1, 6) * DT4
    geff[i] = -C * np.polyfit(tt, np.log(g), 1)[0]
    gtot[i] = gL + gNa * m1[i0] ** 3 * h1[i0] + gK * n1[i0] ** 4
res['B4'] = dict(rel_err_median=float(np.median(np.abs(geff - gtot) / gtot)), rel_err_max=float(np.max(np.abs(geff - gtot) / gtot)),
                 gtot_rest=float(gtot[0]), gtot_max=float(gtot.max()), t_of_max=float(tps[np.argmax(gtot)]))
print('B4', res['B4'], flush=True)
json.dump(res, open('out/b_benchmark.json', 'w'), indent=1)
np.savez_compressed('out/b_benchmark.npz', t=t, V=V, tau2=tau2, G11=G11, Gth=Gth, tps=tps, geff=geff, gtot=gtot,
                    **{f'b1_{k}': v for k, v in b1_curves.items()}, **{f'b3_{k}': v for k, v in b3_curves.items()})
