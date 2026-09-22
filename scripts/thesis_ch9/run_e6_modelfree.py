"""
E6 -- is the propagation kernel model-free?

A weak Gaussian white-noise current xi(t) (intensity SIGP2) is injected into node 1
of the noisy gap->chem motif of Sec. 9.4, and only xi and the voltages are used to
estimate the kernel. For Gaussian xi, Stein's lemma (Furutsu-Novikov) gives exactly

    E[ xi_b V_k(t_b + tau) | s(t_b) ] / Var(xi_b) = E[ dV_k(t_b + tau)/d xi_b | s(t_b) ]

for any s(t_b) fixed before the bin: the input-output covariance IS the
state-conditioned kernel, averaged over whatever is not in s, for any model.
Validation: the same average from pump-probe arms (Eq. 9.1) at states sampled
from the same process. Conditioning sets: the observed potential V3, the hidden
variable of node 3 (HH: n3, Izhikevich: u3), and the time since the last spike
of node 3 (observable, model-free).

usage: python3 run_e6_modelfree.py hh|izh
"""
import numpy as np, json, time, sys
import ch9_circuits as CC
import ch9_modelfree as MF

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'hh'
CFG = {'hh': dict(kw={}, sigma=1.0, M_pp=6000), 'izh': dict(kw=dict(g_gap=0.5, g_chem=0.075), sigma=0.5, M_pp=8000)}[MODEL]
SIGP2 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
DB, XB, DT = 0.1, 10, 0.01            # xi intensity, bin (ms), bin in steps, step
VARXI = SIGP2 / DB
NTR, CHUNK, NCH, BURN = 64, 5000.0, 12, 500.0
LMIN, LMAX = -100, 600                            # lags in bins (-10 .. 60 ms)
NQ = 5
AMP_PP = 0.2                                      # pump-probe pulse amplitude over one bin (q = 0.02)
rng = np.random.default_rng(20260922 + (0 if MODEL == 'hh' else 1))

c = CC.chain(MODEL, [('gap', 1, 0), ('chem', 2, 1)], **CFG['kw']); xeq = c.find_equilibrium()
N = 3; sig = CFG['sigma']
nb = int(round(CHUNK / DB)); lags = np.arange(LMIN, LMAX + 1); tau = lags * DB
nfft = 1 << int(np.ceil(np.log2(nb + LMAX - LMIN + 10)))
valid = np.zeros(nb, bool); valid[-LMIN:nb - LMAX - 1] = True
NOP = np.zeros((NTR, 1, 4))

def tss(spk3, last, tb):
    """time since the last spike of node 3 before each bin start (spikes of this chunk + carry-over)."""
    out = np.empty((NTR, len(tb)))
    for b in range(NTR):
        s = spk3[b][~np.isnan(spk3[b])]
        s = np.concatenate([[last[b]], s])
        k = np.searchsorted(s, tb, side='right') - 1
        out[b] = tb - s[k]
    return out

def xcorr(w, V):
    """C[b, l] = sum_i w[b,i] V[b,i+l] for l in lags."""
    X = np.fft.rfft(w, nfft, axis=1); Y = np.fft.rfft(V, nfft, axis=1)
    C = np.fft.irfft(np.conj(X) * Y, nfft, axis=1)
    return np.concatenate([C[:, nfft + LMIN:], C[:, :LMAX + 1]], axis=1)

t0 = time.time()
# ---- burn-in with the probe noise on
x = np.tile(xeq, (NTR, 1)); eta = sig * rng.standard_normal((NTR, N))
seeds = rng.integers(1, 2 ** 31, NTR)
xi = rng.normal(0, np.sqrt(VARXI), (NTR, int(BURN / DB) + 1))
r = MF.run_xi(c, x, xi, NOP, seeds, BURN, eta, sigma=sig)
x, eta = r['x_end'], r['eta_end']
last = np.full(NTR, -1e4)
for b in range(NTR):
    s = r['spikes'][b, 2]; s = s[~np.isnan(s)]
    if len(s): last[b] = s[-1] - BURN

feats = ('V3', 'hid', 'tss')
edges = {}; Vmean = None
S_all = []                                 # per trial-chunk sums, nodes 1..3
S_grp = {f: np.zeros((NQ, 2, len(lags))) for f in feats}
P_grp = {f: [] for f in feats}; Pn_grp = {f: [] for f in feats}
n_grp = {f: np.zeros((NQ, 2)) for f in feats}
S_joint = {j: np.zeros((3, 3, 2, len(lags))) for j in ('V3xtss', 'V3xhid')}
n_joint = {j: np.zeros((3, 3, 2)) for j in ('V3xtss', 'V3xhid')}
edges3 = {}
rates = np.zeros(N); n_all = 0
for ch in range(NCH):
    seeds = rng.integers(1, 2 ** 31, NTR)
    xi = rng.normal(0, np.sqrt(VARXI), (NTR, nb + 1))
    r = MF.run_xi(c, x, xi, NOP, seeds, CHUNK, eta, sigma=sig)
    V = r['V'][:, :nb, :]; H = r['H'][:, :nb]
    tb = np.arange(nb) * DB
    F = {'V3': V[:, :, 2], 'hid': H, 'tss': tss(r['spikes'][:, 2, :], last, tb)}
    for i in range(N):
        rates[i] += np.sum(~np.isnan(r['spikes'][:, i, :])) / (NTR * CHUNK / 1000.0) / NCH
    for b in range(NTR):
        s = r['spikes'][b, 2]; s = s[~np.isnan(s)]
        last[b] = (s[-1] if len(s) else last[b]) - CHUNK
    x, eta = r['x_end'], r['eta_end']
    if ch == 0:
        Vmean = V.reshape(-1, N).mean(0)
        for f in feats:
            edges[f] = np.quantile(F[f][:, valid], np.linspace(0, 1, NQ + 1)[1:-1])
            edges3[f] = np.quantile(F[f][:, valid], [1 / 3, 2 / 3])
    Vc = V - V[:, valid, :].mean(axis=1, keepdims=True)      # per-trial centring removes the common offset
    w = xi[:, :nb] * valid
    S_all.append(np.stack([xcorr(w, Vc[:, :, k]) for k in range(N)], axis=-1))   # (NTR, lags, N)
    n_all += valid.sum()
    half = np.arange(NTR) % 2
    for f in feats:
        g = np.digitize(F[f], edges[f])
        Cq, nq = [], []
        for q in range(NQ):
            m = (g == q) & valid
            C = xcorr(xi[:, :nb] * m, Vc[:, :, 2])
            Cq.append(C); nq.append(m.sum(1))
            for h in (0, 1):
                S_grp[f][q, h] += C[half == h].sum(0); n_grp[f][q, h] += m[half == h].sum()
        P_grp[f].append(np.stack(Cq, 1)); Pn_grp[f].append(np.stack(nq, 1))
    for jn, (f1, f2) in (('V3xtss', ('V3', 'tss')), ('V3xhid', ('V3', 'hid'))):
        g1 = np.digitize(F[f1], edges3[f1]); g2 = np.digitize(F[f2], edges3[f2])
        for q1 in range(3):
            for q2 in range(3):
                m = (g1 == q1) & (g2 == q2) & valid
                C = xcorr(xi[:, :nb] * m, Vc[:, :, 2])
                for h in (0, 1):
                    S_joint[jn][q1, q2, h] += C[half == h].sum(0); n_joint[jn][q1, q2, h] += m[half == h].sum()
    # correlation of the model-free proxy with the hidden variable
    if ch == 0:
        from scipy.stats import spearmanr
        rho_tss_hid = float(spearmanr(F['tss'][:, valid].ravel()[::50], F['hid'][:, valid].ravel()[::50])[0])
        rho_V3_hid = float(spearmanr(F['V3'][:, valid].ravel()[::50], F['hid'][:, valid].ravel()[::50])[0])
    print(f'[{MODEL}] chunk {ch+1}/{NCH} ({time.time()-t0:.0f}s)', flush=True)

S_all = np.concatenate(S_all)                      # (NTR*NCH, lags, N), each over valid.sum() bins
per = S_all / (valid.sum() * SIGP2)                # per trial-chunk kernel estimate, per unit charge
K_all = per.mean(0); K_se = per.std(0, ddof=1) / np.sqrt(per.shape[0])
Kg = {f: S_grp[f] / (n_grp[f][..., None] * SIGP2) for f in feats}
Kg_se = {}
for f in feats:
    Pc = np.concatenate(P_grp[f]); Pn = np.concatenate(Pn_grp[f])          # (trials, NQ, lags), (trials, NQ)
    w_ = Pn / Pn.sum(0)                                                     # ratio-estimator SE
    kq = Pc.sum(0) / (Pn.sum(0)[:, None] * SIGP2)
    resid = Pc / SIGP2 - Pn[..., None] * kq[None]
    Kg_se[f] = np.sqrt((resid ** 2).sum(0)) / Pn.sum(0)[:, None]
Kj = {j: S_joint[j] / (n_joint[j][..., None] * SIGP2) for j in S_joint}

# ---- cross-fitted fraction of kernel energy that varies with the conditioning variable
pos = tau >= 0
def spread(KA, KB, w):
    mA = np.tensordot(w, KA, 1); mB = np.tensordot(w, KB, 1)
    num = sum(w[q] * np.dot(KA[q][pos] - mA[pos], KB[q][pos] - mB[pos]) for q in range(len(w)))
    return float(num / np.dot(mA[pos], mB[pos]))
res = dict(model=MODEL, sigma=sig, sigp2=SIGP2, T_total_s=NTR * NCH * CHUNK / 1000, rates_Hz=rates.tolist(),
           rho_tss_hid=rho_tss_hid, rho_V3_hid=rho_V3_hid)
res['spread'] = {}
for f in feats:
    w = n_grp[f].sum(1) / n_grp[f].sum()
    res['spread'][f] = spread(Kg[f][:, 0], Kg[f][:, 1], w)
for j in Kj:
    w = n_joint[j].sum(-1).ravel() / n_joint[j].sum()
    res['spread'][j] = spread(Kj[j][:, :, 0].reshape(9, -1), Kj[j][:, :, 1].reshape(9, -1), w)
neg = tau < -0.5
res['acausal_max_abs_z'] = float(np.max(np.abs(K_all[neg, 2] / K_se[neg, 2])))
res['acausal_rms_z'] = float(np.sqrt(np.mean((K_all[neg, 2] / K_se[neg, 2]) ** 2)))
print(json.dumps(res), flush=True)

# ---- pump-probe validation at states sampled from the same process
M = CFG['M_pp']; nchain = 50; nseg = M // nchain
xs = np.tile(x[:1], (nchain, 1)); es = np.tile(eta[:1], (nchain, 1))
xs = x[np.arange(nchain) % NTR].copy(); es = eta[np.arange(nchain) % NTR].copy()
lastc = last[np.arange(nchain) % NTR].copy()
X, E, FT = [], [], []
for sgi in range(nseg):
    seeds = rng.integers(1, 2 ** 31, nchain)
    xi = rng.normal(0, np.sqrt(VARXI), (nchain, 1001))
    r = MF.run_xi(c, xs, xi, NOP[:nchain], seeds, 100.0, es, sigma=sig)
    xs, es = r['x_end'], r['eta_end']
    tl = np.empty(nchain)
    for b in range(nchain):
        s = r['spikes'][b, 2]; s = s[~np.isnan(s)]
        lastc[b] = (s[-1] if len(s) else lastc[b]) - 100.0
        tl[b] = -lastc[b]
    hid_idx = (3 * N + 2) if MODEL == 'hh' else (N + 2)
    X.append(xs.copy()); E.append(es.copy()); FT.append(np.c_[xs[:, 2], xs[:, hid_idx], tl])
X = np.concatenate(X); E = np.concatenate(E); FT = np.concatenate(FT)
T_PP = 60.0; nbp = int(T_PP / DB)
xi = rng.normal(0, np.sqrt(VARXI), (M, nbp + 1)); seeds = rng.integers(1, 2 ** 31, M)
pul = np.zeros((M, 1, 4)); pul[:, 0] = [0, 0.0, 0.095, AMP_PP]
Vp = MF.run_xi(c, X, xi, pul, seeds, T_PP, E, sigma=sig)['V'][:, :, 2]
pul[:, 0, 3] = -AMP_PP
Vm = MF.run_xi(c, X, xi, pul, seeds, T_PP, E, sigma=sig)['V'][:, :, 2]
Gpp = (Vp - Vm) / (2 * AMP_PP * DB)                  # per unit charge, tau = 0 .. 60 ms
# compare window integrals (2.5 ms windows): a shifted spike or reset moves area only at
# window edges, so these are far less heavy-tailed than point values
WIN = 25; nw = nbp // WIN
Ipp = Gpp[:, 1:1 + nw * WIN].reshape(M, nw, WIN).sum(-1) * DB          # (M, nw)
Ist = per[:, lags >= 1, 2][:, :nw * WIN].reshape(per.shape[0], nw, WIN).sum(-1) * DB
a_m, a_se = Ipp.mean(0), Ipp.std(0, ddof=1) / np.sqrt(M)
b_m, b_se = Ist.mean(0), Ist.std(0, ddof=1) / np.sqrt(Ist.shape[0])
z = (a_m - b_m) / np.sqrt(a_se ** 2 + b_se ** 2)
res['win_ms'] = WIN * DB
res['pp_vs_stein_z_rms'] = float(np.sqrt(np.mean(z ** 2))); res['pp_vs_stein_z_max'] = float(np.max(np.abs(z)))
res['pp_vs_stein_rel_L2_windows'] = float(np.linalg.norm(a_m - b_m) / np.linalg.norm(b_m))
res['se_ratio_pp_over_stein_median'] = float(np.median(a_se / b_se))
res['pp_trials'] = int(M); res['stein_hours_of_data'] = NTR * NCH * CHUNK / 3.6e6
gh = np.digitize(FT[:, 1], edges['hid'])
Kh = Kg['hid'].mean(1)[:, lags >= 1][:, :nw * WIN].reshape(NQ, nw, WIN).sum(-1) * DB
Ih = np.stack([Ipp[gh == q].mean(0) for q in range(NQ)]); Ih_se = np.stack([Ipp[gh == q].std(0, ddof=1) / np.sqrt((gh == q).sum()) for q in range(NQ)])
Kh_se = np.sqrt((Kg_se['hid'][:, lags >= 1][:, :nw * WIN].reshape(NQ, nw, WIN) ** 2).sum(-1)) * DB * np.sqrt(WIN)   # upper bound (lags correlated)
Kh_se = np.minimum(Kh_se, b_se[None, :] * np.sqrt(n_grp['hid'].sum() / n_grp['hid'].sum(1))[:, None])
zc = (Ih - Kh) / np.sqrt(Ih_se ** 2 + Kh_se ** 2)
res['cond_hid_z_rms'] = float(np.sqrt(np.mean(zc ** 2))); res['cond_hid_z_max'] = float(np.max(np.abs(zc)))
res['M_pp'] = int(M)
res['Gpp_single_trial_kurtosis_at_peak'] = float((lambda y: ((y - y.mean()) ** 4).mean() / y.var() ** 2)(Gpp[:, np.argmax(np.abs(Gpp.mean(0)))]))
Gpp_mean = Gpp.mean(0); Gpp_se = Gpp.std(0, ddof=1) / np.sqrt(M)
np.savez_compressed(f'out/e6_{MODEL}.npz', tau=tau, K_all=K_all, K_se=K_se, Kg_V3=Kg['V3'], Kg_hid=Kg['hid'],
                    Kg_tss=Kg['tss'], Kj_V3xtss=Kj['V3xtss'], Kj_V3xhid=Kj['V3xhid'], edges_V3=edges['V3'],
                    edges_hid=edges['hid'], edges_tss=edges['tss'], Gpp_mean=Gpp_mean, Gpp_se=Gpp_se,
                    Gpp_hid=np.stack([Gpp[gh == q].mean(0) for q in range(NQ)]), n_grp_hid=n_grp['hid'],
                    Kg_V3_se=Kg_se['V3'], Kg_hid_se=Kg_se['hid'], Kg_tss_se=Kg_se['tss'], rates=rates,
                    Ipp_mean=a_m, Ipp_se=a_se, Ist_mean=b_m, Ist_se=b_se, Ih=Ih, Ih_se=Ih_se, Kh=Kh, K_per=per[:, :, 2].astype(np.float32))
json.dump(res, open(f'out/e6_{MODEL}.json', 'w'), indent=1)
print(json.dumps(res), f'({time.time()-t0:.0f}s)', flush=True)
