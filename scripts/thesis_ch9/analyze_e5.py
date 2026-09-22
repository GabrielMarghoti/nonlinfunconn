"""Analysis of the E5 probes: state-conditioned response functions of the Chapter 5 network."""
import numpy as np, json, sys
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

def cv_r2(S, y, groups, seed=0):
    pred = np.empty_like(y)
    for tr, te in GroupKFold(5).split(S, y, groups):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                          early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
                                          random_state=seed)
        m.fit(S[tr], y[tr]); pred[te] = m.predict(S[te])
    return float(1 - np.mean((y - pred) ** 2) / np.var(y)), pred

def boot_ci(y, pred, groups, n=200, seed=0):
    rng = np.random.default_rng(seed); ug = np.unique(groups); out = []
    for _ in range(n):
        g = rng.choice(ug, len(ug)); idx = np.concatenate([np.where(groups == k)[0] for k in g])
        out.append(1 - np.mean((y[idx] - pred[idx]) ** 2) / np.var(y[idx]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]

def macro(V0, U0, a, ckv, cku):
    N = len(a); order = np.argsort(a); cl = {1: order[:N // 2], 2: order[N // 2:]}; out = {}
    for k, c in cl.items():
        vm, um = V0[:, c].mean(1), U0[:, c].mean(1)
        rv, ru = ckv[:, c].mean(1), cku[:, c].mean(1)
        x, y = (vm - rv.mean()) / rv.std(), (um - ru.mean()) / ru.std()
        out[f'th{k}'] = np.arctan2(y, x); out[f'A{k}'] = np.hypot(x, y)
    return out

def scalar(y, t, W=60.0):
    w = t <= W; dt = t[1] - t[0]
    return y[:, w].sum(1) * dt, np.sqrt((y[:, w] ** 2).sum(1) * dt)

res = {}
# ---- prepared bistable states, N = 120
for name in ('increasing', 'decreasing'):
    s = np.load(f'out/e5_state_{name}_N120_s3.npz'); p = np.load(f'out/e5_probe_{name}_N120_s3.npz')
    ck_t = s['ck_t']; idx = np.searchsorted(ck_t, p['t_probe'])
    V0, U0 = s['ck_v'][idx], s['ck_u'][idx]
    ok = ck_t >= 10000.0
    M = macro(V0, U0, s['a'], s['ck_v'][ok], s['ck_u'][ok])
    y, t = p['y1'], p['t']; R, E = scalar(y, t)
    groups = (p['t_probe'] // 5000).astype(int)
    ym = y.mean(0); w = t <= 60
    r = dict(K=int(len(y)), R_mean=float(R.mean()), R_sd=float(R.std()), norm_mean=float(E.mean()), norm_sd=float(E.std()),
             norm_median=float(np.median(E)), norm_p95=float(np.percentile(E, 95)),
             mean_energy_fraction=float((ym[w] ** 2).sum() / (y[:, w] ** 2).mean(0).sum()),
             kurtosis_R=float(((R - R.mean()) ** 4).mean() / R.var() ** 2))
    sets = {'phase cluster 2': np.c_[np.cos(M['th2']), np.sin(M['th2'])],
            'phase+amp both clusters': np.c_[np.cos(M['th2']), np.sin(M['th2']), M['A2'], np.cos(M['th1']), np.sin(M['th1']), M['A1']],
            'microscopic state (v,u)': np.c_[V0, U0]}
    r['rho_R'] = {}; r['rho_logE'] = {}
    for k, S in sets.items():
        r2, pred = cv_r2(S, R, groups); r['rho_R'][k] = dict(r2=r2, ci95=boot_ci(R, pred, groups))
        r2, pred = cv_r2(S, np.log(E), groups); r['rho_logE'][k] = dict(r2=r2, ci95=boot_ci(np.log(E), pred, groups))
    y2 = p['y2']; lin = np.linalg.norm(y[:, w] - y2[:, w], axis=1) / np.linalg.norm(y[:, w], axis=1)
    big = E > np.percentile(E, 90)
    r['linearity_median'] = float(np.median(lin)); r['linearity_p90'] = float(np.percentile(lin, 90))
    r['linearity_median_top10'] = float(np.median(lin[big])); r['linearity_median_rest'] = float(np.median(lin[~big]))
    E2 = np.sqrt((y2[:, w] ** 2).sum(1) * (t[1] - t[0])); r['E2eps_over_E_top10_median'] = float(np.median(E2[big] / E[big]))
    res[name] = r
    np.savez_compressed(f'out/e5_analysis_{name}_N120_s3.npz', lin=lin, R=R, E=E, th2=M['th2'], A2=M['A2'], th1=M['th1'], ym=ym,
                        ysd=y.std(0), yq=np.percentile(y, [10, 25, 50, 75, 90], axis=0), t=t, y_examples=y[:8])
    print(name, json.dumps(r), flush=True)
json.dump(res, open('out/e5_analysis_N120.json', 'w'), indent=1)

# ---- intermittent network, N = 40: the same network visiting both states
p = np.load('out/e5_probe_intermittent_N40_s2.npz'); s = np.load('out/e5_state_intermittent_N40_s2.npz')
y, t, lab = p['y1'], p['t'], p['label']; R, E = scalar(y, t)
ok = s['ck_t'] >= 10000.0
M = macro(p['V0'], p['U0'], s['a'], s['ck_v'][ok], s['ck_u'][ok])
groups = (p['t_probe'] // 5000).astype(int)
ft = json.load(open('out/e5_probe_intermittent_N40_s2.json'))['frac_time']
pinc = ft['incoherent'] / (ft['incoherent'] + ft['coherent'])
r = {'p_incoherent_time': pinc}
w = t <= 60
for k, nm in ((0, 'coherent'), (1, 'incoherent')):
    yk = y[lab == k]; ym = yk.mean(0)
    r[nm] = dict(K=int(len(yk)), R_mean=float(R[lab == k].mean()), R_sd=float(R[lab == k].std()),
                 norm_median=float(np.median(E[lab == k])), norm_p95=float(np.percentile(E[lab == k], 95)),
                 mean_energy_fraction=float((ym[w] ** 2).sum() / (yk[:, w] ** 2).mean(0).sum()),
                 kurtosis_R=float(((R[lab == k] - R[lab == k].mean()) ** 4).mean() / R[lab == k].var() ** 2),
                 linearity_median=float(np.median(p['lin'][lab == k])))
# between-state fraction of variance of the whole response function, time-weighted mixture
def between_fraction(Y, lab, pinc):
    m0, m1 = Y[lab == 0].mean(0), Y[lab == 1].mean(0)
    v0, v1 = Y[lab == 0].var(0), Y[lab == 1].var(0)
    within = (1 - pinc) * v0 + pinc * v1; between = pinc * (1 - pinc) * (m1 - m0) ** 2
    return float(between.sum() / (between.sum() + within.sum())), float(np.mean(between.sum() / (between.sum() + within.sum())))
r['between_state_fraction_y_timeweighted'] = between_fraction(y[:, w], lab, pinc)[0]
r['between_state_fraction_y_balanced'] = between_fraction(y[:, w], lab, 0.5)[0]
r['between_state_fraction_R_timeweighted'] = between_fraction(R[:, None], lab, pinc)[0]
r['between_state_fraction_E_timeweighted'] = between_fraction(E[:, None], lab, pinc)[0]
r['between_state_fraction_E_balanced'] = between_fraction(E[:, None], lab, 0.5)[0]
sets = {'state label R2pre': np.c_[p['R2pre']],
        'order parameters R1pre,R2pre': np.c_[p['R1pre'], p['R2pre']],
        'phase cluster 2': np.c_[np.cos(M['th2']), np.sin(M['th2'])],
        'label + phase/amp': np.c_[p['R1pre'], p['R2pre'], np.cos(M['th2']), np.sin(M['th2']), M['A2'],
                                   np.cos(M['th1']), np.sin(M['th1']), M['A1']],
        'microscopic state (v,u)': np.c_[p['V0'], p['U0']]}
r['rho_E_pooled_balanced'] = {}
for k, S in sets.items():
    r2, pred = cv_r2(S, np.log(E), groups); r['rho_E_pooled_balanced'][k] = dict(r2=r2, ci95=boot_ci(np.log(E), pred, groups))
res2 = r
np.savez_compressed('out/e5_analysis_intermittent_N40_s2.npz', R=R, E=E, lab=lab, th2=M['th2'], A2=M['A2'],
                    R2pre=p['R2pre'], t=t, ym0=y[lab == 0].mean(0), ym1=y[lab == 1].mean(0),
                    ysd0=y[lab == 0].std(0), ysd1=y[lab == 1].std(0), t_probe=p['t_probe'])
print('intermittent', json.dumps(r, indent=1))
json.dump(res2, open('out/e5_analysis_intermittent.json', 'w'), indent=1)
