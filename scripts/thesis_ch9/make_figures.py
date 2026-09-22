"""
make_figures.py -- all Chapter 9 figures, from the files in out/.
Figures are written to figs/ as PDF, 6.5 in wide (thesis text width 17 cm),
text set by LaTeX so it matches the thesis typeface.
"""
import numpy as np, json, os, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import ch9_analysis as AN

os.makedirs('figs', exist_ok=True)
plt.rcParams.update({
    'text.usetex': True, 'font.family': 'serif', 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 9, 'legend.fontsize': 7.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': 0.7,
    'lines.linewidth': 1.2, 'legend.frameon': False, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02, 'xtick.direction': 'in', 'ytick.direction': 'in',
})
# validated categorical slots (dataviz reference palette); hue = kind, linestyle = position
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
MAGENTA, GOLD = '#d55181', '#c98500'            # Ch. 5 convention: partial sync / unsync
INK, MUTED = '#0b0b0b', '#8a8986'
NODE_C = ['#1f5fb0', '#2a78d6', '#7fb0ea']      # node factors (blue family)
LINK_C = ['#c2491a', '#eb6834']                 # link factors (orange family)
T_PUMP = 5.0
W = 6.5

def panel(ax, s, x=-0.16, y=1.02):
    ax.text(x, y, rf'\textbf{{({s})}}', transform=ax.transAxes, fontsize=9, va='bottom')

def load(tag):
    return np.load(f'out/e2_{tag}.npz')

# ---------------------------------------------------------------------------
def fig_protocol():
    d = load('hh_gap-chem'); t = d['t']; Vr = d['Vref']; D = d['deltas']; tau = d['tau']
    G, G0 = d['G'], d['G0']
    val = json.load(open('out/e1_validation.json'))
    lin = np.load('out/e1_linearity.npz')
    fig, ax = plt.subplots(2, 2, figsize=(W, 4.3), gridspec_kw=dict(hspace=0.55, wspace=0.32))
    a = ax[0, 0]
    for k, (c, ls) in enumerate(zip([BLUE, ORANGE, AQUA], ['-', '-', '-'])):
        a.plot(t, Vr[:, k], color=c, lw=1.0, label=rf'$V_{k+1}$')
    a.axvspan(T_PUMP, T_PUMP + 3, color=MUTED, alpha=0.18, lw=0)
    for dd, lab in [(0.5, r'$\Delta{=}0.5$'), (4.0, r'$4$'), (40.0, r'$40$')]:
        a.annotate('', xy=(T_PUMP + dd, -82), xytext=(T_PUMP + dd, -95),
                   arrowprops=dict(arrowstyle='->', color=INK, lw=0.7), annotation_clip=False)
    a.set_xlim(0, 60); a.set_ylim(-80, 45); a.set_xlabel(r'$t$ (ms)'); a.set_ylabel(r'$V$ (mV)')
    a.legend(loc='upper right', ncol=3, handlelength=1.2, columnspacing=0.8); panel(a, 'a')
    a = ax[0, 1]
    for dd, c, ls in [(0.5, BLUE, '-'), (4.0, ORANGE, '-'), (40.0, AQUA, '-')]:
        k = np.argmin(np.abs(D - dd))
        a.plot(tau, G[k, :, 2], color=c, ls=ls, label=rf'$\Delta = {dd:g}$ ms')
    a.plot(tau, G0[:, 2], color=INK, ls='--', lw=0.9, label='unpumped')
    a.set_yscale('symlog', linthresh=0.05, linscale=0.5)
    a.set_ylim(-20, 400); a.set_yticks([-10, -1, 0, 1, 10, 100])
    a.set_yticklabels([r'$-10$', r'$-1$', r'$0$', r'$1$', r'$10$', r'$10^2$'])
    a.set_xlim(0, 30); a.set_xlabel(r'$t - t_p$ (ms)')
    a.set_ylabel(r'$G_3$ (mV\,cm$^{2}$\,nC$^{-1}$)'); a.legend(loc='upper right', fontsize=7, ncol=2, columnspacing=0.8); panel(a, 'b')
    a = ax[1, 0]
    amps = lin['amps']; nr = lin['norms']            # (namp, ndelta, N)
    dev = np.max(np.abs(nr / nr[0] - 1.0), axis=(1, 2))
    a.loglog(amps[1:], dev[1:], 'o-', color=BLUE, ms=4)
    ref = dev[2] * (amps[1:] / amps[2]) ** 2
    a.loglog(amps[1:], ref, ls=':', color=MUTED, lw=0.9)
    a.text(amps[3], ref[2] * 0.25, r'$\propto q^2$', color=MUTED, fontsize=8)
    a.axvline(0.1, color=MUTED, lw=0.6, ls='--'); a.text(0.105, 2e-3, 'working\namplitude', fontsize=7, color=MUTED)
    a.set_xlabel(r'probe amplitude ($\mu$A\,cm$^{-2}$)'); a.set_ylabel('max.\ relative deviation'); panel(a, 'c')
    a = ax[1, 1]
    labs = [r'$\Delta t{=}0.01$', r'$\Delta t{=}0.005$', 'one-sided\ndifference']
    vals = [val['dt_0.01_vs_0.0025'], val['dt_0.005_vs_0.0025'], val['central_vs_onesided']]
    a.bar(range(3), vals, color=[BLUE, BLUE, ORANGE], width=0.55)
    a.set_yscale('log'); a.set_xticks(range(3)); a.set_xticklabels(labs, fontsize=7.5)
    a.set_ylabel('max.\ relative error'); panel(a, 'd')
    for i, v in enumerate(vals):
        a.text(i, v * 1.4, f'{v:.1e}'.replace('e-0', r'$\times10^{-') + '}$', ha='center', fontsize=7)
    fig.savefig('figs/fig9_1_protocol.pdf'); plt.close(fig)

# ---------------------------------------------------------------------------
def fig_maps():
    from matplotlib.colors import SymLogNorm
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.45), sharey=True, gridspec_kw=dict(wspace=0.06))
    names = [('gap-gap', 'gap $\\to$ gap'), ('gap-chem', 'gap $\\to$ chemical'), ('chem-chem', 'chemical $\\to$ chemical')]
    for i, (n, title) in enumerate(names):
        d = load(f'hh_{n}'); D, tau, G = d['deltas'], d['tau'], d['G'][:, :, 2]
        m = tau <= 30
        Gn = G[:, m] / np.max(np.abs(G[:, m]))
        im = ax[i].imshow(Gn.T, origin='lower', aspect='auto', cmap='RdBu_r',
                          norm=SymLogNorm(linthresh=1e-3, linscale=0.5, vmin=-1, vmax=1),
                          extent=[D[0], D[-1], 0, tau[m][-1]], interpolation='nearest')
        sp = d['spikes_ref'][2]; sp = sp[~np.isnan(sp)]
        for ts in sp:
            ax[i].plot(D, ts - (T_PUMP + D), color=INK, lw=0.5, ls=(0, (2, 2)))
        ax[i].set_ylim(0, 30); ax[i].set_xlim(D[0], D[-1]); ax[i].set_title(title)
        ax[i].set_xlabel(r'$\Delta$ (ms)')
        panel(ax[i], 'abc'[i], x=-0.06 if i else -0.22)
    ax[0].set_ylabel(r'$t - t_p$ (ms)')
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.03, ticks=[-1, -1e-1, -1e-2, 0, 1e-2, 1e-1, 1])
    cb.ax.set_yticklabels([r'$-1$', r'$-10^{-1}$', r'$-10^{-2}$', r'$0$', r'$10^{-2}$', r'$10^{-1}$', r'$1$'], fontsize=6.5)
    cb.set_label(r'$G_3 / \max|G_3|$', fontsize=8)
    fig.savefig('figs/fig9_2_maps.pdf'); plt.close(fig)

# ---------------------------------------------------------------------------
def fig_decomposition():
    dec = json.load(open('out/e2_hh_decomposition.json'))
    fig = plt.figure(figsize=(W, 4.4))
    gs = fig.add_gridspec(2, 3, hspace=0.72, wspace=0.3, height_ratios=[1, 0.85])
    names = [('gap-gap', 'gap $\\to$ gap'), ('gap-chem', 'gap $\\to$ chemical'), ('chem-chem', 'chemical $\\to$ chemical')]
    for i, (n, title) in enumerate(names):
        a = fig.add_subplot(gs[0, i])
        d = load(f'hh_{n}'); dtau = d['tau'][1] - d['tau'][0]; D = d['deltas']
        f, tot = AN.telescoping_factors(d['G'], d['GS'], dtau)
        ref = lambda x: x - x[-1]
        for (k, c, ls) in [('N1', NODE_C[0], '-'), ('N2', NODE_C[1], '--'), ('N3', NODE_C[2], ':')]:
            a.plot(D, ref(f[k]) / np.log(10), color=c, ls=ls, lw=1.0, label=rf'$\mathcal{{N}}_{k[1]}$')
        for (k, c, ls) in [('T21', LINK_C[0], '-'), ('T32', LINK_C[1], '--')]:
            a.plot(D, ref(f[k]) / np.log(10), color=c, ls=ls, lw=1.0, label=rf'$\mathcal{{T}}_{{{k[1:]}}}$')
        a.plot(D, ref(tot) / np.log(10), color=INK, lw=1.4, label='end to end')
        a.set_xlim(-3, 40); a.set_ylim(-3.6, 2.4); a.set_title(title); a.set_xlabel(r'$\Delta$ (ms)')
        if i == 0:
            a.set_ylabel(r'$\log_{10}$ factor (rel.\ to rest)')
        else:
            a.set_yticklabels([])
        a.axhline(0, color=MUTED, lw=0.5)
        panel(a, 'abc'[i], x=-0.2 if i == 0 else -0.06)
        if i == 0:
            h, l = a.get_legend_handles_labels()
            fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.515), ncol=6, fontsize=7.5,
                       handlelength=2.0, columnspacing=1.2)
    a = fig.add_subplot(gs[1, :2])
    labels = ['gap $\\to$ gap', 'gap $\\to$ chemical', 'chemical $\\to$ chemical']
    keys = ['gap-gap', 'gap-chem', 'chem-chem']
    y = np.arange(3)[::-1]
    for j, k in enumerate(keys):
        sh = dec[k]['shares']; left = 0.0
        for fk, c in [('N1', NODE_C[0]), ('N2', NODE_C[1]), ('N3', NODE_C[2]), ('T21', LINK_C[0]), ('T32', LINK_C[1])]:
            v = sh[fk]
            a.barh(y[j], v, left=left if v >= 0 else 0, color=c, height=0.6, edgecolor='white', linewidth=0.8)
            if v >= 0: left += v
    a.set_yticks(y); a.set_yticklabels(labels); a.set_xlim(-0.1, 1.3); a.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    a.set_xlabel(r'share of $\mathrm{Var}_\Delta[\log(\mathrm{end\ to\ end})]$')
    for j, k in enumerate(keys):
        a.text(1.09, y[j], rf"nodes {dec[k]['node_share']*100:.0f}\%", va='center', ha='left', fontsize=7.5, color=INK)
    panel(a, 'd', x=-0.3)
    a = fig.add_subplot(gs[1, 2])
    rng = [[dec[k]['range_log'][f] / np.log(10) for f in ('T21', 'T32')] for k in keys]
    xx = np.arange(3)
    a.bar(xx - 0.17, [r[0] for r in rng], 0.32, color=LINK_C[0], label=r'$\mathcal{T}_{21}$')
    a.bar(xx + 0.17, [r[1] for r in rng], 0.32, color=LINK_C[1], label=r'$\mathcal{T}_{32}$')
    a.set_xticks(xx); a.set_xticklabels(['g$\\to$g', 'g$\\to$c', 'c$\\to$c'])
    a.set_ylabel(r'range of $\log_{10}\mathcal{T}$'); a.legend(loc='upper left'); panel(a, 'e', x=-0.25)
    fig.savefig('figs/fig9_3_decomposition.pdf'); plt.close(fig)

# ---------------------------------------------------------------------------
def fig_models():
    cont = json.load(open('out/e3_continuity.json'))
    fig, ax = plt.subplots(3, 2, figsize=(W, 6.3), gridspec_kw=dict(hspace=0.6, wspace=0.28))
    for col, (m, title, clr) in enumerate([('hh', 'Hodgkin--Huxley', BLUE), ('izh', 'Izhikevich', ORANGE)]):
        d = load(f'{m}_gap-chem'); t = d['t']; D = d['deltas']
        ref = d['spikes_ref']
        a = ax[0, col]
        for k, c in enumerate([INK, MUTED, clr]):
            a.plot(t, d['Vref'][:, k], color=c, lw=0.9, label=rf'$V_{k+1}$')
        a.axvspan(T_PUMP, T_PUMP + 3, color=MUTED, alpha=0.18, lw=0)
        a.set_xlim(0, 40); a.set_title(title); a.set_xlabel(r'$t$ (ms)'); a.set_ylabel(r'$V$ (mV)')
        a.legend(loc='center right', ncol=1, handlelength=1.0, fontsize=7); panel(a, 'ab'[col], x=-0.12, y=1.06)
        a = ax[1, col]
        s = AN.spike_sensitivity(d['spikes'], float(d['q']))
        n3 = int(np.sum(~np.isnan(ref[2])))
        styles = ['o', 's', '^', 'v']
        for n in range(n3):
            a.plot(D, s[:, 2, n], ls='none', marker=styles[n % 4], ms=1.6, color=clr,
                   alpha=[1.0, 0.65, 0.4, 0.3][n % 4], label=rf'spike {n+1}')
        for ts in ref[0][~np.isnan(ref[0])]:
            a.axvline(ts - T_PUMP, color=MUTED, lw=0.6, ls=':')
        a.axhline(0, color=MUTED, lw=0.5); a.set_xlim(-3, 25)
        a.set_xlabel(r'$\Delta$ (ms)'); a.set_ylabel(r'$\partial t^{(3)}_n / \partial q$ (ms\,cm$^2$\,nC$^{-1}$)')
        a.legend(loc='lower right', markerscale=2.5, handletextpad=0.2); panel(a, 'cd'[col], x=-0.12, y=1.06)
    # kernel column
    a = ax[2, 0]
    for m, clr, lab in [('hh', BLUE, 'Hodgkin--Huxley'), ('izh', ORANGE, 'Izhikevich')]:
        d = load(f'{m}_gap-chem'); t = d['t']; D = d['deltas']; G = d['G_abs']
        i = int(round(30.0 / (t[1] - t[0])))
        y = G[:, i, 0] / np.max(np.abs(G[D < 15, i, 0]))
        a.plot(D, y, color=clr, lw=1.0, label=lab)
        for ts in d['spikes_ref'][0][~np.isnan(d['spikes_ref'][0])]:
            a.axvline(ts - T_PUMP, color=clr, lw=0.5, ls=':')
    a.set_xlim(-3, 15); a.set_ylim(-1.15, 1.15); a.set_xlabel(r'$\Delta$ (ms)')
    a.set_ylabel(r'$\Psi_{11}(t_{\mathrm{obs}}, t_p)$ (norm.)'); a.legend(loc='lower left'); panel(a, 'e', x=-0.12, y=1.06)
    a = ax[2, 1]
    for m, clr, lab, mk in [('hh', BLUE, 'Hodgkin--Huxley', 'o'), ('izh', ORANGE, 'Izhikevich', 's')]:
        dd = np.array(cont[m]['deltas']); sep = 2 * dd + 0.02
        a.loglog(sep, cont[m]['jump_node3_spike1'], marker=mk, ms=4, color=clr, label=lab)
    xs = np.array([0.02, 0.7]); a.loglog(xs, 0.009 * xs, ':', color=MUTED, lw=0.9)
    a.text(0.25, 0.0011, r'$\propto$ separation', color=MUTED, fontsize=7.5, rotation=28)
    a.set_xlabel('separation of the two probes (ms)')
    a.set_ylabel(r'jump (ms\,cm$^2$\,nC$^{-1}$)'); a.legend(loc='lower right'); panel(a, 'f', x=-0.12, y=1.06)
    fig.savefig('figs/fig9_4_models.pdf'); plt.close(fig)

# ---------------------------------------------------------------------------
def _cond_mean(x, y, nb=10):
    r = np.argsort(np.argsort(x)) / (len(x) - 1.0)
    edges = np.linspace(0, 1, nb + 1); c, m, e = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (r >= lo) & (r <= hi)
        c.append(0.5 * (lo + hi)); m.append(y[s].mean()); e.append(y[s].std(ddof=1) / np.sqrt(s.sum()))
    return np.array(c), np.array(m), np.array(e)

def fig_conditioning():
    J = json.load(open('out/e4_conditioning.json'))
    fig, ax = plt.subplots(2, 2, figsize=(W, 4.6), gridspec_kw=dict(hspace=0.62, wspace=0.3))
    models = [('hh', 'Hodgkin--Huxley', BLUE), ('izh', 'Izhikevich', ORANGE)]
    # (a) evoked change of the spike count of node 3
    a = ax[0, 0]; ks = np.arange(-4, 5)
    for j, (m, lab, clr) in enumerate(models):
        dist = J[m]['evoked_spike_count_distribution']
        p = np.array([dist.get(str(k), 0.0) for k in ks])
        a.bar(ks + (j - 0.5) * 0.38, p, 0.36, color=clr, label=lab, lw=0)
    a.set_xticks(ks); a.set_xlabel(r'evoked change of spike count, $\Delta n_3$')
    a.set_ylabel('fraction of trials'); a.set_ylim(0, 0.9); a.legend(loc='upper left'); panel(a, 'a', x=-0.14)
    # (b) reducible fraction by conditioning set
    a = ax[0, 1]
    sets = ['V1(t0)', 'V3(t0)', 'V1..V3(t0)', 'V history (6 lags)', 'neural state x', 'x + noise state']
    labs = [r'$V_1$', r'$V_3$', r'$V_{1,2,3}$', r'$V_{1,2,3}$' + '\n' + 'history', r'$\mathbf{x}$', r'$(\mathbf{x},\eta)$']
    xx = np.arange(len(sets))
    for j, (m, lab, clr) in enumerate(models):
        r = np.array([J[m]['rho'][s]['r2'] for s in sets])
        ci = np.array([J[m]['rho'][s]['ci95'] for s in sets])
        off = (j - 0.5) * 0.38
        a.bar(xx + off, r, 0.36, color=clr, lw=0, label=lab)
        a.errorbar(xx + off, r, yerr=[r - ci[:, 0], ci[:, 1] - r], fmt='none', ecolor=INK, elinewidth=0.6, capsize=1.5)
        a.axhline(J[m]['rho_truth_full'], color=clr, ls='--', lw=0.9)
    a.text(-0.5, J['hh']['rho_truth_full'] + 0.003, r'$\rho_{\mathrm{full}}$', color=BLUE, fontsize=7.5, ha='left', va='bottom')
    a.text(-0.5, J['izh']['rho_truth_full'] + 0.003, r'$\rho_{\mathrm{full}}$', color=ORANGE, fontsize=7.5, ha='left', va='bottom')
    a.axhline(0, color=MUTED, lw=0.5)
    a.set_xticks(xx); a.set_xticklabels(labs, fontsize=7.5); a.set_xlim(-0.6, 5.6)
    a.set_ylabel(r'reducible fraction $\rho(\mathcal{S})$'); a.set_ylim(-0.02, 0.14); panel(a, 'b', x=-0.16)
    # (c,d) conditional mean response vs observed voltage and hidden variable of node 3
    spec = {'hh': [(2, r'$V_3(t_0)$', '-', 'o'), (11, r'$n_3(t_0)$', '--', 's')],
            'izh': [(2, r'$v_3(t_0)$', '-', 'o'), (5, r'$u_3(t_0)$', '--', 's')]}
    for j, (m, lab, clr) in enumerate(models):
        a = ax[1, j]; d = np.load(f'out/e4_{m}.npz'); Rm = d['R'].mean(1); X = d['X']
        for col, vl, ls, mk in spec[m]:
            c, mu, se = _cond_mean(X[:, col], Rm)
            a.fill_between(c, mu - se, mu + se, color=clr, alpha=0.15, lw=0)
            a.plot(c, mu, ls=ls, marker=mk, ms=3.2, color=clr if ls == '-' else INK, mfc='white' if ls == '--' else clr,
                   lw=1.0, label=vl)
        a.axhline(d['R'].mean(), color=MUTED, lw=0.6, ls=':')
        a.set_xlabel('quantile of the conditioning variable'); a.set_ylabel(r'$\langle R \mid \mathbf{s}\rangle$ (mV\,ms)')
        a.set_title(lab); a.legend(loc='best'); panel(a, 'cd'[j], x=-0.14, y=1.06)
    fig.savefig('figs/fig9_5_conditioning.pdf'); plt.close(fig)

# ---------------------------------------------------------------------------
def _smooth(x, k):
    x = np.where(np.isnan(x), np.nanmean(x), x)
    return np.convolve(x, np.ones(k) / k, mode='same')

def fig_network():
    J120 = json.load(open('out/e5_analysis_N120.json')); Ji = json.load(open('out/e5_analysis_intermittent.json'))
    fig, ax = plt.subplots(3, 2, figsize=(W, 6.9), gridspec_kw=dict(hspace=0.62, wspace=0.3))
    states = [('increasing', 'partially synchronized', MAGENTA), ('decreasing', 'incoherent', GOLD)]
    # (a) order parameter of the slow cluster in the two coexisting states, N = 120
    a = ax[0, 0]
    for name, lab, clr in states:
        s = np.load(f'out/e5_state_{name}_N120_s3.npz'); tg = s['tg'] / 1000.0
        m = (tg >= 10) & (tg < 40)
        a.plot(tg[m], _smooth(s['R2'], 50)[m], color=clr, lw=0.9, label=lab)
    a.set_ylim(0, 1.05); a.set_xlim(10, 40); a.set_xlabel(r'$t$ (s)'); a.set_ylabel(r'$R_2$ (0.5 s average)')
    a.legend(loc='center right'); panel(a, 'a', x=-0.14)
    # (b) mean response function with interquartile band
    a = ax[0, 1]
    for name, lab, clr in states:
        d = np.load(f'out/e5_analysis_{name}_N120_s3.npz'); t = d['t']
        a.fill_between(t, d['yq'][1], d['yq'][3], color=clr, alpha=0.25, lw=0)
        a.plot(t, d['yq'][2], color=clr, lw=1.1)
        a.plot(t, d['ym'], color=clr, lw=0.9, ls='--')
    a.plot([], [], color=INK, lw=1.1, label='median'); a.plot([], [], color=INK, lw=0.9, ls='--', label='mean')
    a.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=2, borderaxespad=0.1)
    a.axhline(0, color=MUTED, lw=0.5); a.set_xlim(0, 80)
    a.set_xlabel(r'$t - t_p$ (ms)'); a.set_ylabel(r'$y$ (mV per unit charge)'); panel(a, 'b', x=-0.14)
    # (c) distribution of the response norm
    a = ax[1, 0]; bins = np.logspace(-0.5, 2.2, 45)
    for name, lab, clr in states:
        d = np.load(f'out/e5_analysis_{name}_N120_s3.npz')
        a.hist(d['E'], bins=bins, color=clr, histtype='stepfilled', alpha=0.25, lw=0)
        a.hist(d['E'], bins=bins, color=clr, histtype='step', lw=1.1, label=lab)
    a.set_xscale('log'); a.set_yscale('log'); a.set_ylim(0.7, 600); a.set_xlabel(r'response norm $\|y\|$'); a.set_ylabel('number of probes')
    a.legend(loc='upper right'); panel(a, 'c', x=-0.14)
    # (d) phase gating in the partially synchronized state
    a = ax[1, 1]
    for name, lab, clr in states[::-1]:
        d = np.load(f'out/e5_analysis_{name}_N120_s3.npz')
        a.plot(d['th2'], d['E'], ls='none', marker='o', ms=2.0, color=clr, alpha=0.7, mew=0)
    d = np.load('out/e5_analysis_increasing_N120_s3.npz'); bins = np.linspace(-np.pi, np.pi, 9); cb = 0.5 * (bins[1:] + bins[:-1])
    med = [np.median(d['E'][(d['th2'] >= lo) & (d['th2'] < hi)]) for lo, hi in zip(bins[:-1], bins[1:])]
    a.plot(cb, med, color=INK, lw=1.0, label='median (synchronized)')
    a.set_yscale('log'); a.set_xlim(-np.pi, np.pi); a.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    a.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    a.set_xlabel(r'phase of the cluster-2 mean field, $\theta_2$'); a.set_ylabel(r'$\|y\|$')
    panel(a, 'd', x=-0.14)
    # (e) intermittent network, N = 40
    a = ax[2, 0]; s = np.load('out/e5_state_intermittent_N40_s2.npz'); tg = s['tg'] / 1000.0
    r2 = _smooth(s['R2'], 100); m = (tg >= 800) & (tg < 1100)
    a.plot(tg[m], r2[m], color=AQUA, lw=0.7)
    a.fill_between(tg[m], 0, 1.05, where=r2[m] < 0.5, color=GOLD, alpha=0.25, lw=0)
    a.set_ylim(0, 1.05); a.set_xlim(800, 1100); a.set_xlabel(r'$t$ (s)'); a.set_ylabel(r'$R_2$ (1 s average)')
    panel(a, 'e', x=-0.14)
    # (f) what predicts the magnitude of a single response in the intermittent network
    a = ax[2, 1]
    sets = ['state label R2pre', 'order parameters R1pre,R2pre', 'phase cluster 2', 'label + phase/amp', 'microscopic state (v,u)']
    labs = [r'$R_2^{\mathrm{pre}}$', r'$R_{1,2}^{\mathrm{pre}}$', r'$\theta_2$', r'$R^{\mathrm{pre}}_{1,2}$' + '\n' + r'$\theta_{1,2}, A_{1,2}$', r'$(\mathbf{v},\mathbf{u})$']
    rr = np.array([Ji['rho_E_pooled_balanced'][k]['r2'] for k in sets]); ci = np.array([Ji['rho_E_pooled_balanced'][k]['ci95'] for k in sets])
    xx = np.arange(len(sets))
    a.bar(xx, rr, 0.6, color=AQUA, lw=0)
    a.errorbar(xx, rr, yerr=[rr - ci[:, 0], ci[:, 1] - rr], fmt='none', ecolor=INK, elinewidth=0.6, capsize=1.5)
    a.axhline(0, color=MUTED, lw=0.5); a.set_xticks(xx); a.set_xticklabels(labs, fontsize=7.5)
    a.set_ylim(-0.08, 0.62); a.set_ylabel(r'$\rho(\mathcal{S})$ for $\log\|y\|$'); panel(a, 'f', x=-0.14)
    fig.savefig('figs/fig9_6_network.pdf'); plt.close(fig)

# ---------------------------------------------------------------------------
def fig_modelfree():
    from matplotlib import cm
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(W, 6.9))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.72, wspace=0.42)
    ax = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]
    axg = fig.add_subplot(gs[2, :2]); axh = fig.add_subplot(gs[2, 2])
    models = [('hh', 'Hodgkin--Huxley', BLUE, 'Blues'), ('izh', 'Izhikevich', ORANGE, 'Oranges')]
    # (a, b) trial-averaged kernel from the input-output covariance
    for j, (m, lab, clr, cmap) in enumerate(models):
        d = np.load(f'out/e6_{m}.npz'); tau = d['tau']; K = d['K_all'][:, 2]; se = d['K_se'][:, 2]
        a = ax[0][j]
        a.fill_between(tau, K - 2 * se, K + 2 * se, color=clr, alpha=0.3, lw=0)
        a.plot(tau, K, color=clr, lw=1.1)
        a.axhline(0, color=MUTED, lw=0.5); a.axvline(0, color=MUTED, lw=0.5, ls=':')
        a.set_xlim(-10, 60); a.set_xlabel(r'$\tau$ (ms)'); a.set_ylabel(r'$\bar G_{31}(\tau)$')
        a.set_title(lab); panel(a, 'ab'[j], x=-0.22, y=1.06)
    # (c) validation against pump-probe: 2.5 ms window averages
    a = ax[0][2]
    for m, lab, clr, _ in models:
        d = np.load(f'out/e6_{m}.npz')
        a.errorbar(d['Ist_mean'] / 2.5, d['Ipp_mean'] / 2.5, xerr=d['Ist_se'] / 2.5, yerr=d['Ipp_se'] / 2.5, fmt='o',
                   ms=2.4, color=clr, elinewidth=0.5, capsize=0, label=lab)
    lim = [-0.45, 0.45]; a.plot(lim, lim, color=MUTED, lw=0.7, ls='--'); a.set_xlim(lim); a.set_ylim(-0.9, 0.9)
    a.set_xlabel('covariance estimate'); a.set_ylabel('pump--probe estimate'); a.legend(loc='upper left', fontsize=6.5)
    panel(a, 'c', x=-0.22, y=1.06)
    # (d-f) conditional kernels
    spec = [(0, 'hh', 'Kg_V3', r'HH, by $V_3$', 'Blues', False), (1, 'izh', 'Kg_hid', r'Izh., by $u_3$', 'Oranges', False),
            (2, 'izh', 'Kg_tss', r'Izh., by $t_{\mathrm{last}}$', 'Oranges', True)]
    for c_, m, key, title, cmap, rev in spec:
        d = np.load(f'out/e6_{m}.npz'); tau = d['tau']; Kc = d[key].mean(1); nq = Kc.shape[0]
        vals = np.linspace(0.35, 0.95, nq)[::-1] if rev else np.linspace(0.35, 0.95, nq)
        a = ax[1][c_]
        for q in range(nq):
            a.plot(tau, Kc[q], color=getattr(cm, cmap)(vals[q]), lw=1.0, ls='--' if rev else '-',
                   label=('lowest' if q == 0 else 'highest' if q == nq - 1 else None))
        a.axhline(0, color=MUTED, lw=0.5); a.set_xlim(0, 60); a.set_xlabel(r'$\tau$ (ms)')
        a.set_ylabel(r'$\bar G_{31}(\tau \mid s)$'); a.set_title(title)
        a.legend(loc='upper right', fontsize=6.5, handlelength=1.5, title='quintile', title_fontsize=6.5)
        panel(a, 'def'[c_], x=-0.22, y=1.06)
    ax[1][2].set_ylim(ax[1][1].get_ylim())
    # (g) state dependence of the weak-probe kernel captured by each observation
    a = axg
    keys = ['V3', 'hid', 'tss', 'V3xtss', 'V3xhid']
    labs = [r'$V_3$', r'hidden', r'$t_{\mathrm{last}}$', r'$V_3$, $t_{\mathrm{last}}$', r'$V_3$, hidden']
    xx = np.arange(len(keys))
    for j, (m, lab, clr, _) in enumerate(models):
        J = json.load(open(f'out/e6_{m}.json'))
        a.bar(xx + (j - 0.5) * 0.38, [J['spread'][k] for k in keys], 0.36, color=clr, lw=0, label=lab)
    a.set_xticks(xx); a.set_xticklabels(labs, fontsize=7.5); a.axvline(2.5, color=MUTED, lw=0.5, ls=':')
    a.set_ylim(0, 1.25); a.set_ylabel(r'state dependence $D(s)$'); a.legend(loc='upper left', fontsize=6.8)
    panel(a, 'g', x=-0.1, y=1.04)
    # (h) strong stimulus of Sec. 9.4: reducible fraction with spike history
    a = axh
    J4 = json.load(open('out/e4_conditioning.json')); J4c = json.load(open('out/e4c_spikehistory.json'))
    sets = [('V3(t0)', J4), ('V3 + tss3', J4c), ('V3 + hidden3', J4c), ('neural state x', J4)]
    labs = [r'$V_3$', r'$+t_{\mathrm{last}}$', r'$+$hid.', r'$\mathbf{x}$']
    xx = np.arange(len(sets))
    for j, (m, lab, clr, _) in enumerate(models):
        r = [src[m]['rho'][k]['r2'] for k, src in sets]; ci = np.array([src[m]['rho'][k]['ci95'] for k, src in sets])
        off = (j - 0.5) * 0.38
        a.bar(xx + off, r, 0.36, color=clr, lw=0)
        a.errorbar(xx + off, r, yerr=[np.array(r) - ci[:, 0], ci[:, 1] - np.array(r)], fmt='none', ecolor=INK, elinewidth=0.6, capsize=1.2)
        a.axhline(J4[m]['rho_truth_full'], color=clr, ls='--', lw=0.8)
    a.set_xticks(xx); a.set_xticklabels(labs, fontsize=7.5); a.set_ylim(0, 0.13)
    a.set_ylabel(r'$\rho(\mathcal{S})$'); a.set_title('stimulus of Sec. 9.4', fontsize=8); panel(a, 'h', x=-0.3, y=1.04)
    fig.savefig('figs/fig9_7_modelfree.pdf'); plt.close(fig)

# ---------------------------------------------------------------------------
def fig_nonlinearity():
    from matplotlib import cm
    d = np.load('out/e7_pump_sweep.npz'); J = json.load(open('out/e7_pump_sweep.json'))
    D = d['deltas']; P = list(d['pumps'])
    fig, ax = plt.subplots(2, 2, figsize=(W, 4.9), gridspec_kw=dict(hspace=0.55, wspace=0.32))
    # (a) response norm of node 3 across states, for pumps below and above the spike threshold
    a = ax[0, 0]; NV = d['gap-chem_NV']
    show = [2.0, 3.0, 3.25, 3.5, 10.0]
    cols = {2.0: cm.Blues(0.35), 3.0: cm.Blues(0.5), 3.25: cm.Blues(0.62), 3.5: ORANGE, 10.0: '#8f2d0c'}
    for A in show:
        i = P.index(A)
        a.plot(D, NV[i, :, 2], color=cols[A], lw=1.0, label=rf'$A = {A:g}$')
    a.axhline(1, color=MUTED, lw=0.5, ls=':')
    a.set_yscale('log'); a.set_xlim(-3, 60); a.set_xlabel(r'$\Delta$ (ms)')
    a.set_ylabel(r'$\|\delta V_3\| / \|\delta V_3\|_{\mathrm{rest}}$'); a.legend(loc='upper right', fontsize=6.5, ncol=1)
    panel(a, 'a', x=-0.17)
    # (b) range of the response norm across states against pump amplitude
    a = ax[0, 1]
    for (name, lab), clr, mk in zip([('gap-gap', r'gap $\to$ gap'), ('gap-chem', r'gap $\to$ chem.'), ('chem-chem', r'chem. $\to$ chem.')],
                                    [BLUE, ORANGE, AQUA], ['o', 's', '^']):
        y = [J[name]['pumps'][str(A) if str(A) in J[name]['pumps'] else f'{A:g}']['decades_V3'] for A in P]
        a.plot(P, y, marker=mk, ms=3.5, color=clr, lw=1.0, label=lab)
    a.axvspan(3.25, 3.5, color=MUTED, alpha=0.25, lw=0)
    a.text(3.6, 0.3, 'first spike', fontsize=7, color=INK)
    a.set_xlabel(r'pump amplitude $A$ ($\mu$A\,cm$^{-2}$)'); a.set_ylabel(r'range over $\Delta$ (decades)')
    a.set_xlim(0, 15.5); a.set_ylim(0, 7); a.legend(loc='lower right', fontsize=6.8); panel(a, 'b', x=-0.17)
    # (c) propagated signal and membrane response of node 3 (A = 10)
    a = ax[1, 0]; i = P.index(10.0); NS = d['gap-chem_NS']
    a.plot(D, NS[i, :, 2], color=ORANGE, lw=1.0, label=r'propagated signal $S_3$')
    a.plot(D, NV[i, :, 2], color=BLUE, lw=1.0, label=r'membrane response $V_3$')
    a.axhline(1, color=MUTED, lw=0.5, ls=':')
    a.set_yscale('log'); a.set_xlim(-3, 60); a.set_xlabel(r'$\Delta$ (ms)'); a.set_ylabel('norm relative to rest')
    a.legend(loc='lower right', fontsize=6.8); panel(a, 'c', x=-0.17)
    # (d) spike-time shifts of node 3
    a = ax[1, 1]; e = load('hh_gap-chem'); s = AN.spike_sensitivity(e['spikes'], float(e['q'])); De = e['deltas']
    ref = e['spikes_ref']
    for n, (mk, lab, ms) in [(1, ('s', 'rebound spike', 1.6)), (0, ('o', 'first spike', 2.2))]:
        a.plot(De, s[:, 2, n], ls='none', marker=mk, ms=ms, color=[BLUE, ORANGE][n], label=lab)
    for ts in ref[0][~np.isnan(ref[0])]:
        a.axvline(ts - T_PUMP, color=MUTED, lw=0.6, ls=':')
    a.axhline(0, color=MUTED, lw=0.5); a.set_xlim(-3, 25)
    a.set_xlabel(r'$\Delta$ (ms)'); a.set_ylabel(r'$\partial t^{(3)}_n / \partial q$ (ms\,cm$^2$\,nC$^{-1}$)')
    a.legend(loc='upper right', markerscale=2.5, fontsize=6.8); panel(a, 'd', x=-0.17)
    fig.savefig('figs/fig9_3_nonlinearity.pdf'); plt.close(fig)

# ---------------------------------------------------------------------------
def fig_variability():
    from matplotlib import cm
    z = np.load('out/e7_amplitude.npz'); J = json.load(open('out/e7_variability.json'))
    fig, ax = plt.subplots(2, 2, figsize=(W, 4.9), gridspec_kw=dict(hspace=0.55, wspace=0.32))
    # (a) weak stimulus: distribution of the causal response per unit charge
    a = ax[0, 0]; R = z['Rq_0.1'].ravel(); dN = z['dN_0.1'].ravel()
    lim = np.array([-1000, 1000]); bins = np.concatenate([-np.logspace(3, -2, 40), np.logspace(-2, 3, 40)])
    a.hist(R[dN == 0], bins=bins, color=BLUE, histtype='stepfilled', alpha=0.6, lw=0, label='spike count unchanged')
    a.hist(R[dN != 0], bins=bins, color=ORANGE, histtype='stepfilled', alpha=0.9, lw=0, label='spike added or removed')
    a.set_xscale('symlog', linthresh=0.1, linscale=0.4); a.set_yscale('log'); a.set_xlim(-1000, 1000)
    a.set_xticks([-100, -1, 0, 1, 100]); a.set_xticklabels([r'$-10^2$', r'$-1$', r'$0$', r'$1$', r'$10^2$'])
    a.set_xlabel(r'response per unit charge, $R/q$'); a.set_ylabel('number of trials'); a.set_ylim(0.8, 3e3)
    a.legend(loc='upper left', fontsize=6.5); panel(a, 'a', x=-0.17)
    # (b) noise dependence of the trial-to-trial spread
    a = ax[0, 1]; ns = J['noise_sweep']; sig = sorted(ns, key=float); s = np.array([float(x) for x in sig])
    for A, clr, mk, lab in [('0.1', BLUE, 'o', r'weak, $A = 0.1$'), ('10.0', ORANGE, 's', r'strong, $A = 10$')]:
        a.plot(s, [ns[k][A]['sd'] for k in sig], marker=mk, ms=3.5, color=clr, lw=1.0, label=lab)
    for k, x in zip(sig, s):
        a.text(x, 150, f"{ns[k]['rate3_Hz']:.0f}", ha='center', fontsize=6.5, color=MUTED)
    a.text(0.2, 400, r'rate of node 3 (Hz):', fontsize=6.5, color=MUTED)
    a.set_yscale('log'); a.set_ylim(0.008, 1500); a.set_xlim(0.1, 2.15)
    a.set_xlabel(r'noise intensity $\sigma$'); a.set_ylabel(r'SD of $R/q$ across trials'); a.legend(loc='lower right', fontsize=6.5)
    panel(a, 'b', x=-0.17)
    # (c) amplitude dependence: all-or-none outcomes and the part fixed at onset
    a = ax[1, 0]; am = J['amplitude_sweep']; As = sorted(am, key=float); x = np.array([float(v) for v in As])
    a.plot(x, [am[k]['p_spike_change'] for k in As], marker='o', ms=3.5, color=ORANGE, lw=1.0, label='spike count changed')
    a.plot(x, [am[k]['icc'] for k in As], marker='s', ms=3.5, color=INK, lw=1.0, label='intraclass correlation')
    a.set_xscale('log'); a.set_ylim(-0.03, 1.0); a.set_xlabel(r'stimulus amplitude $A$ ($\mu$A\,cm$^{-2}$)')
    a.set_ylabel('fraction'); a.legend(loc='upper left', fontsize=6.5); panel(a, 'c', x=-0.17)
    # (d) dependence on the refractoriness of node 3 at onset
    a = ax[1, 1]; n3 = z['n3_onset']; r = np.argsort(np.argsort(n3)) / (len(n3) - 1); g = np.minimum((r * 10).astype(int), 9)
    cx = (np.arange(10) + 0.5) / 10
    for A, clr in [(3.0, cm.Oranges(0.5)), (5.0, cm.Oranges(0.7)), (10.0, cm.Oranges(0.95))]:
        p = (z[f'dN_{A}'] != 0).mean(1)
        m = np.array([p[g == k].mean() for k in range(10)]); se = np.array([p[g == k].std(ddof=1) / np.sqrt((g == k).sum()) for k in range(10)])
        a.fill_between(cx, m - se, m + se, color=clr, alpha=0.2, lw=0)
        a.plot(cx, m, marker='o', ms=3, color=clr, lw=1.0, label=rf'$A = {A:g}$')
    a.set_ylim(0, 1.05); a.set_xlabel(r'decile of $n_3$ at stimulus onset'); a.set_ylabel(r'P(spike count changed)')
    a.legend(loc='lower left', fontsize=6.5); panel(a, 'd', x=-0.17)
    fig.savefig('figs/fig9_4_variability.pdf'); plt.close(fig)

if __name__ == '__main__':
    which = sys.argv[1:] or ['protocol', 'maps', 'decomposition', 'models']
    for w in which:
        globals()[f'fig_{w}']()
        print('wrote', w)
