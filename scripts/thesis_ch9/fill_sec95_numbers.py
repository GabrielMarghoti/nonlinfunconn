"""Fill the numbers of the model-free section from out/e6_*.json/npz and out/e4c_spikehistory.json."""
import json, numpy as np
O = '/home/claude/ch9/out/'
H = json.load(open(O + 'e6_hh.json')); I = json.load(open(O + 'e6_izh.json'))
C = json.load(open(O + 'e4c_spikehistory.json')); E4 = json.load(open(O + 'e4_conditioning.json'))
A = {m: json.load(open(O + f'e6_acausal_check_{m}.json')) for m in ('hh', 'izh')}
f2 = lambda x: f'{x:.2f}'
def shape(m):
    d = np.load(O + f'e6_{m}.npz'); tau = d['tau']; k = d['K_all'][:, 2]; p = tau >= 0; t = tau[p]; kk = k[p]
    return t, kk
out = {}
r = lambda v: ', '.join(f'{x:.2f}' if x < 1 else f'{x:.1f}' for x in v)
out['RATES_HH'] = r(H['rates_Hz']); out['RATES_IZH'] = r(I['rates_Hz'])
out['T_HOURS'] = f"{H['stein_hours_of_data']:.2f}"
out['MPP_HH'] = f"{H['M_pp']}"; out['MPP_IZH'] = f"{I['M_pp']}"
out['Z_HH'] = f2(H['pp_vs_stein_z_rms']); out['Z_IZH'] = f2(I['pp_vs_stein_z_rms'])
out['ZC_HH'] = f2(H['cond_hid_z_rms']); out['ZC_IZH'] = f2(I['cond_hid_z_rms'])
rz = [a['acausal_rms_z'] for m in A for a in A[m] if a['kind'] == 'real'] + [H['acausal_rms_z'], I['acausal_rms_z']]
out['ACAUSAL'] = f'root-mean-square \\(z\\) between {min(rz):.1f} and {max(rz):.1f} over independent realizations of both models'
Tpp = {'hh': H['M_pp'] * 2 * 0.06, 'izh': I['M_pp'] * 2 * 0.06}          # simulated seconds in the pump-probe arms
out['SE_HH'] = f"{H['se_ratio_pp_over_stein_median'] * np.sqrt(Tpp['hh'] / H['T_total_s']):.0f}"
out['SE_IZH'] = f"{I['se_ratio_pp_over_stein_median'] * np.sqrt(Tpp['izh'] / I['T_total_s']):.0f}"
out['KURT_HH'] = f"{H['Gpp_single_trial_kurtosis_at_peak']:.0f}"; out['KURT_IZH'] = f"{I['Gpp_single_trial_kurtosis_at_peak']:.0f}"
t, k = shape('hh')
m1 = (t < 6); out['HH_P1'] = f'{t[m1][np.argmax(k[m1])]:.1f}'
i_t = np.argmin(k); out['HH_T1'] = f'{t[i_t]:.1f}'
m2 = (t > t[i_t]) & (t < 25); out['HH_P2'] = f'{t[m2][np.argmax(k[m2])]:.1f}'
out['HH_PER'] = f'{(t[m2][np.argmax(k[m2])] - t[m1][np.argmax(k[m1])]):.0f}'
a = np.abs(k); out['HH_DEC'] = f'{t[np.where(a > 0.1 * a.max())[0][-1]]:.0f}'
normH = np.sqrt((k ** 2).sum() * 0.1)
t, k = shape('izh')
ks = np.convolve(k, np.ones(11) / 11, 'same')
out['IZ_P1'] = f'{t[np.argmax(ks)]:.0f}'; out['IZ_T1'] = f'{t[np.argmin(ks)]:.0f}'
zc = t[(t > 5) & (t < 50)][np.where(np.diff(np.sign(ks[(t > 5) & (t < 50)])) != 0)[0]]
out['IZ_ZC'] = f'{zc[0]:.0f}'
normI = np.sqrt((k ** 2).sum() * 0.1); out['NORM_RATIO'] = f'{normH / normI:.1f}'
out['IZ_INT'] = f'{k.sum() * 0.1:.2f}'
S = lambda J, k: f"{J['spread'][k]:.2f}"
out['D_HH_V3'] = S(H, 'V3'); out['D_IZ_V3'] = S(I, 'V3'); out['D_HH_VH'] = S(H, 'V3xhid'); out['D_IZ_VH'] = S(I, 'V3xhid')
out['D_HH_H'] = S(H, 'hid'); out['D_HH_T'] = S(H, 'tss'); out['D_IZ_VT'] = S(I, 'V3xtss')
out['FRAC_VT'] = f"{100 * I['spread']['V3xtss'] / I['spread']['V3xhid']:.0f}\\%"
out['RHO_TU'] = f"\\({I['rho_tss_hid']:.3f}\\)"; out['RHO_TN'] = f"\\({H['rho_tss_hid']:.2f}\\)"
out['E4C_VT'] = f"\\({C['izh']['rho']['V3 + tss3']['r2']:.3f}\\)"; out['E4C_VU'] = f"\\({C['izh']['rho']['V3 + hidden3']['r2']:.3f}\\)"
out['E4C_HT'] = f"\\({C['izh']['rho']['V history + tss3']['r2']:.3f}\\)"
out['FRAC_HT'] = f"{100 * C['izh']['rho']['V history + tss3']['r2'] / E4['izh']['rho']['neural state x']['r2']:.0f}\\%"
out['E4C_HH_VT'] = f"\\({C['hh']['rho']['V3 + tss3']['r2']:.3f}\\)"
P = {m: json.load(open(O + f'e6_pp_check_{m}.json')) for m in ('hh', 'izh')}
out['ZMAX_IZH'] = f"{max(I['pp_vs_stein_z_max'], P['izh']['z_max']):.1f}"
out['Z2_HH'] = f2(P['hh']['z_rms']); out['Z2_IZH'] = f2(P['izh']['z_rms']); out['M2_HH'] = str(P['hh']['M']); out['M2_IZH'] = str(P['izh']['M'])
json.dump(out, open('/home/claude/ch9/v2/fill_values.json', 'w'), indent=1)
s = open('/home/claude/ch9/v2/s5_template.tex').read()
for kk, v in out.items():
    s = s.replace('@@' + kk + '@@', v)
assert '@@' not in s, [x for x in s.split('@@')[1::2]]
open('/home/claude/ch9/v2/s5.tex', 'w').write(s)
print(json.dumps(out, indent=0))
