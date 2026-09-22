import numpy as np, json
import ch9_circuits as CC, ch9_protocol as PP
c = CC.chain('izh', [('gap', 1, 0), ('chem', 2, 1)], g_gap=0.5, g_chem=0.075); x = c.find_equilibrium()
D = np.round(np.arange(-3, 12, 0.25), 3); tp = 5.0 + D; T = 60.0
out = {}
def sens(dt, se, amp=0.1):
    r = PP.probe_arms(c, x, [(0,5.0,3.0,10.0)], 0, tp, amp, 0.2, T, dt=dt, save_every=se)
    s = r['spikes']; return (s[:,0]-s[:,1])/(2*r['q'])
s1, s2, s4 = sens(0.01, 5), sens(0.005, 10), sens(0.0025, 20)
m = ~np.isnan(s4) & ~np.isnan(s1)
scale = np.nanmax(np.abs(s4))
out['sens_err_dt0.01_vs_0.0025 (max/scale)'] = float(np.max(np.abs(s1[m]-s4[m]))/scale)
out['sens_err_dt0.005_vs_0.0025 (max/scale)'] = float(np.max(np.abs(s2[m]-s4[m]))/scale)
sa = sens(0.01, 5, amp=0.05); sb = sens(0.01, 5, amp=0.2)
out['sens_linearity_amp0.05_vs_0.2 (max/scale)'] = float(np.nanmax(np.abs(sa-sb))/scale)
print(json.dumps(out, indent=1)); json.dump(out, open('out/e1_izh_sens_validation.json','w'), indent=1)
np.savez('out/izh_conv_check.npz', D=D, s1=s1, s4=s4)
