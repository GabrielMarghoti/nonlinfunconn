import numpy as np
import ch9_circuits as CC
PUMP = [(0, 5.0, 3.0, 10.0)]
def spikes_of(c, T=150.0):
    x = c.find_equilibrium()
    r = c.run(x, CC.pulse_table(PUMP)[None], [0], T, 0.01, save_every=10)
    s = r['spikes'][0]
    return x, c.eq_drift, [np.round(s[i][~np.isnan(s[i])], 2) for i in range(c.N)]

print("isolated node, pump only:")
c = CC.chain('izh', [], N=1); x, dr, s = spikes_of(c)
print(f"  rest v*={x[0]:.3f} u*={x[1]:.3f}  spikes: {s[0]}")
for g in (0.1, 0.2, 0.3, 0.5, 0.8):
    c = CC.chain('izh', [('gap', 1, 0), ('gap', 2, 1)], g_gap=g); x, dr, s = spikes_of(c)
    print(f"gap-gap  A={g:4.2f}: n_spikes={[len(v) for v in s]}  first={[v[0] if len(v) else None for v in s]}  drift={dr:.1e}")
for B in (0.04, 0.06, 0.07, 0.075, 0.08, 0.085):
    c = CC.chain('izh', [('chem', 1, 0), ('chem', 2, 1)], g_chem=B); x, dr, s = spikes_of(c)
    tonic = B*x[2*3+3*2+1]*(0-x[2])
    print(f"chem-chem B={B:5.3f}: rest={np.round(x[:3],2)} tonic3={tonic:.2f}  n_spikes={[len(v) for v in s]}  drift={dr:.1e}")
