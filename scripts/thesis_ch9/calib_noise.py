import numpy as np
import ch9_circuits as CC
for model, kw in (('hh', {}), ('izh', dict(g_gap=0.5, g_chem=0.075))):
    c = CC.chain(model, [('gap', 1, 0), ('chem', 2, 1)], **kw); x = c.find_equilibrium()
    for sig in ((0.5, 1.0, 1.5, 2.0, 3.0) if model == 'hh' else (0.25, 0.5, 0.75, 1.0, 1.5)):
        B = 8
        r = c.run(np.tile(x, (B,1)), np.zeros((B,1,4)), np.arange(B)+100, 2000.0, 0.01, sigma=sig, tau_noise=5.0, save_every=100, max_spikes=2000)
        n = np.sum(~np.isnan(r['spikes']), axis=-1).mean(0) / 2.0     # spikes per second per node
        V = r['V']; sd = V[:, 100:, :].std(axis=(0,1))
        print(f"{model} sigma={sig:4.2f}: spont. rate (Hz) nodes 1..3 = {np.round(n,2)}   V sd = {np.round(sd,2)}")
