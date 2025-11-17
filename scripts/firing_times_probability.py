import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Model parameters
# -----------------------
N_pre = 40           # number of presynaptic neurons
N_spikes = 5        # spikes per presynaptic neuron
tau = 10.0           # PSP time constant (ms)
w =  np.random.normal(0, 0.2, size=N_pre)# np.ones(N_pre) * 0.1 #   # synaptic weights
V_th = 1.0           # firing threshold
T_max = 200.0        # simulation window (ms)
dt = 0.1             # time step (ms)
N_trials = 1000      # Monte Carlo samples

def time_translation(t_j, t_i):
    """
    Translate times in t_j by subtracting a reference time(s) t_i.

    Behaviors:
    - If t_i is None: return t_j unchanged.
    - If t_i is a scalar: return t_j - t_i (works for scalar or array t_j).
    - If t_i is an array:
        * If t_j has the same length as t_i: return elementwise t_j - t_i.
        * Otherwise: for each element of t_j, subtract the most recent t_i that is <= t_j.
          If no such t_i exists, the t_j element is left unchanged (subtract 0).
    """
    if t_i is None:
        return t_j

    t_j_arr = np.asarray(t_j)
    t_i_arr = np.asarray(t_i)

    # scalar t_i: simple subtraction
    if t_i_arr.size == 1:
        return t_j_arr - float(t_i_arr)

    # both arrays and same length: elementwise subtraction
    if t_j_arr.shape == t_i_arr.shape:
        return t_j_arr - t_i_arr

    # ensure 1D arrays for searchsorted logic
    t_i_sorted = np.sort(t_i_arr)

    # handle scalar t_j
    if t_j_arr.ndim == 0:
        idx = np.searchsorted(t_i_sorted, t_j_arr)
        if idx == 0:
            return float(t_j_arr)  # no earlier t_i, leave unchanged
        return float(t_j_arr) - float(t_i_sorted[idx - 1])

    # vectorized case: for each t_j find previous t_i (if any) and subtract it
    idxs = np.searchsorted(t_i_sorted, t_j_arr)
    prev = np.empty_like(t_j_arr, dtype=float)
    mask = idxs > 0
    prev[mask] = t_i_sorted[idxs[mask] - 1]
    prev[~mask] = 0.0
    return t_j_arr - prev


# -----------------------
# PSP kernel
# -----------------------
def h(t):
    """Exponential PSP kernel."""
    return np.where(t >= 0, np.exp(-t / tau), 0.0)

# -----------------------
# Presynaptic spike distribution
# -----------------------
def sample_presynaptic_times(N_samples):
    """
    Each presynaptic neuron fires once, drawn from a normal distribution
    centered at 50 ms with std 10 ms.
    """
    mu, sigma = 100.0, 10.0
    # sample spike times from a Poisson distribution (mean ~ mu ms)
    #t_pres = np.random.normal(mu, sigma, size=N_pre)
    t_pres = np.random.poisson(lam=mu, size=N_samples).astype(float)
    return np.clip(t_pres, 0, T_max)

# -----------------------
# Compute postsynaptic potential
# -----------------------
time = np.arange(0, T_max, dt)

def compute_V(t_pres):
    """Compute membrane potential V(t) given presynaptic times."""
    V = np.zeros_like(time)
    for j in range(N_pre):
        for spike in range(N_spikes):
            V += w[j] * h(time - t_pres[(j-1)*(spike-1)])
    return V

def find_spike_time(V):
    """Return the first time V crosses threshold."""
    above = np.where(V >= V_th)[0]
    if len(above) == 0:
        return None
    return time[above[0]]

# -----------------------
# Monte Carlo simulation
# -----------------------
spike_times_1st = []
spike_times_2nd = []
dT_pre = []
dT_1st = []
dT_2nd = []
all_presynaptic_times = []

for trial in range(N_trials):
    t_pres = sample_presynaptic_times(N_pre*N_spikes)
    V1 = compute_V(t_pres)
    t_post_1st = find_spike_time(V1)
    if t_post_1st is not None:
        spike_times_1st.append(t_post_1st)
    
    all_presynaptic_times.extend(t_pres.tolist())

    dt_pres = time_translation(t_pres, t_pres)

    dT_pre.extend(dt_pres)


    dt_1st = time_translation(t_pres, t_post_1st)
    dT_1st.extend(dt_1st)


for trial in range(N_trials):
    
    # if no first-spike samples are available, skip this trial
    if len(spike_times_1st) == 0:
        continue
    # sample N_pre presynaptic times from the list of 1st-spike times (with replacement)
    t_pres_2nd = np.random.choice(spike_times_1st, size=N_pre*N_spikes, replace=True)

    V2 = compute_V(t_pres_2nd)
    t_post_2nd = find_spike_time(V2)
    if t_post_2nd is not None:
        spike_times_2nd.append(t_post_2nd)

    dt_2nd = time_translation(t_pres, t_post_2nd)
    dT_2nd.extend(dt_2nd)
    

spike_times_1st = np.array(spike_times_1st)
p_spike_1st = len(spike_times_1st) / N_trials

spike_times_2nd = np.array(spike_times_2nd)
p_spike_2nd = len(spike_times_2nd) / N_trials


print(f"Spike probability: {p_spike_1st:.3f}")
if len(spike_times_1st) > 0:
    print(f"Mean output firing time 1st: {np.mean(spike_times_1st):.2f} ms")
    print(f"Mean output firing time 2nd: {np.mean(spike_times_2nd):.2f} ms")

# -----------------------
# Plot presynaptic & postsynaptic distributions
# -----------------------
plt.figure(figsize=(8, 5))

# Presynaptic distribution (background)
plt.hist(dT_pre, bins=60, density=True,
         color="lightgray", alpha=0.6, label="Presynaptic firing times")

# Postsynaptic distribution (foreground)
if len(spike_times_1st) > 0:
    plt.hist(dT_1st, bins=60, density=True,
             color="steelblue", alpha=0.6, label="1st Postsynaptic firing times")
    plt.hist(dT_2nd, bins=60, density=True,
             color="orange", alpha=0.6, label="2nd Postsynaptic firing times")

plt.xlabel("Time (ms)")
plt.ylabel("Probability density")
plt.title("Presynaptic vs Postsynaptic Firing-Time Distributions")
plt.legend()
plt.tight_layout()
plt.show()
