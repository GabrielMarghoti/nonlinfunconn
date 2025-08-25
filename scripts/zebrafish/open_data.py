
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.signal import detrend, savgol_filter


# -------------------------------
# 1. Load .mat file (v7.3 only)
# -------------------------------

data_dir = "/home/gabriel/projects/nonlinfunconn/data/zebrafish/"
os.makedirs(data_dir, exist_ok=True)

stimdata_tag = "20250625_5_1_data.mat"
spondata_tag = "20250625_5_1_spondata.mat"
# "20250625_6_1_data.mat"
# "20250625_5_1_data.mat"
# "20250625_5_2_data.mat"
# "20250625_6_2_data.mat"


# Parameters for Savitzky–Golay filter
sgv_window_length = 10  # must be odd and < time_len
polyorder = 4       # typical values: 2 or 3


file_path = os.path.join(data_dir, stimdata_tag)

sponfile_path = os.path.join(data_dir, spondata_tag)

print(f"Loading data from: {file_path}")

print(f"Loading spontaneous data from: {sponfile_path}")

output_dir = os.path.join(data_dir, stimdata_tag.replace(".mat", "_results"))
os.makedirs(output_dir, exist_ok=True)

mat = h5py.File(file_path, 'r')
sponmat = h5py.File(sponfile_path, 'r')

# -------------------------------
# 2. Explore keys
# -------------------------------
keys = [k for k in mat.keys() if not k.startswith("__") and not k.startswith("#")]
key = keys[0]
print(f"Using key: {key}, shape: {mat[key].shape}")

sponkeys = [k for k in sponmat.keys() if not k.startswith("__") and not k.startswith("#")]
sponkey = sponkeys[0]
print(f"Using key: {sponkey}, shape: {sponmat[sponkey].shape}")

n_trials = mat[key].shape[0]

# -------------------------------
# 3. Load all trials
# -------------------------------
# First trial to check shape
spondata = np.array(sponmat[sponkey]).T

spondata = spondata[:, 100:]

spontime_len, spon_n_regions = spondata.shape
print(f"Spon data shape: time_len={spontime_len}, n_regions={spon_n_regions}")

spondata = np.nan_to_num(spondata, nan=0.0, posinf=0.0, neginf=0.0)

# ---- Preprocessing ----
# Detrend each region


# Compute F0 as baseline per region (e.g. median across time)
F0 = np.median(spondata, axis=1, keepdims=True)

# ΔF/F0
spondata = (spondata - F0) / (F0 + 1e-8)  # add epsilon to avoid div/0
spondata = savgol_filter(spondata, 
                            window_length=sgv_window_length,
                            polyorder=polyorder, axis=1)

# Normalize per region (z-score)
#mean = trial_data.mean()
#std = trial_data.std() + 1e-8
#trial_data = (trial_data - mean) / std

spondata = detrend(spondata, axis=1, type='linear')


plt.figure(figsize=(24, 12))

spondata_time = np.arange(spondata.shape[1])

# Compute variance per region
variances = np.var(spondata, axis=1)
offset = 400*np.mean(variances)  # Base offset for stacking

# Store tick positions
tick_positions = []

cumulative_offset = 0.0
for region_idx in range(spondata.shape[0]):
    plt.plot(spondata_time, spondata[region_idx]+ cumulative_offset, lw=2)
    # Save current offset for tick
    tick_positions.append(cumulative_offset)

    # Update offset with variance
    cumulative_offset += offset


# Put ticks where the traces were actually offset
plt.yticks(tick_positions, np.arange(1, spondata.shape[0] + 1))

plt.xlabel("Time")
plt.ylabel("Region")
plt.title(f"Traces of regions for spontaneous activity")

plt.savefig(os.path.join(output_dir, f"spontaneous_traces.png"))
plt.close()















### Stimulated data analysis

sample = np.array(mat[mat[key][0,0]])
time_len, n_regions = sample.shape
print(f"Sample shape: time_len={time_len}, n_regions={n_regions}")

# Initialize array: (n_trials, n_regions, time_len)
data = np.zeros((n_trials, n_regions, time_len))

for trial_idx in range(n_trials):
    
    _trial_data = np.array(mat[mat[key][trial_idx, 0]])  # shape: (time, regions)
    trial_data = _trial_data.T  # shape: (regions, time)

    # Replace NaNs/Infs with baseline (e.g. mean or 0)
    trial_data = np.nan_to_num(trial_data, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- Preprocessing ----
    # Detrend each region


    # Compute F0 as baseline per region (e.g. median across time)
    F0 = np.median(trial_data, axis=1, keepdims=True)

    # ΔF/F0
    trial_data = (trial_data - F0) / (F0 + 1e-8)  # add epsilon to avoid div/0
    trial_data = savgol_filter(trial_data, 
                               window_length=sgv_window_length,
                               polyorder=polyorder, axis=1)
    
    # Normalize per region (z-score)
    #mean = trial_data.mean()
    #std = trial_data.std() + 1e-8
    #trial_data = (trial_data - mean) / std

    trial_data = detrend(trial_data, axis=1, type='linear')

    # Save back
    data[trial_idx] = trial_data

    plt.figure(figsize=(12, 12))

    time = np.arange(trial_data.shape[1])

    # Compute variance per region
    variances = np.var(trial_data, axis=1)
    offset = 1000*np.mean(variances)  # Base offset for stacking

    # Store tick positions
    tick_positions = []

    cumulative_offset = 0.0
    for region_idx in range(trial_data.shape[0]):
        # Plot trace with current offset
        plt.plot(time, trial_data[region_idx] + cumulative_offset, lw=2)
        # Save current offset for tick
        tick_positions.append(cumulative_offset)

        # Update offset with variance
        cumulative_offset += offset
        
    plt.xlabel("Time")
    plt.ylabel("Region")
    plt.title(f"Traces of regions (trial {trial_idx})")

    # Put ticks where the traces were actually offset
    plt.yticks(tick_positions, np.arange(1, trial_data.shape[0] + 1))

    plt.savefig(os.path.join(output_dir, f"trial_{trial_idx}_traces.png"))
    plt.close()