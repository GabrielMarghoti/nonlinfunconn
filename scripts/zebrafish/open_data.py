
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.signal import detrend, savgol_filter

import sys


load_cache = '--load-cache' in sys.argv
plot_data = '--plot' in sys.argv

data_dir = "/home/gabriel/projects/nonlinfunconn/data/zebrafish/"
fig_dir = "/home/gabriel/projects/nonlinfunconn/figures/zebrafish/"

os.makedirs(data_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

stimdata_tag = "20250625_5_1_data.mat"
spondata_tag = "20250625_5_1_spondata.mat"
# "20250625_6_1_data.mat"
# "20250625_5_1_data.mat"
# "20250625_5_2_data.mat"
# "20250625_6_2_data.mat"

dt = 0.3

# Parameters for Savitzky–Golay filter
sgv_window_length = 10  # must be odd and < time_len
polyorder = 4       # typical values: 2 or 3


file_path = os.path.join(data_dir, stimdata_tag)

sponfile_path = os.path.join(data_dir, spondata_tag)

print(f"Loading data from: {file_path}")

print(f"Loading spontaneous data from: {sponfile_path}")

output_data_dir = os.path.join(data_dir, stimdata_tag.replace(".mat", "_cache"))
output_fig_dir = os.path.join(fig_dir, stimdata_tag.replace(".mat", "_cache"))
os.makedirs(output_data_dir, exist_ok=True)
os.makedirs(output_fig_dir, exist_ok=True)

stim_data_path = os.path.join(output_data_dir, "stim_data.npy")
spon_data_path = os.path.join(output_data_dir, "spon_data.npy")

if load_cache and os.path.exists(stim_data_path) and os.path.exists(spon_data_path):
    # Load cached processed data
    data = np.load(stim_data_path)
    spondata = np.load(spon_data_path)
    print("Loaded cached data.")
    
else:

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

    # Compute F0 as baseline per region (e.g. median across time)
    F0 = np.median(spondata, axis=1, keepdims=True)

    # ΔF/F0
    spondata = (spondata - F0) / (F0 + 1e-8)  # add epsilon to avoid div/0
    spondata = savgol_filter(spondata, 
                                window_length=sgv_window_length,
                                polyorder=polyorder, axis=1)

    spondata = detrend(spondata, axis=1, type='linear')

##########################################################################################################
##### Stimulated data analysis
##########################################################################################################
    sample = np.array(mat[mat[key][0,0]])
    time_len, n_regions = sample.shape
    print(f"Sample shape: time_len={time_len}, n_regions={n_regions}")

    # Initialize array: (n_trials, n_regions, time_len)
    data = np.zeros((n_trials, n_regions, time_len))

    for trial_idx in range(data.shape[0]):
        
        _trial_data = np.array(mat[mat[key][trial_idx, 0]])  # shape: (time, regions)
        trial_data = _trial_data.T  # shape: (regions, time)

        # Replace NaNs/Infs with baseline (e.g. mean or 0)
        trial_data = np.nan_to_num(trial_data, nan=0.0, posinf=0.0, neginf=0.0)

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


    # Normalize all data by the global maximum value across all trials and regions
    global_max = np.max(np.abs(data))
    if global_max > 0:
        data = data / global_max
        spondata = spondata / global_max

    # Save processed data to output_data_dir
    np.save(os.path.join(output_data_dir, "stim_data.npy"), data)
    np.save(os.path.join(output_data_dir, "spon_data.npy"), spondata)


if plot_data:
    for trial_idx in range(data.shape[0]):
        
        plt.figure(figsize=(12, 12))

        time =  dt*np.arange(data.shape[2])

        # Compute variance per region
        variances = np.var(data[trial_idx], axis=1)
        offset = 1000*np.mean(variances)  # Base offset for stacking

        # Store tick positions
        tick_positions = []

        cumulative_offset = 0.0
        for region_idx in range(data[trial_idx].shape[0]):
            # Plot trace with current offset
            plt.plot(time, data[trial_idx][region_idx] + cumulative_offset, lw=2)
            # Save current offset for tick
            tick_positions.append(cumulative_offset)

            # Update offset with variance
            cumulative_offset += offset
            
        plt.xlabel("Time (s)")
        plt.ylabel("Region")
        plt.title(f"Traces of regions (trial {trial_idx})")

        # Put ticks where the traces were actually offset
        plt.yticks(tick_positions, np.arange(1, data[trial_idx].shape[0] + 1))

        plt.savefig(os.path.join(output_fig_dir, f"trial_{trial_idx}_traces.png"))
        plt.close()


    plt.figure(figsize=(24, 12))

    spondata_time = dt*np.arange(spondata.shape[1])

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

    plt.xlabel("Time (s)")
    plt.ylabel("Region")
    plt.title(f"Traces of regions for spontaneous activity")

    plt.savefig(os.path.join(output_fig_dir, f"spontaneous_traces.png"))
    plt.close()
