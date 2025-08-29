
import sys

import numpy as np
import os
import matplotlib.pyplot as plt
import pickle # for cache saving/loading

from scipy.signal import detrend, savgol_filter

import nonlinfunconn as nlfc # for non-linear kernels

from nonlinfunconn.models.nm import NM

plot_data = '--plot' in sys.argv
load_cache = '--load-cache' in sys.argv

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

#regions to analyze

regions = np.array([34, 35, 36, 44, 45]) # np.array([2, 4, 6, 8, 10, 16, 20, 28, 30, 34, 35, 36, 44, 45, 50])

trials = np.array([5,7,8,11,12, 19]) #

dt = 0.3

fitting_window = 120

lowering_fit_resolution_step = 4

file_path = os.path.join(data_dir, stimdata_tag)

sponfile_path = os.path.join(data_dir, spondata_tag)

print(f"Loading data from: {file_path}")

print(f"Loading spontaneous data from: {sponfile_path}")

output_data_dir = os.path.join(data_dir, stimdata_tag.replace(".mat", "_fitted_model_results"))
output_fig_dir = os.path.join(fig_dir, stimdata_tag.replace(".mat", "_fitted_model_results"))
os.makedirs(output_data_dir, exist_ok=True)
os.makedirs(output_fig_dir, exist_ok=True)

stim_data_path = os.path.join(os.path.join(data_dir, stimdata_tag.replace(".mat", "_cache")), "stim_data.npy")
spon_data_path = os.path.join(os.path.join(data_dir, stimdata_tag.replace(".mat", "_cache")), "spon_data.npy")

if os.path.exists(stim_data_path) and os.path.exists(spon_data_path):
    # Load cached processed data
    data = np.load(stim_data_path)
    spondata = np.load(spon_data_path)
    print("Loaded cached signal data.")
    
else:
    print("Cache not found, run open_data.py file for preprocessing.")

# downsample the dataset
data = data[trials][:, regions, :]
spondata = spondata[regions, :]

n_regions = regions.shape[0] # data.shape[1]


model_parameters = {
    "tau": 1.0,                 # Membrane capacitance  [pF]
    "w": 1.0,                   # Synaptic weight
    "beta": 0.1,                # Steepness of the sigmoid function
    "xth": 1.0,                  # Threshold potential for synapse activation
}
if load_cache:
    # Check if the cache file exists
    cache_file_path = os.path.join(output_data_dir, f"fitted_parameters_{regions}.pkl")
    
    if os.path.exists(cache_file_path):
        # Load parameters after fitting from the .pkl file
        with open(cache_file_path, "rb") as f:  # 'rb' not 'r'
            fitted_parameters = pickle.load(f)
        model_parameters.update(fitted_parameters)
        print(f"Loaded cached fitted parameters from '{cache_file_path}'.")
        
    else:
        print(f"Warning: Cache file '{cache_file_path}' does not exist. Proceeding without loading cached parameters.")


                
# Set main diagonal of fitted_parameters['w'] to 0
if isinstance(model_parameters['w'], np.ndarray):
    np.fill_diagonal(model_parameters['w'], 0)
elif isinstance(model_parameters['w'], list):
    w_array = np.array(model_parameters['w'])
    np.fill_diagonal(w_array, 0)
    model_parameters['w'] = w_array
    
nm_kernels = nlfc.GreenFunctions(
    model = NM(n_regions, model_parameters),
    x = data[:, :, 0:fitting_window:lowering_fit_resolution_step],
    dt = dt,
)



print("NEGF fitting")
min_constrain_dict = {
                    "tau": 0.0001  ,             
                    "w": -1000,                   # Synaptic weight
                    "beta": 0.00001,                # Steepness of the sigmoid function
                    "xth": -100.0,                  # Threshold potential for synapse activation
                }
max_constrain_dict = {
                    "tau":1000  ,             
                    "w": 1000,                   # Synaptic weight
                    "beta": 1000,                # Steepness of the sigmoid function
                    "xth": 100.0,                  # Threshold potential for synapse activation
                }

fitted_parameters = nm_kernels.ADAM_fit(
    x = data[:, :, 0:fitting_window:lowering_fit_resolution_step],
    dt = dt,
    #target_nodes=np.arange(0, n_regions),  
    max_iters=1000,
    constrain=(min_constrain_dict, max_constrain_dict),
    rms_tol=1e-6,
    learning_rate = 5e-3,
    beta1 = 0.9,
    beta2 = 0.99,
    eps = 1e-3,
    parameter_to_fit_list=['tau', 'w', 'beta', 'xth'],
    #loss_method='correlation',
    verbose = True,
)



print('FITTING DONE')
print("fitted parameters: ", fitted_parameters['w'])

# Set main diagonal of fitted_parameters['w'] to 0
if isinstance(fitted_parameters['w'], np.ndarray):
    np.fill_diagonal(fitted_parameters['w'], 0)
elif isinstance(fitted_parameters['w'], list):
    w_array = np.array(fitted_parameters['w'])
    np.fill_diagonal(w_array, 0)
    fitted_parameters['w'] = w_array

# Save parameters after fitting as a tab-delimited text file
with open(os.path.join(output_data_dir, f"fitted_parameters_{regions}.txt"), "w") as f:
    f.write("Parameter\tValue\n")
    for key, value in fitted_parameters.items():
        f.write(f"{key}\t{value}\n")
f.close()

# Save parameters after fitting in a JSON format for easier loading
with open(os.path.join(output_data_dir, f"fitted_parameters_{regions}.pkl"), "wb") as f:
    pickle.dump(fitted_parameters, f)

f.close()
print(f"Fitted parameters saved to {output_data_dir}")

# Compute green functions using the higher time resolution, but the fitted parameters
nm_kernels_trials = nlfc.GreenFunctions(
    model = NM(n_regions, fitted_parameters),
    x = data,
    dt = dt,
)

# Compute green functions using the higher time resolution, but the fitted parameters
nm_kernels_spon = nlfc.GreenFunctions(
    model = NM(n_regions, fitted_parameters),
    x = np.array([spondata]),
    dt = dt,
)



predspondata = np.zeros_like(spondata)

time_len = spondata.shape[1]

for i in np.arange(0, n_regions):
    
    predspondata[i, :] = np.full_like(predspondata[i, :], spondata[i, 0])

    for j in range(n_regions):
        
        delta_j = spondata[j, :] - spondata[j, 0]
        predspondata[i] += nlfc.utils.nontt_conv(nm_kernels_spon.g[0][i, j], delta_j, dt=dt)
        
        if (np.all(abs(nm_kernels_spon.g[0][i, j]) < 1e-02)): 
            continue
        #nlfc.utils.plots.t_t_heatmap(x, g[i, j, :, :], os.path.join(ie_dir, f'negf_g_heatmap_neuron_pair_{i}_{j}_stimulation_{str(ie)}.png'))
        
        nlfc.utils.plots.time_level_curves( dt*np.arange(time_len), nm_kernels_spon.g[0][i, j], nm_kernels_spon.g[0][i, j,  -1], xlabel=None, ylabel="g(t,t')", title=f"Neurons : {i}<-{j}", save_path= os.path.join(output_fig_dir,f'fitted_negf_direct_g_neurons_neuron_pair_{i}<-{j}.png'))
        
        #nlfc.utils.plots.t_t_heatmap(time_fit, G[:, :, i, 0], os.path.join(ie_dir, f'negf_G{G_degree}_heatmap_neuron_pair_{labels[n_regions[i]]}_{labels[stim]}_stimulation_{str(ie)}.png'))
        #nlfc.utils.plots.time_level_curves(time_fit, G[i, j], G[0][i, j][-1, :], xlabel=None, ylabel="G(t,t')", title=f"Neurons : {i}<-{j}", save_path= os.path.join(stimulus_fig_path,f'fitted_negf_G_neurons_neuron_pair_{i}<-{j}.png'))





if plot_data:
    plt.figure(figsize=(24, 12))

    spondata_time = dt*np.arange(time_len)

    # Compute variance per region
    variances = np.var(spondata, axis=1)
    offset = 400*np.mean(variances)  # Base offset for stacking

    # Store tick positions
    tick_positions = []

    cumulative_offset = 0.0
    for region_idx in range(n_regions):
        color = 'black' if region_idx % 2 == 0 else 'gray'
        plt.plot(spondata_time, spondata[region_idx]+ cumulative_offset, lw=2, color=color, label='data' if region_idx==0 else "")
        plt.plot(spondata_time, predspondata[region_idx]+ cumulative_offset, lw=2, ls='--', color=color, label='model prediction' if region_idx==0 else "")
        # Save current offset for tick
        tick_positions.append(cumulative_offset)

        # Update offset with variance
        cumulative_offset += offset


    # Put ticks where the traces were actually offset
    plt.yticks(tick_positions, regions)

    plt.xlabel("Time")
    plt.ylabel("Region")
    plt.title(f"Traces of regions for spontaneous activity")

    plt.savefig(os.path.join(output_fig_dir, f"spontaneous_traces_predictions.png"))
    plt.close()

    # plot training dataset
    output_fig_dir_stim_trials = os.path.join(output_fig_dir, "stimulated_trials")
    os.makedirs(output_fig_dir_stim_trials, exist_ok=True)
    for trial_idx in range(data.shape[0]):
        trial_data = data[trial_idx]
        
        n_regions, time_len = trial_data.shape
        
        pred_trial_data  = np.zeros_like(trial_data)
        
        for i in range(n_regions):
            pred_trial_data[i] = np.full_like(pred_trial_data[i, :], trial_data[i, 0])
            for j in range(n_regions):
                
                delta_j = trial_data[j, :] - trial_data[j, 0]
                pred_trial_data[i] += nlfc.utils.nontt_conv(nm_kernels_trials.g[trial_idx][i, j], delta_j, dt=dt)
            
        plt.figure(figsize=(12, 12))

        data_time = dt*np.arange(time_len)

        # Compute variance per region
        variances = np.var(trial_data, axis=1)
        offset = 400*np.mean(variances)  # Base offset for stacking

        # Store tick positions
        tick_positions = []

        cumulative_offset = 0.0
        for region_idx in range(n_regions):
            color = 'black' if region_idx % 2 == 0 else 'gray'
            plt.plot(data_time, trial_data[region_idx]+ cumulative_offset, lw=2, color=color, label='data' if region_idx==0 else "")
            plt.plot(data_time, pred_trial_data[region_idx]+ cumulative_offset, lw=2, ls='--', color=color, label='model prediction' if region_idx==0 else "")
            # Save current offset for tick
            tick_positions.append(cumulative_offset)

            # Update offset with variance
            cumulative_offset += offset


        # Put ticks where the traces were actually offset
        plt.yticks(tick_positions, regions)

        plt.xlabel("Time")
        plt.ylabel("Region")
        plt.title(f"Traces of regions for stimulated activity trial #{trials[trial_idx]}")

        plt.savefig(os.path.join(output_fig_dir_stim_trials, f"stimulated_traces_predictions_trial{trial_idx}.png"))
        plt.close()



