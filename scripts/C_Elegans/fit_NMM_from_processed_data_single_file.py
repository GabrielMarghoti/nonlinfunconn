#
# code for fit of convolution kernels, combination of linear (exponentials) or non-linear (NEGF) kernels
#
# inspired by 
# https://github.com/leiferlab/pumpprobe/tree/main/scripts/fconnectivity/fit_responses_constrained_stim_eci
# https://github.com/leiferlab/pumpprobe/tree/main/scripts/fconnectivity/figures/compare_connectomes/funatlas_vs_correlations2
# Kunert et al., PRE 89 052805 (2014) for parameter estimation

import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pickle # for cache saving/loading
import gc

import nonlinfunconn as nlfc # for non-linear kernels
from nonlinfunconn.models.nm import NM

load_cache = "--load-cache" in sys.argv

only_labeled_neurons = "--only-labeled-neurons" in sys.argv

# default 
figures_path = "figures/C_elegans_pumpprobre_exp/"
data_path = "data/C_elegans_pumpprobre_exp/"
for arg in sys.argv:
    _arg = arg.split(":")
    if _arg[0] == "--worm-type": 
        worm_type = str(_arg[1])
    if _arg[0] == "--dataset-path":
        stim_neu_path = _arg[1]
    if _arg[0] == "--figures-path":
        figures_path = _arg[1]
        
# Load the kunert_ODE_parameters dictionary from a pickled file
kunert_ODE_parameters_file_path = os.path.join(data_path, "kunert_ODE_parameters.pkl")
with open(kunert_ODE_parameters_file_path, "rb") as f:
    kunert_ODE_parameters = pickle.load(f)

kunert_ODE_parameters['C'] = kunert_ODE_parameters['C']*200 # Helps the fitting to find best C, which converts the membrane potential to  the actual signal proportional to the model V, effectivelly a proportional constant that multiplies the capacitance

print(f"ODE model parameters loaded from {kunert_ODE_parameters_file_path}")

# Iterate over the folders whcih 

print(" Processing specific stimulated neuron at ", stim_neu_path)     

with open(os.path.join(stim_neu_path, 'processed_data.pkl'), 'rb') as f:
    loaded_data = pickle.load(f)

# Load pump probe experiment processed data
worm_type          = loaded_data["worm_type"]
responding_neurons = loaded_data["responding_neurons"]
stim_neuron        = loaded_data["stim_neuron"]
neuron_labels      = loaded_data["neuron_labels"]
neuron_positions   = loaded_data["neuron_positions"]
signal_raw         = loaded_data["signal_raw"]
signal_smooth      = loaded_data["signal_smooth"]
time_fit           = loaded_data["time_fit"]
time_plt           = loaded_data["time_plt"]
dt                 = loaded_data["dt"]
stim_begin_idx     = loaded_data["stim_begin_idx"]

# Close the loaded data file
f.close()

n_stimuli, n_neurons, time_len = signal_smooth.shape

# Skip datasets where the number of stimulations is not between 2 and 3
#if not (2 <= num_stimulations <= 4): 
#    continue

stim_neuron_label = neuron_labels[stim_neuron]

output_data_dir = os.path.join(stim_neu_path, f"fit_negf")

output_figure_dir = os.path.join(figures_path, os.path.relpath(output_data_dir, data_path))
            
# Find first neighbors (nodes connected to stim via either gap or syn)
first_neighbors = np.where((kunert_ODE_parameters["gamma_g"][:, stim_neuron] > 0) | (kunert_ODE_parameters["gamma_s"][:, stim_neuron] > 0))[0]
responsive_first_neighbors = [first_neighbors[i] for i in range(len(first_neighbors)) if first_neighbors[i] in responding_neurons] # filter nonresponsive first neigbors

# Find second neighbors (nodes connected to first neighbors via either gap or syn)
second_neighbors = np.where((np.sum(list(kunert_ODE_parameters["gamma_g"][:, responsive_first_neighbors]), axis=1) > 0) | 
                            (np.sum(list(kunert_ODE_parameters["gamma_s"][:, responsive_first_neighbors]), axis=1) > 0))[0]

responsive_second_neighbors = [second_neighbors[i] for i in range(len(second_neighbors)) if second_neighbors[i] in responding_neurons] # filter nonresponsive first neigbors

# Create a set of allowed nodes (stim + first + second neighbors)
allowed_nodes = set([stim_neuron]).union(set(responsive_first_neighbors)).union(set(responsive_second_neighbors))

# Filter responding list to only include allowed nodes # check again if is responsive
first_second_responsive_nodes = [node for node in responding_neurons if node in allowed_nodes]


responding_neurons = first_second_responsive_nodes # = list(responding_neurons) # consider all responsive neurons

n_responding_neurons = len(responding_neurons)

print('Number of responding neurons: ', n_responding_neurons)

responding_neurons_labels = np.array(neuron_labels)[responding_neurons]  # Create an array of labels for responding_neurons indexes
labeled_neurons = [label != "" for label in responding_neurons_labels]  # Create a boolean list for non-empty labels

n_responding_neurons_labeled = len(np.array(responding_neurons)[labeled_neurons])


responses_correlations = np.zeros((n_responding_neurons, n_stimuli, n_stimuli))
trial_variations = np.zeros(n_responding_neurons)

for i in range(n_responding_neurons):
    neu_i = responding_neurons[i]
    responses_correlations[i, :, :] = np.corrcoef(
                signal_smooth[:, neu_i, stim_begin_idx:]
            )
    # Calculate the standard deviation of the correlation matrix for each neuron
    trial_variations[i] = np.std(responses_correlations[i, :, :])

# Find the neuron with the most variation in trial correlations
response_neuron_toplot_idx = np.argmax(trial_variations)
response_neuron_toplot = responding_neurons[response_neuron_toplot_idx] 

# Responding neurons positions
responding_positions = neuron_positions[responding_neurons]
responding_positions = np.array(responding_positions, dtype=np.float64)  # Ensure numeric type

# Handle cases where positions are None or invalid
valid_positions_mask = ~np.isnan(responding_positions[:,0])
valid_positions = responding_positions[valid_positions_mask]

# Ensure valid_positions is not empty
if valid_positions.size == 0:
    raise ValueError("No valid positions found for responding neurons.")

# Initialize distance matrix with NaN values
distance_matrix_responding_neurons = np.full((len(responding_positions), len(responding_positions)), np.nan)

# Compute distances only for valid positions
valid_distance_matrix = np.linalg.norm(
    valid_positions[:, np.newaxis, :] - valid_positions[np.newaxis, :, :], axis=-1
)

# Fill the valid distances into the main distance matrix
for i, valid_i in enumerate(np.where(valid_positions_mask)[0]):
    for j, valid_j in enumerate(np.where(valid_positions_mask)[0]):
        distance_matrix_responding_neurons[valid_i, valid_j] = valid_distance_matrix[i, j]

# Ensure the directories for figures and data exist
os.makedirs(output_data_dir, exist_ok=True)
os.makedirs(output_figure_dir, exist_ok=True)


stimulus_data_path = [os.path.join(output_data_dir, f'trial_{ie_idx}') for ie_idx in range(n_stimuli)]
stimulus_fig_path = [os.path.join(output_figure_dir, f'trial_{ie_idx}') for ie_idx in range(n_stimuli)]


# save connectome based network considering only responsive neurons over all stimulations
resp_gamma_g = kunert_ODE_parameters["gamma_g"][responding_neurons][:, responding_neurons] 
resp_gamma_s = kunert_ODE_parameters["gamma_s"][responding_neurons][:, responding_neurons]
resp_Es = kunert_ODE_parameters["Es"][responding_neurons][:, responding_neurons]

# Plot gamma_g and gamma_s as heatmaps
#nlfc.utils.netplots.connect_matrices_heatmap(resp_gamma_g, resp_gamma_s, responding_neurons_labels, os.path.join(output_figure_dir, 'gamma_g_gamma_s_heatmaps.png'))
# plot neural network
nlfc.utils.netplots.neural_network(resp_gamma_g, resp_gamma_s, resp_Es, responding_neurons_labels, save_path=os.path.join(output_figure_dir, f'Neural_Network_responding_only_connectome_parameters.png'))

# nodes at real positions
#nlfc.utils.netplots.neural_network(gamma_g, gamma_s, Es, responding_labels, positions=responding_neurons_positions[:, [0,1]], save_path=os.path.join(output_figure_dir, f'Neural_Network_responding_only_before_fit_real_positions.png'))

model_parameters = kunert_ODE_parameters.copy()

if load_cache:
    # Check if the cache file exists
    cache_file_path = os.path.join(output_data_dir, "fitted_parameters.pkl")
    
    if os.path.exists(cache_file_path):
        # Load parameters after fitting from the .pkl file
        with open(cache_file_path, "rb") as f:  # 'rb' not 'r'
            fitted_parameters = pickle.load(f)
        model_parameters.update(fitted_parameters)
        
    else:
        print(f"Warning: Cache file '{cache_file_path}' does not exist. Proceeding without loading cached parameters.")
        model_parameters["gamma_g"] = resp_gamma_g
        model_parameters["gamma_s"] = resp_gamma_s
        model_parameters["Es"] = resp_Es

if model_parameters["gamma_g"].shape[0] != n_responding_neurons:
    model_parameters["gamma_g"] = resp_gamma_g
    model_parameters["gamma_s"] = resp_gamma_s
    model_parameters["Es"] = resp_Es

Y_nonlin_fit = np.zeros_like(signal_smooth[:, responding_neurons, stim_begin_idx:])

G_degree = 2
# initialize the greenfunctions class, computing the direct green functions of ecery tryal and every neuron pair interaction
lif_gf = nlfc.GreenFunctions(
    model = LIF(n_responding_neurons, model_parameters),
    x = signal_smooth[:, responding_neurons, stim_begin_idx::],
    dt = dt,
)

G  = lif_gf.total_G(G_degree)

for ie_idx in range(n_stimuli):

    os.makedirs(stimulus_fig_path[ie_idx], exist_ok=True)
    # save plot the NEGF for each stimulation and each neuron pair (consider source only the stim neuron)
    for i in range(n_responding_neurons):
        for j in range(n_responding_neurons):
            neu_i = responding_neurons[i]
            neu_j = responding_neurons[j]
            if i == 0 or (np.all(lif_gf.g[ie_idx][i, j] == 0)): continue
            nlfc.utils.plots.time_level_curves(time_fit, lif_gf.g[ie_idx][i, j], lif_gf.g0[i, j, -1], xlabel=None, ylabel="g(t,t')", title=f"Neurons : {neuron_labels[neu_i]}<-{neuron_labels[neu_j]}", save_path= os.path.join(stimulus_fig_path[ie_idx],f'before_fit_negf_direct_g_neurons_neuron_pair_{neuron_labels[neu_i]}<-{neuron_labels[neu_j]}.png'))
            nlfc.utils.plots.time_level_curves(time_fit, G[ie_idx][i, j], G[0][i, j][-1, :], xlabel=None, ylabel="G(t,t')", title=f"Neurons : {neuron_labels[neu_i]}<-{neuron_labels[neu_j]}", save_path= os.path.join(stimulus_fig_path[ie_idx],f'before_fit_negf_G_neurons_neuron_pair_{neuron_labels[neu_i]}<-{neuron_labels[neu_j]}.png'))



#########################################################################################################################################################

# FIT NEGF

# Lower the sampling rate so fitting is not so time consuming

lowering_resolution_step = 5
fitting_window = 80

print("NEGF fitting")
min_constrain_dict = {
                "C": 0.0,          
                "gamma": 0.0,  
                "beta": 0.0, 
                "gamma_g": 0.0,
                "gamma_s": 0.0,
                "E_s": -12000,  # the data is not mV so such parameters might have another dimension
                "E_c": -12000,  
                }
max_constrain_dict = {
                "C": np.inf,          
                "gamma": np.inf,  
                "beta": np.inf, 
                "gamma_g": np.inf,
                "gamma_s": np.inf,
                "E_s": 10000,
                "E_c": 10000,  
                }

fitted_parameters = lif_gf.ADAM_fit(
    x=signal_smooth[:, responding_neurons, stim_begin_idx:stim_begin_idx+fitting_window:lowering_resolution_step],
    dt=lowering_resolution_step * dt,
    target_nodes=np.arange(1, n_responding_neurons),  # Exclude index 0 (stimulated neuron)
    max_iters=1000,
    constrain=(min_constrain_dict, max_constrain_dict),
    rms_tol=1e-6,
    learning_rate = 5e-3,
    beta1 = 0.9,
    beta2 = 0.99,
    eps = 1e-3,
    parameter_to_fit_list=['C', 'gamma', 'gamma_g', 'gamma_s', 'E_c', 'E_s', 'beta', 'Vth'],
    #loss_method='correlation'
)

# Compute green functions using the higher time resolution, but the fitted parameters
lif_gf = nlfc.GreenFunctions(
    model = LIF(n_responding_neurons, fitted_parameters),
    x = signal_smooth[:, responding_neurons, stim_begin_idx::],
    dt = dt,
)
print('FITTING DONE')

# Save parameters after fitting as a tab-delimited text file
with open(os.path.join(output_data_dir, "fitted_parameters.txt"), "w") as f:
    f.write("Parameter\tValue\n")
    for key, value in fitted_parameters.items():
        f.write(f"{key}\t{value}\n")
f.close()

# Save parameters after fitting in a JSON format for easier loading
with open(os.path.join(output_data_dir, "fitted_parameters.pkl"), "wb") as f:
    pickle.dump(fitted_parameters, f)

f.close()

#########################################################################################################################################################

# plot neural network after fitting
nlfc.utils.netplots.neural_network(fitted_parameters["gamma_g"], fitted_parameters["gamma_s"], fitted_parameters["E_s"], responding_neurons_labels, save_path=os.path.join(output_figure_dir, f'Neural_Network_responding_only_fitted_parameters.png'))

# nodes at real positions
#nlfc.utils.netplots.neural_network(fitted_parameters["gamma_g"], fitted_parameters["gamma_s"], fitted_parameters["E_s"], responding_neurons_labels, positions=responding_neurons_positions[:, [0,1]], save_path=os.path.join(output_figure_dir, f'Neural_Network_responding_only_after_fit_real_positions.png'))

G  = lif_gf.total_G(G_degree)

sig_corr = np.zeros((n_responding_neurons-1))
for ie_idx in range(n_stimuli):

    # save plot the NEGF for each stimulation and each neuron pair (consider source only the stim neuron)
    for i in np.arange(1, n_responding_neurons):
        neu_i = responding_neurons[i]
        Y_nonlin_fit[ie_idx][i, :] = np.full_like(Y_nonlin_fit[ie_idx][i, :], signal_smooth[ie_idx][neu_i, stim_begin_idx])
        for j in range(n_responding_neurons):
            neu_j = responding_neurons[j]
            delta_j = signal_smooth[ie_idx][neu_j, stim_begin_idx:] - signal_smooth[ie_idx][neu_j, stim_begin_idx]
            Y_nonlin_fit[ie_idx, i] += nlfc.utils.nontt_conv(lif_gf.g[ie_idx][i, j], delta_j, dt=dt)
            
            if (np.all(abs(lif_gf.g[ie_idx][i, j]) < 1e-02)): 
                continue
            #nlfc.utils.plots.t_t_heatmap(x, g[ie_idx][i, j, :, :], os.path.join(ie_dir, f'negf_g_heatmap_neuron_pair_{i}_{j}_stimulation_{str(ie)}.png'))
            nlfc.utils.plots.time_level_curves(time_fit, lif_gf.g[ie_idx][i, j], lif_gf.g0[i, j, -1], xlabel=None, ylabel="g(t,t')", title=f"Neurons : {neuron_labels[neu_i]}<-{neuron_labels[neu_j]}", save_path= os.path.join(stimulus_fig_path[ie_idx],f'fitted_negf_direct_g_neurons_neuron_pair_{neuron_labels[neu_i]}<-{neuron_labels[neu_j]}.png'))
            #nlfc.utils.plots.t_t_heatmap(time_fit, G[:, :, i, 0], os.path.join(ie_dir, f'negf_G{G_degree}_heatmap_neuron_pair_{labels[responding_neurons[i]]}_{labels[stim]}_stimulation_{str(ie)}.png'))
            nlfc.utils.plots.time_level_curves(time_fit, G[ie_idx][i, j], G[0][i, j][-1, :], xlabel=None, ylabel="G(t,t')", title=f"Neurons : {neuron_labels[neu_i]}<-{neuron_labels[neu_j]}", save_path= os.path.join(stimulus_fig_path[ie_idx],f'fitted_negf_G_neurons_neuron_pair_{neuron_labels[neu_i]}<-{neuron_labels[neu_j]}.png'))
        sig_corr[i-1] += np.corrcoef(signal_smooth[ie_idx, responding_neurons[i], stim_begin_idx:], Y_nonlin_fit[ie_idx, i])[0, 1]/n_stimuli

# Find the neuron with the most variation in trial correlations
response_neuron_toplot_idx = np.argmax(sig_corr)+1
response_neuron_toplot = responding_neurons[response_neuron_toplot_idx] 


    
###############
# PREPARE PANELS PLOT
###############


# Load linear kernel data
with open(os.path.join(stim_neu_path, 'lin_kernels_fit_responding_neurons.pkl'), 'rb') as f:
    loaded_data = pickle.load(f)

lin_kernel = loaded_data["lin_kernel"]
fit_y      = loaded_data["fit_y"]
time_fit   = loaded_data["time_fit"]
dt         = loaded_data["dt"]


color_map = plt.get_cmap("tab10", n_stimuli)
nrows = int(np.ceil(np.sqrt(len(responding_neurons))))
ncols = int(np.ceil(len(responding_neurons) / nrows))
while nrows * ncols < len(responding_neurons):
    ncols += 1
    nrows = int(np.ceil(len(responding_neurons) / ncols))

fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 12)) 
fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(16, 6)) 
for a in np.ravel(ax): 
    # a.set_xticks([])
    # a.set_yticks([])
    a.twinx().set_yticks([])
    
for i in range(nrows):
    for j in range(ncols):
        if i == nrows - 1:  # Set xlabel only for the last row
            ax[i, j].set_xlabel('time (s)')
        if j == 0:  # Set ylabel only for the first column
            ax[i, j].set_ylabel('Signal (a.u)')
for a in np.ravel(ax2): 
    # a.set_xticks([])
    # a.set_yticks([])
    a.twinx().set_yticks([])
if nrows == 1: 
    ax = np.array([ax])
    a.set_xlabel('time (s)')


ax2[0].set_ylabel('Signal (a.u)')

for i, neu_i in enumerate(responding_neurons):

    # Plot only for detected responses
    i_plot = np.where(responding_neurons==neu_i)[0][0]
    ax_r = i_plot//ncols
    ax_c = i_plot%ncols

    panel_title = f"Neuron {neu_i}: {neuron_labels[neu_i]}"

    for ie_idx in range(n_stimuli):
        stim_color = color_map(ie_idx)  # Get a distinct color for each ie plot line

        y_smooth_plt = signal_smooth[ie_idx, neu_i]
        y_smooth = signal_smooth[ie_idx, neu_i][stim_begin_idx:]

        lbl = f'Response to stimulus #{ie_idx}'
        
        lw = 1
        if neu_i == stim_neuron: 
            lbl += "*"
            lw = 2
            ax2[0].set_title("Stimulated "+ panel_title, fontsize=10)
            ax2[0].plot(time_plt, y_smooth_plt, label=lbl, c=stim_color, lw=lw)
            ax2[0].set_xlim(time_plt[0], time_plt[-1])
            ax2[0].set_ylim(np.nanmin(signal_smooth[:, neu_i, :]), np.nanmax(signal_smooth[:, neu_i, :]))
            ax2[0].axvline(0, c="k", alpha=0.5, label="stim. time")
            ax2[0].axvspan(0, time_fit[fitting_window], color="gray", alpha=0.15, label="Fitting")

        elif neu_i == response_neuron_toplot:
            ax2[1].set_title(panel_title, fontsize=10)
            ax2[1].plot(time_plt, y_smooth_plt, label=lbl, c=stim_color, lw=lw)
            ax2[1].set_xlim(time_plt[0], time_plt[-1])
            ax2[1].set_ylim(np.nanmin(signal_smooth[:, neu_i, :]), np.nanmax(signal_smooth[:, neu_i, :]))
            ax2[1].plot(time_fit, Y_nonlin_fit[ie_idx,i], label="Nonlinear kernel pred.", lw=2, ls='--', c=stim_color)
            ax2[1].axvline(0, c="k", alpha=0.5, label="stim. time")
            ax2[1].axvspan(0, time_fit[fitting_window], color="gray", alpha=0.15, label="Fitting")

        ax[ax_r, ax_c].plot(time_plt, y_smooth_plt, label=lbl, c=stim_color, lw=lw)

        if neu_i != stim_neuron:
            ax[ax_r, ax_c].plot(time_fit, Y_nonlin_fit[ie_idx, i], label="Nonlinear kernel pred.", lw=2, ls=':', c=stim_color)
            



    fit_ls = "-"
    fit_lbl = "Linear kernel pred." #"|".join([str(nbp - 1) for nbp in n_branch_params])
    if neu_i == stim_neuron:
        panel_title = "Stimulated "+ panel_title
    elif neu_i == response_neuron_toplot:
        ax2[1].plot(time_fit, fit_y[i], label=fit_lbl, c='black', lw=1, ls=fit_ls)
        ax2[1].legend(loc='upper left', bbox_to_anchor=(0, 1))
    if neu_i != stim_neuron: 
        ax[ax_r, ax_c].plot(time_fit, fit_y[i], label=fit_lbl, c='black', lw=1, ls=fit_ls)

    ax[ax_r, ax_c].set_xlim(time_plt[0], time_plt[-1])
    ax[ax_r, ax_c].set_ylim(np.nanmin(signal_smooth[:, neu_i, :]), np.nanmax(signal_smooth[:, neu_i, :]))
    ax[ax_r, ax_c].axvline(0, c="k", alpha=0.8)
    ax[ax_r, ax_c].axvspan(0, time_fit[fitting_window], color="gray", alpha=0.15, label="Fitting")
    

    ax[ax_r, ax_c].set_title(panel_title, fontsize=10)
    if i_plot == len(responding_neurons) - 1:  # Add legend only for the last panel
        handles, labels_plt = ax[ax_r, ax_c].get_legend_handles_labels()
        fig.legend(handles, labels_plt, loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=3)

# Save plot with neuron index in filename
filename = f"panels_mult_stimulation_fits_{fitting_window}_{lowering_resolution_step}.png"
filename2 = f"response_neuron_toplot_response_{fitting_window}_{lowering_resolution_step}.png"
fig.savefig(os.path.join(output_figure_dir, filename), bbox_inches="tight")
plt.close(fig)        # Plot heatmaps for each neuron pair
fig2.savefig(os.path.join(output_figure_dir, filename2), bbox_inches="tight")
plt.close(fig2)        # Plot heatmaps for each neuron pair

# Scatter plot gamma_g and gamma_s as a function of the distance matrix
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Flatten the matrices for scatter plotting
distances = distance_matrix_responding_neurons.flatten()
gamma_g_values = fitted_parameters["gamma_g"].flatten()
gamma_s_values = fitted_parameters["gamma_s"].flatten()


# Filter out None or NaN values from distances and corresponding gamma_g_values
valid_indices = ~np.isnan(distances) & ~np.isnan(gamma_g_values) & ~np.isnan(gamma_s_values)

# Plot gamma_g vs distance only for valid values
ax[0].scatter(distances[valid_indices], gamma_g_values[valid_indices], alpha=0.8, label="gamma_g")
ax[0].set_xlabel("Distance")
ax[0].set_ylabel("gamma_g")
ax[0].set_title("gamma_g vs Distance")
ax[0].grid(True)

# Plot gamma_s vs distance
ax[1].scatter(distances[valid_indices], gamma_s_values[valid_indices], alpha=0.8, label="gamma_s", color="orange")
ax[1].set_xlabel("Distance")
ax[1].set_ylabel("gamma_s")
ax[1].set_title("gamma_s vs Distance")
ax[1].grid(True)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(os.path.join(output_figure_dir, "gamma_vs_distance_scatter.png"), bbox_inches="tight")
plt.close(fig)

plt.close('all')

gc.collect()
