#
# code for correlation between fitted adjacency matrix and the annatomical connectome of C. Elegans worm wild (wt) type and mutant (unc31)with no wireless connection receptors

import numpy as np
import matplotlib.pyplot as plt
import os, sys, json
import pickle # for cache saving/loading

import nonlinfunconn as nlfc # for non-linear kernels
from nonlinfunconn.models.lif import LIF

plot = True

# Parse command-line arguments
save = "--no-save" not in sys.argv

only_labeled_neurons = "--only-labeled-neurons" in sys.argv


figures_path = "figures/C_elegans_pumpprobre_exp/"
data_path = "data/C_elegans_pumpprobre_exp/"


worm_type_wt_path = os.path.join(data_path, "worm_type_wt")

worm_type_unc31_path = os.path.join(data_path, "worm_type_unc31")

# Load the kunert_ODE_parameters dictionary from a pickled file
kunert_ODE_parameters_file_path = os.path.join(data_path, "kunert_ODE_parameters.pkl")
with open(kunert_ODE_parameters_file_path, "rb") as f:
    kunert_ODE_parameters = pickle.load(f)


wt_worms_datasets_paths_list = [os.path.join(worm_type_wt_path, folder) for folder in os.listdir(worm_type_wt_path) if os.path.isdir(os.path.join(worm_type_wt_path, folder))]

unc31_worms_datasets_paths_list = [os.path.join(worm_type_unc31_path, folder) for folder in os.listdir(worm_type_unc31_path) if os.path.isdir(os.path.join(worm_type_unc31_path, folder))]

wt_unc31_worms_datasets_paths_list = wt_worms_datasets_paths_list + unc31_worms_datasets_paths_list

gamma_g_connectome_wt = []
gamma_g_connectome_unc31 = []

gamma_s_connectome_wt = []
gamma_s_connectome_unc31 = []

gamma_g_fitted_wt = []
gamma_g_fitted_unc31 = []

gamma_s_fitted_wt = []
gamma_s_fitted_unc31 = []

signal_correlation_wt = []
signal_correlation_unc31 = []

distances_total = []

# Iterate over the folders which contains each experiment data
for (worm_idx, worm_dataset_path) in enumerate(wt_unc31_worms_datasets_paths_list):

    print("Processing dataset worm #", worm_idx, " : ", worm_dataset_path)     
    
    stimulated_neuron_paths_list = [os.path.join(worm_dataset_path, folder) for folder in os.listdir(worm_dataset_path) if os.path.isdir(os.path.join(worm_dataset_path, folder))]
    
    for (stim_idx, stim_neu_path) in enumerate(stimulated_neuron_paths_list):

        #try:
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

        responding_neurons_labels = np.array(neuron_labels)[responding_neurons]  # Create an array of labels for responding_neurons indexes
        labeled_neurons = [label != "" for label in responding_neurons_labels]  # Create a boolean list for non-empty labels

        n_responding_neurons_labeled = len(np.array(responding_neurons)[labeled_neurons])

        #if n_responding_neurons_labeled > 15 or n_responding_neurons_labeled < 2:
        #    print(f"   Skipping dataset {stim_neu_path} with {n_responding_neurons_labeled} labeled responding neurons.")
        #    continue

        if only_labeled_neurons:
            if any(label == '' for label in np.array(neuron_labels)[responding_neurons]):
                print(f"   Skipping dataset {stim_neu_path} with some responding neuron not identified.")
                continue

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
        

        resp_gamma_g = kunert_ODE_parameters["gamma_g"][responding_neurons][:, responding_neurons] 
        resp_gamma_s = kunert_ODE_parameters["gamma_s"][responding_neurons][:, responding_neurons]
        resp_Es = kunert_ODE_parameters["Es"][responding_neurons][:, responding_neurons]


        cache_file_path = os.path.join(output_data_dir, "fitted_parameters.pkl")
        if worm_type == "wt":
            worm_type_path = worm_type_wt_path
        elif worm_type == "unc31":
            worm_type_path = worm_type_unc31_path
            
        old_cache_file_path = os.path.join( "data/C_elegans_pumpprobre_exp/" + os.path.relpath(worm_dataset_path, worm_type_path) + f"/stim_neu_{stim_neuron_label}_{n_stimuli}x/fit_negf_consider_not_labeled_neurons", "fitted_parameters.pkl")
        
        if os.path.exists(cache_file_path):
            # Load parameters after fitting from the .pkl file
            with open(cache_file_path, "rb") as f:  # 'rb' not 'r'
                fitted_parameters = pickle.load(f)
            f.close()
        elif os.path.exists(old_cache_file_path):
            with open(old_cache_file_path, "rb") as f:  
                fitted_parameters = pickle.load(f)
            print('Found cache in old directory')
            # Close the loaded data file
            f.close()

        else:
            print(f"Warning: Cache file '{cache_file_path}' does not exist. Proceedingto next dataset.")
            continue


        if worm_type == "wt":
            gamma_g_connectome_wt.extend(resp_gamma_g[labeled_neurons][:, labeled_neurons].flatten())
            gamma_s_connectome_wt.extend(resp_gamma_s[labeled_neurons][:, labeled_neurons].flatten())
            gamma_g_fitted_wt.extend(fitted_parameters["gamma_g"][labeled_neurons][:, labeled_neurons].flatten())
            gamma_s_fitted_wt.extend(fitted_parameters["gamma_s"][labeled_neurons][:, labeled_neurons].flatten())
        elif worm_type == "unc31":
            gamma_g_connectome_unc31.extend(resp_gamma_g[labeled_neurons][:, labeled_neurons].flatten())
            gamma_s_connectome_unc31.extend(resp_gamma_s[labeled_neurons][:, labeled_neurons].flatten())
            gamma_g_fitted_unc31.extend(fitted_parameters["gamma_g"][labeled_neurons][:, labeled_neurons].flatten())
            gamma_s_fitted_unc31.extend(fitted_parameters["gamma_s"][labeled_neurons][:, labeled_neurons].flatten())
    

        # Compute green functions using the higher time resolution, but the fitted parameters
        lif_gf = nlfc.GreenFunctions(
            model = LIF(n_responding_neurons, fitted_parameters),
            x = signal_smooth[:, responding_neurons, stim_begin_idx::],
            dt = dt,
        )


        Y_nonlin_fit = np.zeros_like(signal_smooth[:, responding_neurons, stim_begin_idx:])
        signal_correlation = np.zeros((n_stimuli, n_responding_neurons))
        for ie_idx in range(n_stimuli):
            for i in range(n_responding_neurons):
                neu_i = responding_neurons[i]
                Y_nonlin_fit[ie_idx][i, :] = np.full_like(Y_nonlin_fit[ie_idx][i, :], signal_smooth[ie_idx][neu_i, stim_begin_idx])
                for j in range(n_responding_neurons):
                    neu_j = responding_neurons[j]
                    delta_j = signal_smooth[ie_idx][neu_j, stim_begin_idx:] - signal_smooth[ie_idx][neu_j, stim_begin_idx]
                    Y_nonlin_fit[ie_idx, i] += nlfc.utils.nontt_conv(lif_gf.g[ie_idx][i, j], delta_j, dt=dt)
                    
                signal_correlation[ie_idx, i] = np.corrcoef(
                    signal_smooth[ie_idx, responding_neurons[i], stim_begin_idx:], 
                    Y_nonlin_fit[ie_idx, i]
                )[0, 1]
                
                
        if worm_type == "wt":
            signal_correlation_wt.extend(signal_correlation.flatten())
        elif worm_type == "unc31":
            signal_correlation_unc31.extend(signal_correlation.flatten())

        #    pass
        #except Exception as e:
        #    print(f"Error processing dataset {stim_neu_path}: {e}")
        #    continue           

# Scatter plot gamma_g and gamma_s: connectome vs fitted
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].scatter(gamma_g_connectome_wt, gamma_g_fitted_wt, alpha=0.8, color="blue", label="WT")
ax[0].scatter(gamma_g_connectome_unc31, gamma_g_fitted_unc31, alpha=0.8, color="orange", label="unc31")
ax[0].set_xlabel("gamma_g (connectome) [pS]")
ax[0].set_ylabel("gamma_g (fitted) [pS]")
ax[0].set_title("gamma_g: Connectome vs Fitted")
ax[0].grid(True)
ax[0].legend(loc="upper left")

ax[1].scatter(gamma_s_connectome_wt, gamma_s_fitted_wt, alpha=0.8, color="blue", label="WT")
ax[1].scatter(gamma_s_connectome_unc31, gamma_s_fitted_unc31, alpha=0.8, color="orange", label="unc31")
ax[1].set_xlabel("gamma_s (connectome) [pS]")
ax[1].set_ylabel("gamma_s (fitted) [pS]")
ax[1].set_title("gamma_s: Connectome vs Fitted")
ax[1].grid(True)
ax[1].legend(loc="upper left")

plt.savefig(os.path.join(figures_path, "gammas_connectome_vs_fitted_scatter_plot.png"), bbox_inches="tight")
plt.close(fig)

# Bar plot with the correlation of the data
gamma_g_corr_wt = np.corrcoef(gamma_g_connectome_wt, gamma_g_fitted_wt)[0, 1]
gamma_g_corr_unc31 = np.corrcoef(gamma_g_connectome_unc31, gamma_g_fitted_unc31)[0, 1]

gamma_s_corr_wt = np.corrcoef(gamma_s_connectome_wt, gamma_s_fitted_wt)[0, 1]
gamma_s_corr_unc31 = np.corrcoef(gamma_s_connectome_unc31, gamma_s_fitted_unc31)[0, 1]

signal_correlation_wt_mean    = np.nanmean(signal_correlation_wt)
signal_correlation_unc31_mean = np.nanmean(signal_correlation_unc31)

wt_values = [signal_correlation_wt_mean, gamma_g_corr_wt, gamma_s_corr_wt]
unc31_values = [signal_correlation_unc31_mean, gamma_g_corr_unc31, gamma_s_corr_unc31]

labels = ["Signal", "Gap junction", "Chem. Syn."]

x = np.arange(len(labels))  # the label locations
width = 0.35  # width of the bars

fig, ax = plt.subplots(figsize=(7, 5))
bars1 = ax.bar(x - width/2, wt_values, width, label='WT', color='blue', alpha=0.8)
bars2 = ax.bar(x + width/2, unc31_values, width, label='UNC31', color='orange', alpha=0.8)

# Formatting
ax.set_ylabel("Correlation Coefficient")
ax.set_ylim(0, 1)
ax.set_title("Correlation Measures: WT vs UNC31")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(axis="y")

plt.tight_layout()
plt.savefig(os.path.join(figures_path, "correlations_barplot_combined.png"))
plt.close(fig)