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
from nonlinfunconn.models.lif import LIF

load_cache = "--load-cache" in sys.argv

only_labeled_neurons = "--only-labeled-neurons" in sys.argv

# default 
figures_path = "figures/C_elegans_pumpprobre_exp/"
data_path = "data/C_elegans_pumpprobre_exp/"

stim_neu_path = "/home/gabrielm/projects/nonlinfunconn-main/data/C_elegans_pumpprobre_exp/worm_type_unc31/20220113_101730/stim_neu_AVDR/"

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

responding_neurons_labels = np.array(neuron_labels)[responding_neurons]  # Create an array of labels for responding_neurons indexes
labeled_neurons = [label != "" for label in responding_neurons_labels]  # Create a boolean list for non-empty labels

n_responding_neurons_labeled = len(np.array(responding_neurons)[labeled_neurons])

responding_neurons_positions = neuron_positions[responding_neurons]



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
nlfc.utils.netplots.neural_network(resp_gamma_g[labeled_neurons][:, labeled_neurons], resp_gamma_s[labeled_neurons][:, labeled_neurons], resp_Es[labeled_neurons][:, labeled_neurons], responding_neurons_labels[labeled_neurons], save_path=os.path.join(output_figure_dir, f'Neural_Network_responding_only_connectome_parameters.png'))

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

g = lif_gf.g
g0 = lif_gf.g0

# plot neural network after fitting

model_parameters["gamma_g"] = np.where((abs(model_parameters["gamma_g"]) < 1), 0.0, model_parameters["gamma_g"])
model_parameters["gamma_s"] = np.where((abs(model_parameters["gamma_s"]) < 1), 0.0, model_parameters["gamma_s"])
nlfc.utils.netplots.neural_network(model_parameters["gamma_g"][labeled_neurons][:, labeled_neurons], model_parameters["gamma_s"][labeled_neurons][:, labeled_neurons], model_parameters["E_s"][labeled_neurons][:, labeled_neurons], np.array(neuron_labels)[np.array(responding_neurons)[labeled_neurons]], save_path=os.path.join(output_figure_dir, f'Neural_Network_responding_only_after_fit.png'))
Adj = model_parameters["gamma_g"][labeled_neurons][:, labeled_neurons] + model_parameters["gamma_s"][labeled_neurons][:, labeled_neurons]
# plot neural network after fitting
nlfc.utils.netplots.neural_network(Adj,np.zeros_like(Adj), np.zeros_like(Adj), np.array(neuron_labels)[np.array(responding_neurons)[labeled_neurons]], save_path=os.path.join(output_figure_dir, f'Neural_Network_Adj_responding_only_after_fit.png'))

K0= np.zeros_like(g0[:, :, 0, :])
DyCon0 = np.zeros_like(model_parameters["gamma_g"])
for i in range(n_responding_neurons):
    for j in range(n_responding_neurons):
            K0[i, j] = np.nansum(g0[i, j], axis=0) * dt
            DyCon0[i, j] = np.nansum(K0[i,j], axis=0) * dt

K = np.zeros_like(g[:, :, :, 0, :])
for ie_idx in range(n_stimuli):
    DyCon = np.zeros_like(model_parameters["gamma_s"])
    os.makedirs(stimulus_fig_path[ie_idx], exist_ok=True)
    # save plot the NEGF for each stimulation and each neuron pair (consider source only the stim neuron)
    for j in range(n_responding_neurons):
        for i in np.arange(1, n_responding_neurons):
            if i==j: continue
            neu_i = responding_neurons[i]
            neu_j = responding_neurons[j]  
            delta_j = signal_smooth[ie_idx][neu_j, stim_begin_idx:] - signal_smooth[ie_idx][neu_j, stim_begin_idx]
            if j==0:
                Y_ji = lif_gf.g[ie_idx][i, j]
            else:
                Y_ji = nlfc.utils.nontt_conv(lif_gf.g[ie_idx][i, j], lif_gf.g[ie_idx][j, 0], dt=dt)
            
            #Y_ji = np.where((Y_ji < -100) | (Y_ji > 100), np.nan, Y_ji)  # set clipped values to nan
            K[ie_idx][i, j] =   np.nansum(Y_ji, 0) * dt #np.nansum(lif_gf.g[ie_idx][i, j], 0) * dt # nlfc.utils.nontt_conv(lif_gf.g[ie_idx][i, j], delta_j, dt=dt)
            DyCon[i, j] = np.nansum(K[ie_idx][i, j], 0) * dt

    DyCon = np.where((abs(DyCon) < 1), 0.0, DyCon)
    print('DyCon: ', DyCon)      
    nlfc.utils.netplots.dynamics_network(DyCon[labeled_neurons][:, labeled_neurons], np.array(neuron_labels)[np.array(responding_neurons)[labeled_neurons]], save_path=os.path.join(stimulus_fig_path[ie_idx], f'Green_function_cumulative_sum_stim_{ie_idx}.png'))

    #nlfc.utils.netplots.dynamics_network_3d(DyCon[labeled_neurons][:, labeled_neurons], np.array(neuron_labels)[np.array(responding_neurons)[labeled_neurons]], positions=responding_neurons_positions[labeled_neurons], save_path=os.path.join(stimulus_fig_path[ie_idx], f'Green_function_cumulative_sum_stim_{ie_idx}_3D_NET.png'))

        
        
for i in range(n_responding_neurons):
    for j in range(n_responding_neurons):
        neu_i = responding_neurons[i]
        neu_j = responding_neurons[j]  

        plt.figure(figsize=(6, 4), dpi=200)
        plt.plot(time_fit[-1] - time_fit[:], K0[i, j], label="K₀", color="black", linewidth=2.4)
        for ie_idx in range(n_stimuli):

            color = np.random.rand(3).tolist() 
            plt.plot(time_fit[-1] - time_fit[:], K[ie_idx][i, j], label=f"Stimulation #{ie_idx}", color=color, linewidth=2)
            plt.xlabel("T - t′ (s)") 
            plt.ylabel("K(t')")
            plt.legend()
            plt.title("Cumulative Green function amplitude")

        plt.savefig(os.path.join(output_figure_dir,f'Cumulative_K_neuron_pair_{neuron_labels[neu_i]}<-{neuron_labels[neu_j]}.png'), bbox_inches='tight')
        plt.close()

    



