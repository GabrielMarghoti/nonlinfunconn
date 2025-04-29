#
# code for correlation between fitted adjacency matrix and the annatomical connectome of C. Elegans worm wild (wt) type and mutant (unc31)with no wireless connection receptors

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm
import os, sys, time, json
import pickle # for cache saving/loading

import pumpprobe as pp
import wormdatamodel as wormdm
import wormbrain as wormb

import nonlinfunconn as nlfc # for non-linear kernels
from nonlinfunconn.models.lif import LIF

plot = True

# Parse command-line arguments
save = "--no-save" not in sys.argv
sig_green = "--signal:green" in sys.argv
skip_if_not_manually_located = "--skip-if-not-manually-located" in sys.argv
matchless_nan_th = None
matchless_nan_th_from_file = "--matchless-nan-th-from-file" in sys.argv
matchless_nan_th_added_only = "--matchless-nan-th-added-only" in sys.argv
merge = "--no-merge" not in sys.argv
ds_exclude_tags =  None
kwar_fit_lineal_model  = "--fit-linear" in sys.argv

load_cache = "--load-cache" in sys.argv

aconn_ds_i = None # default is loading from funatlas, if aconn_ds_i is set, it will load from the specified dataset
# default 
figures_folder = "figures/C_elegans_pumpprobre_exp/"
data_folder = "data/C_elegans_pumpprobre_exp/"

ds_list_path =  "/home/gabrielm/paper_reproduction/ds_list_full.txt"

ds_list_spont_path = "/home/gabrielm/paper_reproduction/ds_list_ctrl_wt.txt"

signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": "sg_causal", 
                 "smooth_n": 13, "smooth_poly": 1,
                 "photobl_appl":True,            
                 "matchless_nan_th_from_file": matchless_nan_th_from_file,
                 "matchless_nan_th": matchless_nan_th,
                 "matchless_nan_th_added_only": matchless_nan_th_added_only}

def rolling_window(a, window):
    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window-1
    pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
    a = np.pad(a, pad,mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def parse_neuron_positions(filepath):
    neuron_positions = {}
    with open(filepath, 'r') as file:
        lines = file.readlines()
        # First line contains neuron labels
        neuron_labels = lines[0].strip().split()
        if neuron_labels[0].startswith("#"):
            neuron_labels[0] = neuron_labels[0][1:]
        # Remaining lines contain the coordinates
        for label, coord_line in zip(neuron_labels, lines[1:]):
            coords = tuple(map(float, coord_line.strip().split()))
            neuron_positions[label] = coords
    return neuron_positions

def load_ds_list(fname,tags=None,exclude_tags=None,return_tags=False):
    '''Loads the list of dataset folder names given the filename of a 
    text file containing a folder name per line. Comments start with # (both
    for whole line and for annotations after the folder name).
    
    Parameters
    ----------
    fname: str
        Name of the txt file containing the list of datasets.
    tags: str (optional)
        Space-separated tags to select datasets. Default: None.
    exclude_tags: str (optional)
        Space-separated tags to exclude in the dataset selection. Default:
        None.
    
    Returns
    -------
    ds_list: list of str
        List of the dataset folder names.
    '''
    
    f = open(fname, "r")
    ds_list = []
    ds_tags_lists = []
    if tags is not None:
        tags = tags.split(" ")
    if exclude_tags is not None:
        exclude_tags = exclude_tags.split(" ")

    for l in f.readlines():
        if l[0] not in ["#", "\n"]:  # Ignore commented lines
            # Remove commented annotations
            l2 = l.split("#")[0]
            # Get tags
            if len(l.split("#")) == 0:
                # There are no tags
                continue
            tgs = l.split("#")[1].split(" ")
            tgs = [t.strip() for t in tgs]
            if tags is not None:
                ok = all(t in tgs or t == "" for t in tags)
                if not ok:
                    continue
            if exclude_tags is not None:
                not_ok = any(t in tgs and t != "" for t in exclude_tags)
                if not_ok:
                    continue

            # Remove blank spaces, newlines, and tabs
            l2 = l2.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")

            # Complete folder path with last / if necessary
            if l2[-1] != "/":
                l2 += "/"

            ds_tags_lists.append(tgs)
            ds_list.append(l2)
    f.close()

    if return_tags:
        return ds_list, ds_tags_lists
    else:
        return ds_list


ds_list, ds_tags = load_ds_list(ds_list_path, return_tags=True)
ds_list_spont, ds_spont_tags = load_ds_list(ds_list_spont_path, return_tags=True)

# get connectome from pp.Funatlas class
funa = pp.Funatlas.from_datasets(ds_list,merge_bilateral=merge,signal="green",
                                 signal_kwargs = signal_kwargs,
                                 enforce_stim_crosscheck=False,
                                 ds_tags=ds_tags,ds_exclude_tags=ds_exclude_tags,
                                 verbose=False)

aconn_chem, aconn_elec = funa.get_aconnectome_from_file() # get the anatomical connectome with the correct atlas index for neuros
num_neurons = aconn_chem.shape[0]

# Find the kernel parameters based on Kunert C.Elegans model
f = open('/home/gabrielm/paper_reproduction/kunertPRE2014/params.json','r')
params = json.load(f)
f.close()

def get_genetic_prediction():
    #Downlaod the Excel workbook from the paper
    import shutil
    import tempfile
    import urllib.request
    url = 'https://doi.org/10.1371/journal.pcbi.1007974.s003'
    with urllib.request.urlopen(url) as response:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            shutil.copyfileobj(response, tmp_file) #store it in a temporary location

    import pandas as pd
    WS = pd.read_excel(tmp_file.name, sheet_name='5. Sign prediction')
    WS_np = np.array(WS)

    # Location of various data within the excel worksheet
    pre_syn_col = 0
    post_syn_col = 3
    pred_col = 16

    first_row = 2
    last_row = 3639

    pre = WS_np[first_row:last_row, pre_syn_col] #Presynaptic neuron
    post = WS_np[first_row:last_row, post_syn_col] #postsynaptic neuron
    sign = WS_np[first_row:last_row, pred_col] #Sign prediction
    # strip out the 0's from neuron names to match our formatting so that we get VB1 instead of VB01
    pre = pd.Series(pre).str.replace('0','').to_numpy()
    post = pd.Series(post).str.replace('0','').to_numpy()
    return pre, post, sign

pre, post, pred = get_genetic_prediction()
sign = np.ones((funa.n_neurons,funa.n_neurons))
for k in np.arange(len(pre)):
    if pred[k] == "-":
        aj,ai = funa.ids_to_i([pre[k],post[k]])
        sign[ai,aj] = -1     # row are postsynaptic neurons, columns presynaptic neurons

# Get the composite aconnectome via the Funatlas
if aconn_ds_i is None:
    Gsyn, Ggap = funa.get_aconnectome_from_file(chem_th=0, gap_th=0, exclude_white=False, average=True)
    #Gsyn, Ggap = Gsyn.T, Ggap.T  # Transpose the adjacency matrices, my default is to have the post-synaptic neurons index as rows
else:
    aconn_folder = funa.module_folder
    aconn_fname = funa.aconn_sources[aconn_ds_i]["fname"]
    Gsyn, Ggap = funa._get_aconnectome_witvliet(aconn_folder + aconn_fname)
    #Gsyn, Ggap = Gsyn.T, Ggap.T # Transpose the adjacency matrices, my default is to have the post-synaptic neurons index as rows


# Cell
Ci = params['C'] # Membrane capacitance 1 F

Ci = 1e12*Ci  # pF

Ci = 100*Ci #  to change the dynamics for the time scale of calcium concentration/fluorescence instead of membrane potentials # to review latter

Gcell = 1e12*params['Gcell'] # Leakage conductance of membrane [pS]
Ecell = params['Ecell']*1000 # Leakage potential [mV]

# Electrical synapses
ggap = 1e12*params['ggap'] # conductivity of electrical synapse [pS]

# Chemical synapses
gsyn = 1e12*params['gsyn'] # "conductivity" of chemical synapse [pS]
ar = params['ar'] # activation rate of synapses [s^-1]
ad = params['ad'] # deactivation rate of synapses [s^-1]
beta = params['beta']/1000 # width of synaptic activation [mV^-1]
esynexc = params['esynexc']*1000 # reverse potential for excitatory synapses
esyninh = params['esyninh']*1000 # reverse potential for inhibitory synapses


# Build the Esyn array of the synaptic reverse potentials
# The index is presynaptic neuron, which determines the neurotransmitter and
# hence the sign of the synapse.
Esyn = np.ones((funa.n_neurons,funa.n_neurons))*esynexc
Esyn[sign<0] = esyninh

# Dict with neurons positions
anatlas_positions = parse_neuron_positions(funa.module_folder + "anatlas_neuron_positions.txt")

kunert_parameters = {
    "C": Ci,          
    "gamma": Gcell,  
    "E_c": Ecell,  
    "beta": beta, 
    "a_r": ar,  
    "a_d": ad,  
}


gamma_g_connectome_wt = []
gamma_g_connectome_unc31 = []

gamma_s_connectome_wt = []
gamma_s_connectome_unc31 = []

gamma_g_fitted_wt = []
gamma_g_fitted_unc31 = []

gamma_s_fitted_wt = []
gamma_s_fitted_unc31 = []

distances_total = []

# Iterate over the folders whcih contains each experiment data
for (i_folder, folder) in enumerate(ds_list):

    # Create functional connectome
    fconn = pp.Fconn.from_file(folder)
    #shift_vol = fconn.shift_vol

    sig = wormdm.signal.Signal.from_file(
                folder,"green",
                matchless_nan_th=matchless_nan_th,
                matchless_nan_th_from_file=matchless_nan_th_from_file,
                matchless_nan_th_added_only=matchless_nan_th_added_only)
    sig.appl_photobl()
    # Smooth and calculate the derivative of the signal (derivative needed for
    # detection of responses)
    sig.remove_spikes()
    #sig.median_filter()

    #sig.get_smoothed(127,None,3,"sg_causal")
    sig.smooth(n=110,i=None,poly=4,mode="sg")

    # Get the neurons coordinates of the reference volume and load the matches
    # to determine what neuron was targeted
    cervelli = wormb.Brains.from_file(folder,ref_only=True)
    labels = cervelli.get_labels(0)

    print("Processing dataset", i_folder, ":", folder)     

    stim_neurons_analyzed = set()   
    
    for stim in fconn.stim_neurons[fconn.stim_neurons > 0]:
        # Get the stimulation neurons
        if stim in stim_neurons_analyzed:
            continue
        stim_neurons_analyzed.add(stim)

        #stim = np.bincount(fconn.stim_neurons[fconn.stim_neurons > 0]).argmax()
        stimulations_idx = np.where(fconn.stim_neurons == stim)[0]

        num_stimulations = len(stimulations_idx)

        # Skip datasets where the number of stimulations is not between 2 and 3
        if not (2 <= num_stimulations <= 4): 
            continue

        stim_neuron_label = labels[stim]

        print("Analyzing source neuron", stim, ":", stim_neuron_label)     

        responding = set()  # Initialize a set to store all responding neurons across all ie stimulation loops
        Y_total = []
        Y_smooth_total = []

        shift_vol = None
        i0 = max(0, fconn.i0s[stimulations_idx[0]])  # start of the stimulation
        i1 = fconn.i1s[stimulations_idx[0]]          # end of the stimulation
        shift_vol = fconn.shift_vols[0]
        time_plt = (np.arange(i1 - i0) - shift_vol) * fconn.Dt  
        time_plt_len = len(time_plt)
        
        time_fit = np.arange(time_plt_len- shift_vol) * fconn.Dt  
        time_fit_len = len(time_fit)

        #  Set output directories
        fig_dir = figures_folder + "_".join(ds_tags[i_folder]) + f"/stim_neu_{stim_neuron_label}_{num_stimulations}x/paths_dyn_connectome"
        data_dir = data_folder + "_".join(ds_tags[i_folder]) + f"/stim_neu_{stim_neuron_label}_{num_stimulations}x/paths_dyn_connectome"
        if kwar_fit_lineal_model:
            fig_dir += "_equilibirum"
            data_dir += "_equilibirum"
        fig_dir   += "/"
        data_dir  += "/"


        cache_dir = data_folder + "_".join(ds_tags[i_folder]) + f"/stim_neu_{stim_neuron_label}_{num_stimulations}x/fit_negf/"

        ie_dir_list = []

        for ie in stimulations_idx:  # stimulation index only though cases which the most stimulated neuron is stimulated
            responding_ie = fconn.resp_neurons_by_stim[ie]
            i0 = max(0, fconn.i0s[ie])  # start of the stimulation
            i1 = fconn.i1s[ie]         # end of the stimulation

            responding.update(responding_ie)  # Add the responding neurons to the set

            n_responding_ie = len(responding_ie)
            
            # Ensure output directory exists
            ie_dir = fig_dir + f"n_resp_neurons_{n_responding_ie}_ie_trial{ie}/"

            ie_dir_list.append(ie_dir)
                
            Y = sig.get_segment(i0, i1, shift_vol, unsmoothed_data=True, baseline_mode="constant")[0:time_plt_len, :].transpose()
            Y_smooth = sig.get_segment(i0, i1, shift_vol, unsmoothed_data=False, baseline_mode="constant")[0:time_plt_len, :].transpose()

            Y_total.append(Y[np.newaxis, ...])  # Add a new axis to ensure 3D structure
            Y_smooth_total.append(Y_smooth[np.newaxis, ...])  # Add a new axis to ensure 3D structure
        
        if stim not in responding: 
            print('Stim. neuron not responsive')
            continue 
            #responding.update([stim]) 

        responding = list(responding)
        
        # Insert the stimulated neuron at the beginning of the responding list
        responding.remove(stim)
        responding.insert(0, stim)
        

        # Find first neighbors (nodes connected to stim via either gap or syn)
        first_neighbors = np.where((Ggap[:, stim] > 0) | (Gsyn[:, stim] > 0))[0]
        responsive_first_neighbors = [first_neighbors[i] for i in range(len(first_neighbors)) if first_neighbors[i] in responding] # filter nonresponsive first neigbors

        # Find second neighbors (nodes connected to first neighbors via either gap or syn)
        second_neighbors = np.where((np.sum(list(Ggap[:, responsive_first_neighbors]), axis=1) > 0) | 
                                    (np.sum(list(Gsyn[:, responsive_first_neighbors]), axis=1) > 0))[0]
        
        responsive_second_neighbors = [second_neighbors[i] for i in range(len(second_neighbors)) if second_neighbors[i] in responding] # filter nonresponsive first neigbors

        # Create a set of allowed nodes (stim + first + second neighbors)
        allowed_nodes = set([stim]).union(set(responsive_first_neighbors)).union(set(responsive_second_neighbors))
        
        # Filter responding list to only include allowed nodes # check again if is responsive
        first_second_responsive_nodes = [node for node in responding if node in allowed_nodes]
        
        responding = first_second_responsive_nodes # = list(responding) # consider all responsive neurons
        
        n_responding = len(responding)

        if n_responding > 20 or n_responding < 4:
            print(f"Skipping dataset {folder} with {n_responding} responding neurons.")
            continue

        if any(label == '' for label in np.array(labels)[responding]):
            print(f"Skipping dataset {folder} with some responding neuron not identified.")
            continue

        # plot complete neural network 
        #nlfc.utils.netplots.neural_network((Ggap*ggap), (Gsyn*gsyn), Esyn, np.array(labels), positions=None, save_path=os.path.join(fig_dir, f'Neural_Network_total.png'))

        Y_total = np.concatenate(Y_total, axis=0)  # Concatenate along the new axis to maintain 3D structure
        Y_smooth_total = np.concatenate(Y_smooth_total, axis=0)  # Concatenate along the new axis to maintain 3D structure

        responses_correlations = np.zeros((n_responding, num_stimulations, num_stimulations))
        trial_variations = np.zeros(n_responding)

        for i in range(n_responding):
            neu_i = responding[i]
            responses_correlations[i, :, :] = np.corrcoef(
                        Y_smooth_total[:, neu_i, shift_vol:]
                    )
            # Calculate the standard deviation of the correlation matrix for each neuron
            trial_variations[i] = np.std(responses_correlations[i, :, :])

        # Find the neuron with the most variation in trial correlations
        most_variable_neuron_idx = np.argmax(trial_variations)
        most_variable_neuron = responding[most_variable_neuron_idx] 

        print(f"Neuron with most variation: {most_variable_neuron} ({labels[most_variable_neuron]})")


        if np.mean(responses_correlations[0]) < 0.55:
            print(f"Skipping dataset {folder} with low stimuli correlations for the target.")
            continue
        
        # Responding parameters positions
        # review: consider the most similar labels if it does not match exactly
        responding_positions = []
        for i in responding:
            for key in anatlas_positions:
                if labels[i].startswith(key):
                    responding_positions.append(anatlas_positions[key])
                    break
                elif key.startswith(labels[i]):
                    responding_positions.append(anatlas_positions[key])
                    break
                elif labels[i].startswith(key[:2]):
                    responding_positions.append(anatlas_positions[key])
                    break
                elif key.startswith(labels[i][:2]):
                    responding_positions.append(anatlas_positions[key])
                    break
        if len(responding_positions) != len(responding):
            print(f"Warning: Mismatch in responding positions for dataset {folder}.")
            continue

        responding_positions = np.array(responding_positions)

        responding_positions = np.array(responding_positions)
        distance_matrix = np.linalg.norm(responding_positions[:, np.newaxis, :] - responding_positions[np.newaxis, :, :], axis=-1)

        # Ensure the directories for figures and data exist
        os.makedirs(fig_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

       # Check if the cache file exists
        cache_file_path = os.path.join(cache_dir, "fitted_parameters.pkl")
        if os.path.exists(cache_file_path):
            # Load parameters after fitting from the .pkl file
            with open(cache_file_path, "rb") as f:  # 'rb' not 'r'
                model_parameters = pickle.load(f)
        else:
            print(f"Warning: Cache file '{cache_file_path}' does not exist. Skipping.")
            continue
        # save connectome based network considering only responsive neurons over all stimulations
        gamma_g = (Ggap*ggap)[responding][:, responding] 
        gamma_s = (Gsyn*gsyn)[responding][:, responding]
        Es = Esyn[responding][:, responding]

        kunert_parameters.update({
            "gamma_g": gamma_g,
            "gamma_s": gamma_s,
            "E_s": Es
        })
        print('ds_tags: ', ds_tags[i_folder])
        if "wt" in ds_tags[i_folder]:
            gamma_g_connectome_wt.extend(gamma_g.flatten())
            gamma_s_connectome_wt.extend(gamma_s.flatten())
            gamma_g_fitted_wt.extend(model_parameters["gamma_g"].flatten())
            gamma_s_fitted_wt.extend(model_parameters["gamma_s"].flatten())
        elif "unc31" in ds_tags[i_folder]:
            gamma_g_connectome_unc31.extend(gamma_g.flatten())
            gamma_s_connectome_unc31.extend(gamma_s.flatten())
            gamma_g_fitted_unc31.extend(model_parameters["gamma_g"].flatten())
            gamma_s_fitted_unc31.extend(model_parameters["gamma_s"].flatten())
       
        
# Scatter plot gamma_g and gamma_s: connectome vs fitted
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].scatter(gamma_g_connectome_wt, gamma_g_fitted_wt, alpha=0.8, color="blue", label="WT")
ax[0].scatter(gamma_g_connectome_unc31, gamma_g_fitted_unc31, alpha=0.8, color="orange", label="unc31")
ax[0].set_xlabel("gamma_g (connectome)")
ax[0].set_ylabel("gamma_g (fitted)")
ax[0].set_title("gamma_g: Connectome vs Fitted")
ax[0].grid(True)
ax[0].legend(loc="upper left")

ax[1].scatter(gamma_s_connectome_wt, gamma_s_fitted_wt, alpha=0.8, color="blue", label="WT")
ax[1].scatter(gamma_s_connectome_unc31, gamma_s_fitted_unc31, alpha=0.8, color="orange", label="unc31")
ax[1].set_xlabel("gamma_s (connectome)")
ax[1].set_ylabel("gamma_s (fitted)")
ax[1].set_title("gamma_s: Connectome vs Fitted")
ax[1].grid(True)
ax[1].legend(loc="upper left")

plt.savefig(os.path.join(figures_folder, "gammas_connectome_vs_fitted_scatter_plot.png"), bbox_inches="tight")
plt.close(fig)

# Bar plot with the correlation of the data
gamma_g_corr_wt = np.corrcoef(gamma_g_connectome_wt, gamma_g_fitted_wt)[0, 1]
gamma_g_corr_unc31 = np.corrcoef(gamma_g_connectome_unc31, gamma_g_fitted_unc31)[0, 1]
gamma_s_corr_wt = np.corrcoef(gamma_s_connectome_wt, gamma_s_fitted_wt)[0, 1]
gamma_s_corr_unc31 = np.corrcoef(gamma_s_connectome_unc31, gamma_s_fitted_unc31)[0, 1]

labels = ["gamma_g (WT)", "gamma_g (unc31)", "gamma_s (WT)", "gamma_s (unc31)"]
correlations = [gamma_g_corr_wt, gamma_g_corr_unc31, gamma_s_corr_wt, gamma_s_corr_unc31]

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(labels, correlations, color=["blue", "orange", "blue", "orange"], alpha=0.8)
ax.set_ylabel("Correlation Coefficient")
ax.set_title("Correlation of Connectome vs Fitted Gammas")
ax.grid(axis="y")

plt.savefig(os.path.join(figures_folder, "gammas_connectome_vs_fitted_correlation_bar_plot.png"), bbox_inches="tight")
plt.close(fig)
