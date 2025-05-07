#
# code for fit of convolution kernels, combination of linear (exponentials) or non-linear (NEGF) kernels
#
# inspired by 
# https://github.com/leiferlab/pumpprobe/tree/main/scripts/fconnectivity/fit_responses_constrained_stim_eci
# https://github.com/leiferlab/pumpprobe/tree/main/scripts/fconnectivity/figures/compare_connectomes/funatlas_vs_correlations2
# Kunert et al., PRE 89 052805 (2014) for parameter estimation

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

ds_list_path = (
    "/home/gabrielm/paper_reproduction/ds_list_unc31.txt" if "--unc31" in sys.argv 
    else "/home/gabrielm/paper_reproduction/ds_list_wt.txt" if "--wt" in sys.argv 
    else "/home/gabrielm/paper_reproduction/ds_list_full.txt"
)
ds_list_spont_path = "/home/gabrielm/paper_reproduction/ds_list_ctrl_wt.txt"


for arg in sys.argv:
    _arg = arg.split(":")
    if _arg[0] == "--matchless-nan-th": 
        matchless_nan_th = float(_arg[1])
    elif _arg[0] == "--folder:":
        figures_folder = _arg[1]
    if _arg[0] == "--aconn-ds-i": 
        aconn_ds_i=int(_arg[1])

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

Ci = 500*Ci #  to change the dynamics for the time scale of calcium concentration/fluorescence instead of membrane potentials # to review latter

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


gamma_g_total = []

gamma_s_total = []

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
        fig_dir = figures_folder + "_".join(ds_tags[i_folder]) + f"/stim_neu_{stim_neuron_label}_{num_stimulations}x/fit_negf"
        data_dir = data_folder + "_".join(ds_tags[i_folder]) + f"/stim_neu_{stim_neuron_label}_{num_stimulations}x/fit_negf"
        if kwar_fit_lineal_model:
            fig_dir += "_equilibirum"
            data_dir += "_equilibirum"
        fig_dir   += "/"
        data_dir  += "/"

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

        responding_labels = np.array(labels)[responding]  # Create an array of labels for responding indexes
        labeled_neurons = [label != "" for label in responding_labels]  # Create a boolean list for non-empty labels


        n_responding_labeled = len(np.array(responding)[labeled_neurons])

        if n_responding > 12 or n_responding_labeled < 4:
            print(f"Skipping dataset {folder} with {n_responding} responding neurons.")
            continue

        #if any(label == '' for label in np.array(labels)[responding]):
        #    print(f"Skipping dataset {folder} with some responding neuron not identified.")
        #    continue

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


        if np.mean(responses_correlations[0]) < 0.6:
            print(f"Skipping dataset {folder} with low stimuli correlations.")
            continue
        
        # Responding parameters positions
        # review: consider the most similar labels if it does not match exactly
        responding_positions = []
        for i in range(len(responding)):
            if not labeled_neurons[i]: continue
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
                
        if len(responding_positions) != np.sum(labeled_neurons):
            print(f"Warning: Mismatch in responding positions for dataset {folder}.")
            continue

        responding_positions = np.array(responding_positions)

        responding_positions = np.array(responding_positions)
        distance_matrix = np.linalg.norm(responding_positions[:, np.newaxis, :] - responding_positions[np.newaxis, :, :], axis=-1)

        # Ensure the directories for figures and data exist
        os.makedirs(fig_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)


        # save connectome based network considering only responsive neurons over all stimulations
        gamma_g = (Ggap*ggap)[responding][:, responding] 
        gamma_s = (Gsyn*gsyn)[responding][:, responding]
        Es = Esyn[responding][:, responding]

        # Plot gamma_g and gamma_s as heatmaps
        nlfc.utils.netplots.connect_matrices_heatmap(gamma_g, gamma_s, responding_labels, os.path.join(fig_dir, 'gamma_g_gamma_s_heatmaps.png'))
        # plot neural network
        nlfc.utils.netplots.neural_network(gamma_g, gamma_s, Es, responding_labels, save_path=os.path.join(fig_dir, f'Neural_Network_responding_only_original_parameters.png'))
        
        # nodes at real positions
        #nlfc.utils.netplots.neural_network(gamma_g, gamma_s, Es, responding_labels, positions=responding_positions[:, [0,1]], save_path=os.path.join(fig_dir, f'Neural_Network_responding_only_before_fit_real_positions.png'))
    
        kunert_parameters.update({
            "gamma_g": gamma_g,
            "gamma_s":  gamma_s,
            "E_s": Es
        })

        model_parameters = kunert_parameters.copy()

        if load_cache:
            # Check if the cache file exists
            cache_file_path = os.path.join(data_dir, "fitted_parameters.pkl")
            if os.path.exists(cache_file_path):
                # Load parameters after fitting from the .pkl file
                with open(cache_file_path, "rb") as f:  # 'rb' not 'r'
                    model_parameters = pickle.load(f)
            else:
                print(f"Warning: Cache file '{cache_file_path}' does not exist. Proceeding without loading cached parameters.")


        Y_nonlin_fit = np.zeros_like(Y_smooth_total[:, responding, shift_vol:])

        G_degree = 2

        # initialize the greenfunctions class, computing the direct green functions of ecery tryal and every neuron pair interaction
        lif_gf = nlfc.GreenFunctions(
            model = LIF(n_responding, model_parameters),
            x = Y_smooth_total[:, responding, shift_vol::],
            dt = fconn.Dt,
        )

        G  = lif_gf.total_G(G_degree)
        
        for ie_idx, ie in enumerate(stimulations_idx):
             
            os.makedirs(ie_dir_list[ie_idx], exist_ok=True)
            # save plot the NEGF for each stimulation and each neuron pair (consider source only the stim neuron)
            for i in range(n_responding):
                for j in range(n_responding):
                    neu_i = responding[i]
                    neu_j = responding[j]
                    if i == 0 or (np.all(lif_gf.g[ie_idx][i, j] == 0)): continue
                    nlfc.utils.plots.time_level_curves(time_fit, lif_gf.g[ie_idx][i, j], lif_gf.g0[i, j, -1], xlabel=None, ylabel="g(t,t')", title=f"Neurons : {labels[neu_i]}<-{labels[neu_j]}", save_path= os.path.join(ie_dir_list[ie_idx],f'before_fit_negf_direct_g_neurons_neuron_pair_{labels[neu_i]}<-{labels[neu_j]}.png'))
                    nlfc.utils.plots.time_level_curves(time_fit, G[ie_idx][i, j], G[0][i, j][-1, :], xlabel=None, ylabel="G(t,t')", title=f"Neurons : {labels[neu_i]}<-{labels[neu_j]}", save_path= os.path.join(ie_dir_list[ie_idx],f'before_fit_negf_G_neurons_neuron_pair_{labels[neu_i]}<-{labels[neu_j]}.png'))



        #########################################################################################################################################################
        
        # FIT NEGF
        
        # Lower the sampling rate so fitting is not so time consuming

        lowering_resolution_step = 2
        fitting_window = 60

        print("NEGF fitting")
        min_constrain_dict = {
                        "C": 0.0,          
                        "gamma": 0.0,  
                        "beta": 0.0, 
                        "a_r": 0.0,  
                        "a_d": 0.0,  
                        "gamma_g": 0.0,
                        "gamma_s": 0.0,
                        "E_s": -120,
                        "E_c": -120,  
                        }
        max_constrain_dict = {
                        "C": np.inf,          
                        "gamma": np.inf,  
                        "beta": np.inf, 
                        "a_r": np.inf,  
                        "a_d": np.inf,  
                        "gamma_g": np.inf,
                        "gamma_s": np.inf,
                        "E_s": 100,
                        "E_c": 100,  
                        }

        fitted_parameters = lif_gf.ADAM_fit(
            x=Y_smooth_total[:, responding, shift_vol:shift_vol+fitting_window:lowering_resolution_step],
            dt=lowering_resolution_step * fconn.Dt,
            fit_linear_model=kwar_fit_lineal_model,
            target_nodes=np.arange(1, n_responding),  # Exclude index 0 (stimulated neuron)
            max_iters=200,
            constrain=(min_constrain_dict, max_constrain_dict),
            rms_tol=1e-2,
            parameter_to_fit_list=['C', 'gamma', 'E_c', 'gamma_g', 'gamma_s', 'E_s'],
            #loss_method='correlation'
        )
        # Compute green functions using the higher time resolution, but the fitted parameters
        lif_gf = nlfc.GreenFunctions(
            model = LIF(n_responding, fitted_parameters),
            x = Y_smooth_total[:, responding, shift_vol::],
            dt = fconn.Dt,
        )
        print('FIT DONE')

        #########################################################################################################################################################
        
        # plot neural network after fitting
        nlfc.utils.netplots.neural_network(fitted_parameters["gamma_g"], fitted_parameters["gamma_s"], fitted_parameters["E_s"], responding_labels, save_path=os.path.join(fig_dir, f'Neural_Network_responding_only_after_fit.png'))
       
        # nodes at real positions
        #nlfc.utils.netplots.neural_network(fitted_parameters["gamma_g"], fitted_parameters["gamma_s"], fitted_parameters["E_s"], responding_labels, positions=responding_positions[:, [0,1]], save_path=os.path.join(fig_dir, f'Neural_Network_responding_only_after_fit_real_positions.png'))
    
        ######
        # Plot
        ######        
        nrows = int(np.ceil(np.sqrt(num_neurons)))
        ncols = int(np.ceil(num_neurons / nrows))
        while nrows * ncols < num_neurons:
            ncols += 1
            nrows = int(np.ceil(num_neurons / ncols))
    
        if plot:
            print("plotting")
            try:
                fig.clear()
            except:
                pass
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols,figsize=(16,12))
            for a in np.ravel(ax): a.set_xticks([]);a.set_yticks([])
            if nrows==1: ax = np.array([ax])

        # plot signals for visualization
        for ie_idx in range(num_stimulations):
            os.makedirs(ie_dir_list[ie_idx], exist_ok=True)
             
            ie = stimulations_idx[ie_idx]

            fig, ax = plt.subplots(figsize=(10, 6))  # Create figure and axis
            fig_smooth, ax_smooth = plt.subplots(figsize=(10, 6))  # Create smoothed figure and axis

            for neu_j in range(fconn.n_neurons):  # Iterate over all neurons

                # Assign colors based on neuron type
                if neu_j == stim:
                    color, lw = "red", 3  # Stimulated neuron
                elif neu_j in responding:
                    color, lw = "blue", 2  # Responsive neurons
                else:
                    color, lw = "gray", 0.5  # Non-responsive neurons

                ax.plot(time_plt, Y_total[ie_idx, neu_j], color=color, linewidth=lw, alpha=0.6)
                ax_smooth.plot(time_plt, Y_smooth_total[ie_idx, neu_j], color=color, linewidth=lw, alpha=0.6)  # Corrected to use smoothed data
            for i in range(n_responding):
                neu_i = responding[i]
                neu_j = stim
                if np.all(Y_nonlin_fit[ie_idx, :, i] == 0.0) : continue
                #ax.plot(time_fit, Y_nonlin_fit[ie_idx, :, i], color=color, linewidth=lw, alpha=0.8, ls='--')  # using conectome estimated propagated signals
                ax_smooth.plot(time_fit, Y_nonlin_fit[ie_idx, i], color="blue", linewidth=2, alpha=0.9, ls='--')
        
            # Create custom legend handles
            stim_handle = mlines.Line2D([], [], color="red", linewidth=2.5, label="Stimulated")
            responsive_handle = mlines.Line2D([], [], color="blue", linewidth=2, label="Responsive")
            non_responsive_handle = mlines.Line2D([], [], color="gray", linewidth=1, label="Non-responsive")

            # Add legends to both plots
            ax.legend(handles=[stim_handle, responsive_handle, non_responsive_handle], loc="upper right")
            ax_smooth.legend(handles=[stim_handle, responsive_handle, non_responsive_handle], loc="upper right")

            # Set plot labels & titles
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Delta F/F")
            ax.set_title(f"Neuron Responses to Stimulation {ie}")

            ax_smooth.set_xlabel("Time (s)")
            ax_smooth.set_ylabel("Delta F/F")
            ax_smooth.set_title(f"Neuron Responses to Stimulation {ie} (Smoothed)")

            # Mark key time points with vertical lines
            ax.axvline(0, color="k", alpha=0.5, linestyle="--")  # Stimulation onset
            #ax.axvline(fconn.next_stim_after_n_vol[ie] * fconn.Dt, color="k", alpha=0.5, linestyle="--")

            ax_smooth.axvline(0, color="k", alpha=0.5, linestyle="--")  # Stimulation onset
            #ax_smooth.axvline(fconn.next_stim_after_n_vol[ie] * fconn.Dt, color="k", alpha=0.5, linestyle="--")

            # Set y-limits (before tight_layout)
            ax.set_ylim(-31, 31)
            ax_smooth.set_ylim(-61, 61)

            # Final plot adjustments
            fig.tight_layout()
            fig_smooth.tight_layout()

            # Save figures with correct filenames
            fig.savefig(f"{ie_dir_list[ie_idx]}signals.png", bbox_inches="tight")
            fig_smooth.savefig(f"{ie_dir_list[ie_idx]}signals_smooth.png", bbox_inches="tight")  # Fixed filename

            # Close figures properly
            plt.close(fig)
            plt.close(fig_smooth)

        #G = lif_gf.total_G(G_degree) 
        #G0 = lif_gf.total_G(G_degree, linear_model=True)

        G  = lif_gf.total_G(G_degree)

        for ie_idx, ie in enumerate(stimulations_idx):

            # save plot the NEGF for each stimulation and each neuron pair (consider source only the stim neuron)
             for i in range(n_responding):
                neu_i = responding[i]
                Y_nonlin_fit[ie_idx][i, :] = np.full_like(Y_nonlin_fit[ie_idx][i, :], Y_smooth_total[ie_idx][neu_i, shift_vol])
                for j in range(n_responding):
                    neu_j = responding[j]
                    delta_j = Y_smooth_total[ie_idx][neu_j, shift_vol:] - Y_smooth_total[ie_idx][neu_j, shift_vol]
                    Y_nonlin_fit[ie_idx, i] += nlfc.utils.nontt_conv(lif_gf.g[ie_idx][i, j], delta_j, dt=fconn.Dt)
                    
                    if i == 0 or (np.all(abs(lif_gf.g[ie_idx][i, j]) < 1e-04)): 
                        continue
                    #nlfc.utils.plots.t_t_heatmap(x, g[ie_idx][i, j, :, :], os.path.join(ie_dir, f'negf_g_heatmap_neuron_pair_{i}_{j}_stimulation_{str(ie)}.png'))
                    nlfc.utils.plots.time_level_curves(time_fit, lif_gf.g[ie_idx][i, j], lif_gf.g0[i, j, -1], xlabel=None, ylabel="g(t,t')", title=f"Neurons : {labels[neu_i]}<-{labels[neu_j]}", save_path= os.path.join(ie_dir_list[ie_idx],f'fitted_negf_direct_g_neurons_neuron_pair_{labels[neu_i]}<-{labels[neu_j]}.png'))
                    #nlfc.utils.plots.t_t_heatmap(time_fit, G[:, :, i, 0], os.path.join(ie_dir, f'negf_G{G_degree}_heatmap_neuron_pair_{labels[responding[i]]}_{labels[stim]}_stimulation_{str(ie)}.png'))
                    nlfc.utils.plots.time_level_curves(time_fit, G[ie_idx][i, j], G[0][i, j][-1, :], xlabel=None, ylabel="G(t,t')", title=f"Neurons : {labels[neu_i]}<-{labels[neu_j]}", save_path= os.path.join(ie_dir_list[ie_idx],f'fitted_negf_G_neurons_neuron_pair_{labels[neu_i]}<-{labels[neu_j]}.png'))
                     
        ###############
        # PREPARE PANELS PLOT
        ###############

        color_map = cm.get_cmap("tab10", num_stimulations)  # Use tab10 or any other colormap
        nrows = int(np.ceil(np.sqrt(len(responding))))
        ncols = int(np.ceil(len(responding) / nrows))
        while nrows * ncols < len(responding):
            ncols += 1
            nrows = int(np.ceil(len(responding) / ncols))
    
        try:
            fig.clear()
        except:
            pass
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

        for i, neu_i in enumerate(responding):

            # Plot only for detected responses
            i_plot = np.where(responding==neu_i)[0][0]
            ax_r = i_plot//ncols
            ax_c = i_plot%ncols

            panel_title = f"Neuron {neu_i}: {labels[neu_i]}"

            k = []
        
            for ie_idx, ie in enumerate(stimulations_idx):
                stim_color = color_map(ie_idx)  # Get a distinct color for each ie plot line

                y_plt = Y_total[ie_idx, neu_i]
                y = Y_total[ie_idx, neu_i][shift_vol:]
                y_smooth_plt = Y_smooth_total[ie_idx, neu_i]
                y_smooth = Y_smooth_total[ie_idx, neu_i][shift_vol:]

                # Get the unconstrained parameters to build a cleaned-up version of the stimulated neuron's activity (FOR LIN KERNEL)
                stim_unc_par_dict = fconn.fit_params_unc[ie][stim]
                stim_unc_par = fconn.get_irrarray_from_params(stim_unc_par_dict)
                
                stim_y = pp.Fconn.eci(time_fit, stim_unc_par)  # stim_y is an exponential kernel
    
                fconn.clear_fit_results(stim=ie,neu=neu_i,mode="constrained")
                
                params_, n_branch_params, _ = fconn.fit_eci_branching(
                                time_fit,y_smooth,stim_y,dt=fconn.Dt,
                                n_hops_min=2,n_hops_max=3,
                                n_branches_max=2,#3,
                                rms_limits=[None,None],auto_stop=True,rms_tol=1e-2,
                                method="trf",routine="least_squares")
                
                if params_ is None: 
                    params_dict = {"params": [0,1], 
                            "n_branches": 1, 
                            "n_branch_params": [2]}
                    fconn.fit_params[ie][neu_i] = params_dict
                    continue
                
                params_dict = {"params": np.array(params_), 
                            "n_branches": len(n_branch_params), 
                            "n_branch_params": n_branch_params}
                fconn.fit_params[ie][neu_i] = params_dict
                

                params = fconn.get_irrarray_from_params(params_dict)
                
                k_trial = pp.Fconn.eci(time_fit,params)
                k.append(k_trial)

                fit_y_trial =  pp.convolution(stim_y, k_trial, fconn.Dt,8)

                #lbl = str(neu_i)
                lbl = f'Response to stim. #{ie}'
                fit_lbl = "Trial kernel pred." #"|".join([str(nbp - 1) for nbp in n_branch_params])
                lw = 1
                if neu_i == stim: 
                    lbl += "*"
                    lw = 2
                    ax2[0].set_title("Stimulated "+ panel_title, fontsize=10)
                    ax2[0].plot(time_plt, y_smooth_plt, label=lbl, c=stim_color, lw=lw)
                    #ax2[0].plot(time, y_plt, c=stim_color, lw=lw, alpha=0.2)
                    ax2[0].set_xlim(time_plt[0], time_plt[-1])
                    ax2[0].set_ylim(np.nanmin(Y_smooth_total[:, neu_i, :]), np.nanmax(Y_smooth_total[:, neu_i, :]))
                    ax2[0].axvline(0, c="k", alpha=0.5, label="stim. time")
                    ax2[0].axvspan(0, time_fit[fitting_window], color="gray", alpha=0.15, label="Training")
                    #ax2[0].axvline(fconn.next_stim_after_n_vol[ie] * fconn.Dt, c="k", alpha=0.5)

                elif neu_i == most_variable_neuron:
                    ax2[1].set_title(panel_title, fontsize=10)
                    ax2[1].plot(time_plt, y_smooth_plt, label=lbl, c=stim_color, lw=lw)
                    #ax2[1].plot(time_fit, fit_y_trial, label=fit_lbl, c=stim_color, lw=1, ls=':')
                    #ax2[0].plot(time, y_plt, c=stim_color, lw=lw, alpha=0.2)
                    ax2[1].set_xlim(time_plt[0], time_plt[-1])
                    ax2[1].set_ylim(np.nanmin(Y_smooth_total[:, neu_i, :]), np.nanmax(Y_smooth_total[:, neu_i, :]))
                    ax2[1].plot(time_fit, Y_nonlin_fit[ie_idx,i], label="Nonlinear kernel pred.", lw=2, ls='--', c=stim_color)
                    ax2[1].axvline(0, c="k", alpha=0.5, label="stim. time")
                    ax2[1].axvspan(0, time_fit[fitting_window], color="gray", alpha=0.15, label="Training")

                ax[ax_r, ax_c].plot(time_plt, y_smooth_plt, label=lbl, c=stim_color, lw=lw)
                #ax[ax_r, ax_c].plot(time_fit, fit_y_trial, label=fit_lbl, c=stim_color, lw=1, ls=':')

                #rf_plt = rf
                #rf_plt /= np.max(np.abs(rf_plt)) / np.max(np.abs(fit_y))
                stim_y_plt = stim_y / np.sum(stim_y) * np.abs(np.sum(y))
                if neu_i != stim:
                    ax[ax_r, ax_c].plot(time_fit, Y_nonlin_fit[ie_idx, i], label="Nonlinear kernel pred.", lw=2, ls=':', c=stim_color)
                    
                # ax[ax_r, ax_c].plot(x, rf_plt, label="rf", lw=2, c="k")
                # ax[ax_r, ax_c].plot(x, stim_y_plt, label=f"st stimulation {ie}", lw=2, c=stim_color, alpha=0.6)

                #params = fconn.get_irrarray_from_params(params_dict)
                #rf = pp.Fconn.eci(x, params)
                #fit_y = km[neu_i, stim,  :] # pp.convolution(stim_y, rf, fconn.Dt, 8)
                #fit_ls = ":"
                #fit_lbl = "Av. kernel fit" #"|".join([str(nbp - 1) for nbp in n_branch_params])
                #ax[ax_r, ax_c].plot(np.linspace(0,60,120), fit_y, label=fit_lbl, c=stim_color, lw=1, ls=fit_ls)

                
            # Find the minimum length of arrays in k
            min_len = min(len(item) for item in k)

            # Trim each array to min_len using list comprehension
            k_trimmed = [item[:min_len] for item in k]

            # Compute the average
            lin_kernel = np.average(np.array(k_trimmed), axis=0)

            fit_y =  pp.convolution(stim_y, lin_kernel, fconn.Dt,8)
            fit_ls = "-"
            fit_lbl = "Linear kernel pred." #"|".join([str(nbp - 1) for nbp in n_branch_params])
            if neu_i == stim:
                panel_title = "Stimulated "+ panel_title
                #ax2[0].plot(time_fit, stim_y, label="Filtered Stim.", c='yellow', lw=1)
                #ax2[0].legend()
            elif neu_i == most_variable_neuron:
                ax2[1].plot(time_fit, fit_y, label=fit_lbl, c='black', lw=1, ls=fit_ls)
                ax2[1].legend(loc='upper left', bbox_to_anchor=(0, 1))
            if neu_i != stim: 
                ax[ax_r, ax_c].plot(time_fit, fit_y, label=fit_lbl, c='black', lw=1, ls=fit_ls)

            ax[ax_r, ax_c].set_xlim(time_plt[0], time_plt[-1])
            ax[ax_r, ax_c].set_ylim(np.nanmin(Y_smooth_total[:, neu_i, :]), np.nanmax(Y_smooth_total[:, neu_i, :]))
            ax[ax_r, ax_c].axvline(0, c="k", alpha=0.8)
            ax[ax_r, ax_c].axvspan(0, time_fit[fitting_window], color="gray", alpha=0.15, label="Training")
            

            ax[ax_r, ax_c].set_title(panel_title, fontsize=10)
            if i_plot == len(responding) - 1:  # Add legend only for the last panel
                handles, labels_plt = ax[ax_r, ax_c].get_legend_handles_labels()
                fig.legend(handles, labels_plt, loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=3)


        # Save plot with neuron index in filename
        filename = f"panels_mult_stimulation_fits.png"
        filename2 = f"most_variable_neuron_response.png"
        fig.savefig(os.path.join(fig_dir, filename), bbox_inches="tight")
        plt.close(fig)        # Plot heatmaps for each neuron pair
        fig2.savefig(os.path.join(fig_dir, filename2), bbox_inches="tight")
        plt.close(fig2)        # Plot heatmaps for each neuron pair




        # Scatter plot gamma_g and gamma_s as a function of the distance matrix
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Flatten the matrices for scatter plotting
        distances = distance_matrix.flatten()
        gamma_g_values = fitted_parameters["gamma_g"][labeled_neurons][:, labeled_neurons].flatten()
        gamma_s_values = fitted_parameters["gamma_s"][labeled_neurons][:, labeled_neurons].flatten()

        # Plot gamma_g vs distance
        valid_indices = [i for i, d in enumerate(distances) if d is not None]
        filtered_distances = [distances[i] for i in valid_indices]
        gamma_g_values = [gamma_g_values[i] for i in valid_indices]
        gamma_s_values = [gamma_s_values[i] for i in valid_indices]
        
        ax[0].scatter(filtered_distances, gamma_g_values, alpha=0.8, label="gamma_g")
        ax[0].set_xlabel("Distance")
        ax[0].set_ylabel("gamma_g")
        ax[0].set_title("gamma_g vs Distance")
        ax[0].grid(True)

        # Plot gamma_s vs distance
        ax[1].scatter(distances, gamma_s_values, alpha=0.8, label="gamma_s", color="orange")
        ax[1].set_xlabel("Distance")
        ax[1].set_ylabel("gamma_s")
        ax[1].set_title("gamma_s vs Distance")
        ax[1].grid(True)

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "gamma_vs_distance_scatter.png"), bbox_inches="tight")
        plt.close(fig)

        gamma_g_total.extend(fitted_parameters["gamma_g"][labeled_neurons][:, labeled_neurons].flatten())
        gamma_s_total.extend(fitted_parameters["gamma_s"][labeled_neurons][:, labeled_neurons].flatten())
        distances_total.extend(distances.flatten())

        # Save data

        # Save parameters before fitting as a tab-delimited text file
        params_before_fitting = model_parameters

        with open(os.path.join(data_dir, "params_before_fitting.txt"), "w") as f:
            f.write("Parameter\tValue\n")
            for key, value in params_before_fitting.items():
                f.write(f"{key}\t{value}\n")

        # Save parameters after fitting as a tab-delimited text file
        with open(os.path.join(data_dir, "fitted_parameters.txt"), "w") as f:
            f.write("Parameter\tValue\n")
            for key, value in fitted_parameters.items():
                f.write(f"{key}\t{value}\n")

        # Save parameters after fitting in a JSON format for easier loading
        with open(os.path.join(data_dir, "fitted_parameters.pkl"), "wb") as f:
            pickle.dump(fitted_parameters, f)

        # Save gamma_g, gamma_s, and Esyn as separate tab-delimited text files
        np.savetxt(os.path.join(data_dir, "gamma_g.txt"), gamma_g, delimiter="\t", fmt="%.6f")
        np.savetxt(os.path.join(data_dir, "gamma_s.txt"), gamma_s, delimiter="\t", fmt="%.6f")
        np.savetxt(os.path.join(data_dir, "Esyn.txt"), Esyn, delimiter="\t", fmt="%.6f")

        np.savetxt(os.path.join(data_dir, "gamma_g_after_fitting.txt"), fitted_parameters['gamma_g'], delimiter="\t", fmt="%.6f")
        np.savetxt(os.path.join(data_dir, "gamma_s_after_fitting.txt"), fitted_parameters['gamma_s'], delimiter="\t", fmt="%.6f")
        np.savetxt(os.path.join(data_dir, "E_s_after_fitting.txt"), fitted_parameters['E_s'], delimiter="\t", fmt="%.6f")
        plt.close('all')
# Scatter plot gamma_g and gamma_s as a function of the distance matrix
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Flatten the matrices for scatter plotting
D = 10 # mm²/s
distances = distances_total
distances_squared = [d**2 for d in distances]
distances_exp_squared = [np.exp(-d**2)/(4*D) for d in distances]
gamma_g_values = gamma_g_total
gamma_s_values = gamma_s_total

# Plot gamma_g vs distance
ax[0].scatter(distances, gamma_g_values, alpha=0.8, label="gamma_g", color="blue")
ax[0].scatter(distances, gamma_s_values, alpha=0.8, label="gamma_s", color="orange")
ax[0].set_xlabel("r")
ax[0].set_ylabel("gamma")
ax[0].legend()
ax[0].set_title("coupling vs Distance")
ax[0].grid(True)

# Plot gamma_g vs distance
ax[1].scatter(distances_exp_squared, gamma_g_values, alpha=0.8, label="gamma_g", color="blue")
ax[1].scatter(distances_exp_squared, gamma_s_values, alpha=0.8, label="gamma_s", color="orange")
ax[1].set_xlabel("exp(-r^2/(4D))")
ax[1].set_ylabel("gamma")
ax[1].set_title("coupling vs exp(-Distance^2/(4D))")
ax[1].grid(True)

coupling_vs_distance_file_name = (
    "gamma_vs_distance_scatter_all_trials_unc_31.png" if "--unc31" in sys.argv 
    else "gamma_vs_distance_scatter_all_trials_wt.png" if "--wt" in sys.argv 
    else "gamma_vs_distance_scatter_all_trials_all.png"
)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, coupling_vs_distance_file_name), bbox_inches="tight")
plt.close(fig)
