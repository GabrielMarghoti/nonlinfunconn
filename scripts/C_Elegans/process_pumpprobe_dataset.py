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
skip_processed = "--skip-processed" in sys.argv

worm_type = 'all'
data_path = None
figures_path = None

for arg in sys.argv:
    _arg = arg.split(":")
    
    if _arg[0] == "--worm-type": 
        worm_type = str(_arg[1])
    elif _arg[0] == "--data-path":
        data_path = _arg[1]
    elif _arg[0] == "--figures-path":
        figures_path = _arg[1]

if data_path:
    data_folder = data_path
if figures_path:
    figures_folder = figures_path

aconn_ds_i = None # default is loading from funatlas, if aconn_ds_i is set, it will load from the specified dataset
# default 
figures_folder = "figures/C_elegans_pumpprobre_exp/"
data_folder = "data/C_elegans_pumpprobre_exp/"#

ds_list_path = (
    "/home/gabrielm/paper_reproduction/ds_list_unc31.txt" if worm_type == "unc31"
    else "/home/gabrielm/paper_reproduction/ds_list_wt.txt" if worm_type == "wt"
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

kunert_ODE_parameters = {
    "C": Ci,          
    "gamma": Gcell,  
    "E_c": Ecell,  
    "beta": beta, 
    "a_r": ar,  
    "a_d": ad,  
    "gamma_g": Ggap*ggap,
    "gamma_s":Gsyn*gsyn,
    "Es": Esyn,
    }


# Save the kunert_ODE_parameters dictionary as a pickled file
kunert_ODE_parameters_file_path = os.path.join(data_folder, "kunert_ODE_parameters.pkl")
with open(kunert_ODE_parameters_file_path, "wb") as f:
    pickle.dump(kunert_ODE_parameters, f)
print(f"kunert_ODE_parameters saved to {kunert_ODE_parameters_file_path}")

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

        try:
            # Get the stimulation neurons
            if stim in stim_neurons_analyzed:
                continue
            stim_neurons_analyzed.add(stim)

            #stim = np.bincount(fconn.stim_neurons[fconn.stim_neurons > 0]).argmax()
            stimulations_idx = np.where(fconn.stim_neurons == stim)[0]

            num_stimulations = len(stimulations_idx)

            # Skip datasets where the number of stimulations is not between 2 and 3
            if not (2 <= num_stimulations <= 5): 
                continue

            stim_neuron_label = labels[stim]

            if stim_neuron_label == "":
                print("Stim neuron label is empty, skipping...")
                continue
            
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
            data_dir = data_folder + f"worm_type_{worm_type}/{ds_tags[i_folder][0]}" + f"/stim_neu_{stim_neuron_label}/"

            if os.path.exists(os.path.join(data_dir, "processed_data.pkl")) and os.path.exists(os.path.join(data_dir, "lin_kernels_fit_responding_neurons.pkl")) and skip_processed: continue

            ie_dir_list = []

            for ie in stimulations_idx:  # stimulation index only though cases which the most stimulated neuron is stimulated
                responding_ie = fconn.resp_neurons_by_stim[ie]
                i0 = max(0, fconn.i0s[ie])  # start of the stimulation
                i1 = fconn.i1s[ie]         # end of the stimulation

                responding.update(responding_ie)  # Add the responding neurons to the set

                n_responding_ie = len(responding_ie)
                    
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
        

            Y_total = np.concatenate(Y_total, axis=0)  # Concatenate along the new axis to maintain 3D structure
            Y_smooth_total = np.concatenate(Y_smooth_total, axis=0)  # Concatenate along the new axis to maintain 3D structure

            positions = [(None, None, None)] * len(labels)  # Initialize positions with None values
            for i in range(len(labels)):
                for key in anatlas_positions:
                    if labels[i].startswith(key) | key.startswith(labels[i]) | labels[i].startswith(key[:2]) |  key.startswith(labels[i][:2]):
                        positions[i] = anatlas_positions[key]
                        break
            
            n_responding = len(responding)

            if n_responding > 20 or n_responding < 2:
                continue       

            positions = np.array(positions)

            # Ensure the directories for figures and data exist

            os.makedirs(data_dir, exist_ok=True)


            # Create a data dictionary with necessary variables
            data_dict = {
                "worm_type": worm_type,
                "responding_neurons": responding,
                "stim_neuron": stim,
                "neuron_labels": labels,
                "neuron_positions": positions,
                "signal_raw": Y_total,
                "signal_smooth": Y_smooth_total,
                "time_fit": time_fit,
                "time_plt": time_plt,
                "dt": fconn.Dt,
                "stim_begin_idx": shift_vol,
            }

            # Save the data dictionary as a pickled file
            pickle_file_path = os.path.join(data_dir, "processed_data.pkl")
            with open(pickle_file_path, "wb") as f:
                pickle.dump(data_dict, f)
            print(f"Data dictionary saved to {pickle_file_path}")
            f.close()

            lin_kernel = np.zeros((len(responding), time_fit_len))
            fit_y = np.zeros((len(responding), time_fit_len))
            for i, neu_i in enumerate(responding):


                k = []
            
                for ie_idx, ie in enumerate(stimulations_idx):

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
                    else:
                        params_dict = {"params": np.array(params_), 
                                    "n_branches": len(n_branch_params), 
                                    "n_branch_params": n_branch_params}
                    
                    params = fconn.get_irrarray_from_params(params_dict)
                    
                    k_trial = pp.Fconn.eci(time_fit,params)
                    k.append(k_trial)

                    fit_y_trial =  pp.convolution(stim_y, k_trial, fconn.Dt,8)

                # Find the minimum length of arrays in k
                min_len = min(len(item) for item in k)

                # Trim each array to min_len using list comprehension
                k_trimmed = [item[:min_len] for item in k]

                # Compute the average
                lin_kernel[i] = np.average(np.array(k_trimmed), axis=0)

                fit_y[i] =  pp.convolution(stim_y, lin_kernel, fconn.Dt,8)

            # Save the linear kernel data
            lin_kernel_data = {
                "lin_kernel": lin_kernel,
                "fit_y": fit_y,
                "time_fit": time_fit,
                "dt": fconn.Dt,
            }

            lin_kernel_file_path = os.path.join(data_dir, f"lin_kernels_fit_responding_neurons.pkl")
            with open(lin_kernel_file_path, "wb") as f:
                pickle.dump(lin_kernel_data, f)
            f.close()
            print(f"Linear kernel data saved to {lin_kernel_file_path}")

            # Clear variables to free memory
            del Y_total, Y_smooth_total, lin_kernel, fit_y, data_dict, lin_kernel_data

            pass
        except Exception as e:
            print(f"Error processing folder {folder}: {e}. Stim neuron: {stim}.")
            continue
