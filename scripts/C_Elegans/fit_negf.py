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
import matplotlib.colors as mcolors
from scipy.integrate import simps

import os, sys, time, json
import pumpprobe as pp
import wormdatamodel as wormdm
import wormbrain as wormb
import tqdm

import gc

import nonlinfunconn as nlfc # for non-linear kernels

plot = True

# Parse command-line arguments
save = "--no-save" not in sys.argv
sig_green = "--signal:green" in sys.argv
skip_if_not_manually_located = "--skip-if-not-manually-located" in sys.argv
matchless_nan_th = None
matchless_nan_th_from_file = "--matchless-nan-th-from-file" in sys.argv
matchless_nan_th_added_only = "--matchless-nan-th-added-only" in sys.argv
merge = "--no-merge" not in sys.argv
ds_exclude_tags = "mutant" if "--unc31" in sys.argv else None

aconn_ds_i = None # default is loading from funatlas, if aconn_ds_i is set, it will load from the specified dataset
# default 
output_folder = "figures/"

ds_list_path = "/home/gabrielm/paper_reproduction/ds_list_full.txt"
ds_list_spont_path = "/home/gabrielm/paper_reproduction/ds_list_ctrl_wt.txt"

for arg in sys.argv:
    _arg = arg.split(":")
    if _arg[0] == "--matchless-nan-th": 
        matchless_nan_th = float(_arg[1])
    elif _arg[0] == "--folder:":
        output_folder = _arg[1]
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

#occ1,occ2 = funa.get_occurrence_matrix(req_auto_response=True)
#occ3 = funa.get_observation_matrix(req_auto_response=True)
#km = funa.get_kernels_map(occ2,occ3,filtered=True,include_flat_kernels=False)


#aconn_chem, aconn_elec = funa.get_aconnectome_from_file() # get the anatomical connectome with the correct atlas index for neuros
#num_neurons = aconn_chem.shape[0]


############################################################################################################################################
########### #NEGF kernels
############################################################################################################################################

# Find the kernel parameters based on Kunert C.Elegans model
f = open('/home/gabrielm/paper_reproduction/kunertPRE2014/params.json','r')
params = json.load(f)
f.close()

f = open('/home/gabrielm/paper_reproduction/kunertPRE2014/aconnectome.json','r')
content = json.load(f)
Neurotrans_ = np.array(content['chemical_sign'])
f.close()
f = open('/home/gabrielm/paper_reproduction/kunertPRE2014/neurons.txt','r')
neu_id_ = []
for line in f.readlines():
    ni = line.split("\t")[1]
    if ni[-1]=="\n": ni=ni[:-1]
    neu_id_.append(ni)
f.close()

# Transfer over the neurotransmitter information to the funatlas reference frame
Neurotrans = np.ones(funa.n_neurons)
for i_n in np.arange(len(Neurotrans_)):
    ai = funa.ids_to_i(neu_id_[i_n])
    Neurotrans[ai] = Neurotrans_[i_n]
    
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
        sign[ai,aj] = -1

# Get the composite aconnectome via the Funatlas
if aconn_ds_i is None:
    Gsyn, Ggap = funa.get_aconnectome_from_file(chem_th=0,gap_th=0,exclude_white=False,average=True)
else:
    aconn_folder = funa.module_folder
    aconn_fname = funa.aconn_sources[aconn_ds_i]["fname"]
    Gsyn, Ggap = funa._get_aconnectome_witvliet(aconn_folder+aconn_fname)

# Number of neurons
num_neurons = len(Neurotrans)

#If non-interacting, set all elements to zero
"""
if params['interacting'] == 0:
    Gsyn[:,:] = 0
    Ggap[:,:] = 0
"""

# Cell
Ci = params['C'] # Membrane capacitance 1 pF

Ci = Ci*500  ############### review, this is necessary for better time scale, otherwise exponentials decrease to fast (the kernel decay is miliseconds)

Gcell = params['Gcell'] # Leakage conductance of membrane [pS]
Ecell = params['Ecell']*1000 # Leakage potential [mV]

# Electrical synapses
ggap = params['ggap'] # conductivity of electrical synapse [pS]

# Chemical synapses
gsyn = params['gsyn'] # "conductivity" of chemical synapse [pS]
ar = params['ar'] # activation rate of synapses [s^-1]
ad = params['ad'] # deactivation rate of synapses [s^-1]
beta = params['beta']/1000 # width of synaptic activation [mV^-1]
esynexc = params['esynexc']*1000 # reverse potential for excitatory synapses
esyninh = params['esyninh']*1000 # reverse potential for inhibitory synapses

# Print the parameters
"""
print(f"Gcell: {Gcell}")
print(f"Ecell: {Ecell}")
print(f"ggap: {ggap}")
print(f"gsyn: {gsyn}")
print(f"ar: {ar}")
print(f"ad: {ad}")
print(f"beta: {beta}")
print(f"esynexc: {esynexc}")
print(f"esyninh: {esyninh}")
"""

# Build the Esyn array of the synaptic reverse potentials
# The index is presynaptic neuron, which determines the neurotransmitter and
# hence the sign of the synapse.
Esyn = np.ones((funa.n_neurons,funa.n_neurons))*esynexc
Esyn[sign<0] = esyninh

# Iterate over the folders whcih contains each experiment data
for (i_folder, folder) in enumerate(ds_list):

    #if '20211104_163944' not in folder: continue # use only folder of waterfall fig1

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
    sig.median_filter()

    #sig.get_smoothed(127,None,3,"sg_causal")
    sig.smooth(n=60,i=None,poly=4,mode="sg")

    # Get the neurons coordinates of the reference volume and load the matches
    # to determine what neuron was targeted
    cervelli = wormb.Brains.from_file(folder,ref_only=True)
    labels = cervelli.get_labels(0)


    # Check that the targets have been manually located, and, if not, ask for 
    # confirmation.
    if not fconn.manually_located_present:
        if not skip_if_not_manually_located:
            cont = input("Continue? (y/n)") == "y"
            if not cont: quit() 
        else:
            print("\tSkipping this dataset.")
            quit()
            
#### Get the neuron stimulated most times

    if len(fconn.stim_neurons > 0) == 0: continue 
    stim = np.bincount(fconn.stim_neurons[fconn.stim_neurons > 0]).argmax()
    stimulations_idx = np.where(fconn.stim_neurons == stim)[0]

    num_stimulations = len(stimulations_idx)

    if num_stimulations < 4: continue # consider only neurons stimulated at least 4 times

    stim_neuron_label = labels[stim]

#####

    print("Processing dataset", i_folder, ":", folder)

    # Ensure output directory exists
    main_dir = output_folder + "_".join(ds_tags[i_folder]) + f"_stim_neu_{stim_neuron_label}_{num_stimulations}x/"
    os.makedirs(main_dir, exist_ok=True)

    # plot neural network complete
    nlfc.utils.netplots.neural_network((Ggap*ggap/Ci), (Gsyn*gsyn/Ci), Esyn, np.array(labels), os.path.join(main_dir, f'Neural_Network_total.png'))

    responding = set()  # Initialize a set to store all responding neurons across all ie stimulation loops
    Y_total = []
    Y_smooth_total = []

    for ie in stimulations_idx:  # stimulation index only though cases which the most stimulated neuron is stimulated
        responding_ie = fconn.resp_neurons_by_stim[ie]
        
        #########################################
        # CASES IN WHICH TO SKIP THIS STIMULATION
        #########################################
        # If the targeting has failed
        """
        if stim == -2:
            stimulations_idx = np.delete(stimulations_idx, np.where(stimulations_idx == ie))
            num_stimulations = len(stimulations_idx)
            continue
        # If no response was detected in the stimulated neuron, because you'd risk
        # finding weird kernels just because there is no activity as input.
        if stim not in responding_ie:
            stimulations_idx = np.delete(stimulations_idx, np.where(stimulations_idx == ie))
            num_stimulations = len(stimulations_idx)
            continue
        # If the fit of the stimulated neuron's activity did not converge.    
        if fconn.fit_params_unc[ie][stim]["n_branches"]==0:
            stimulations_idx = np.delete(stimulations_idx, np.where(stimulations_idx == ie))
            num_stimulations = len(stimulations_idx)
            continue
        """
        responding.update(responding_ie)  # Add the responding neurons to the set

        n_responding_ie = len(responding_ie)
        
        i0 = max(0, fconn.i0s[ie])  # start of the stimulation
        i1 = fconn.i1s[ie]         # end of the stimulation
        shift_vol = fconn.shift_vols[ie]  # Negative-time interval (in steps) to consider before each stimulus.
        time = (np.arange(i1 - i0) - shift_vol) * fconn.Dt  # Dt is in seconds, convert time steps to times domain (s)
        # notice the slice for the signal is based on the stimulated neuron signal only

        # Ensure output directory exists
        ie_dir = main_dir + f"n_resp_neurons_{n_responding_ie}_ie_trial{ie}/"
        os.makedirs(ie_dir, exist_ok=True)

        i1p = shift_vol+fconn.next_stim_after_n_vol[ie]

        x = time[shift_vol:i1p]

        # Determine the range for baseline subtraction. Keep the full shift_vol
        # interval if the neuron was not responding before. But shorten it
        # if the neuron was responding to the previous stimulation. This latter
        # case is more sensitive to the noise, but avoids systematic wrong
        # baselines due to ongoing dynamics in the shift_vol segment.
        """
        # MUST CONSIDER CORRECTION IF THE RESPONSIVE NEURONS HAVE PREVIOUS STIMULATION ACTIVATION (far from equilibrium), which error it introduces? it imposes linearity?
        for neu_i in np.arange(n_responding_ie):
            if ie>0:
                if neu_i in fconn.resp_neurons_by_stim[ie-1]:
                    Y = sig.get_segment(i0,i1,shift_vol,unsmoothed_data=True,
                                        baseline_mode="constant",
                                        baseline_range=[shift_vol-4,shift_vol])[:,neu_i]#, FIXME FIXME FIXME
                                        #normalize="none")[:,neu_j]
                else:
                    Y = sig.get_segment(i0,i1,shift_vol,unsmoothed_data=True,
                                        baseline_mode="constant")[:,neu_i]#, FIXME FIXME FIXME
                                        #normalize="none")[:,neu_j]
            else:
                Y = sig.get_segment(i0,i1,shift_vol,unsmoothed_data=True,
                                        baseline_mode="constant")[:,neu_i]#, FIXME FIXME FIXME
                                        #normalize="none")[:,neu_j]
        """
        Y = sig.get_segment(i0, i1, shift_vol, unsmoothed_data=True, baseline_mode="constant")
        Y_smooth = sig.get_segment(i0, i1, shift_vol, unsmoothed_data=False, baseline_mode="constant")
        """
        for i in range(len(Y_smooth[0])): # Normalize signals
            y = Y_smooth[:, i]
            loc_std = np.sqrt(np.nanmedian(np.nanvar(rolling_window(y, 8), axis=-1)))
            Y_smooth /= loc_std
        """
        Y_total.append(Y[np.newaxis, ...])  # Add a new axis to ensure 3D structure
        Y_smooth_total.append(Y_smooth[np.newaxis, ...])  # Add a new axis to ensure 3D structure
        
    if stim not in responding:
        responding.update([stim]) 

    responding = list(responding)
    
    responding.remove(stim)
    responding.insert(0, stim)

    n_responding = len(responding)

    Y_total = np.concatenate(Y_total, axis=0)  # Concatenate along the new axis to maintain 3D structure
    Y_smooth_total = np.concatenate(Y_smooth_total, axis=0)  # Concatenate along the new axis to maintain 3D structure

    responses_correlations = np.zeros((n_responding, num_stimulations, num_stimulations))
    trial_variations = np.zeros(n_responding)
    common_trials = np.zeros(n_responding, dtype=int)

    for i in range(n_responding):
        neu_i = responding[i]
        responses_correlations[i, :, :] = np.corrcoef(
                    Y_smooth_total[:, shift_vol:i1p, neu_i]
                )
        # Calculate the standard deviation of the correlation matrix for each neuron
        trial_variations[i] = np.std(responses_correlations[i, :, :])

    # Find the neuron with the most variation in trial correlations
    most_variable_neuron_idx = np.argmax(trial_variations)
    most_variable_neuron = responding[most_variable_neuron_idx] 

    print(f"Neuron with most variation: {most_variable_neuron} ({labels[most_variable_neuron]})")
    

    # save connectome based network considering only responsive neurons over all stimulations
    gamma_g = (Ggap*ggap/Ci)[responding][:, responding] 
    gamma_s = (Gsyn*gsyn/Ci)[responding][:, responding]
    Es = Esyn[responding][:, responding]
    
    # Plot gamma_g and gamma_s as heatmaps
    nlfc.utils.netplots.connect_matrices_heatmap(gamma_g, gamma_s, os.path.join(ie_dir, 'gamma_g_gamma_s_heatmaps.png'))
    # plot neural network
    nlfc.utils.netplots.neural_network(gamma_g, gamma_s, Es, np.array(labels)[responding], os.path.join(ie_dir, f'Neural_Network_responding_only.png'))
    
    ####
    # Plot
    ####
    nrows = max(1,int(np.sqrt(num_neurons)))
    ncols = int(np.sqrt(num_neurons))+2
    if plot:
        print("plotting")
        try:
            fig.clear()
        except:
            pass
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols,figsize=(15,10))
        for a in np.ravel(ax): a.set_xticks([]);a.set_yticks([])
        if nrows==1: ax = np.array([ax])

    # plot signals for visualization
    for ie_idx in range(num_stimulations):
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
                color, lw = "gray", 1  # Non-responsive neurons

            ax.plot(time, Y_total[ie_idx, :, neu_j], color=color, linewidth=lw, alpha=0.7)
            ax_smooth.plot(time, Y_smooth_total[ie_idx, :, neu_j], color=color, linewidth=lw, alpha=0.7)  # Corrected to use smoothed data

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
        ax.axvline(fconn.next_stim_after_n_vol[ie] * fconn.Dt, color="k", alpha=0.5, linestyle="--")

        ax_smooth.axvline(0, color="k", alpha=0.5, linestyle="--")  # Stimulation onset
        ax_smooth.axvline(fconn.next_stim_after_n_vol[ie] * fconn.Dt, color="k", alpha=0.5, linestyle="--")

        # Set y-limits (before tight_layout)
        ax.set_ylim(-31, 31)
        ax_smooth.set_ylim(-61, 61)

        # Final plot adjustments
        fig.tight_layout()
        fig_smooth.tight_layout()

        # Save figures with correct filenames
        fig.savefig(f"{ie_dir}signals_{ie}.png", bbox_inches="tight")
        fig_smooth.savefig(f"{ie_dir}signals_{ie}_smooth.png", bbox_inches="tight")  # Fixed filename

        # Close figures properly
        plt.close(fig)
        plt.close(fig_smooth)


    ###############
    # PREPARE PANELS PLOT
    ###############

    color_map = cm.get_cmap("tab10", num_stimulations)  # Use tab10 or any other colormap

    nrows = max(1, int(np.sqrt(n_responding)))
    ncols = int(np.sqrt(n_responding)) + 2
    try:
        fig.clear()
    except:
        pass
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 12)) 
    fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(12, 6)) 
    for a in np.ravel(ax): 
        a.set_xticks([])
        a.set_yticks([])
        a.twinx().set_yticks([])
    for a in np.ravel(ax2): 
        a.set_xticks([])
        a.set_yticks([])
        a.twinx().set_yticks([])
    if nrows == 1: 
        ax = np.array([ax])


    for i, neu_i in enumerate(responding):

        # Plot only for detected responses
        i_plot = np.where(responding==neu_i)[0][0]
        ax_r = i_plot//ncols
        ax_c = i_plot%ncols

        panel_title = f"Neuron {neu_i}: {labels[neu_i]}"

        k = []
       
        for ie_idx, ie in enumerate(stimulations_idx):
            stim_color = color_map(ie_idx)  # Get a distinct color for each ie plot line

            i0 = max(0, fconn.i0s[ie])  # start of the stimulation
            i1 = fconn.i1s[ie]         # end of the stimulation
            shift_vol = fconn.shift_vols[ie]  # Negative-time interval (in steps) to consider before each stimulus.

            i1p = shift_vol+fconn.next_stim_after_n_vol[ie]

            time = (np.arange(i1 - i0) - shift_vol) * fconn.Dt  # Dt is in seconds, convert time steps to times domain (s)
            x = time[shift_vol:i1p]

            y_plt = Y_total[ie_idx, :,neu_i]
            y = Y_total[ie_idx, :,neu_i][shift_vol:i1p]
            y_smooth_plt = Y_smooth_total[ie_idx, :,neu_i]
            y_smooth = Y_smooth_total[ie_idx, :,neu_i][shift_vol:i1p]
            
            """
            ############################################################################################################        
            # Compute NEGF kernels
            ############################################################################################################
            # Compute the direct NEGF kernel
            # Initialize the NEGF kernel class
            nonlin_kernel = nlfc.negf.LIF(
                Vs = Y_smooth_total[ie, shift_vol:i1p, responding],  # Membrane potential dynamics (sliced signals)
                num_neurons = n_responding,
                gamma_g = gamma_g, 
                gamma_s = gamma_s, 
                gamma = Gcell/Ci, 
                C = Ci,  
                beta= beta,  
                E_c = Ecell, 
                E_s = Es, 
                a_r = ar, 
                a_d = ad, 
                dt = fconn.Dt,
                ts=x,
                Veq= Y_smooth_total[ie, shift_vol, responding]
            )
            
            G_degree = 2
            g = nonlin_kernel.compute_direct_negf() 
            G = nonlin_kernel.compute_effective_negf(g, G_degree) # until second neighbors
            """
            # Get the unconstrained parameters to build a cleaned-up version of the stimulated neuron's activity (FOR LIN KERNEL)
            stim_unc_par_dict = fconn.fit_params_unc[ie][stim]
            stim_unc_par = fconn.get_irrarray_from_params(stim_unc_par_dict)
            
            stim_y = pp.Fconn.eci(x, stim_unc_par)  # stim_y is an exponential kernel

            # Data segmentation based on stimulation window
            i0 = max(0, fconn.i0s[ie])  # start of the stimulation
            i1 = fconn.i1s[ie]         # end of the stimulation
            shift_vol = fconn.shift_vols[ie]  # Negative-time interval (in steps) to consider before each stimulus.
            # notice the slice for the signal is based on the stimulated neuron signal only
            

            #loc_std = sig.get_loc_std(y_smooth,4)
            
            fconn.clear_fit_results(stim=ie,neu=neu_i,mode="constrained")
            
            n_hops_min = 2
            
            params_, n_branch_params, _ = fconn.fit_eci_branching(
                            x,y_smooth,stim_y,dt=fconn.Dt,
                            n_hops_min=1,n_hops_max=3,
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
            
            k.append(pp.Fconn.eci(x,params))

            """
            for j in range(n_responding):
                if Gsyn[i, j] == 0 and Ggap[i, j]==0: continue # avoid computing null kernells (no connection)
                nonlin_fit_y[:, i] += nlfc.nontt_conv(g[:, :, i, j], Y_smooth[ie, shift_vol:i1p, j])
            """

            #lbl = str(neu_i)
            lbl = f'stim. {ie}'
            lw = 1
            if neu_i == stim: 
                lbl += "*"
                lw = 2
                ax2[0].set_title(panel_title, fontsize=10)
                ax2[0].plot(time, y_smooth_plt, label=lbl, c=stim_color, lw=lw)
                #ax2[0].plot(time, y_plt, c=stim_color, lw=lw, alpha=0.2)
                ax2[0].set_xlim(time[0], time[-1])
                ax2[0].set_ylim(np.nanmin(Y_smooth_total[:, :, neu_i]), np.nanmax(Y_smooth_total[:, :, neu_i]))
                ax2[0].axvline(0, c="k", alpha=0.5)
                #ax2[0].axvline(fconn.next_stim_after_n_vol[ie] * fconn.Dt, c="k", alpha=0.5)

            elif neu_i == most_variable_neuron:
                ax2[1].set_title(panel_title, fontsize=10)
                ax2[1].plot(time, y_smooth_plt, label=lbl, c=stim_color, lw=lw)
                #ax2[0].plot(time, y_plt, c=stim_color, lw=lw, alpha=0.2)
                ax2[1].set_xlim(time[0], time[-1])
                ax2[1].set_ylim(np.nanmin(Y_smooth_total[:, :, neu_i]), np.nanmax(Y_smooth_total[:, :, neu_i]))
                ax2[1].axvline(0, c="k", alpha=0.5)

            ax[ax_r, ax_c].plot(time, y_smooth_plt, label=lbl, c=stim_color, lw=lw)

            #rf_plt = rf
            #rf_plt /= np.max(np.abs(rf_plt)) / np.max(np.abs(fit_y))
            stim_y_plt = stim_y / np.sum(stim_y) * np.abs(np.sum(y))

            # ax[ax_r, ax_c].plot(x, nonlin_fit_y[:, i], label="FIT NEGF" + "|" + "g", lw=2, ls=fit_ls, c="r")
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

        fit_y =  pp.convolution(lin_kernel,Y_smooth_total[ie_idx, :,stim], fconn.Dt,8)
        fit_ls = ":"
        fit_lbl = "Av. kernel fit" #"|".join([str(nbp - 1) for nbp in n_branch_params])
        if neu_i == stim:
            panel_title = "Stimulated "+ panel_title
            ax2[0].plot(time, stim_y, label="Filtered Stim.", c='gray', lw=1)
        elif neu_i == most_variable_neuron:
            ax2[1].twinx().plot(np.linspace(0, 60, len(fit_y)), fit_y, label=fit_lbl, c='gray', lw=1, ls=fit_ls)
            ax2[1].legend()
        ax[ax_r, ax_c].twinx().plot(np.linspace(0, 60, len(fit_y)), fit_y, label=fit_lbl, c='gray', lw=1, ls=fit_ls)

        ax[ax_r, ax_c].set_xlim(time[0], time[-1])
        ax[ax_r, ax_c].set_ylim(np.nanmin(Y_smooth_total[:, :, neu_i]), np.nanmax(Y_smooth_total[:, :, neu_i]))
        ax[ax_r, ax_c].axvline(0, c="k", alpha=0.8)

        ax[ax_r, ax_c].set_title(panel_title, fontsize=10)

        if i_plot == len(responding) - 1:  # Add legend only for the last panel
            ax[ax_r, ax_c].legend()


    # Save plot with neuron index in filename
    filename = f"panels_mult_stimulation_fits.png"
    filename2 = f"most_variable_neuron_response.png"
    fig.savefig(os.path.join(main_dir, filename), bbox_inches="tight")
    plt.close(fig)        # Plot heatmaps for each neuron pair
    fig2.savefig(os.path.join(main_dir, filename2), bbox_inches="tight")
    plt.close(fig2)        # Plot heatmaps for each neuron pair





    """
    # save plot the NEGF for each stimulation and each neuron pair (consider source only the stim neuron)
    for i in range(n_responding):
        neu_i = responding[i]
        neu_j = stim
        if np.all(g[1:, 1:, i, 0] == 0.0) : continue
        #nlfc.utils.plots.t_t_heatmap(x, g[:, :, i, j], os.path.join(ie_dir, f'negf_g_heatmap_neuron_pair_{i}_{j}_stimulation_{str(ie)}.png'))
        nlfc.utils.plots.time_level_curves(x, g[:, :, i, j], rf, os.path.join(ie_dir,f'negf_direct_g_neurons_neuron_pair_{i}_{0}.png'))

    for i in range(n_responding):
        neu_i = responding[i]
        neu_j = stim
        if np.all(G[1:, 1:, i, 0] == 0.0) : continue
        #nlfc.utils.plots.t_t_heatmap(x, G[:, :, i, j], os.path.join(ie_dir, f'negf_G{G_degree}_heatmap_neuron_pair_{i}_{j}_stimulation_{str(ie)}.png'))
        nlfc.utils.plots.time_level_curves(x, G[:, :, i, 0], rf, os.path.join(ie_dir, f'negf_effective_G_neurons_neuron_pair_{i}_{0}.png' ))
    """

"""
    for i in range(n_responding):
        neu_i = responding[i]
            
        # Plot the data
        ax.plot(time, Y[:, neu_i], label=f"Neu.{neu_i}:{labels[neu_i]} raw", lw=1)
        ax.plot(time, Y_smooth[:, neu_i], label=f"Neu.{neu_i}:{labels[neu_i]} smooth", lw=1)
        
        params = fconn.get_irrarray_from_params(params_dict)
        rf = pp.Fconn.eci(x,params)
        fit_y = pp.convolution(stim_y,rf,fconn.Dt,8)
        fit_ls = "-"
        fit_lbl = "FIT"+ "|".join([str(nbp-1) for nbp in n_branch_params])
        
        rf_plt = rf
        rf_plt /= np.max(np.abs(rf_plt))/np.max(np.abs(fit_y))
        stim_y_plt = stim_y/np.sum(stim_y)*np.abs(np.sum(y))
        
        ax.plot(x,fit_y,label=fit_lbl,lw=2,ls=fit_ls)
        ax.plot(x,nonlin_fit_y[:, i],label="FIT NEGF"+"|"+"g",lw=2,ls=fit_ls, c="r")
        ax.plot(x,rf_plt,label="kernel (rf)",lw=2,c="k")
        ax.plot(x,stim_y_plt,label="stim (st)",lw=2,c="yellow",alpha=0.5)
        
        ax.set_xlim(time[0],time[-1])
        ax.set_ylim(min(y_plt),max(y_plt))
        ax.axvline(0,c="k",alpha=0.5)
        ax.axvline(fconn.next_stim_after_n_vol[ie]*fconn.Dt,c="k",alpha=0.5)
        ax.legend()
"""
      
        