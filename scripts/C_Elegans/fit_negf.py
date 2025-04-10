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
ds_exclude_tags =  None
kwar_fit_negf   = "--fit-negf" in sys.argv

aconn_ds_i = None # default is loading from funatlas, if aconn_ds_i is set, it will load from the specified dataset
# default 
output_folder = "figures/"

ds_list_path =  "/home/gabrielm/paper_reproduction/ds_list_unc31.txt" if "--unc31" in sys.argv else "/home/gabrielm/paper_reproduction/ds_list_full.txt"
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

aconn_chem, aconn_elec = funa.get_aconnectome_from_file() # get the anatomical connectome with the correct atlas index for neuros
num_neurons = aconn_chem.shape[0]


############################################################################################################################################
########### #NEGF kernels
############################################################################################################################################

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

Ci = Ci*1e+12  ## kunert Farad for capacitance F what I noticed got numerical problems due to finite-precision of floating-point, so I convert to 1 pF = 1e-12 F

Ci = Ci*500 # trick to change the dynamics for the time scale of calcium concentration/fluorescence instead of membrane potentials # to review latter

Gcell = params['Gcell']*1e+12 # Leakage conductance of membrane [pS]
Ecell = params['Ecell']*1000 # Leakage potential [mV]

# Electrical synapses
ggap = params['ggap']*1e+12 # conductivity of electrical synapse [pS]

# Chemical synapses
gsyn = params['gsyn']*1e+12 # "conductivity" of chemical synapse [pS]
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

# Print the parameters from kunert model
"""
print("Parameters:")
print(f"Ci: {Ci}")
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

# Iterate over the folders whcih contains each experiment data
for (i_folder, folder) in enumerate(ds_list):

    #if '20211104_163944' not in folder: continue # use only folder of waterfall fig1
    if '20220511_150909' in folder: continue # problem with this data

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
        if not (2 <= num_stimulations <= 3): 
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

        # Ensure output directory exists
        if kwar_fit_negf:
            main_dir = output_folder + "_".join(ds_tags[i_folder]) + f"/stim_neu_{stim_neuron_label}_{num_stimulations}x/negf_fit/"
        else:
            main_dir = output_folder + "_".join(ds_tags[i_folder]) + f"/stim_neu_{stim_neuron_label}_{num_stimulations}x/equilibirum_gf_fit/"

        ie_dir_list = []
        
        for ie in stimulations_idx:  # stimulation index only though cases which the most stimulated neuron is stimulated
            responding_ie = fconn.resp_neurons_by_stim[ie]
            i0 = max(0, fconn.i0s[ie])  # start of the stimulation
            i1 = fconn.i1s[ie]         # end of the stimulation

            responding.update(responding_ie)  # Add the responding neurons to the set

            n_responding_ie = len(responding_ie)
            
            # Ensure output directory exists
            ie_dir = main_dir + f"n_resp_neurons_{n_responding_ie}_ie_trial{ie}/"

            ie_dir_list.append(ie_dir)
                
            Y = sig.get_segment(i0, i1, shift_vol, unsmoothed_data=True, baseline_mode="constant")[0:time_plt_len, :]
            Y_smooth = sig.get_segment(i0, i1, shift_vol, unsmoothed_data=False, baseline_mode="constant")[0:time_plt_len, :]

            Y_total.append(Y[np.newaxis, ...])  # Add a new axis to ensure 3D structure
            Y_smooth_total.append(Y_smooth[np.newaxis, ...])  # Add a new axis to ensure 3D structure
        
        if stim not in responding:
            responding.update([stim]) 

        responding = list(responding)
        
        responding.remove(stim)
        responding.insert(0, stim)

        # Consider only second order neighbors of stimulated node

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

        if n_responding > 6 or n_responding <2:
            print(f"Skipping dataset {folder} with {n_responding} responding neurons.")
            continue

        os.makedirs(main_dir, exist_ok=True)

        # plot complete neural network 
        #nlfc.utils.netplots.neural_network((Ggap*ggap), (Gsyn*gsyn), Esyn, np.array(labels), positions=None, save_path=os.path.join(main_dir, f'Neural_Network_total.png'))

        Y_total = np.concatenate(Y_total, axis=0)  # Concatenate along the new axis to maintain 3D structure
        Y_smooth_total = np.concatenate(Y_smooth_total, axis=0)  # Concatenate along the new axis to maintain 3D structure

        responses_correlations = np.zeros((n_responding, num_stimulations, num_stimulations))
        trial_variations = np.zeros(n_responding)
        common_trials = np.zeros(n_responding, dtype=int)

        for i in range(n_responding):
            neu_i = responding[i]
            responses_correlations[i, :, :] = np.corrcoef(
                        Y_smooth_total[:, shift_vol:, neu_i]
                    )
            # Calculate the standard deviation of the correlation matrix for each neuron
            trial_variations[i] = np.std(responses_correlations[i, :, :])

        # Find the neuron with the most variation in trial correlations
        most_variable_neuron_idx = np.argmax(trial_variations)
        most_variable_neuron = responding[most_variable_neuron_idx] 

        print(f"Neuron with most variation: {most_variable_neuron} ({labels[most_variable_neuron]})")
        

        # save connectome based network considering only responsive neurons over all stimulations
        gamma_g = (Ggap*ggap)[responding][:, responding] 
        gamma_s = (Gsyn*gsyn)[responding][:, responding]
        Es = Esyn[responding][:, responding]
        
        # Plot gamma_g and gamma_s as heatmaps
        nlfc.utils.netplots.connect_matrices_heatmap(gamma_g, gamma_s, np.array(labels)[responding], os.path.join(main_dir, 'gamma_g_gamma_s_heatmaps.png'))
        # plot neural network
        nlfc.utils.netplots.neural_network(gamma_g, gamma_s, Es, np.array(labels)[responding], positions=None, save_path=os.path.join(main_dir, f'Neural_Network_responding_only.png'))
        

        Y_nonlin_fit = np.zeros_like(Y_smooth_total[:, shift_vol:, responding])

        nonlin_kernel = nlfc.models.LIF(
            num_neurons=n_responding,
            C=Ci,
            gamma_g=(Ggap)[responding][:, responding], 
            gamma_s=(Gsyn)[responding][:, responding], 
            gap_conductance=ggap,
            syn_conductance=gsyn,
            gamma=Gcell, 
            beta=beta,  
            E_c=Ecell, 
            E_s= Esyn[responding][:, responding], 
            a_r=ar, 
            a_d=ad, 
        )
        G_degree = 2

        # FIT NEGF
        print("NEGF fitting")

        lowering_resolution_step = 10
        p = nonlin_kernel.fit(Y_smooth_total[:, shift_vol::lowering_resolution_step, responding], dt=lowering_resolution_step*fconn.Dt, fit_linear_model= not kwar_fit_negf , max_iters=50, include_adj_matrix=True)
        
        print('FIT DONE')

        # plot neural network after fitting
        nlfc.utils.netplots.neural_network(nonlin_kernel.gamma_g*nonlin_kernel.gap_cond, nonlin_kernel.gamma_s*nonlin_kernel.syn_cond, nonlin_kernel.E_s, np.array(labels)[responding], positions=None, save_path=os.path.join(main_dir, f'Neural_Network_responding_only_after_fit.png'))
    
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
            os.makedirs(ie_dir, exist_ok=True)
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

                ax.plot(time_plt, Y_total[ie_idx, :, neu_j], color=color, linewidth=lw, alpha=0.6)
                ax_smooth.plot(time_plt, Y_smooth_total[ie_idx, :, neu_j], color=color, linewidth=lw, alpha=0.6)  # Corrected to use smoothed data
            for i in range(n_responding):
                neu_i = responding[i]
                neu_j = stim
                if np.all(Y_nonlin_fit[ie_idx, :, i] == 0.0) : continue
                #ax.plot(time_fit, Y_nonlin_fit[ie_idx, :, i], color=color, linewidth=lw, alpha=0.8, ls='--')  # using conectome estimated propagated signals
                ax_smooth.plot(time_fit, Y_nonlin_fit[ie_idx, :, i], color="blue", linewidth=2, alpha=0.9, ls='--')
        
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


        for ie_idx, ie in enumerate(stimulations_idx):
            # Ensure output directory exists
            ie_dir = main_dir + f"n_resp_neurons_{n_responding_ie}_ie_trial{ie}/"
            os.makedirs(ie_dir, exist_ok=True)

            g, Y_nonlin_fit[ie_idx, :, :] = nonlin_kernel.compute_direct_negf(Vs=Y_smooth_total[ie_idx][shift_vol:, responding], dt=fconn.Dt, return_estimated_V=True)
            G = nonlin_kernel.compute_effective_negf(g, G_degree) # until second neighbors
            G0 = nonlin_kernel.compute_effective_negf(nonlin_kernel.g0, G_degree) # until second neighbors

            # save plot the NEGF for each stimulation and each neuron pair (consider source only the stim neuron)
            for i in range(n_responding):
                for j in range(n_responding):
                    neu_i = responding[i]
                    neu_j = responding[j]
                    if i == 0 or (gamma_g[i, j] == 0 and gamma_s[i, j] == 0): continue
                    #nlfc.utils.plots.t_t_heatmap(x, g[:, :, i, j], os.path.join(ie_dir, f'negf_g_heatmap_neuron_pair_{i}_{j}_stimulation_{str(ie)}.png'))
                    nlfc.utils.plots.time_level_curves(time_fit, g[:, :, i, j], nonlin_kernel.g0[-1, :, i, j], xlabel=None, ylabel="g(t,t')", title=f"Neurons : {labels[neu_i]}<-{labels[neu_j]}", save_path= os.path.join(ie_dir_list[ie_idx],f'fitted_negf_direct_g_neurons_neuron_pair_{labels[neu_i]}<-{labels[neu_j]}.png'))
                    #nlfc.utils.plots.t_t_heatmap(time_fit, G[:, :, i, 0], os.path.join(ie_dir, f'negf_G{G_degree}_heatmap_neuron_pair_{labels[responding[i]]}_{labels[stim]}_stimulation_{str(ie)}.png'))
                    nlfc.utils.plots.time_level_curves(time_fit, G[:, :, i, j], G0[-1, :, i, j], xlabel=None, ylabel="G(t,t')", title=f"Neurons : {labels[neu_i]}<-{labels[neu_j]}", save_path= os.path.join(ie_dir_list[ie_idx],f'fitted_negf_G_neurons_neuron_pair_{labels[neu_i]}<-{labels[neu_j]}.png'))
    

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

                y_plt = Y_total[ie_idx, :,neu_i]
                y = Y_total[ie_idx, :,neu_i][shift_vol:]
                y_smooth_plt = Y_smooth_total[ie_idx, :,neu_i]
                y_smooth = Y_smooth_total[ie_idx, :,neu_i][shift_vol:]

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
                lbl = f'stim. {ie}'
                fit_lbl = "Trial kernel fit" #"|".join([str(nbp - 1) for nbp in n_branch_params])
                lw = 1
                if neu_i == stim: 
                    lbl += "*"
                    lw = 2
                    ax2[0].set_title(panel_title, fontsize=10)
                    ax2[0].plot(time_plt, y_smooth_plt, label=lbl, c=stim_color, lw=lw)
                    #ax2[0].plot(time, y_plt, c=stim_color, lw=lw, alpha=0.2)
                    ax2[0].set_xlim(time_plt[0], time_plt[-1])
                    ax2[0].set_ylim(np.nanmin(Y_smooth_total[:, :, neu_i]), np.nanmax(Y_smooth_total[:, :, neu_i]))
                    ax2[0].axvline(0, c="k", alpha=0.5)
                    #ax2[0].axvline(fconn.next_stim_after_n_vol[ie] * fconn.Dt, c="k", alpha=0.5)

                elif neu_i == most_variable_neuron:
                    ax2[1].set_title(panel_title, fontsize=10)
                    ax2[1].plot(time_plt, y_smooth_plt, label=lbl, c=stim_color, lw=lw)
                    #ax2[1].plot(time_fit, fit_y_trial, label=fit_lbl, c=stim_color, lw=1, ls=':')
                    #ax2[0].plot(time, y_plt, c=stim_color, lw=lw, alpha=0.2)
                    ax2[1].set_xlim(time_plt[0], time_plt[-1])
                    ax2[1].set_ylim(np.nanmin(Y_smooth_total[:, :, neu_i]), np.nanmax(Y_smooth_total[:, :, neu_i]))
                    ax2[1].plot(time_fit, Y_nonlin_fit[ie_idx, :, i], label="FIT NEGF" + "|" + "g", lw=2, ls='--', c=stim_color)
                    ax2[1].axvline(0, c="k", alpha=0.5)

                ax[ax_r, ax_c].plot(time_plt, y_smooth_plt, label=lbl, c=stim_color, lw=lw)
                #ax[ax_r, ax_c].plot(time_fit, fit_y_trial, label=fit_lbl, c=stim_color, lw=1, ls=':')

                #rf_plt = rf
                #rf_plt /= np.max(np.abs(rf_plt)) / np.max(np.abs(fit_y))
                stim_y_plt = stim_y / np.sum(stim_y) * np.abs(np.sum(y))

                ax[ax_r, ax_c].plot(time_fit, Y_nonlin_fit[ie_idx, :, i], label="FIT NEGF" + "|" + "g", lw=2, ls=':', c=stim_color)
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
            fit_lbl = "Av. kernel fit" #"|".join([str(nbp - 1) for nbp in n_branch_params])
            if neu_i == stim:
                panel_title = "Stimulated "+ panel_title
                #ax2[0].plot(time_fit, stim_y, label="Filtered Stim.", c='yellow', lw=1)
                ax2[0].legend()
            elif neu_i == most_variable_neuron:
                ax2[1].plot(time_fit, fit_y, label=fit_lbl, c='black', lw=1, ls=fit_ls)
                ax2[1].legend()
            ax[ax_r, ax_c].plot(time_fit, fit_y, label=fit_lbl, c='black', lw=1, ls=fit_ls)

            ax[ax_r, ax_c].set_xlim(time_plt[0], time_plt[-1])
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

        # Save parameters before fitting as a tab-delimited text file
        params_before_fitting = {
            "Ci": Ci,
            "Gcell": Gcell,
            "Ecell": Ecell,
            "ggap": ggap,
            "gsyn": gsyn,
            "ar": ar,
            "ad": ad,
            "beta": beta,
            "esynexc": esynexc,
            "esyninh": esyninh,
        }
        with open(os.path.join(main_dir, "params_before_fitting.txt"), "w") as f:
            f.write("Parameter\tValue\n")
            for key, value in params_before_fitting.items():
                f.write(f"{key}\t{value}\n")

        # Save nonlin_kernel attributes after fitting as a tab-delimited text file
        params_after_fitting = {
            "C": nonlin_kernel.C,
            "gamma": nonlin_kernel.gamma,
            "beta": nonlin_kernel.beta,
            "E_c": nonlin_kernel.E_c,
            "a_r": nonlin_kernel.a_r,
            "a_d": nonlin_kernel.a_d,
        }
        with open(os.path.join(main_dir, "params_after_fitting.txt"), "w") as f:
            f.write("Parameter\tValue\n")
            for key, value in params_after_fitting.items():
                f.write(f"{key}\t{value}\n")

        # Save gamma_g, gamma_s, and Esyn as separate tab-delimited text files
        np.savetxt(os.path.join(main_dir, "gamma_g.txt"), gamma_g, delimiter="\t", fmt="%.6f")
        np.savetxt(os.path.join(main_dir, "gamma_s.txt"), gamma_s, delimiter="\t", fmt="%.6f")
        np.savetxt(os.path.join(main_dir, "Esyn.txt"), Esyn, delimiter="\t", fmt="%.6f")

        np.savetxt(os.path.join(main_dir, "gamma_g_after_fitting.txt"), nonlin_kernel.gamma_g*nonlin_kernel.gap_cond, delimiter="\t", fmt="%.6f")
        np.savetxt(os.path.join(main_dir, "gamma_s_after_fitting.txt"), nonlin_kernel.gamma_s*nonlin_kernel.syn_cond, delimiter="\t", fmt="%.6f")
        np.savetxt(os.path.join(main_dir, "E_s_after_fitting.txt"), nonlin_kernel.E_s, delimiter="\t", fmt="%.6f")


