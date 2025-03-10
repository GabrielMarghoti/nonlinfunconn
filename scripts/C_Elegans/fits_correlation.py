#
# code for fit of convolution kernels, combination of linear (exponentials) or non-linear (NEGF) kernels
#
# inspired by 
# https://github.com/leiferlab/pumpprobe/tree/main/scripts/fconnectivity/fit_responses_constrained_stim_eci
# https://github.com/leiferlab/pumpprobe/tree/main/scripts/fconnectivity/figures/compare_connectomes/funatlas_vs_correlations2
# Kunert et al., PRE 89 052805 (2014) for parameter estimation

import numpy as np
import matplotlib.pyplot as plt
import os, sys, time, json
import pumpprobe as pp
import wormdatamodel as wormdm
import wormbrain as wormb

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

# If non-interacting, set all elements to zero
#if params['interacting'] == 0:
#    Gsyn[:,:] = 0
#    Ggap[:,:] = 0

# Cell
Ci = params['C'] # Membrane capacitance [F]

Ci = Ci*200  ############### review, this is necessary for better time scale, otherwise exponentials decrease to fast


Gcell = params['Gcell'] # Leakage conductance of membrane [S]
Ecell = params['Ecell'] # Leakage potential [V]

# Electrical synapses
ggap = params['ggap'] # conductivity of electrical synapse [Siemens]

# Chemical synapses
gsyn = params['gsyn'] # "conductivity" of chemical synapse [Siemens]
ar = params['ar'] # activation rate of synapses [s^-1]
ad = params['ad'] # deactivation rate of synapses [s^-1]
beta = params['beta'] # width of synaptic activation [V^-1]
esynexc = params['esynexc'] # reverse potential for excitatory synapses
esyninh = params['esyninh'] # reverse potential for inhibitory synapses

# Build the Esyn array of the synaptic reverse potentials
# The index is presynaptic neuron, which determines the neurotransmitter and
# hence the sign of the synapse.
Esyn = np.ones((funa.n_neurons,funa.n_neurons))*esynexc
Esyn[sign<0] = esyninh

# Iterate over the folders whcih contains each experiment data
for (i_folder, folder) in enumerate(ds_list):

    if i_folder>2: break   # remove to process all datasets
    print("Processing dataset", i_folder, ":", folder)
    # Ensure output directory exists
    fits_dir = output_folder + "_".join(ds_tags[i_folder]) + "_fits/"
    os.makedirs(fits_dir, exist_ok=True)


    tubatura = pp.Pipeline("fits_correlation.py",folder=fits_dir)
    tubatura.open_logbook_f()
    tubatura.log("",False)
    tubatura.log("",False)
    tubatura.log("## Fitting the responses with stim_constrained ExponentialConvolutions.")
    tubatura.log('Command used: python '+" ".join(sys.argv),False)

    # Load the signal
    if not sig_green:
        sig = wormdm.signal.Signal.from_signal_and_reference(fits_dir)
    else:
        tubatura.log("Using green signal.")
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
    sig.smooth(n=127,i=None,poly=7,mode="sg")

    # Get the neurons coordinates of the reference volume and load the matches
    # to determine what neuron was targeted
    cervelli = wormb.Brains.from_file(folder,ref_only=True)
    labels = cervelli.get_labels(0)

    # Create functional connectome
    fconn = pp.Fconn.from_file(folder)
    #shift_vol = fconn.shift_vol

    # Check that the targets have been manually located, and, if not, ask for 
    # confirmation.
    if not fconn.manually_located_present:
        tubatura.log("Targets have not been manually located/confirmed.")
        if not skip_if_not_manually_located:
            cont = input("Continue? (y/n)") == "y"
            if not cont: quit() 
        else:
            print("\tSkipping this dataset.")
            quit()
            
    tubatura.log("Fitting with n_branches_max = 2")

    for ie in np.arange(fconn.n_stim): # stimulation index
        i0 = max(0,fconn.i0s[ie])  # start of the stimulation
        i1 = fconn.i1s[ie]         # end of the stimulation
        shift_vol = fconn.shift_vols[ie]  # Negative-time interval (in steps) to consider before each stimulus.
        time = (np.arange(i1-i0)-shift_vol)*fconn.Dt  # Dt is in seconds, convert time steps to  times domain (s)
        # notice the slice for the signal is based on the stimulated neuron signal only
        
        # Get the indices of the stimulated neuron and of the responding neurons.
        stim = fconn.stim_neurons[ie]

        responding_original =  fconn.resp_neurons_by_stim[ie]
        n_responding_original = len(responding_original)
        responding = responding_original
        n_responding = n_responding_original
        # but fit everything - nope
        #responding = np.arange(fconn.n_neurons)  # fit all neurons
        #n_responding = len(responding)
        
        Y = np.zeros((time.shape[0], num_neurons)) # store the signal of all neurons in the network
        
        # Get the unconstrained parameters to build a cleaned-up version of the
        # stimulated neuron's activity.
        stim_unc_par_dict = fconn.fit_params_unc[ie][stim]
        
        #########################################
        # CASES IN WHICH TO SKIP THIS STIMULATION
        #########################################
        # If the targeting has failed
        if stim == -2:
            tubatura.log("skipping "+str(ie)+" failed target");continue
        # If no response was detected in the stimulated neuron, because you'd risk
        # finding weird kernels just because there is no activity as input.
        if stim not in responding_original:
            tubatura.log("skipping "+str(ie)+" stimulated neuron did not respond");continue
        # If the fit of the stimulated neuron's activity did not converge.    
        if stim_unc_par_dict["n_branches"]==0:
            tubatura.log("skipping "+str(ie)+" fit absent");continue
        
        stim_unc_par = fconn.get_irrarray_from_params(stim_unc_par_dict)
        
        for j in np.arange(n_responding):    
            
            neu_j = responding[j]
            #if neu_j==stim: continue  # get segment of stimulated neuron
            
            # Skip the following checks on i1 and simply fit on the time axis
            # before the next stimulus. This also avoids fits of the next response.
            if ie<1:
                i1p = None
            elif ie<fconn.n_stim-2:
                inplus2 = neu_j in fconn.resp_neurons_by_stim[ie+2]
                inplus1 = neu_j in fconn.resp_neurons_by_stim[ie+1]
                i1p = None
                if inplus2: i1p = shift_vol+np.sum(fconn.next_stim_after_n_vol[ie:ie+2])
                if inplus1: i1p = shift_vol+fconn.next_stim_after_n_vol[ie]
            elif ie==fconn.n_stim-2:
                inplus1 = neu_j in fconn.resp_neurons_by_stim[ie+1]
                i1p = None
                if inplus1: i1p = shift_vol+fconn.next_stim_after_n_vol[ie]
            else:
                i1p = None
            #i1p = shift_vol+fconn.next_stim_after_n_vol[ie]
            if ie == fconn.n_stim-1: i1p = None
                    
            x = time[shift_vol:i1p]
            # Determine the range for baseline subtraction. Keep the full shift_vol
            # interval if the neuron was not responding before. But shorten it
            # if the neuron was responding to the previous stimulation. This latter
            # case is more sensitive to the noise, but avoids systematic wrong
            # baselines due to ongoing dynamics in the shift_vol segment.
            if ie>0:
                if neu_j in fconn.resp_neurons_by_stim[ie-1]:
                    y = sig.get_segment(i0,i1,shift_vol,unsmoothed_data=True,
                                        baseline_mode="constant",
                                        baseline_range=[shift_vol-4,shift_vol])[:,neu_j]#, FIXME FIXME FIXME
                                        #normalize="none")[:,neu_j]
                else:
                    y = sig.get_segment(i0,i1,shift_vol,unsmoothed_data=True,
                                        baseline_mode="constant")[:,neu_j]#, FIXME FIXME FIXME
                                        #normalize="none")[:,neu_j]
            else:
                y = sig.get_segment(i0,i1,shift_vol,unsmoothed_data=True,
                                        baseline_mode="constant")[:,neu_j]#, FIXME FIXME FIXME
                                        #normalize="none")[:,neu_j]
            
            #y = sig.get_segment(i0,i1,shift_vol)[:,neu_j]
            if np.all(np.isinf(y)) or np.all(np.isnan(y)): continue
            y[np.isinf(y)] = y[np.where(np.isinf(y))[0]-1]
            y_plt = y

            Y[:,neu_j] = y.copy()

############################################################################################################        
#        Compute NEGF kernels
############################################################################################################
        # Compute the direct NEGF kernel
        # Initialize the NEGF kernel class
        nonlin_kernel = nlfc.negf.LIF(
            Vs = Y[:, responding],  # Membrane potential dynamics (sliced signals)
            num_neurons = n_responding,
            gamma_g = (Ggap*ggap/Ci)[responding][:, responding], 
            gamma_s = (Gsyn*gsyn/Ci)[responding][:, responding], 
            gamma = Gcell/Ci, 
            C = Ci,  
            beta= beta,  
            E_c = Ecell, 
            E_s = Esyn[responding][:, responding], 
            a_r = ar, 
            a_d = ad, 
            dt = fconn.Dt,
            t_s=time
        )
        
        g = nonlin_kernel.compute_direct_negf(Vs=Y[:, responding], dt=fconn.Dt) 
        G = nonlin_kernel.compute_effective_negf(g, 2) # until second neighbors

        # Plot heatmaps for each neuron pair
        for i in range(n_responding):
            for j in range(n_responding):
                if np.all(G[:, :, i, j] == 0.0) : continue
                nlfc.utils.plots.t_t_heatmap(time, G[:, :, i, j], os.path.join(fits_dir, f'negf_G_heatmap_neuron_pair_{i}_{j}_stimulation_{str(ie)}.png'))
                nlfc.utils.plots.time_level_curves(time, G[:, :, i, j], os.path.join(fits_dir, f'negf_G_level_curves_neuron_pair_{i}_{j}_stimulation_{str(ie)}.png'))

        gc.collect()





        ###############
        # PREPARE PLOTS
        ###############
            
        nrows = max(1,int(np.sqrt(n_responding_original)))
        ncols = int(np.sqrt(n_responding_original))+2
        if plot:
            print("plotting")
            try:
                fig.clear()
            except:
                pass
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols,figsize=(15,10))
            for a in np.ravel(ax): a.set_xticks([]);a.set_yticks([])
            if nrows==1: ax = np.array([ax])



        for j in np.arange(n_responding):
            neu_j = responding[j]
            if neu_j==stim: continue

            y_plt = Y[:,neu_j]
            y = Y[:,neu_j][shift_vol:i1p]
            #if y.shape[0] == 0: continue
            
            stim_y = pp.Fconn.eci(x,stim_unc_par)    # stim_y is an exponential kernel????????????
                    
            loc_std = sig.get_loc_std(y,4)
            
            fconn.clear_fit_results(stim=ie,neu=neu_j,mode="constrained")
            
            rms_calc_lim = min(int(30/fconn.Dt),len(x))
            
            n_hops_min = 2
            
            params_, n_branch_params, _ = fconn.fit_eci_branching(
                            x,y,stim_y,dt=fconn.Dt,
                            n_hops_min=1,n_hops_max=3,
                            n_branches_max=2,#3,
                            rms_limits=[None,None],auto_stop=True,rms_tol=1e-2,
                            method="trf",routine="least_squares")
            
            if params_ is None: 
                tubatura.log("constrained params is None. stim "+str(ie)+" neuron "+str(neu_j))
                params_dict = {"params": [0,1], 
                        "n_branches": 1, 
                        "n_branch_params": [2]}
                fconn.fit_params[ie][neu_j] = params_dict
                continue
            
            params_dict = {"params": np.array(params_), 
                        "n_branches": len(n_branch_params), 
                        "n_branch_params": n_branch_params}
            fconn.fit_params[ie][neu_j] = params_dict


            if neu_j in responding_original and plot:
                #print("plotting")
                # Plot only for detected responses
                j_plot = np.where(responding_original==neu_j)[0][0]
                ax_r = j_plot//ncols
                ax_c = j_plot%ncols
                
                #lbl = str(neu_j)
                lbl = str(neu_j)+":"+labels[neu_j]
                lw = 1
                if neu_j==stim: 
                    lbl+="*"
                    lw = 3
                    
                ax[ax_r,ax_c].plot(time,y_plt,label=lbl,lw=lw)
                
                params = fconn.get_irrarray_from_params(params_dict)
                rf = pp.Fconn.eci(x,params)
                fit_y = pp.convolution(stim_y,rf,fconn.Dt,8)
                fit_ls = "-"
                fit_lbl = "|".join([str(nbp-1) for nbp in n_branch_params])
                
                rf_plt = rf
                rf_plt /= np.max(np.abs(rf_plt))/np.max(np.abs(fit_y))
                stim_y_plt = stim_y/np.sum(stim_y)*np.abs(np.sum(y))
                
                ax[ax_r,ax_c].plot(x,fit_y,label=fit_lbl,lw=2,ls=fit_ls)
                ax[ax_r,ax_c].plot(x,rf_plt,label="rf",lw=2,c="k")
                ax[ax_r,ax_c].plot(x,stim_y_plt,label="st",lw=2,c="yellow",alpha=0.5)
                
                ax[ax_r,ax_c].set_xlim(time[0],time[-1])
                ax[ax_r,ax_c].set_ylim(min(y_plt),max(y_plt))
                ax[ax_r,ax_c].axvline(0,c="k",alpha=0.5)
                ax[ax_r,ax_c].axvline(fconn.next_stim_after_n_vol[ie]*fconn.Dt,c="k",alpha=0.5)
                ax[ax_r,ax_c].legend()
            
        if plot:
            if sig_green: sig_type="g_"
            else: sig_type=""
            plt.figure(1)
            plt.tight_layout()
            plt.savefig(fits_dir+sig_type+"eci_con_stim"+str(ie)+".png",bbox_inches="tight")
            plt.close(fig=fig)





"""
    # Get the direct anatomical connectome

    def get_signal_correlations(ds_list=None):
        r_act = np.ones((n_neurons,n_neurons))*np.nan
        count = np.ones((n_neurons,n_neurons))
        
        if ds_list is None: ds_list = np.arange(len(self.ds_list))
        
        for i_ds in ds_list:
            r_act_ = np.corrcoef(self.sig[i_ds].data.T)
            # Translate it in the atlas reference frame
            for i in np.arange(r_act_.shape[0]):
                ai = self.atlas_i[i_ds][i]
                if ai<0: continue
                for j in np.arange(r_act_.shape[1]):
                    aj = self.atlas_i[i_ds][j]
                    if aj<0: continue
                    if np.isnan(r_act[ai,aj]):
                        r_act[ai,aj] = r_act_[i,j]
                    else:
                        r_act[ai,aj] += r_act_[i,j]
                    count[ai,aj] += 1
        
        r_act[count!=0] = r_act[count!=0]/count[count!=0]
        
        return r_act


    # Fit ECI parameters for each stimulus
    for stim_idx in range(fconn.n_stim):
        stim_neuron = fconn.stim_neurons[stim_idx]
        responding_neurons = fconn.resp_neurons_by_stim[stim_idx]

        # Skip failed or non-responsive stimulations
        if stim_neuron == -2 or stim_neuron not in responding_neurons:
            tubatura.log(f"Skipping stimulus {stim_idx}, invalid or non-responsive.")
            continue

        # Extract and fit ECI parameters
        stim_unc_params = fconn.get_irrarray_from_params(fconn.fit_params_unc[stim_idx][stim_neuron])
        for resp_neuron in responding_neurons:
            if resp_neuron == stim_neuron:
                continue
            
            x = np.arange(fconn.i1s[stim_idx] - fconn.i0s[stim_idx]) * fconn.Dt
            y = signal.get_segment(fconn.i0s[stim_idx], fconn.i1s[stim_idx], unsmoothed_data=True)[:, resp_neuron]
            if np.all(np.isnan(y)):
                continue
            
            stim_response = pp.Fconn.eci(x, stim_unc_params)
            
            params, branch_params, _ = fconn.fit_eci_branching(
                x, y, stim_response, dt=fconn.Dt, n_branches_max=2
            )

            if params is None:
                tubatura.log(f"Constrained params is None for stim {stim_idx}, neuron {resp_neuron}")
                continue
            
            fconn.fit_params[stim_idx][resp_neuron] = {
                "params": np.array(params),
                "n_branches": len(branch_params),
                "n_branch_params": branch_params
            }

# Fit NEGF parameters fitting
#        params_nonlin, branch_params_nonlin, _ = NEGF_LIF.fit(
#            x, y, dt=fconn.Dt, n_branches_max=2
#        )



    if save_results:
        fconn.to_file(folder)

    # Compute correlation matrices
    funatlas = pp.Funatlas.from_datasets("ds_list.txt", merge_bilateral=True, signal="green")


    # Get the kernel-derived correlations
    conv_lin_kernel_stim = funa.get_kernels_map(occ2,occ3,filtered=True,include_flat_kernels=True)
    #ec_conv_stim[q>0.05]=np.nan
    ck = funa.get_correlation_from_kernels_map(km,occ3,set_unknown_to_zero=False)


    corr_lin_kernels = funatlas.get_correlation_from_kernels_map(conv_lin_kernel_stim,occ3,js=None, set_unknown_to_zero=False)
    corr_NEGF_kernels = get_correlation_from_NEGF()

    # Compute correlations
    r_spont_stim = np.corrcoef(spontcorr[~np.isnan(stimcorr)], stimcorr[~np.isnan(stimcorr)])[0, 1]
    print("Correlation between spontaneous and stimulus-driven activity:", r_spont_stim)

"""            
   