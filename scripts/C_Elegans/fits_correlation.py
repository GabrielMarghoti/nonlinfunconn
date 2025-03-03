#
# code for fit of convolution kernels, combination of linear (exponentials) or non-linear (NEGF) kernels
#
# inspired by 
# fit_responses_constrained_stim_eci
# funatlas_vs_correlations2
#

import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pumpprobe as pp
import wormdatamodel as wormdm
import wormbrain as wormb
import mistofrutta as mf

# Parse command-line arguments
save_results = "--no-save" not in sys.argv
use_green_signal = "--signal:green" in sys.argv
skip_unconfirmed_targets = "--skip-if-not-manually-located" in sys.argv
matchless_nan_th = None
matchless_nan_th = None
matchless_nan_th_from_file = "--matchless-nan-th-from-file" in sys.argv
matchless_nan_th_added_only = "--matchless-nan-th-added-only" in sys.argv


for arg in sys.argv:
    if arg.startswith("--matchless-nan-th:"):
        matchless_nan_th = float(arg.split(":")[1])
    elif arg.startswith("--folder:"):
        folder = float(arg.split(":")[1])


# Validate arguments
if (matchless_nan_th is not None) and not use_green_signal:
    raise ValueError("--matchless-nan-th can only be used with --signal:green")

ds_list = "ds_list_full.txt"
ds_list_spont = "ds_list_ctrl_wt.txt"
                  
                  
signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": "sg_causal", 
                 "smooth_n": 13, "smooth_poly": 1,
                 "photobl_appl":True,            
                 "matchless_nan_th_from_file": matchless_nan_th_from_file,
                 "matchless_nan_th": matchless_nan_th,
                 "matchless_nan_th_added_only": matchless_nan_th_added_only}

# Ensure output directory exists
fits_dir = folder + "responses/fits/"
os.makedirs(fits_dir, exist_ok=True)

# Initialize logging pipeline
tubatura = pp.Pipeline("fit_responses_constrained_stim_eci.py", folder=folder)
tubatura.open_logbook_f()
tubatura.log("## Fitting responses with stimulus-constrained Exponential Convolutions")
tubatura.log('Command used: python ' + " ".join(sys.argv))

# Load signal data
if not use_green_signal:
    signal = wormdm.signal.Signal.from_signal_and_reference(folder)
else:
    tubatura.log("Using green signal.")
    signal = wormdm.signal.Signal.from_file(
        folder, "green", matchless_nan_th=matchless_nan_th
    )
    signal.appl_photobl()

# Preprocess signal
signal.remove_spikes()

# Load neuron coordinate data
brains = wormb.Brains.from_file(folder, ref_only=True)
labels = brains.get_labels(0)

# Load functional connectivity data
fconn = pp.Fconn.from_file(folder)

# Validate manually located targets
if not fconn.manually_located_present:
    tubatura.log("Targets have not been manually located/confirmed.")
    if skip_unconfirmed_targets:
        print("Skipping this dataset.")
        sys.exit()
    if input("Continue? (y/n)") != "y":
        sys.exit()

tubatura.log("Fitting with n_branches_max = 2")

# Load Funatlas for actual data
funa = pp.Funatlas.from_datasets(ds_list,merge_bilateral=merge,signal="green",
                                 signal_kwargs = signal_kwargs,
                                 enforce_stim_crosscheck=False,
                                 ds_tags=ds_tags,ds_exclude_tags=ds_exclude_tags,
                                 verbose=False)

# Get the direct anatomical connectome
aconn = funa.aconn_chem + funa.aconn_gap

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
        

        
        params_nonlin, branch_params_nonlin, _ = NEGF_LIF.fit(
            x, y, dt=fconn.Dt, n_branches_max=2
        )


        if params is None:
            tubatura.log(f"Constrained params is None for stim {stim_idx}, neuron {resp_neuron}")
            continue
        
        fconn.fit_params[stim_idx][resp_neuron] = {
            "params": np.array(params),
            "n_branches": len(branch_params),
            "n_branch_params": branch_params
        }

if save_results:
    fconn.to_file(folder)

# Compute correlation matrices
funatlas = pp.Funatlas.from_datasets("ds_list.txt", merge_bilateral=True, signal="green")
stimcorr = funatlas.get_signal_correlations()
spontcorr = funatlas.get_signal_correlations(spontaneous=True)

# Compute correlations
r_spont_stim = np.corrcoef(spontcorr[~np.isnan(stimcorr)], stimcorr[~np.isnan(stimcorr)])[0, 1]
print("Correlation between spontaneous and stimulus-driven activity:", r_spont_stim)
