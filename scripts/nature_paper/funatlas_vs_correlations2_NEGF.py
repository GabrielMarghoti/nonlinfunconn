import numpy as np
import matplotlib.pyplot as plt
import os, sys, json
import pumpprobe as pp

# Set plot font sizes for better visualization
plt.rc("xtick", labelsize=18)
plt.rc("ytick", labelsize=18)
plt.rc("axes", labelsize=18)

# Check command-line arguments for configuration options
merge = "--no-merge" not in sys.argv  # Whether to merge bilateral data
print("Merge:", merge)
unc31 = "--unc31" in sys.argv  # Specific dataset filtering option
strain = "" if not unc31 else "unc31"
ds_tags = None if not unc31 else "unc31"
ds_exclude_tags = "mutant" if not unc31 else None

# Handle optional thresholds for missing data
matchless_nan_th = None
matchless_nan_th_from_file = "--matchless-nan-th-from-file" in sys.argv
matchless_nan_th_added_only = "--matchless-nan-th-added-only" in sys.argv

# Parse additional threshold values from command-line arguments
for s in sys.argv:
    sa = s.split(":")
    if sa[0] == "--matchless-nan-th":
        matchless_nan_th = float(sa[1])

# Define folder path for anatomical simulation data
acfolder = "/home/gabrielm/paper_reproduction/simulations/activity_connectome_sign2/"

# Load anatomical connectivity correlation matrix (merged or unmerged)
if not merge:
    actconncorr = np.loadtxt(acfolder+"activity_connectome_correlation_no_merge.txt")
else:
    actconncorr = np.loadtxt(acfolder+"activity_connectome_bilateral_merged_correlation.txt")

# Define dataset lists for different experimental conditions
ds_list = "/home/gabrielm/paper_reproduction/ds_list_full.txt"  # Full dataset
ds_list_spont = "/home/gabrielm/paper_reproduction/ds_list_ctrl_wt.txt"  # Control wild-type dataset

# Signal processing parameters
signal_kwargs = {
    "remove_spikes": True,  # Remove signal spikes
    "smooth": True,  # Apply smoothing
    "smooth_mode": "sg_causal",  # Smoothing method
    "smooth_n": 13,  # Window size for smoothing
    "smooth_poly": 1,  # Polynomial order for smoothing
    "photobl_appl": True,  # Apply photobleaching correction
    "matchless_nan_th_from_file": matchless_nan_th_from_file,
    "matchless_nan_th": matchless_nan_th,
    "matchless_nan_th_added_only": matchless_nan_th_added_only
}

# Load functional connectivity data using Funatlas
funa = pp.Funatlas.from_datasets(
    ds_list, merge_bilateral=merge, signal="green",
    signal_kwargs=signal_kwargs,
    enforce_stim_crosscheck=False,
    ds_tags=ds_tags, ds_exclude_tags=ds_exclude_tags,
    verbose=False
)

# Load spontaneous activity dataset
funa_spont = pp.Funatlas.from_datasets(
    ds_list_spont, merge_bilateral=merge, signal="green",
    signal_kwargs=signal_kwargs
)

# Extract occurrence matrices for stimulus response analysis
occ1, occ2 = funa.get_occurrence_matrix(req_auto_response=True)
occ3 = funa.get_observation_matrix(req_auto_response=True)
_, inclall_occ2 = funa.get_occurrence_matrix(req_auto_response=True, inclall=True)

# Compute statistical significance using Kolmogorov-Smirnov test
q, p = funa.get_kolmogorov_smirnov_q(inclall_occ2, return_p=True, strain=strain)

# Compute kernel-based functional connectivity
km = funa.get_kernels_map(occ2, occ3, filtered=True, include_flat_kernels=True)
km[q > 0.05] = np.nan  # Set non-significant values to NaN
ck = funa.get_correlation_from_kernels_map(km, occ3, set_unknown_to_zero=False)
ck[q > 0.05] = np.nan  # Set non-significant values to NaN

# Symmetrize kernel-based correlation matrix
nanmask = np.isnan(ck) * np.isnan(ck.T)
ck = 0.5 * (np.nansum([ck, ck.T], axis=0))
ck[nanmask] = np.nan

# Compute correlation matrices for different conditions
stimcorr = funa.get_signal_correlations()  # Stimulated activity
spontcorr = funa_spont.get_signal_correlations()  # Spontaneous activity
aconn = funa.aconn_chem + funa.aconn_gap  # Direct anatomical connectivity
Aconn = funa.get_effective_aconn4(gain_1=646.4)  # Effective anatomical connectivity
Aconnsym = funa.corr_from_eff_causal_conn(Aconn)  # Symmetric effective connectivity

# Resymmetrize anatomical correlation matrix
nanmask = np.isnan(actconncorr) * np.isnan(actconncorr.T)
actconncorr = 0.5 * (np.nansum([actconncorr, actconncorr.T], axis=0))
actconncorr[nanmask] = np.nan

# Create mask to exclude diagonal and NaN values
ondiag = np.zeros_like(q, dtype=bool)
np.fill_diagonal(ondiag, True)
excl = np.isnan(spontcorr) | np.isnan(ck) | np.isnan(actconncorr) | ondiag | np.isnan(stimcorr)
np.savetxt("excl_vs_correlations.txt", excl)
excl_min = np.isnan(spontcorr) | ondiag

############################################################
# COMPUTE CORRELATION COEFFICIENTS WITH SPONTANEOUS ACTIVITY
############################################################

# Anatomical connectome correlation
excl = excl_min
spontcorr_ = spontcorr[~excl]
aconn_ = aconn[~excl]
r_spontcorr_aconn = np.corrcoef([spontcorr_, aconn_])[0, 1]

# Effective anatomical connectome correlation
excl = excl_min
spontcorr_ = spontcorr[~excl]
Aconnsym_ = Aconnsym[~excl]
r_spontcorr_Aconnsym = np.corrcoef([spontcorr_, Aconnsym_])[0, 1]

# Kernel-derived correlation
excl = excl_min | np.isnan(ck)
spontcorr_ = spontcorr[~excl]
ck_ = ck[~excl]
r_spontcorr_ck = np.corrcoef([spontcorr_, ck_])[0, 1]

# Biophysical model-derived anatomical correlation
excl = excl_min
spontcorr_ = spontcorr[~excl]
actconncorr_ = actconncorr[~excl]
r_spontcorr_actconncorr = np.corrcoef([spontcorr_, actconncorr_])[0, 1]

# Stimulated activity correlation
excl = excl_min | np.isnan(stimcorr)
spontcorr_ = spontcorr[~excl]
stimcorr_ = stimcorr[~excl]
r_spontcorr_stimcorr = np.corrcoef([spontcorr_, stimcorr_])[0, 1]

###############
# PRINT RESULTS
###############
print("TAKE THESE TWO VALUES FOR THE BAR PLOT funatlas_vs_correlation_combined_bar_plot.py")
print('Anatomical connectome vs spontaneous correlations:', r_spontcorr_aconn)
print('Connectome-derived correlation (linear) vs spontaneous correlations:', r_spontcorr_Aconnsym)
print('Connectome-derived correlation (biophysical) vs spontaneous correlations:', r_spontcorr_actconncorr)
print('Kernel-derived correlation vs spontaneous correlations:', r_spontcorr_ck)
print('Stimulated correlation vs spontaneous correlations:', r_spontcorr_stimcorr)
