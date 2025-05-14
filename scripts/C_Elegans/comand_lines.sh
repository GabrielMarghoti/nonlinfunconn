
# fit exp kernels and NEGF, compare the final correlation with observed spontaneous activity

# process the raw pumpprobe data to .pkl datasets which does not require the pumpprobe package
python3 scripts/C_Elegans/process_pumpprobe_dataset.py  --signal:green --no-merge  --worm-type:wt   --skip-processed
python3 scripts/C_Elegans/process_pumpprobe_dataset.py  --signal:green --no-merge  --worm-type:unc31   --skip-processed

# fit negf and save the fitted parameters, from wt or unc31 worm types, --load-cache looks for previously fitted parameters 
python3 scripts/C_Elegans/fit_negf_from_processed_data.py  --load-cache --worm-type:wt
python3 scripts/C_Elegans/fit_negf_from_processed_data.py  --load-cache --worm-type:unc31


python3 scripts/C_Elegans/corr_gammas.py 



# For fitting of a specific dataset (maybe to focus on high resolution fitting)
python3 scripts/C_Elegans/fit_negf_from_processed_data_single_file.py  --load-cache --dataset-path:/home/gabrielm/projects/nonlinfunconn-main/data/C_elegans_pumpprobre_exp/worm_type_unc31/20220113_101730/stim_neu_AVDR/


######################## OLD ############################

# Unc31 worms, without wirelless connections. fit exp kernels and NEGF, compare the predicted signals with observed activity
python3 scripts/C_Elegans/fit_negf.py  --folder:figures --signal:green --no-merge --load-cache --unc31 


# WT (wild type) worms, with wirelless connections. fit exp kernels and NEGF, compare the predicted signals with observed activity
python3 scripts/C_Elegans/fit_negf.py  --folder:figures --signal:green --no-merge --load-cache --wt


# All worms:
python3 scripts/C_Elegans/fit_negf.py  --folder:figures --signal:green --no-merge --load-cache

# All worms, consider not labeled neurons (nut responsive) for the fitting
python3 scripts/C_Elegans/fit_negf.py  --folder:figures --signal:green --no-merge --load-cache --use-not-labeled-neurons

# Correlation between fitted adjancy matrix and connetome
python3 scripts/C_Elegans/corr_gammas.py  --folder:figures --signal:green --use-not-labeled-neurons --no-merge







