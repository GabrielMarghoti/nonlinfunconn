
# fit exp kernels and NEGF, compare the final correlation with observed spontaneous activity


# Unc31 worms, without wirelless connections. fit exp kernels and NEGF, compare the predicted signals with observed activity
python3 scripts/C_Elegans/fit_negf.py  --folder:figures --signal:green --no-merge --load-cache --unc31 


# WT (wild type) worms, with wirelless connections. fit exp kernels and NEGF, compare the predicted signals with observed activity
python3 scripts/C_Elegans/fit_negf.py  --folder:figures --signal:green --no-merge --load-cache --wt


# All worms:
python3 scripts/C_Elegans/fit_negf.py  --folder:figures --signal:green --no-merge --load-cache



# Correlation between fitted adjancy matrix and connetome
python3 scripts/C_Elegans/corr_gammas.py  --folder:figures --signal:green --no-merge