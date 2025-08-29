
# preprocess and plot the smooth data
.venv/bin/python3.10 scripts/zebrafish/open_data.py  --load-cache --plot

# fit the neural mass model to the stimulated activity data
.venv/bin/python3.10 scripts/zebrafish/fit_NMM.py --plot