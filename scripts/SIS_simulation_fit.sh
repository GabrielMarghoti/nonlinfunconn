
# Simulate the SIS network dynamics
python3 SIS\ model/SIS_simulation.py --plot_data

# Fit linear exponential kernels for functional connectivity as in "Neural signal propagation atlas of Caenorhabditis elegans 2023"
python3 fit_exp_kernel.py --plot_data --data_folder SIS\ model/trials_data/