import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
import argparse
import nonlinfunconn as nlfc
from nonlinfunconn import convolution, irrarray

def eci(x, p, power_t=None):
    """
    Evaluate the Exponential Convolution Integral (ECI) for given parameters.

    Args:
        x (array-like): Input time values.
        p (array-like or irrarray): Parameters for the exponential convolution.
        power_t (array-like, optional): Powers of t in terms like t^n exp(-gt). Defaults to None.

    Returns:
        array-like: The evaluated exponential convolution.
    """
    if not isinstance(p, irrarray):
        # If p is not an irrarray, handle it as a simple array
        if power_t is None:
            ecj = nlfc.ExponentialConvolution(p[1:], p[0])
        else:
            # Handle cases where power_t is provided
            ecj = nlfc.ExponentialConvolution(p[1], p[0])
            if power_t[1] > 0:
                for q in np.arange(power_t[1]):
                    ecj.convolve_exp(p[1])
            for h in np.arange(len(p) - 2):
                for q in np.arange(power_t[h] + 1):
                    ecj.convolve_exp(p[h])
    else:
        # If p is an irrarray, handle branching logic
        if power_t is None:
            ecj = nlfc.ExponentialConvolution(p(branch=0)[1:], p(branch=0)[0])
            for b in np.arange(len(p.first_index["branch"]) - 1)[1:]:
                branch_par = p(branch=b)
                ecj.branch_path(branch_par[1], branch_par[0])
                for h in np.arange(len(branch_par))[2:]:
                    ecj.convolve_exp(branch_par[h], branch=b)
        else:
            # Handle cases where power_t is provided and p is an irrarray
            p_b0 = p(branch=0)
            pt_b0 = power_t(branch=0)  # power_t must also be an irrarray

            ecj = nlfc.ExponentialConvolution([p_b0[1]], p_b0[0])
            if pt_b0[1] > 0:
                for q in np.arange(pt_b0[1]):
                    ecj.convolve_exp(p_b0[1])
            for h in np.arange(len(p_b0))[2:]:
                for q in np.arange(pt_b0[h] + 1):
                    ecj.convolve_exp(p[h])

            for b in np.arange(len(p.first_index["branch"]) - 1)[1:]:
                branch_par = p(branch=b)
                pt_bi = power_t(branch=b)
                ecj.branch_path(branch_par[1], branch_par[0])
                if pt_bi[1] > 0:
                    for q in np.arange(pt_bi[1]):
                        ecj.convolve_exp(branch_par[1], branch=b)
                for h in np.arange(len(branch_par))[2:]:
                    for q in np.arange(pt_bi[h] + 1):
                        ecj.convolve_exp(branch_par[h], branch=b)

    y = ecj.eval(x)
    del ecj  # Clean up the nonlinfunconn.ExponentialConvolution object

    return y

def load_trial_data(trial_folder):
    """
    Load DeltaX and t_eval from saved text files.

    Args:
        trial_folder (str): Path to the folder containing trial data.

    Returns:
        tuple: (t_eval, DeltaX) as numpy arrays.
    """
    DeltaX = np.loadtxt(os.path.join(trial_folder, 'DeltaX.txt'))
    t_eval = np.loadtxt(os.path.join(trial_folder, 't_eval.txt'))
    return t_eval, DeltaX

def save_kernel(trial_folder, kernel_params):
    """
    Save estimated kernel parameters to a text file.

    Args:
        trial_folder (str): Path to the folder where the kernel parameters will be saved.
        kernel_params (array-like or irrarray): Kernel parameters to save.
    """
    file_path = os.path.join(trial_folder, 'fitted_kernel.txt')

    # Extract values from the irrarray object if necessary
    if isinstance(kernel_params, irrarray):
        kernel_values = kernel_params[:]  # Convert irrarray to a flat numpy array
    else:
        kernel_values = kernel_params  # Assume it's already a numpy array

    np.savetxt(file_path, kernel_values)
    print(f"Kernel parameters saved to {file_path}")

def plot_kernels(t_eval, kernel_params, trial_folder, node_idx):
    """
    Plot and save the estimated kernel.

    Args:
        t_eval (array-like): Time values for the x-axis.
        kernel_params (array-like or irrarray): Kernel parameters to evaluate and plot.
        trial_folder (str): Path to the folder where the plot will be saved.
        node_idx (int): Index of the node for which the kernel is being plotted.
    """
    # Evaluate the kernel using the eci function
    kernel = eci(t_eval, kernel_params)

    # Plot the kernel
    plt.figure(figsize=(10, 5))
    plt.plot(t_eval, kernel, label='Estimated Kernel')
    plt.xlabel("Time")
    plt.ylabel("Kernel Response")
    plt.title(f"Fitted Exponential Kernel (Multiple Branches) for Node {node_idx}")
    plt.legend()
    plt.grid()

    # Save the plot
    plot_path = os.path.join(trial_folder, f'fitted_kernel_{node_idx}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Kernel plot saved to {plot_path}")

def fit_eci_branching(x, y, stim, dt, n_hops_min=3, n_hops_max=5, n_branches_max=5,
                      rms_limits=[None, None], auto_stop=False, rms_tol=1e-2,
                      method=None, routine="least_squares"):
    """
    Fit an Exponential Convolution Integral (ECI) model with branching.

    Args:
        x (array-like): Time values.
        y (array-like): Response signal.
        stim (array-like): Stimulated node signal.
        dt (float): Time step.
        n_hops_min (int): Minimum number of hops for fitting. Defaults to 3.
        n_hops_max (int): Maximum number of hops for fitting. Defaults to 5.
        n_branches_max (int): Maximum number of branches for fitting. Defaults to 5.
        rms_limits (list): Limits for RMS calculation. Defaults to [None, None].
        auto_stop (bool): Whether to stop early if RMS improvement is below tolerance. Defaults to False.
        rms_tol (float): Tolerance for RMS improvement. Defaults to 1e-2.
        method (str): Optimization method. Defaults to None.
        routine (str): Optimization routine ("minimize" or "least_squares"). Defaults to "least_squares".

    Returns:
        array-like: Fitted parameters.
    """
    rms = []  # Store RMS values for each iteration

    # Normalize the response signal
    y_norm = np.sum(y)
    if np.isnan(y_norm) or np.isinf(y_norm) or y_norm == 0:
        return None
    else:
        yb = y / y_norm

    # Normalize the stimulus signal
    stim_norm = np.sum(stim)
    stimb = stim / stim_norm

    p0_tot_prev_ = np.array([])  # Store previous total parameters
    p_prev = np.array([])  # Store previous parameters
    n_in_prev = np.array([], dtype=int)  # Store previous number of parameters per branch

    # Iterate over the number of branches
    for n_branches in np.arange(0, n_branches_max):
        rms_cur_b = []  # Store RMS values for the current branch

        # Iterate over the number of hops
        for i in np.arange(n_hops_min, n_hops_max + 1):
            # Initialize parameters for the current branch
            p0_cur_b_ = 0.2 + np.arange(i) * 0.03 + n_branches * 0.02
            if n_branches == 0:
                A0 = 1.0
            else:
                A0 = (-1) ** n_branches
            p0_cur_b_ = np.append(A0, p0_cur_b_)

            # Combine previous and current parameters
            p0_tot_ = np.append(p0_tot_prev_, p0_cur_b_)
            p0_tot = irrarray(p0_tot_, [np.append(n_in_prev, i + 1)], ["branch"])

            # Set bounds for optimization
            lower_bounds = -np.inf * np.ones_like(p0_tot)
            upper_bounds = np.inf * np.ones_like(lower_bounds)
            special_idx = np.append(0, np.cumsum(n_in_prev))
            for q in np.arange(len(lower_bounds)):
                if q not in special_idx:
                    lower_bounds[q] = 0.0

            # Perform optimization
            if routine == "minimize":
                error = lambda p, x, y: np.sum(np.power(convolution(stimb, eci(x, p), dt, 8) - y, 2))
                res = minimize(error, p0_tot, args=(x, yb), method=method)
                p_cur_b = res.x
            elif routine == "least_squares":
                residuals = lambda p, x, y: convolution(stimb, eci(x, p), dt, 8) - y
                res = least_squares(residuals, p0_tot, args=(x, yb), method=method, bounds=(lower_bounds, upper_bounds))
                p_cur_b = res.x

            # Calculate RMS for the current iteration
            rms_cur_b.append(np.sqrt(np.sum(np.power((eci(x, p_cur_b) - yb)[rms_limits[0]:rms_limits[1]], 2))))

            # Stop early if RMS improvement is below tolerance
            if auto_stop and i > n_hops_min:
                delta_rms_rel = np.abs(rms_cur_b[-1] - rms_cur_b[-2]) / rms_cur_b[-2]
                if delta_rms_rel < rms_tol:
                    break

        p0_tot_prev_ = p0_tot_
        rms.append(rms_cur_b[-1])

        # Stop early if RMS improvement is below tolerance
        if auto_stop and n_branches > 0:
            delta_rms_rel = (rms[-1] - rms[-2]) / rms[-2]
            if np.abs(delta_rms_rel) < rms_tol and delta_rms_rel < 0:
                n_in_prev = np.append(n_in_prev, i + 1)
                p_prev = p_cur_b
                break
            elif np.abs(delta_rms_rel) < rms_tol and delta_rms_rel >= 0:
                rms.pop(-1)
                break
            else:
                n_in_prev = np.append(n_in_prev, i + 1)
                p_prev = p_cur_b
        else:
            n_in_prev = np.append(n_in_prev, i + 1)
            p_prev = p_cur_b

    # Adjust parameters based on normalization
    special_idx = np.append(0, np.cumsum(n_in_prev))
    for j in special_idx[:-1]:
        p_prev[j] *= y_norm / stim_norm

    return p_prev

def main():
    """
    Main function to fit the data stored in a folder parsed with parse.add_argument(), in the command line.
    """
    parser = argparse.ArgumentParser(description="Run the ODE simulation with optional plotting.")
    parser.add_argument("--plot_data", action="store_true", help="Enable plotting of the results.")
    parser.add_argument("--data_folder", type=str, default="trials_data", help="Path to the folder containing trial data.")
    args = parser.parse_args()

    trial_base_folder = args.data_folder
    if not os.path.exists(trial_base_folder):
        print(f"No trial data found in {trial_base_folder}.")
        return

    # Process each trial folder
    for trial_idx in range(0,len(next(os.walk(args.data_folder))[1])):
        trial_folder = os.path.join(trial_base_folder, f'trial_{trial_idx}')
        if not os.path.exists(trial_folder):
            continue

        # Load trial data
        t_eval, DeltaX = load_trial_data(trial_folder)

        # Fit the ECI model with branching
        num_nodes = 4
        kernel_params = []
        for node_idx in range(0, num_nodes):
            kernel_params.append(fit_eci_branching(t_eval, DeltaX[node_idx, :], DeltaX[0, :], dt=t_eval[1] - t_eval[0],
                                          n_hops_min=1, n_hops_max=3, n_branches_max=2,
                                          rms_limits=[None, None], auto_stop=True, rms_tol=1e-2,
                                          method="trf", routine="least_squares"))
                
            # Plot the data if requested
            if args.plot_data:
                plot_kernels(t_eval, kernel_params[-1], trial_folder, node_idx)

        # Save the fitted kernel parameters
       # save_kernel(trial_folder, kernel_params)


if __name__ == "__main__":
    main()