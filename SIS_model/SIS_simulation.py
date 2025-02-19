""" 
Simulates the SIS network, saves the data for different in silico trials in different folders.
Each trial differs by added noise and stimulus intensity.

Run command line:
python3 SIS_simulation

To save the data and plot the dynamics:
python3 SIS_simulation --plot_data 
"""

import numpy as np
from scipy.integrate import solve_ivp
import os
import matplotlib.pyplot as plt
import argparse

def nonlinear_equation_stimuli(t, X, N, A, stim_neurons, stimuli):
    """Nonlinear ODE system with external stimulus and internal noise."""
    dX = np.zeros_like(X)
    noise = 0.1*np.random.rand(N)  # Generate normalized noise between 0 and 1
    
    for i in range(N):
        sum_term = np.sum(A[i] * (1 - X[i]) * X) - A[i, i] * (1 - X[i]) * X[i]
        
        stim = stimuli if (i in stim_neurons) and (2 < t < 8) else 0.0
        
        dX[i] = sum_term - X[i] + stim + noise[i]  # Add noise directly to the equation
    
    return dX

def save_data_to_txt(folder_path, filename, data):
    """Save data to a .txt file."""
    file_path = os.path.join(folder_path, filename)
    np.savetxt(file_path, data)
    print(f"Data saved to {file_path}")

def plot_trial_data(t_eval, DeltaX, trial_idx, trial_folder):
    """Plot the data for a single trial and save as PNG."""
    plt.figure(figsize=(10, 6))
    for i in range(DeltaX.shape[0]):
        plt.plot(t_eval, DeltaX[i, :], label=f'Node {i+1}')
    plt.title(f'Trial {trial_idx} - Node Responses')
    plt.xlabel('Time')
    plt.ylabel('Node State')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(trial_folder, f'trial_{trial_idx}.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ODE simulation with optional plotting.")
    parser.add_argument("--plot_data", action="store_true", help="Enable plotting of the results.")
    args = parser.parse_args()

    N = 4  # Number of nodes
    A = np.array([[0, 1,   0,   0], 
                  [0, 0,   1,   0], 
                  [0, 0,   0,   1], 
                  [0, 0,   0,   0]])
    resolution = 500
    t_span = (0, 20)
    t_eval = np.linspace(t_span[0], t_span[1], resolution)
    N_trials = 10
    
    # Create a directory to store the trial data
    if not os.path.exists('SIS model/trials_data'):
        os.makedirs('SIS model/trials_data')

    # Solve trial by trial and save data
    for trial_idx in range(N_trials):
        # Create a folder for the current trial
        trial_folder = os.path.join('SIS model/trials_data', f'trial_{trial_idx}')
        if not os.path.exists(trial_folder):
            os.makedirs(trial_folder)

        # Simulate the system with noise added within the ODE
        sol = solve_ivp(lambda t, X: nonlinear_equation_stimuli(t, X, N, A, [3], np.array([0.01 + 0.2 * trial_idx])), 
                        t_span, np.zeros(N), t_eval=t_eval)
        DeltaX = sol.y  
        
        # Save the data for the current trial as .txt files
        save_data_to_txt(trial_folder, 'DeltaX.txt', DeltaX)
        save_data_to_txt(trial_folder, 't_eval.txt', t_eval)

        # Plot the data if requested and save as PNG
        if args.plot_data:
            plot_trial_data(t_eval, DeltaX, trial_idx, trial_folder)
