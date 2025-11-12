import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import networkx as nx

# --- 1. Define the Kuramoto model ODE (Direct Summation) ---
def kuramoto_model_direct(t, thetas, K, omegas, A):
    """
    Defines the differential equations for the Kuramoto model
    using direct N^2 summation.
    
    Arguments:
        t: time
        thetas: array of phases (N,)
        K: coupling strength
        omegas: array of natural frequencies (N,)
    """
    N = len(thetas)
    dthetas_dt = np.zeros_like(thetas)
    
    # This is the N^2 loop
    for i in range(N):
        sum_sin = 0.0
        for j in range(N):
            sum_sin += A[i,j]*np.sin(thetas[j] - thetas[i])
        
        dthetas_dt[i] = omegas[i] + (K / N) * sum_sin
        
    return dthetas_dt

# --- 2. Main simulation loop ---
def simulate_kuramoto_direct_sum(N=50, sim_time=15, K_values=None, A=None, Kmax=8.0):
    """
    Runs the simulation for a range of K values using the direct summation model.
    """
    if K_values is None:
        K_values = np.linspace(0,Kmax, 41)
        
    if A is None:
        A = np.ones((N, N)) 
        np.fill_diagonal(A, 0)
        
    final_order_parameters = []
    
    
    # Set up simulation parameters
    t_span = [0, sim_time]
    t_eval = np.linspace(sim_time - 5, sim_time, 100) 
    
    # Initialize oscillators
    omegas =  np.random.normal(0, 1, N) # np.ones(N)#
    
    print(f"Running simulation for K values (Direct Summation, N={N})...")
    
    F1n = []
    F2n = []
    for K in K_values:
        thetas_0 = np.random.uniform(0, 2 * np.pi, N)
        
        # Run the ODE solver
        sol = solve_ivp(
            kuramoto_model_direct,  
            t_span, 
            thetas_0, 
            args=(K, omegas, A), 
            t_eval=t_eval,
            method='RK45'
        )
        
        # --- Post-processing for this K value ---
        
        # Get the phase trajectories from the solution
        # phase_trajectories has shape (N, num_time_steps)

        phase_trajectories = sol.y
        thetas_0 = phase_trajectories[:, -1]
        # 1. Calculate Order Parameter (r)
        Z_over_time = np.mean(np.exp(1j * phase_trajectories), axis=0)
        r_over_time = np.abs(Z_over_time)
        final_r = np.mean(r_over_time)
        final_order_parameters.append(final_r)
        
        # 2. Calculate Average Coupling Force (|F|_avg)
        # We must recalculate the force term for the final trajectories

        forces1n = np.zeros((N,N))
        forces2n = np.zeros((N,N,N))
        # Loop over the 100 saved time steps
        num_steps = phase_trajectories.shape[1]
        start_idx = int(np.ceil(0.95 * num_steps))
        # ensure at least one time step is processed
        if start_idx >= num_steps:
            start_idx = max(0, num_steps - 1)
        counter = 0
        for t_step in range(start_idx, num_steps):
            current_thetas = phase_trajectories[:, t_step]
            
            # Run the N^2 loop again to find the force on each oscillator
            for i in [1]:#range(N):
                for j in range(N):
                    if j == i:
                        forces1n[i,j]   += omegas[j]
                        continue
                    dif_ij = current_thetas[j] - current_thetas[i]
                    
                    forces1n[i,j]   += ((K/(N))*A[i,j]*np.sin(dif_ij))  ##(1 / N)*
                    for k in range(N):
                        dif_ik = current_thetas[k] - current_thetas[i]
                        dif_jk = current_thetas[k] - current_thetas[j]
                        forces2n[i,j,k] += ((K*K/(N*N))*A[i,j]*np.cos(dif_ij)*(A[j,k]*np.sin(dif_jk)-A[i,k]*np.sin(dif_ik)))  
                        
            counter += 1


        F1n.append(forces1n/counter)
        F2n.append(forces2n/counter)
        print(K/K_values[-1]*100,"% done", end='\r')

    return K_values, final_order_parameters, F1n, F2n

# --- 3. Run the simulation and plot the results ---
if __name__ == "__main__":
    # N=50 is a good compromise for speed. N=100 will be noticeably slower.
    N = 60

    A, A_name = None, "all_to_all"
    '''
    # create a Watts–Strogatz small-world network for the given N
    k = min(6, N-1)
    if k % 2 == 1:
        k -= 1
    p = 0.2
    seed = 42
    G = nx.watts_strogatz_graph(N, k, p, seed=seed)
    A = nx.to_numpy_array(G, dtype=float)
    np.fill_diagonal(A, 0.0)
    A_name = f"smallworld_k{k}_p{p}"
    '''

    '''
    # create a 1D ring lattice where each node connects only to its first neighbors (periodic)
    A = np.zeros((N, N), dtype=float)
    for i in range(N):
        A[i, (i - 1) % N] = 1.0
        A[i, (i + 1) % N] = 1.0
    np.fill_diagonal(A, 0.0)
    A_name = f"ring_first_neighbors_N{N}"
    
    '''

    # create a 1D ring lattice where each node connects only to its second neighbors (periodic)
    A = np.zeros((N, N), dtype=float)
    for i in range(N):
        A[i, (i - 1) % N] = 1.0
        A[i, (i + 1) % N] = 1.0
        A[i, (i - 2) % N] = 1.0
        A[i, (i + 2) % N] = 1.0
    np.fill_diagonal(A, 0.0)
    A_name = f"ring_second_neighbors_N{N}"

    Kmax = 1/np.max(np.abs(np.linalg.eigvals(A))) * 1000.0

    K_vals, r_vals, forces1n, forces2n = simulate_kuramoto_direct_sum(N=N, A=A, Kmax=Kmax)
    K_len = np.zeros(len(K_vals))

    f1n_ieqj = np.zeros(K_len.shape)
    f1n_idifj = np.zeros(K_len.shape)

    f2n_ieqjeqk   = np.zeros(K_len.shape)
    f2n_ieqjdifk  = np.zeros(K_len.shape)
    f2n_idifjeqk  = np.zeros(K_len.shape)
    f2n_jdifieqk  = np.zeros(K_len.shape)
    f2n_idifjdifk = np.zeros(K_len.shape)
    for idx in range(len(K_vals)):
        _f1n_ieqj = []
        _f1n_idifj = []
        _f2n_ieqjeqk   = []
        _f2n_ieqjdifk  = []
        _f2n_idifjeqk  = []
        _f2n_jdifieqk  = []
        _f2n_idifjdifk = []
        for i in [1]:#range(N):
            _f1n_ieqj.append(forces1n[idx][i,i])
            for j in range(N):
                if j != i:
                    _f1n_idifj.append(forces1n[idx][i,j])
                for k in range(N):
                    if k == j and j == i:
                        _f2n_ieqjeqk.append(forces2n[idx][i,j,k])
                    elif k != j and j == i:
                        _f2n_ieqjdifk.append(forces2n[idx][i,j,k])
                    elif k == j and j != i:
                        _f2n_idifjeqk.append(forces2n[idx][i,j,k])
                    elif k == i and j != i:
                        _f2n_jdifieqk.append(forces2n[idx][i,j,k])
                    elif k != j and j != i:
                        _f2n_idifjdifk.append(forces2n[idx][i,j,k])
    
        f1n_ieqj[idx] = np.sum(_f1n_ieqj)
        f1n_idifj[idx] =  np.sum(_f1n_idifj)

        f2n_ieqjeqk[idx]   =  np.sum(_f2n_ieqjeqk)
        f2n_ieqjdifk[idx]  =  np.sum(_f2n_ieqjdifk)
        f2n_idifjeqk[idx]  =  np.sum(_f2n_idifjeqk)
        f2n_jdifieqk[idx]  =  np.sum(_f2n_jdifieqk)
        f2n_idifjdifk[idx] =  np.sum(_f2n_idifjdifk)


    plt.figure(figsize=(10, 6))
    
    # Plot in three stacked panels: (1) order parameter, (2) 1st-neighbor forces, (3) 2nd-neighbor forces
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 12))
    fig.suptitle("Kuramoto Model (Direct Summation): Order vs. Forces", fontsize=16)

    # Panel 1: Order parameter
    ax1.plot(K_vals, r_vals, 'o-', color='black', label='Order Parameter (r)')
    ax1.set_ylabel('Order Parameter (r)', color='black', fontsize=12)
    ax1.tick_params(axis='y', colors='black')
    ax1.grid(True)
    ax1.legend(fontsize=10)

    # Panel 2: 1st-neighbor forces
    ax2.plot(K_vals, f1n_ieqj, 's-', color='red', label='1st neigh. i=j')
    ax2.plot(K_vals, f1n_idifj, '--', color='orange', label='1st neigh. i!=j')
    ax2.set_ylabel('Avg. 1st-neigh. interaction', fontsize=12)
    ax2.grid(True)
    ax2.legend(fontsize=10)

    # Panel 3: 2nd-neighbor forces (multiple categories)
    ax3.plot(K_vals, f2n_ieqjeqk, linestyle='-',  color='#1f77b4', linewidth=1.6, label='2nd neigh. i=j=k')
    ax3.plot(K_vals, f2n_ieqjdifk, linestyle='--', color='#2ca02c', linewidth=1.6, label='2nd neigh. i=j!=k')
    ax3.plot(K_vals, f2n_idifjeqk, linestyle=':',  color='#17becf', linewidth=1.6, label='2nd neigh. i!=j=k')
    ax3.plot(K_vals, f2n_jdifieqk, linestyle='-.', color='#9467bd', linewidth=1.6, label='2nd neigh. k=i!=j')
    ax3.plot(K_vals, f2n_idifjdifk, linestyle=(0, (1, 1)), color='#8c564b', linewidth=1.6, label='2nd neigh. i!=j!=k')
    ax3.set_ylabel('Avg. 2nd-neigh. interaction', fontsize=12)
    ax3.set_xlabel("Coupling Strength (K)", fontsize=12)
    ax3.grid(True)
    ax3.legend(fontsize=9, ncol=1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'kuramoto_direct_r_and_F_N{N}_A{A_name}.png')
    
    print("\nPlot saved as 'kuramoto_direct_r_and_F.png'")