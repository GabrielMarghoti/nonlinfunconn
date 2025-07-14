import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===================================================================
# Part 1: Simulate the LIF Network (Corrected)
# ===================================================================

def simulate_lif_network(
    N=100,            # Number of neurons
    T=1000.0,         # Total simulation time in ms
    dt=0.1,           # Simulation time step in ms
    p_connect=0.1,    # Connection probability
    w_exc=0.8,        # Excitatory synaptic weight (mV)
    w_inh=-1.2,       # Inhibitory synaptic weight (mV)
    frac_exc=0.8,     # Fraction of excitatory neurons
    I_bg=0.6,         # Background current in mV/ms (the crucial new parameter)
    I_bg_std=0.2,     # Std deviation of background current noise
    ):
    """
    Simulates a randomly connected network of Leaky Integrate-and-Fire neurons.
    
    This version includes a background current to drive network activity.
    """
    print("--- Starting Network Simulation ---")
    
    # --- Model Parameters ---
    V_th = -55.0  # Spike threshold (mV)
    V_r = -75.0   # Reset potential (mV)
    V_L = -70.0   # Leak/rest potential (mV)
    tau_m = 20.0  # Membrane time constant (ms). This is 1/g_L from your notes.
    
    # --- Simulation setup ---
    n_steps = int(T / dt)
    
    # --- Network Connectivity (Synaptic weights E_ij) ---
    n_exc = int(N * frac_exc)
    weights = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j or np.random.rand() > p_connect:
                continue
            if j < n_exc:
                weights[i, j] = w_exc
            else:
                weights[i, j] = w_inh
    
    # --- State variables ---
    V = np.full(N, V_L)
    V_trace = np.zeros((N, n_steps))
    spike_times = [[] for _ in range(N)]
    
    # --- Main Simulation Loop ---
    for t_step in tqdm(range(n_steps), desc="Simulating"):
        V_trace[:, t_step] = V
        
        I_syn = np.zeros(N)
        if t_step > 0:
            # Find which neurons spiked at t-1
            spiked_indices = [i for i, times in enumerate(spike_times) if times and times[-1] >= (t_step - 1) * dt - dt/2]
            if spiked_indices:
                I_syn = np.sum(weights[:, spiked_indices], axis=1)

        # === THIS IS THE KEY CORRECTION ===
        # Add background current (constant drive + noise)
        background_input = I_bg + np.random.randn(N) * I_bg_std
        
        # Update membrane potential using Euler method
        # dV/dt = -(V - V_L) / tau_m + I_background
        dV = (-(V - V_L) / tau_m + background_input) * dt
        V += dV + I_syn # Add leak, background, and synaptic "kick"
        # ================================
        
        spiked_neurons = np.where(V >= V_th)[0]
        if len(spiked_neurons) > 0:
            V[spiked_neurons] = V_r
            current_time = t_step * dt
            for idx in spiked_neurons:
                spike_times[idx].append(current_time)
                
    print("--- Simulation Finished ---")
    
    params = {
        'N': N, 'T': T, 'dt': dt, 'V_th': V_th, 'V_r': V_r, 'V_L': V_L, 
        'tau_m': tau_m
    }

    return spike_times, weights, V_trace, params

# ===================================================================
# Part 2: Compute and Plot Propagation Kernels 
# ===================================================================

def compute_and_plot_kernels(spike_times, weights, V_trace, params):
    """
    Computes and visualizes the propagation kernels based on simulation results.
    """
    print("\n--- Computing and Plotting Kernels ---")
    
    # Check if any spikes occurred at all
    total_spikes = sum(len(s) for s in spike_times)
    if total_spikes == 0:
        print("CRITICAL: No spikes were generated in the simulation. Cannot plot kernels.")
        print("Try increasing I_bg or w_exc in the simulation parameters.")
        return

    N = params['N']
    T = params['T']
    dt = params['dt']
    tau_m = params['tau_m']
    V_th = params['V_th']
    V_L = params['V_L']
    
    # --- 1. Plot Raster of network activity ---
    plt.figure(figsize=(12, 6))
    plt.eventplot(spike_times, colors='gray', linelengths=0.75)
    plt.title('Network Activity (Raster Plot)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.xlim(0, T)
    plt.ylim(-1, N)
    plt.grid(alpha=0.0)
    
    # --- 2. First-Order Kernel (g_ij * ΔV_j) ---
    exc_connections = np.where(weights > 0)
    if len(exc_connections[0]) == 0:
        print("No excitatory connections found to plot first-order kernel.")
        i_target, j_source = -1, -1
    else:
        # Try to find a pair where the source actually spiked
        i_target, j_source = -1, -1
        for i_cand, j_cand in zip(*exc_connections):
            if spike_times[j_cand]:
                i_target, j_source = i_cand, j_cand
                break
    
    if j_source == -1:
         print(f"Could not find a spiking source neuron. Cannot plot first-order kernel.")
    else:
        E_ij = weights[i_target, j_source]
        t_spike_j = spike_times[j_source][0]
        t_kernel = np.arange(t_spike_j, T, dt)
        first_order_psp = E_ij * np.exp(-(t_kernel - t_spike_j) / tau_m)
        
        plt.figure(figsize=(12, 6))
        time_axis = np.arange(0, T, dt)
        plt.plot(time_axis, V_trace[i_target, :], 'b-', label=f'Actual V(t) of Neuron {i_target}', alpha=0.7)
        plt.plot(t_kernel, first_order_psp + V_trace[i_target, int(t_spike_j/dt)-1], 'r--', 
                 label=f'First-Order Kernel Effect from j={j_source}')
        
        plt.axvline(t_spike_j, color='gray', linestyle=':', label=f'Spike from j={j_source}')
        plt.title(f'First-Order Propagation Kernel ($g_{{ij}}$)')
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Potential (mV)')
        plt.legend()
        plt.xlim(max(0, t_spike_j - 3*tau_m), t_spike_j + 10*tau_m)
        plt.ylim(params['V_r'] - 2, params['V_th'] + 5)
        plt.grid(alpha=0.0)


    # --- 3. Second-Order Kernel (g_ij * g_jk * ΔV_k) ---
    k, j, i = -1, -1, -1
    exc_conn_indices = np.transpose(np.where(weights > 0))
    np.random.shuffle(exc_conn_indices) # Randomize to find different paths
    for j_cand, k_cand in exc_conn_indices:
        if not spike_times[k_cand]: continue
        i_candidates = np.where(weights[:, j_cand] > 0)[0]
        if len(i_candidates) > 0 and spike_times[j_cand]:
            k, j, i = k_cand, j_cand, i_candidates[0]
            break
            
    if k == -1:
        print("Could not find a suitable 2-step path k->j->i. Skipping second-order kernel plot.")
    else:
        E_jk = weights[j, k]
        E_ij = weights[i, j]
        
        causal_pairs = []
        for t_k in spike_times[k]:
            for t_j in spike_times[j]:
                if t_j > t_k and (t_j - t_k) < 3 * tau_m:
                    causal_pairs.append((t_k, t_j))
                    break # Only take the first j spike after a k spike

        if not causal_pairs:
            print(f"No causal spike pairs found for path {k}->{j}->{i}. Skipping plot.")
        else:
            print(f"Found {len(causal_pairs)} causal spike pairs for path {k}->{j}->{i}.")
            total_second_order_effect = np.zeros(int(T/dt))
            
            for t_k, t_j in causal_pairs:
                start_idx = int(t_j / dt)
                if start_idx >= len(total_second_order_effect): continue
                
                t_kernel_2nd = np.arange(t_j, T, dt)
                # The decay depends on the full path from k
                second_order_psp = (E_ij * E_jk)/((V_th-V_L)**2) * np.exp(-(t_kernel_2nd - t_k) / tau_m)
                
                end_idx = start_idx + len(second_order_psp)
                if end_idx > len(total_second_order_effect):
                    end_idx = len(total_second_order_effect)
                    second_order_psp = second_order_psp[:end_idx-start_idx]

                total_second_order_effect[start_idx:end_idx] += second_order_psp

            plt.figure(figsize=(12, 6))
            time_axis = np.arange(0, T, dt)
            base_voltage = np.interp(time_axis, time_axis, V_trace[i, :]) 
            plt.plot(time_axis, base_voltage, 'b-', label=f'Actual V(t) of Neuron {i}', alpha=0.6)
            plt.plot(time_axis, total_second_order_effect + params['V_L'], 'r--', 
                     label=f'Summed Second-Order Effect from path {k}->{j}')
            
            for idx, (t_k, t_j) in enumerate(causal_pairs[:5]):
                 plt.axvline(t_k, color='green', linestyle=':', alpha=0.8, label=f'Spike k={k}' if idx==0 else "")
                 plt.axvline(t_j, color='orange', linestyle=':', alpha=0.8, label=f'Spike j={j}' if idx==0 else "")

            plt.title(f'Second-Order Propagation Kernel ($g_{{ij}} * g_{{jk}}$) for path {k}→{j}→{i}')
            plt.xlabel('Time (ms)')
            plt.ylabel('Membrane Potential (mV)')
            plt.legend()
            plt.grid(alpha=0.0)


# ===================================================================
# Main Execution Block
# ===================================================================

if __name__ == '__main__':
    # Run the simulation with parameters that ensure activity
    spike_data, weight_matrix, voltage_trace, sim_params = simulate_lif_network(
        N=50, 
        T=5000.0, 
        w_exc=0.5,
        I_bg=0.7, # Set background current
        I_bg_std=0.5 # Add some noise
    )
    
    # Analyze and plot the results
    compute_and_plot_kernels(spike_data, weight_matrix, voltage_trace, sim_params)
    
    plt.tight_layout()
    plt.show()