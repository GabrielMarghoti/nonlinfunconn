import os
import numpy as np
import sympy as sp
import random
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit

# Specify the folder where you want to save the figures
data_path = os.getcwd()+'/dados/quali/HH_neurons_stocas'  # Change this to your desired folder

# Specify the folder where you want to save the figures
figures_path = os.getcwd()+'/figuras/quali/HH_neurons_stocas'  # Change this to your desired folder

def theta(x):
    return np.where(x < 0, 0.0, 1.0)

def generate_ar1_noise(size, alpha=0.5, mean=0, std=1):
    noise = np.zeros(size)
    noise[0] = np.random.normal(mean, std)
    for t in range(1, size):
        noise[t] = alpha * noise[t - 1] + np.random.normal(mean, std * np.sqrt(1 - alpha**2))
    return noise

def conv(X, Y, interval):
    """Calculate the convolution of two signals X and Y over a given interval, ignoring NaNs."""
    # Filter out NaN values and compute the product directly
    product_sum = sum(x * y for x, y in zip(X, Y) if not (np.isnan(x) or np.isnan(y)))
    
    # Scale factor based on interval
    scale = (interval[-1] - interval[0]) / len(interval)
    return scale * product_sum

def exp_kernel_func(t, *params):
    # exponentials combinations with two parameters each, amplitude, decay const
    exponential_combination = np.zeros(len(t))
    for p_idx in range(0,len(params),2):
        exponential_combination += params[p_idx]* np.exp(-params[p_idx+1] * t)
    return exponential_combination*np.heaviside(t, 1.0)

def fit_exp_kernel(t, x, y):
    def convolved_model(t, *params):
        # Convolution function: convolves x with the kernel
        resol = len(t)
        kernel = exp_kernel_func(t, *params)
        convolved_signal = np.zeros(resol)
        for t_idx in range(1,resol):
            convolved_signal[t_idx] = conv(x[:t_idx], kernel[(t_idx-1)::-1], t[:t_idx])
        return convolved_signal
        
    initial_guess = np.random.rand(1*2) # be multiple of two, otherwise change func exp_kernel_func
    
    # Fit the parameters a and b to minimize the difference between fit_func and the actual data y
    params_opt, params_cov = curve_fit(convolved_model, t, y, p0=initial_guess)
    
    return params_opt, params_cov

# Ion gates functions
def alpha_m(V): return 0.1 * (V + 40.0) / (1.0 - np.exp(-0.1 * (V + 40.0)))
def beta_m(V):  return 4.0 * np.exp(-(V + 65.0) / 18.0)
def alpha_h(V): return 0.07 * np.exp(-(V + 65.0) / 20.0)
def beta_h(V):  return 1.0 / (1.0 + np.exp(-0.1 * (V + 35.0)))
def alpha_n(V): return 0.01 * (V + 55.0) / (1.0 - np.exp(-0.1 * (V + 55.0)))
def beta_n(V):  return 0.125 * np.exp(-(V + 65.0) / 80.0)

# Synapses functions
def alpha_S(Veq, beta, V_th):
    return 1 / (1 + np.exp(-beta * (Veq - V_th)))

def d_alpha_S(Veq, beta, V_th):
    exp_term = np.exp(-beta * (Veq - V_th))
    return (beta * exp_term) / (1 + exp_term)**2


def simulate_neuron_data(ti, tf, dt, I_ext_funcs, trial_label=0,
                                N      =   4,
                                C_m     =   1.0, 
                                g_Na   = 120.0, 
                                g_K    =  36.0, 
                                g_L    =   0.5, 
                                E_Na   =  50.0, 
                                E_K    = -77.0, 
                                E_L    = -60.0, 
                                g_gap  =   0.0, 
                                g_che  =   0.0,
                                E_s     =   0.0,
                                beta   =   0.125,
                                a_r    =   5.0,
                                a_d    =   5.0,
                                V_th   = -50.0):    
    W_elec = np.array([[0, 0, 0, 0],
                    [g_gap, 0, 0, 0],
                    [g_gap, 0, 0, 0],
                    [0, g_gap, g_gap, 0]])
    W_chem = np.array([[0, 0, 0, 0],
                    [g_che, 0, 0, 0],
                    [g_che, 0, 0, 0],
                    [0, g_che, g_che, 0]])
    

    E_s = np.zeros((N, N))
    E_s[3, 2] = -70.0          #inhibition
    #E_s[3, 1] = -70.0          #inhibition
    a_r = np.full((N, N), 5.0)
    a_d = np.full((N, N), 5.0) 
    beta = np.full((N, N), 0.125)
    V_th = np.full((N, N), -55.0)  # mV
    
    # Differential equations for the network of Hodgkin-Huxley neurons
    def hh_network_derivs(u, t, I_ext_funcs, N, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, W_elec, W_chem, E_s, beta, a_r, a_d, V_th):
        dudt = []
        for i in range(N):
            V, m, h, n = u[(4+N)*i:((4+N)*i + 4)] 
            s = u[((4+N)*i + 4):((4+N)*(i+1))]  # Each neuron's state: V, m, h, n, s
            
            # External current with noise
            I_ext = I_ext_funcs[i](t)
            
            # Ion channel current_span
            I_Na = g_Na * m**3 * h * (V - E_Na)
            I_K  = g_K * n**4 * (V - E_K)
            I_L  = g_L * (V - E_L)

            # Synaptic current from other neurons
            I_syn = sum((W_elec[i, j] * (V - u[(4+N)*j]) + W_chem[i, j]*s[j]*(V - E_s[i, j])) for j in range(N))
            
            # Differential equations for neuron i
            dVdt = (I_ext - I_syn - I_Na - I_K - I_L) / C_m
            dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
            dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
            dndt = alpha_n(V) * (1 - n) - beta_n(V) * n
            dsdt = np.zeros(N)
            for j in range(N):
                dsdt[j] = (a_r[i, j] * (1 / (1 + np.exp(-beta[i, j] * (u[(4+N)*j] - V_th[i, j])))) * (1 - s[j])- a_d[i, j] * s[j])

            dudt.extend([dVdt, dmdt, dhdt, dndt])
            dudt.extend(dsdt)
        return dudt

    # Initial conditions for each neuron
    initial_conditions = []
    for _ in range(N):
        V0 = -66.0
        m0 = alpha_m(V0) / (alpha_m(V0) + beta_m(V0))
        h0 = alpha_h(V0) / (alpha_h(V0) + beta_h(V0))
        n0 = alpha_n(V0) / (alpha_n(V0) + beta_n(V0))
        s0 = 0.1*np.ones(N)
        initial_conditions.extend([V0, m0, h0, n0])
        initial_conditions.extend(s0)
    

    # Solve the differential equations
    sol = odeint(hh_network_derivs, initial_conditions, t_span, (I_ext_funcs, N, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, W_elec, W_chem, E_s, beta, a_r, a_d, V_th))
    
    os.makedirs(data_path, exist_ok=True)  # Create the folder if it doesn't exist
    
    os.makedirs(figures_path, exist_ok=True)  # Create the folder if it doesn't exist
    fig, axes = plt.subplots(N, 1, figsize=(10, 10), sharex=True, dpi=200) 

    resol = len(sol[:,0])
    # Plot membrane potential and external current for each neuron in separate panels
    Vs = np.zeros((resol, N))
    Delta_Vs = np.zeros((len(sol[:,0]), N))
    for i in range(N):
        axes2 = axes[i].twinx()
        axes2.set_ylim(-30,70)
        axes[i].set_ylim(-85,42)
        V = sol[:, (N + 4) * i]  # Membrane potential of neuron i
        Vs[:, i] = V
        Delta_Vs[:, i] = V - initial_conditions[(N + 4) * i]

        I_ext = I_ext_funcs[i](t_span)  # External current for neuron i
        
         # Save membrane potential (V) and external current (I_ext) to .txt files
        np.savetxt(os.path.join(data_path, f"neuron_{i+1}_V_trial_{trial_label}_gNa_{g_Na}_ggap_{g_gap}_gche_{g_che}.txt"), np.column_stack((t_span, V)), 
                   header="Time(ms) Membrane_Potential(mV)", fmt="%.4f")
        np.savetxt(os.path.join(data_path, f"neuron_{i+1}_I_ext_trial_{trial_label}_gNa_{g_Na}_ggap_{g_gap}_gche_{g_che}.txt"), np.column_stack((t_span, I_ext)), 
                   header=f'Time(ms) External_Current', fmt="%.4f")

        axes2.plot(t_span, I_ext, color='red', alpha=0.6, linewidth=0.6)  
        axes[i].plot(t_span, V, color='blue')
        
        axes[i].set_ylabel(f'$V_{{{i + 1}}}$ (mV)', fontsize="12")
        axes2.set_ylabel(f'Ext. Current $(uA/C_m^2)$', fontsize="12")

    axes[0].axvspan(15, 20, color='red', alpha=0.25)
        
    # Add text annotation for "stimulus"
    axes[0].text(10.0, 42-15, "Stimulus", color="red", ha="center", fontsize=12)

    # Label the shared x-axis
    axes[-1].set_xlabel('Time (ms)', fontsize="12")

    plt.tight_layout()  # Adjust layout

    # Save the figure in both format_span
    plt.savefig(os.path.join(figures_path, f'neuron_network_trial_{trial_idx}_gNa_{g_Na}_ggap_{g_gap}_gche_{g_che}.png'), format='png', dpi=200)
    #plt.savefig(os.path.join(figures_path, f'neuron_network_trial_{trial_idx}_gNa_{g_Na}_ggap_{g_gap}_gche_{g_che}.pdf'), format='pdf', dpi=200)

    return Vs, Delta_Vs


def NEGF_HH_model(V, t_span,
                            N      =   4,
                            C_m     =   1.0, 
                            g_Na   = 120.0, 
                            g_K    =  36.0, 
                            g_L    =   0.5, 
                            E_Na   =  50.0, 
                            E_K    = -77.0, 
                            E_L    = -60.0, 
                            g_gap  =   0.0, 
                            g_che  =   0.0,
                            E_s     =   0.0,
                            beta   =   0.125,
                            a_r    =   5.0,
                            a_d    =   5.0,
                            V_th   = -50.0):

    resol = len(t_span)

    beta = beta*np.ones((N,N))
    a_r = a_r*np.ones((N,N))
    a_d = a_d*np.ones((N,N))
    E_s = E_s*np.ones((N,N))
    V_th = V_th*np.ones((N,N))

    W_elec = np.array([[0, 0, 0, 0],
                    [g_gap, 0, 0, 0],
                    [g_gap, 0, 0, 0],
                    [0, g_gap, g_gap, 0]])
    W_chem = np.array([[0, 0, 0, 0],
                    [g_che, 0, 0, 0],
                    [g_che, 0, 0, 0],
                    [0, g_che, g_che, 0]])
      
    V0 = V[0, :]
    m0 = np.zeros_like(V0)
    h0 = np.zeros_like(V0)
    n0 = np.zeros_like(V0)
    s0 = np.zeros((N, N))
    for i in range(N):
        m0[i] = alpha_m(V0[i]) / (alpha_m(V0[i]) + beta_m(V0[i]))
        h0[i] = alpha_h(V0[i]) / (alpha_h(V0[i]) + beta_h(V0[i]))
        n0[i] = alpha_n(V0[i]) / (alpha_n(V0[i]) + beta_n(V0[i]))
        for j in range(N): 
            s0[i, j] = 0.1

    Delta_Vs = np.zeros_like(V)

    for t in range(resol):
        Delta_Vs[t, :] = V[t, :] - V0

    # Defining intermediate calculations for each gating variable n, m, h
    n_inf = alpha_n(V) / (alpha_n(V) + beta_n(V))
    tau_n = 1.0 / (alpha_n(V) + beta_n(V))
    n_inf0 = alpha_n(V[0, :]) / (alpha_n(V[0, :]) + beta_n(V[0, :]))

    m_inf = alpha_m(V) / (alpha_m(V) + beta_m(V))
    tau_m = 1.0 / (alpha_m(V) + beta_m(V))
    m_inf0 = alpha_m(V[0, :]) / (alpha_m(V[0, :]) + beta_m(V[0, :]))

    h_inf = alpha_h(V) / (alpha_h(V) + beta_h(V))
    tau_h = 1.0 / (alpha_h(V) + beta_h(V))
    h_inf0 = alpha_h(V[0, :]) / (alpha_h(V[0, :]) + beta_h(V[0, :]))

    # Calculating tau_i
    tau_i = np.zeros(N)
    for i in range(N):
        tau_i[i] = (C_m) / (g_L + (g_Na * m0[i] ** 3 * h0[i]) + (g_K * n0[i] ** 4) + np.sum((W_chem*s0)[i,:]) + np.sum(W_elec[i,:]))

    # Allocate arrays
    sigma0 = np.zeros((resol, resol, N, N))
    gg0 = np.zeros((resol, resol, N, N))
    gs0 = np.zeros((resol, resol, N, N))
    g0 = np.zeros((resol, resol, N, N))

    # First nested loop for sigma0, gg0, gs0
    for j in range(N):
        for i in range(N):
            if W_chem[i, j]!= 0 or W_elec[i, j]!=0:
                for t in range(resol):
                    for t_prime in range(resol):
                        sigma0[t, t_prime, i, j] = theta(t_span[t] - t_span[t_prime]) * a_r[i, j] * (1 - s0[i, j]) * d_alpha_S(V0[j], beta[i, j], V_th[i, j]) * np.exp(-(t_span[t] - t_span[t_prime]) * (a_d[i, j] - a_r[i, j] / (1 + np.exp(-beta[i, j] * (V0[j] - V_th[i, j])))))
                        gg0[t, t_prime, i, j] = theta(t_span[t] - t_span[t_prime]) * W_elec[i, j] * np.exp(-(t_span[t] - t_span[t_prime]) / tau_i[i])
                        gs0[t, t_prime, i, j] = theta(t_span[t] - t_span[t_prime]) * W_chem[i, j] * (E_s[i, j] - V0[i]) * np.exp(-(t_span[t] - t_span[t_prime]) / tau_i[i])

    for i in range(N):
        for j in range(N):
            if W_chem[i, j]!= 0 or W_elec[i, j]!=0:
                for t in range(resol):
                    for t_prime in range(t):
                        g0[t, t_prime, i, j] = gg0[t, t_prime, i, j] + conv(gs0[t, t_prime:t+1, i, j], sigma0[t_prime:t+1, t_prime, i, j], t_span[t_prime:t+1])
           
    # Intermediate arrays for gating variables
    sigma_n_V = np.zeros((resol, resol, N))
    kappa_gK_V = np.zeros((resol, resol, N))
    sigma_m_V = np.zeros((resol, resol, N))
    sigma_h_V = np.zeros((resol, resol, N))
    kappa_gNa_V = np.zeros((resol, resol, N))
                            
    Chi_K = np.zeros((resol, resol, N))
    Chi_Na = np.zeros((resol, resol, N))
    Chi_i = np.zeros((resol, resol, N))
    conv_chi_V_i = np.zeros((resol, N))

    conv_sigma_n_V = np.zeros((resol, N))  # n approx
    conv_sigma_m_V = np.zeros((resol, N))  # m approx
    conv_sigma_h_V = np.zeros((resol, N))  # h approx

    # Loop through iterations (assumes a ProgressBar setup; replace with a standard range if not using ProgressBar)
    for _ in range(4):  # Example: from 1 to 4 iterations
        if g_Na == 0 and g_K == 0:
            break
        for i in range(N):
            for t in range(resol):
                for t_prime in range(t + 1):
                    if Delta_Vs[t_prime, i] != 0.0:
                        # Potassium Channel
                        sigma_n_V[t, t_prime, i] = (n_inf[t_prime, i] - (n_inf0[i] + conv_sigma_n_V[t_prime, i]))/(tau_n[t_prime, i] * Delta_Vs[t_prime, i])
                        kappa_gK_V[t, t_prime, i] = g_K * ((conv_sigma_n_V[t_prime, i] + n0[i]) ** 3)*((n_inf[t_prime, i] - n_inf0[i]) - conv_sigma_n_V[t_prime, i])/(tau_n[t_prime, i] * Delta_Vs[t_prime, i])

                        # Sodium Channel
                        sigma_m_V[t, t_prime, i] = (m_inf[t_prime, i] - (m_inf0[i] + conv_sigma_m_V[t_prime, i]))/(tau_m[t_prime, i] * Delta_Vs[t_prime, i])
                        sigma_h_V[t, t_prime, i] = (h_inf[t_prime, i] - (h_inf0[i] + conv_sigma_h_V[t_prime, i]))/(tau_h[t_prime, i] * Delta_Vs[t_prime, i])
                        kappa_gNa_V[t, t_prime, i] = g_Na * (conv_sigma_m_V[t_prime, i] + m0[i]) ** 2 * ((conv_sigma_h_V[t_prime, i] + h0[i]) *((m_inf[t_prime, i] - m_inf0[i] - conv_sigma_m_V[t_prime, i]) /(tau_m[t_prime, i] * Delta_Vs[t_prime, i])) + conv_sigma_m_V[t_prime, i] *((h_inf[t_prime, i] - h_inf0[i] - conv_sigma_h_V[t_prime, i])/(tau_h[t_prime, i] * Delta_Vs[t_prime, i])))

                    # Update Chi_K and Chi_Na
                    Chi_K[t, t_prime, i] = -np.exp(-(t - t_prime) / tau_i[i]) * (V[t_prime, i] - E_K) / C_m
                    Chi_Na[t, t_prime, i] = -np.exp(-(t - t_prime) / tau_i[i]) * (V[t_prime, i] - E_Na) / C_m

                # Convolve sigma values with Delta_V for each gating variable
                conv_sigma_n_V[t, i] = conv(sigma_n_V[t, :t+1, i], Delta_Vs[:t+1, i],t_span[:t+1])
                conv_sigma_m_V[t, i] = conv(sigma_m_V[t, :t+1, i], Delta_Vs[:t+1, i],t_span[:t+1])
                conv_sigma_h_V[t, i] = conv(sigma_h_V[t, :t+1, i], Delta_Vs[:t+1, i],t_span[:t+1])

            # Additional convolution for Chi_i with Delta_V over time
            for t in range(resol):
                for t_prime in range(t + 1):
                    Chi_i[t, t_prime, i] = conv(Chi_K[t, t_prime:t+1, i], kappa_gK_V[t_prime:t+1, t_prime, i],t_span[t_prime:t+1]) + conv(Chi_Na[t, t_prime:t+1, i], kappa_gNa_V[t_prime:t+1, t_prime, i],t_span[t_prime:t+1])
                conv_chi_V_i[t, i] = conv(Chi_i[t, :t+1, i], Delta_Vs[:t+1, i], t_span[:t+1])
    
    # Allocation for final matrices Chi_s, nonli_pi, g, and Gamma
    Chi_s = np.zeros((resol, resol, N, N))
    nonli_pi = np.zeros((resol, resol, N, N))
    g = np.zeros((resol, resol, N, N))
    conv_Chi_s_V = np.zeros((resol, N, N))  # DeltaS

    # Iterative loop for sigma approximation
    for _ in range(3):
        for j in range(N):
            for i in range(N):
                if W_chem[i, j]!= 0 or W_elec[i, j]!=0:
                    for t in range(resol):
                        for t_prime in range(t):
                            if V[t_prime, j] == V0[j]:
                                Chi_s[t, t_prime, i, j] = 0.0
                            else:
                                Chi_s[t, t_prime, i, j] = (sigma0[t, t_prime, i, j] / d_alpha_S(V0[j], beta[i, j], V_th[i, j]))*((alpha_S(V[t_prime, j], beta[i, j], V_th[i, j]) - alpha_S(V0[j], beta[i, j], V_th[i, j])) / Delta_Vs[t_prime, j])*(1 - (conv_Chi_s_V[t_prime, i, j]) / (1 - s0[i, j]))
                    for t in range(resol):    
                        # Convolution of Chi_s with Delta_V over time
                        conv_Chi_s_V[t, i, j] = conv(Chi_s[t, :t+1, i, j], Delta_Vs[:t+1, j],t_span[:t+1])
                    
                    # Nonlinear interaction calculation for g
                    for t in range(resol):
                        for t_prime in range(t):
                            nonli_pi[t, t_prime, i, j] = conv(gs0[t, t_prime:t+1, i, j], (1 - (Delta_Vs[t_prime:t+1, i] / (E_s[i, j] - V0[i]))) * Chi_s[t_prime:t+1, t_prime, i, j], t_span[t_prime:t+1])
                            g[t, t_prime, i, j] = gg0[t, t_prime, i, j] + nonli_pi[t, t_prime, i, j]

    # Gamma calculations and convolutions
    Gamma0   = g0  # First neighbors
    Gamma    = g    # First neighbors
    gamma_ast_chi = np.zeros_like(g)

    Lambda_2      = np.zeros_like(g)

    for t in range(resol):
        for t_prime in range(t):
            Gamma0[t, t_prime, -1, 0] += conv(g0[t, t_prime:t+1, -1, 2], Gamma0[t_prime:t+1, t_prime, 2, 0], t_span[t_prime:t+1]) + conv(g0[t, t_prime:t+1, -1, 1], Gamma0[t_prime:t+1, t_prime, 1, 0], t_span[t_prime:t+1])
            Gamma[t, t_prime, -1, 0] += conv(g[t, t_prime:t+1, -1, 2], Gamma[t_prime:t+1, t_prime, 2, 0], t_span[t_prime:t+1]) + conv(g[t, t_prime:t+1, -1, 1], Gamma[t_prime:t+1, t_prime, 1, 0], t_span[t_prime:t+1])

    # Final convolution over Gamma
    for i in range(N):
        for j in range(N):
            if W_chem[i, j]!= 0 or W_elec[i, j]!=0:
                if j != i:
                    for t in range(resol):
                        for t_prime in range(t + 1):
                            gamma_ast_chi[t, t_prime, i, j] = conv(g[t, t_prime:t+1, i, j], Chi_i[t_prime:t+1, t_prime, j], t_span[t_prime:t+1])
    for i in range(N):
        Lambda_2[:, :, i, i] = Chi_i[:, :, i]
        for j in range(N):
            Lambda_2[:, :, i, j] = gamma_ast_chi[:, :, i, j] + Gamma[:, :, i, j]

    estimated_V = V[0, -1]*np.ones(resol)
    for t in range(resol):
        for j in range(N):
            estimated_V[t] += conv(Lambda_2[t, :t+1, -1, j], Delta_Vs[:t+1, j], t_span[:t+1])

    return estimated_V

########################################################################################################################

########################################################################################################################

if __name__ == "__main__":
    
    N_trials = 2
    
    ti=0 
    tf= 100 
    dt=0.5
    N=4 
    # Parameters for noise and stimulus
    I_mean = -5.0         # Mean input current, uA/C_m^2
    I_std =  2.0           # Standard deviation of noise, uA/C_m^2
    t_span = np.arange(ti, tf, dt)  # Simulation time vector
    resol = np.size(t_span)
        
    V_data      = np.zeros((4, N_trials, resol, N))
    Delta_V_data      = np.zeros((4, N_trials, resol, N))

    V_data_avr  = np.zeros((4, resol, N))
    Delta_V_data_avr  = np.zeros((4, resol, N))

    for trial_idx in tqdm(range(N_trials), 
              desc="Processing Trials", 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
              colour="cyan"):
        # Generate noisy external current_span for each neuron and each trial, the noise is qhat distinguish different experiment_span
        I_ext_funcs = []
        for i in range(4):
            if i==0:
                I_stim = [I_mean if x < 15 or x > 20 else I_mean+15 for x in t_span]
            else:
                I_stim = I_mean
            I_ext = I_stim + I_std * generate_ar1_noise(size=len(t_span))
            I_ext_func = interp1d(t_span, I_ext, kind="linear", fill_value="extrapolate")
            I_ext_funcs.append(I_ext_func)        
        trial_label = random.sample(range(1, 10000),1)[0]
        # CHEMICAL HH
        V_data[0, trial_idx, :, :], Delta_V_data[0, trial_idx, :, :] = simulate_neuron_data(ti=ti, tf=tf, dt=dt, N=N, I_ext_funcs=I_ext_funcs, trial_label=trial_label, 
                                                                g_gap  =   0.0, 
                                                                g_che  =   0.5,)
        # GAP JUNCTION HH
        V_data[1, trial_idx, :, :], Delta_V_data[1, trial_idx, :, :] = simulate_neuron_data(ti=ti, tf=tf, dt=dt, N=N, I_ext_funcs=I_ext_funcs, trial_label=trial_label, 
                                                                g_gap  =   0.5, 
                                                                g_che  =   0.0,)
        # CHEMICAL LIF
        V_data[2, trial_idx, :, :], Delta_V_data[2, trial_idx, :, :] = simulate_neuron_data(ti=ti, tf=tf, dt=dt, N=N, I_ext_funcs=I_ext_funcs, trial_label=trial_label, 
                                                                g_Na   =   0.0,
                                                                g_K   =   0.0,
                                                                g_gap  =   0.0, 
                                                                g_che  =   0.5,)
        # GAP JUNCTION LIF
        V_data[3, trial_idx, :, :], Delta_V_data[3, trial_idx, :, :] = simulate_neuron_data(ti=ti, tf=tf, dt=dt, N=N, I_ext_funcs=I_ext_funcs, trial_label=trial_label, 
                                                                g_Na   =   0.0,
                                                                g_K   =   0.0,
                                                                g_gap  =  0.5,)
        
    fitted_lin_params = []
    fitted_curves  = np.zeros((5, 4, N_trials,resol)) #model(lin, HH_CHEM, HH_ELEC, LIF_CHEM, LIF_ELEC) data(HH_CHEM, HH_ELEC, LIF_CHEM, LIF_ELEC)
    for data_idx in  tqdm(range(4), 
              desc="Processing Data", 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
              colour="green"):
        for t_idx in range(resol):
            for i in range(N):
                V_data_avr[data_idx, t_idx, i]       = np.mean(V_data[data_idx, :, t_idx, i])
                Delta_V_data_avr[data_idx, t_idx, i] = np.mean(Delta_V_data[data_idx, :, t_idx, i])
    
        exp_parameters, _  = fit_exp_kernel(t_span,  Delta_V_data_avr[data_idx, : 0], Delta_V_data_avr[data_idx, :, -1])
        fitted_lin_params.append(exp_parameters)

        for trial_idx in tqdm(range(N_trials), 
              desc="Processing Trials (NEGF)", 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
              colour="blue"):
            fitted_curves[0, data_idx, trial_idx, 0] = V_data[data_idx, trial_idx, 0, 0]
            for t_idx in range(1,len(t_span)):
                fitted_curves[0, data_idx, trial_idx, t_idx] = V_data[data_idx, trial_idx, 0, 0] + conv(Delta_V_data[data_idx, trial_idx, :t_idx, 0], exp_kernel_func(t_span[(t_idx-1)::-1], *exp_parameters),t_span[:t_idx])
               
            fitted_curves[1, data_idx, trial_idx, :] = NEGF_HH_model(V_data[data_idx, trial_idx, :, :], t_span,
                                                                g_gap  =   0.0, 
                                                                g_che  =   0.5,)
            fitted_curves[2, data_idx, trial_idx, :] = NEGF_HH_model(V_data[data_idx, trial_idx, :, :], t_span,
                                                                g_gap  =   0.5, 
                                                                g_che  =   0.0,)
            fitted_curves[3, data_idx, trial_idx, :] = NEGF_HH_model(V_data[data_idx, trial_idx, :, :], t_span,
                                                                g_Na   =   0.0,
                                                                g_K    =   0.0,
                                                                g_gap  =   0.0, 
                                                                g_che  =   0.5,)
            fitted_curves[4, data_idx, trial_idx, :] = NEGF_HH_model(V_data[data_idx, trial_idx, :, :], t_span,
                                                                g_Na   =   0.0,
                                                                g_K    =   0.0,
                                                                g_gap  =  0.5, 
                                                                g_che  =   0.0,)
            
    # Plot fitted result_span
    # Define the model and data names
    model_names = ["Decay. Exp. Kernel", "HH Chem. NEGF", "HH Elec. NEGF", "LIF Chem. NEGF", "LIF Elec. NEGF"]
    data_names = ["HH Chem.", "HH Elec.", "LIF Chem.", "LIF Elec."]

    colors = plt.cm.gist_rainbow(np.linspace(0, 1, N_trials))

    fig, axes = plt.subplots(5, 4, figsize=(10, 10), dpi=200)

    for data_idx in range(4):
        for model_idx in range(5):
            axes[model_idx, data_idx].set_ylim(-85,42)
            for trial_idx in range(N_trials):
                # Color stimulus period
                axes[model_idx, data_idx].axvspan(15, 20, color='red', alpha=0.10)
                # Scatter plot for observed data
                axes[model_idx, data_idx].scatter(t_span, V_data[data_idx, trial_idx, :, -1], color=colors[trial_idx], edgecolors='k', alpha=0.90, s=3, linewidths=0.3)
                # Line plot for fitted curves
                axes[model_idx, data_idx].plot(t_span, fitted_curves[model_idx, data_idx, trial_idx, :], color=colors[trial_idx], alpha=0.90)

    for data_idx in range(4):
        axes[0, data_idx].set_title(f'{data_names[data_idx]}')
        axes[-1, data_idx].set_xlabel(f'Time (ms)')
        for model_idx in range(N):
            axes[model_idx, data_idx].set_xticks([])

    for model_idx in range(5):
        axes[model_idx, 0].set_ylabel(f'{model_names[model_idx]}')
        for data_idx in range(1, N):
            axes[model_idx, data_idx].set_yticks([])        
    
    # Global title and y-axis label
    fig.suptitle('Data', fontsize=16)
    fig.text(0.01, 0.5, 'Fitted Model', va='center', rotation='vertical', fontsize=16)  # Move further left

    # Adjust the layout to increase left margin and reduce spacing between panels
    plt.subplots_adjust(left=0.1, right=0.98, top=0.94, bottom=0.06, wspace=0.02, hspace=0.02)

    plt.savefig(os.path.join(figures_path, f'data_to_model_fit_N_trials{N_trials}.pdf'), format='pdf', dpi=200)
    plt.savefig(os.path.join(figures_path, f'data_to_model_fit_N_trials{N_trials}.png'), format='png', dpi=200)
