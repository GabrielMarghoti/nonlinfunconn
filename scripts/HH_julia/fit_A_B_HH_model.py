import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm

def conv(X, Y, interval):
    """
    Perform a convolution-like operation between two arrays X and Y over a given interval.
    
    Args:
    - X (numpy array): First array.
    - Y (numpy array): Second array.
    - interval (numpy array): Array defining the time interval.
    
    Returns:
    - result (float): The result of the convolution-like operation.
    """
    # Handle missing values by masking NaNs
    mask = ~np.isnan(X) & ~np.isnan(Y)
    
    # Element-wise multiplication of X and Y, skipping missing values
    product = X[mask] * Y[mask]
    
    # Calculate the scaling factor based on the interval
    scale = (interval[-1] - interval[0]) / len(interval)
    
    # Perform the summation
    result = scale * np.sum(product)
    
    return result
            
# Define the functions for alpha and beta
def alpha_m(V):
    return 0.1 * (V + 40.0) / (1.0 - np.exp(-0.1 * (V + 40.0)))

def beta_m(V):
    return 4.0 * np.exp(-(V + 65.0) / 18.0)

def alpha_h(V):
    return 0.07 * np.exp(-(V + 65.0) / 20.0)

def beta_h(V):
    return 1.0 / (1.0 + np.exp(-0.1 * (V + 35.0)))

def alpha_n(V):
    return 0.01 * (V + 55.0) / (1.0 - np.exp(-0.1 * (V + 55.0)))

def beta_n(V):
    return 0.125 * np.exp(-(V + 65.0) / 80.0)

# The Hodgkin-Huxley model for solving with solve_ivp
def diff_eq_model(t, u, *params):
    # Unpack parameters
    N, C_m, g_Na_bar, g_K_bar, g_L_bar, E_Na, E_K, E_L,  A, B, Es, a_r, a_d, beta, V_th, I_ext = params[0]
    stim = params[1]
        
    # Reshape u into a (N, 4+N) array
    u = u.reshape((N, 4 + N))
    du = np.zeros_like(u)
        
    # Iterate over neurons
    for i in range(N):
        # Extract neuron variables: V, m, h, n
        V, m, h, n = u[i, :4]

        # Initialize I_ext (External current)
        I_ext_i = 0.0
        if i == 0 and (4.0 <= t <= 5.0):  # External stimulus only to the first neuron between t=4 and t=5
            I_ext_i += stim

        # Add coupling terms from pre-synaptic neurons
        for j in range(N):
            I_ext_i += -A[i, j] * (u[i, 0] - u[j, 0]) - B[i, j] * u[i, 4 + j] * (u[i, 0] - Es[i, j])
            du[i, 4 + j] = (a_r[i, j] * (1 / (1 + np.exp(-beta[i, j] * (u[j, 0] - V_th[i, j])))) * (1 - u[i, 4 + j])- a_d[i, j] * u[i, 4 + j])

        # Calculate the membrane potential derivative
        du[i, 0] = (I_ext_i - g_Na_bar * m**3 * h * (V - E_Na) - g_K_bar * n**4 * (V - E_K) - g_L_bar * (V - E_L)) / C_m
            
        # Update gating variables (m, h, n)
        du[i, 1] = alpha_m(V) * (1.0 - m) - beta_m(V) * m
        du[i, 2] = alpha_h(V) * (1.0 - h) - beta_h(V) * h
        du[i, 3] = alpha_n(V) * (1.0 - n) - beta_n(V) * n

    # Flatten du to match solve_ivp's required shape
    return du.flatten()

def NEGF_HH_model(resolution, N, DeltaV, V, C_m, g_Na_bar, g_K_bar, g_L_bar, E_Na, E_K, E_L, A, B, Es, a_r, a_d, beta, V_th, I_ext, V0, gS0, gg0, gs0):
    # Initialize arrays
    # Nonequilibrium self-interaction Green functions
    sigma_n_V = np.zeros((resolution, resolution, N))
    kappa_gK_V = np.zeros((resolution, resolution, N))
        
    sigma_m_V = np.zeros((resolution, resolution, N))
    sigma_h_V = np.zeros((resolution, resolution, N))
    kappa_gNa_V = np.zeros((resolution, resolution, N))
        
    conv_sigma_n_V = np.zeros((resolution, N))
    conv_sigma_m_V = np.zeros((resolution, N))
    conv_sigma_h_V = np.zeros((resolution, N))
        
    Chi_K = np.zeros((resolution, resolution, N))
    Chi_Na = np.zeros((resolution, resolution, N))

    Chi_i = np.zeros((resolution, resolution, N))
    conv_chi_V_i = np.zeros((resolution, N))
        
        
    # Nonequilibrium interaction Green Functions (synapse)
    Chi_s = np.zeros((resolution, resolution, N, N))
    gamma = np.zeros((resolution, resolution, N, N))
    conv_Chi_s_V = np.zeros((resolution, N, N))
        
    # Compute n_inf, tau_n, and n_inf0
    n_inf = alpha_n(V) / (alpha_n(V) + beta_n(V))
    tau_n = 1.0 / (alpha_n(V) + beta_n(V))
    n_inf0 = alpha_n(V[0, :]) / (alpha_n(V[0, :]) + beta_n(V[0, :]))

    # Compute m_inf, tau_m, and m_inf0
    m_inf = alpha_m(V) / (alpha_m(V) + beta_m(V))
    tau_m = 1.0 / (alpha_m(V) + beta_m(V))
    m_inf0 = alpha_m(V[0, :]) / (alpha_m(V[0, :]) + beta_m(V[0, :]))

    # Compute h_inf, tau_h, and h_inf0
    h_inf = alpha_h(V) / (alpha_h(V) + beta_h(V))
    tau_h = 1.0 / (alpha_h(V) + beta_h(V))
    h_inf0 = alpha_h(V[0, :]) / (alpha_h(V[0, :]) + beta_h(V[0, :]))

    # Initialize tau_i
    tau_i = np.zeros(N)
        
    for i in range(N):
        tau_i[i] = (C_m) / (g_L_bar + (g_Na_bar * (m0[i]**3) * h0[i]) + (g_K_bar * (n0[i]**4)) + np.sum(A * gS0) + np.sum(B * np.ones(N)))

        # Iterative method to approximate sigma
        for itr in range(6):
            for i in range(N):
                for t in range(resolution):
                    for t_prime in range(t + 1):
                        if DeltaV[t_prime, i] != 0.0:
                            # Potassium Channel
                            sigma_n_V[t, t_prime, i] = (n_inf[t_prime, i] - (n_inf0[i] + conv_sigma_n_V[t_prime, i])) / (tau_n[t_prime, i] * DeltaV[t_prime, i])
                            kappa_gK_V[t, t_prime, i] = g_K_bar * (conv_sigma_n_V[t_prime, i] ** 3) * ((n_inf[t_prime, i] - n_inf0[i]) - conv_sigma_n_V[t_prime, i]) / (tau_m[t_prime, i] * DeltaV[t_prime, i])

                            # Sodium Channel
                            sigma_m_V[t, t_prime, i] = (m_inf[t_prime, i] - (m_inf0[i] + conv_sigma_m_V[t_prime, i])) / (tau_m[t_prime, i] * DeltaV[t_prime, i])
                            sigma_h_V[t, t_prime, i] = (h_inf[t_prime, i] - (h_inf0[i] + conv_sigma_h_V[t_prime, i])) / (tau_h[t_prime, i] * DeltaV[t_prime, i])
                            kappa_gNa_V[t, t_prime, i] = (
                                g_K_bar * conv_sigma_m_V[t_prime, i] ** 2
                            ) * (
                                conv_sigma_h_V[t_prime, i] * (
                                    (m_inf[t_prime, i] - m_inf0[i] - conv_sigma_m_V[t_prime, i]) / (tau_m[t_prime, i] * DeltaV[t_prime, i])
                                ) + conv_sigma_m_V[t_prime, i] * (
                                    (h_inf[t_prime, i] - h_inf0[i] - conv_sigma_h_V[t_prime, i]) / (tau_h[t_prime, i] * DeltaV[t_prime, i])
                                )
                            )

                        Chi_K[t, t_prime, i] = -np.exp(-(t - t_prime) / tau_i[i]) * (V[t_prime, i] - E_K) / C_m
                        Chi_Na[t, t_prime, i] = -np.exp(-(t - t_prime) / tau_i[i]) * (V[t_prime, i] - E_Na) / C_m

                    # Convolution for sigma
                    conv_sigma_n_V[t, i] = np.convolve(sigma_n_V[t, :t + 1, i], DeltaV[:t + 1, i], mode='valid')[0]
                    conv_sigma_m_V[t, i] = np.convolve(sigma_m_V[t, :t + 1, i], DeltaV[:t + 1, i], mode='valid')[0]
                    conv_sigma_h_V[t, i] = np.convolve(sigma_h_V[t, :t + 1, i], DeltaV[:t + 1, i], mode='valid')[0]

                # Chi_i computation
                for t in range(resolution):
                    for t_prime in range(t):
                        Chi_i[t, t_prime, i] = np.convolve(
                            Chi_K[t, t_prime:t + 1, i], kappa_gK_V[t_prime:t + 1, t_prime, i], mode='valid'
                        )[0] + np.convolve(
                            Chi_Na[t, t_prime:t + 1, i], kappa_gNa_V[t_prime:t + 1, t_prime, i], mode='valid'
                        )[0]
                    conv_chi_V_i[t, i] = np.convolve(Chi_i[t, :t + 1, i], DeltaV[:t + 1, i], mode='valid')[0]

        # Nonequilibrium interaction Green Functions
        # Iterative method to approximate Chi_s and gamma
        for itr in range(10):
            for j in range(N):
                for i in range(j, N):
                    for t in range(resolution):
                        for t_prime in range(t - 1):
                            if V[t_prime, j] == V0[j]:
                                Chi_s[t, t_prime, i, j] = 0.0
                            else:
                                Chi_s[t, t_prime, i, j] = (
                                    sigma0[t, t_prime, i, j] / d_alpha_S(V0[j], beta[i, j], V_th[i, j])
                                ) * (
                                    (alpha_S(V[t_prime, j], beta[i, j], V_th[i, j]) - alpha_S(V0[j], beta[i, j], V_th[i, j])) / DeltaV[t_prime, j]
                                ) * (1 - conv_Chi_s_V[t_prime, i, j] / (1 - gS0[i, j]))

                    for t in range(resolution):
                        conv_Chi_s_V[t, i, j] = np.convolve(Chi_s[t, :t + 1, i, j], DeltaV[:t + 1, j], mode='valid')[0]

                    for t in range(resolution):
                        for t_prime in range(t - 1):
                            gamma[t, t_prime, i, j] = gg0[t, t_prime, i, j] + np.convolve(
                                gs0[t, t_prime:t + 1, i, j],
                                (1 - (DeltaV[t_prime:t + 1, i] / (Es[i, j] - V0[i]))) * Chi_s[t_prime:t + 1, t_prime, i, j],
                                mode='valid'
                            )[0]

        return Chi_i, gamma
            
            
def sum_conv_g_ij_delta_x_j(DeltaX, *params):
    """
    Estimate x_i based on the convolution of g_ij and delta_x_j.
    
    Args:
    - nonlinear_term (numpy array): Array containing the nonlinear terms, combination of DeltaX_j and DeltaX_i states.
    - params (list): List of parameters to be optimized.
    
    Returns:
    - estimated_x_i_from_convolutions (numpy array): Array of estimated x_i values.
    """
    
    N = 4
    C_m = 1.0          
    g_Na_bar = 120.0       
    g_K_bar = 36.0       
    g_L_bar = 0.5        
    E_Na = 50.0          
    E_K = -77.0            
    E_L = -55.0  
    Es = np.zeros((N, N))
    a_r = np.full((N, N), 5.0)
    a_d = np.full((N, N), 5.0) 
    beta = np.full((N, N), 0.125)
    V_th = np.full((N, N), -50.0)  # mV
    
    resolution = 1000
    t_eval = np.linspace(0, 30, resolution)
    
    A = np.reshape(params[0:(N*N)], (N, N))  
    B = np.reshape(params[N*N:], (N, N))  
    
    Chi, gamma = NEGF_HH_model(resolution, N, DeltaV, n_inf, n_inf0, tau_n, m_inf, m_inf0, tau_m, h_inf, h_inf0, tau_h, g_K_bar, V, E_K, E_Na, C_m, tau_i, alpha_S, d_alpha_S, V0, beta, V_th, gS0, gg0, gs0, Es)

    estimated_x_i_from_convolutions = np.zeros((N, resolution))
    for t in range(1, resolution):
        for i in range(N):
            A[i, i] = 0.0 
            for j in range(N):
                estimated_x_i_from_convolutions[i, t] +=  conv(A[i, j] * np.exp(-(t_eval[t] - t_eval[:t])), DeltaX[(i), :t], t_eval[:t])
       
    return estimated_x_i_from_convolutions.flatten()

# Example usage
if __name__ == "__main__":
    
    original_params = np.array([])
    # Neuron Parameters
    C_m = 1.0              # Membrane capacitance (uF/cm^2)
    g_Na_bar = 120.0       # Maximum Na+ conductance (mS/cm^2)
    g_K_bar = 36.0         # Maximum K+ conductance (mS/cm^2)
    g_L_bar = 0.5          # Leakage conductance (mS/cm^2)
    E_Na = 50.0            # Nernst reversal potential for Na+ (mV)
    E_K = -77.0            # Nernst reversal potential for K+ (mV)
    E_L = -55.0            # Leakage reversal potential (mV)

    # Synapses parameters
    N = 4  # Number of neurons
    
    np.append(original_params,[N, C_m, g_Na_bar, g_K_bar, g_L_bar, E_Na, E_K, E_L])

    # Coupling matrices (Adjacency matrices)
    # A: Gap junction coupling matrix (S/F)
    A = np.array([[0.0, 0.0, 0.0, 0.0], 
                  [0.1, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.1, 0.0]])  # No coupling in gap junctions
    np.append(original_params, A.flatten)
    
    # B: Chemical synapse coupling matrix (S/F)
    B = np.array([[0.0, 0.0, 0.0, 0.0], 
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.5, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]])  # Chemical synapse coupling
    np.append(original_params, B.flatten)
    
    # Synapse Nernst potentials (Es) matrix
    Es = np.zeros((N, N))  # Nernst potential (mV) for synaptic coupling (can modify values based on specific neurons)
    np.append(original_params, Es.flatten)
    
    # Synaptic channel conductance activity rates
    a_r = np.full((N, N), 5.0)  # Synaptic activation rate
    np.append(original_params, a_r.flatten)
    a_d = np.full((N, N), 5.0)  # Synaptic deactivation rate
    np.append(original_params, a_d.flatten)
    
    # Beta parameter matrix for synapse
    beta = np.full((N, N), 0.125)
    np.append(original_params, beta.flatten)
    
    # Threshold potential matrix for synapse (V_th)
    V_th = np.full((N, N), -50.0)  # mV
    np.append(original_params, V_th.flatten)
    
    # External current and stimulus
    I_ext = 0.0  # External current in μA/cm^2
    np.append(original_params, I_ext)
    stim = 8.0   # Stimulus current in μA/cm^2

    # Optional modifications:
    # To introduce specific values for inhibition or excitation (e.g., for Es, V_th, a_r):
    # Es[2, 1] = -70.0  # Inhibition for neuron 2 to neuron 1 (synaptic reversal potential)
    # a_r[2, 1] = 1.0   # Adjust rate for specific synapse
    # V_th[2, 1] = -10.0  # Specific threshold potential for neuron 2 to neuron 1

    X0 = np.random.rand(N)                                  # Initial random state for nodes
    resolution = 1000
    tolerance = 1e-6                                        # tol for equilibrium values estimation
    max_iterations = 200
    t_span = (0, 30)                                        # Time span for integration
    t_eval = np.linspace(t_span[0], t_span[1], resolution)  # Evaluation points for the solution

    original_params = N, C_m, g_Na_bar, g_K_bar, g_L_bar, E_Na, E_K, E_L, A, B, Es, a_r, a_d, beta, V_th, I_ext
    
    original_params_stacked = np.hstack(original_params)
    
    # Find equilibrium
    X0 = np.random.rand(N, N+4).flatten()  # Initial random state for nodes
    for iteration in tqdm(range(max_iterations), desc="Iterating"):
        sol = solve_ivp(hh_model, t_span, X0, args=(original_params, 0.0), t_eval=t_eval, method='RK45')
        change = np.linalg.norm(sol.y[:, -1] - sol.y[:, -2])
        if change < tolerance:
            break
        else:
            X0 = sol.y[:, -1]  # Update initial condition
    Xeq = X0.reshape((N, 4 + N))[:, 0]  # Final equilibrium state

    sol = solve_ivp(hh_model, t_span, X0, args=(original_params, stim), t_eval=t_eval, method='RK45')
    
    DeltaX = sol.y.reshape((N, 4 + N, resolution))[:, 0, :] +  np.random.normal(0, 1.0, (N, resolution)) - np.tile(Xeq.reshape(-1, 1), (1, len(sol.t)))  # Add random fluctuation

    plt.figure(figsize=(10, 6))
    for i in range(N):
        plt.plot(sol.t, DeltaX[i], label=f'Node {i+1}')
    plt.title('Time Evolution of the System for Each Node')
    plt.xlabel('Time')
    plt.ylabel('Node State')
    plt.legend()
    plt.grid(True)
        
    plt.show()

    nonlinear_term = np.zeros((N*N, resolution))
    for i in range(N):
        for j in range(N):
            for t_idx in range(resolution):
                nonlinear_term[(i*N+j), t_idx] = (1 - Xeq[i])*(1 - (DeltaX[i, t_idx] / (1 - Xeq[i])))*DeltaX[j, t_idx]
    
    # Initial guess for matrix A        
    A_guess = np.array([[0.0, 0.5, 0.5, 0.5],  
                        [0.5, 0.0, 0.5, 0.5],
                        [0.5, 0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.5, 0.0]])
    B_guess = np.array([[0.0, 0.5, 0.5, 0.5],  
                        [0.5, 0.0, 0.5, 0.5],
                        [0.5, 0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.5, 0.0]])

    #guess_params = C_m, g_Na_bar, g_K_bar, g_L_bar, E_Na, E_K, E_L, N, A_guess, B_guess, Es, a_r, a_d, beta, V_th, I_ext
    
    # Fit parameters using observed values
    fitted_params, pcov = curve_fit(sum_conv_g_ij_delta_x_j, DeltaX.flatten(), DeltaX.flatten(), p0=np.hstack((A_guess.flatten(), B_guess.flatten())), bounds=(np.zeros((2*N*N)), np.ones((2*N*N))))

    fitted_curve = sum_conv_g_ij_delta_x_j(nonlinear_term, fitted_params)
    
    A_fitted = np.reshape(fitted_params[0:N*N], (N, N)) 
    print("Fitted parameters for the network:")
    print(A_fitted)


line_colors = ['b', 'g', 'r', 'c']  # Blue, Green, Red, Cyan for line fits

plt.figure(figsize=(10, 24))  # Adjusted height for better visibility in 4x1 layout

# Creating subplots
for i in range(N):
    plt.subplot(N, 1, i + 1)  # Create a 4x1 grid of subplots, selecting the (i+1)th one
    # Plot fitted curves for the current node
    plt.plot(t_eval, fitted_curve[i * resolution:(i + 1)*resolution], label=f'Fit', color=line_colors[i])
    # Scatter plot for the observed data for the current node
    plt.scatter(t_eval, DeltaX[i, :], label=f'Observed', color='gray', alpha=0.6)  # Adjusting index for observed data
    plt.ylabel(f'Node {i + 1} State')

    # Setting titles and labels
    plt.legend()
    plt.grid(True)

plt.xlabel('Time')
plt.show()


