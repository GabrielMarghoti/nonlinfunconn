import numpy as np
from typing import Optional, Tuple, Union

from nonlinfunconn import convolution
from ..utils.nontt_conv import  nontt_conv
from ..utils.expandtoarray import expandtoarray


class LIF:
    """
    Leaky Integrate-and-Fire (LIF) model class.
    Defines the default parameters and any internal functions needed to simulate or compute Green's functions.
    """
    
    def __init__(self, num_nodes: int, params: dict):
        """
        Initialize the LIF model with parameters passed as a dictionary.
        The first argument is the number of nodes (neurons) in the network.
        The second argument is a dictionary of parameters. The keys in the dictionary should match the expected parameter names.
        The parameters are:
        - V0: Initial membrane potential, can be assumed to be the equilibrium
        - S0: Initial synaptic state, as default we set as the equilibrium value
        - C: Membrane capacitance
        - gamma_g: Conductance for gap junctions
        - gamma_s: Synaptic decay for chemical synapses
        - gamma: Membrane potential decay rate
        - beta: Inverse synaptic timescale
        - Vth: Threshold potential for spiking
        - E_c: Equilibrium membrane potential
        - E_s: Synaptic reversal potential (default value is excitatory)
        - a_r: Synaptic rise time constant
        - a_d: Synaptic decay time constant

        The parameters are used to set up the model. If a parameter is not provided, a default value is used.
        The parameters are expanded to the appropriate shape based on the number of neurons in the network.
        """
        # Default parameters
        default_params = {
            "C": 1000,       # Membrane capacitance  [pF]
            "gamma_g": 100,  # Conductance for gap juctions [pS]
            "gamma_s": 100,  # Synaptic decay for chemical synapses [pS]
            "gamma": 10,     # Membrane potential decay rate  [pS]
            "beta": 0.125,   # Inverse synaptic timescale   [mV^-1]
            "Vth": None,     # Threshold potential for synapse activation, middle of sigmoig function
            "E_c": -60.0,    # Equilibrium membrane potential [mV]
            "E_s": 0.0,      # Synaptic reversal potential (default value is excitatory)
            "a_r": 1.0,      # Synaptic rise time constant
            "a_d": 5.0,      # Synaptic decay time constant
        }

        self.parameters = default_params.copy()  # Copy default parameters to instance variable
        # Update default parameters with provided ones
        self.parameters.update(params)

        # Assign parameters to instance variables
        self.num_neurons = num_nodes

        V0      = None  #FIXME
        S0      = None  #FIXME

        Vth     = self.parameters["Vth"]

        # Expand parameters to appropriate shapes
        self.C       = expandtoarray(self.parameters["C"]      ,  (num_nodes))
        self.gamma   = expandtoarray(self.parameters["gamma"]  ,  (num_nodes))
        self.E_c     = expandtoarray(self.parameters["E_c"]    ,  (num_nodes))
        self.gamma_g = expandtoarray(self.parameters["gamma_g"], (num_nodes, num_nodes))
        self.gamma_s = expandtoarray(self.parameters["gamma_s"], (num_nodes, num_nodes))
        self.E_s     = expandtoarray(self.parameters["E_s"]    , (num_nodes, num_nodes))
        self.beta    = expandtoarray(self.parameters["beta"]   , (num_nodes, num_nodes))
        self.a_r     = expandtoarray(self.parameters["a_r"]    , (num_nodes, num_nodes))
        self.a_d     = expandtoarray(self.parameters["a_d"]    , (num_nodes, num_nodes))
        
        # find Vth as the equilibrium value, so the chemical synapse as term phi = 0.5, half oppened channels
        _Veq, _Seq = self.find_eq_self_consistent() # self.find_equilibrium(np.zeros((num_neurons)))
        if Vth is None:
            Vth = _Veq
        self.Vth = expandtoarray(Vth, (num_nodes, num_nodes))

        self.parameters.update({'Vth': self.Vth})

        # the first assumption for initial conditions is the equilibrium values
        if V0 == None:
            self.V0 = _Veq
        else:
            self.V0 = V0
        
        if S0 == None:
            self.S0 = expandtoarray(_Seq, (num_nodes, num_nodes))
        else:
            self.S0 = expandtoarray(S0, (num_nodes, num_nodes))

        self.parameters.update({
            "C": self.C,       # Membrane capacitance  [pF]
            "gamma_g": self.gamma_g,  # Conductance for gap juctions [pS]
            "gamma_s": self.gamma_s,  # Synaptic decay for chemical synapses [pS]
            "gamma": self.gamma,     # Membrane potential decay rate  [pS]
            "beta": self.beta,   # Inverse synaptic timescale   [mV^-1]
            "Vth": self.Vth,     # Threshold potential for spiking
            "E_c": self.E_c,    # Equilibrium membrane potential [mV]
            "E_s": self.E_s,      # Synaptic reversal potential (default value is excitatory)
            "a_r": self.a_r,      # Synaptic rise time constant
            "a_d": self.a_d,      # Synaptic decay time constant
        })


    def Veq_step(self, V, S):
        """
        Compute the updated membrane potentials self-consistently.
        
        Parameters:
        - V: 1D NumPy array of membrane potentials.
        - S: 1D NumPy array of synaptic activations.
        
        Returns:
        - Updated 1D NumPy array of membrane potentials.
        """
        # Reshape V for element-wise operations
        V_exp = np.expand_dims(V, axis=0)  # Shape (1, N)
        
        # Compute new membrane potential
                
        assert np.all(np.isfinite(self.gamma_g)), "NaNs in gamma_g"
        assert np.all(np.isfinite(self.gamma_s)), "NaNs in gamma_s"
        assert np.all(np.isfinite(self.gamma)), "NaNs in gamma"
        assert np.all(np.isfinite(V)), "NaNs in V"
        assert np.all(np.isfinite(V_exp)), "NaNs in V_exp"
        Y = self.E_c \
            - np.sum((self.gamma_s * S / self.gamma) * (V_exp - self.E_s), axis=1) \
            - np.sum((self.gamma_g / self.gamma) * (V_exp - V[:, None]), axis=1)
        return Y

    def find_eq_self_consistent(self, maxit=1000, damp=1e-2, tol=1e-2):
        """
        Find equilibrium membrane potentials using an iterative self-consistent method.

        Parameters:
        - maxit: Maximum number of iterations (default: 100000).
        - damp: Initial damping factor for stability (default: 1e-2).
        - tol: Convergence tolerance (default: 1e-2).

        Returns:
        - V: 1D NumPy array of equilibrium membrane potentials.
        - S_eq: 1D NumPy array of synaptic activations at rest.
        """
        # Compute steady-state synaptic activation (precomputed value)
        S_eq = 0.5 * self.a_r / (0.5 * self.a_r + self.a_d)
        
        # Initialize membrane potential
        V = np.copy(self.E_c)
        
        for i in range(maxit):
            V_old = V.copy()
            
            # Compute new potential with damping factor
            V_new = self.Veq_step(V_old, S_eq)
            V_new = np.nan_to_num(V_new, nan=0.0, posinf=1e10, neginf=-1e10)
            V = V_old + damp * (V_new - V_old)
            V = np.nan_to_num(V, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # Check for convergence
            dV = np.linalg.norm(V - V_old, ord=1) / np.linalg.norm(V_old, ord=1)
            if dV < tol:
                break
            
            # Adaptive damping: Increase if changes are small, decrease if unstable
            if i % 1000 == 0:
                damp = min(0.01, damp * 1.1)  # Gradually increase damping for stability
        
        else:
            print(f"Warning: Did not converge within {maxit} iterations.")
            V = np.copy(self.E_c)

        return V, S_eq

    def heaviside(self, t: np.ndarray) -> np.ndarray:
        """
        Heaviside step function.

        Parameters:
            t (np.ndarray): Input array.

        Returns:
            np.ndarray: Heaviside step function applied to the input.
        """
        return np.where(t >= 0, 1.0, 0.0)

    
    def synaptic_activation(self, V: np.ndarray, beta: np.ndarray, Vth: np.ndarray) -> np.ndarray:
        """
        Compute the synaptic activation function.

        Parameters:
            V (np.ndarray): Membrane potential.
            beta (np.ndarray): Synaptic activation steepness.
            Vth (np.ndarray): Threshold potential.

        Returns:
            np.ndarray: Synaptic activation values.
        """
        return 1 / (1 + np.exp(-beta * (V - Vth)))

    def d_synaptic_activation(self, V: np.ndarray, beta: np.ndarray, Vth: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the synaptic activation function.

        Parameters:
            V (np.ndarray): Membrane potential.
            beta (np.ndarray): Synaptic activation steepness.
            Vth (np.ndarray): Threshold potential.

        Returns:
            np.ndarray: Derivative of the synaptic activation function.
        """
        exp_term = np.exp(-beta * (V - Vth))
        return (beta * exp_term) / (1 + exp_term) ** 2

    def compute_direct_equilibrium_green_functions(self, time_len=None, dt=None, p=None):
        # Update class attributes 
        if p is not None:
            for key, value in p.items():
                if key in self.parameters:
                    setattr(self, key, expandtoarray(value, getattr(self, key).shape))

        self.time_len = time_len if time_len is not None else self.time_len
        self.dt = dt if dt is not None else self.dt
        
        green_functions_shape = (self.num_neurons, self.num_neurons, self.time_len, self.time_len)

        # Initialize Green's function arrays
        self.sigma0 = np.zeros(green_functions_shape)
        self.gg0 = np.zeros(green_functions_shape)
        self.gs0 = np.zeros(green_functions_shape)
        self.g0 = np.zeros(green_functions_shape)

        self.ts = np.arange(0, self.time_len * self.dt, self.dt)
        ts_diff =  self.ts[:, np.newaxis] - self.ts # (time_len, time_len)

        heaviside_func = self.heaviside(ts_diff)

        # Precompute gamma_sum[i] = scalar per neuron
        gamma_sum = (self.gamma / self.C) \
            + np.sum((self.gamma_g / self.C[:, None]), axis=1) \
            + np.sum((self.gamma_s / self.C[:, None]) * self.S0, axis=1)
        
        for i in range(self.num_neurons):
            arg = -ts_diff * gamma_sum[i]
            arg = np.clip(arg, -500, 500)  # Cap the range
            exp_factor_gs_gg = np.where(ts_diff > 0, np.exp(arg), 0)

            for j in range(self.num_neurons):
                if i == j or (self.gamma_g[i, j] == 0 and self.gamma_s[i, j] == 0):
                    continue  # Skip self-interaction & null kernels

                # Fetch parameters
                a_r, a_d, beta = self.a_r[i, j], self.a_d[i, j], self.beta[i, j]
                Veq, Vth = self.V0[j], self.Vth[i, j]

                # Synaptic exponential kernel
                exp_factor_synaptic = np.where(heaviside_func > 0,np.exp(-ts_diff * (a_d - a_r / (1 + np.exp(-beta * (Veq - Vth))))), 0)

                # Synaptic kernel amplitude
                synaptic_factor = a_r * (1 - self.S0[i, j]) * self.d_synaptic_activation(Veq, beta, Vth)

                # Assign kernels
                self.sigma0[i, j] = synaptic_factor * np.where(heaviside_func > 0, exp_factor_synaptic, 0)
                self.gg0[i, j] = (self.gamma_g[i, j] / self.C[i]) * np.where(heaviside_func > 0, exp_factor_gs_gg, 0)
                self.gs0[i, j] = (self.gamma_s[i, j] / self.C[i]) * (self.E_s[i, j] - self.V0[i]) * np.where(heaviside_func > 0,exp_factor_gs_gg  , 0)   
                 
                self.g0 = self.gg0 + nontt_conv(self.gs0[i, j], self.sigma0[i, j], self.dt)

        return self.g0

    def compute_direct_green_functions(self, Vs,  dt = None, neu_i=None, neu_j=None, iteration_index_MAX=4, p=None, return_estimated_V=False):
        """
        Compute the nonequilibrium Green's functions for the LIF network.

        Parameters:
            dt (float): Time step for simulation.
            Vs (np.ndarray): Membrane potential dynamics (time series).
            iteration_index_MAX (int): Max iterations for Neumann series.
            return_estimated_V (bool): If True, also return estimated voltages.
        """
        
        num_neurons, time_len = Vs.shape
        
        dt = dt if dt is not None else self.dt
        
        # Update class attributes 
        if p is not None:
            for key, value in p.items():
                setattr(self, key, expandtoarray(value, getattr(self, key).shape))

        V0 = Vs[:, 0]  # first time point for each neuron
        delta_Vs = Vs - V0[:, None]  # (neurons, time)

        # SHAPE: (num_neurons, num_neurons, time_len, time_len)
        green_shape = (num_neurons, num_neurons, time_len, time_len)

        sigma = np.zeros(green_shape)
        pi = np.zeros(green_shape)
        g = np.zeros(green_shape)

        delta_Ss = np.zeros((num_neurons, num_neurons, time_len))

        self.compute_direct_equilibrium_green_functions(time_len, dt)

        for i in range(num_neurons):
            for j in range(num_neurons):
                if self.gamma_g[i, j] == 0 and self.gamma_s[i, j] == 0:
                    continue  # Skip non-connected neurons

                synaptic_diff = np.zeros_like(delta_Vs[j])
                non_zero = np.abs(delta_Vs[j]) >= 1e-4

                synaptic_diff[non_zero] = (
                    self.synaptic_activation(Vs[j, :], self.beta[i, j], self.Vth[i, j]) - 
                    self.synaptic_activation(V0[j, None], self.beta[i, j], self.Vth[i, j])
                )[non_zero] / delta_Vs[j][non_zero]
                
                prev_delta_S = np.copy(delta_Ss[i, j])

                syn_act_j = self.d_synaptic_activation(V0[j], self.beta[i, j], self.Vth[i, j])
                for _ in range(iteration_index_MAX):
                    div_factor = (1 - (delta_Ss[i, j] / (1 - self.S0[i, j])))
                    valid_indexes = np.abs(div_factor) >= 1e-4
                    sigma[i, j][:, valid_indexes] = (
                        self.sigma0[i, j][:, valid_indexes] / syn_act_j
                    ) * (synaptic_diff * div_factor)[None, valid_indexes]

                    # Update delta_Ss
                    delta_Ss[i, j] = nontt_conv(sigma[i, j], delta_Vs[j], dt)
                    if np.all(np.abs(delta_Ss[i, j] - prev_delta_S) < 1e-3):
                        break

                    prev_delta_S = np.copy(delta_Ss[i, j])

                # Compute π and g Green functions
                factor   = 1 - (delta_Vs[i] / (self.E_s[i, j] - V0[i]))
                pi[i, j] = nontt_conv(self.gs0[i, j], factor[None, :] * sigma[i, j], dt)
                g[i, j]  = self.gg0[i, j] + pi[i, j]

            
        if return_estimated_V:
            est_V = np.zeros_like(Vs)
            for i in range(num_neurons):
                est_V[i, :] += V0[i]
                for j in range(num_neurons):
                    est_V[i] += nontt_conv(g[i, j], delta_Vs[j], dt)

            return g, est_V

        return g
