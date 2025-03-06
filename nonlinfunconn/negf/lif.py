import numpy as np
from scipy.optimize import minimize, least_squares
from typing import Optional, Tuple, Union
from nonlinfunconn import convolution
from ..utils.irrarray import irrarray

import numpy as np
from typing import Union, Tuple

import numpy as np
from typing import Union, Tuple

class LIF:
    """
    Leaky Integrate-and-Fire (LIF) neural network model with Green's function computation.
    """

    def __init__(
        self,
        dt: float = 1.0,
        num_neurons: int = None,
        Veq: Union[float, np.ndarray] = 0.0,  # Equilibrium membrane potential
        Seq: Union[float, np.ndarray] = 0.0,  # Equilibrium synaptic state
        Vs: np.ndarray = None,  # Membrane potential dynamics (time series)
        Ss: np.ndarray = None,  # Synaptic state dynamics (time series)
        gamma_g: Union[float, np.ndarray] = 10.0,  # Conductance decay rate
        gamma_s: Union[float, np.ndarray] = 10.0,  # Synaptic decay rate
        gamma: Union[float, np.ndarray] = 10.0,  # Membrane potential decay rate
        beta: Union[float, np.ndarray] = 125,  # Inverse synaptic timescale
        V_th: Union[float, np.ndarray] = 0.0,  # Threshold potential for spiking
        E_c: Union[float, np.ndarray] = 0.0,  # Equilibrium membrane potential
        E_s: Union[float, np.ndarray] = 0.0,  # Synaptic reversal potential
        a_r: Union[float, np.ndarray] = 1.0,  # Synaptic rise time constant
        a_d: Union[float, np.ndarray] = 5.0,  # Synaptic decay time constant
        C: Union[float, np.ndarray] = 1.0,  # Membrane capacitance
    ):
        """
        Initialize the LIF model.

        Parameters:
        - dt: Time step for simulation.
        - num_neurons: Number of neurons in the network.
        - Veq: Equilibrium membrane potential (scalar or array).
        - Seq: Equilibrium synaptic state (scalar or array).
        - Vs: Predefined membrane potential dynamics (or initialized to Veq).
        - Ss: Predefined synaptic state dynamics (or initialized to Seq).
        - gamma_g: Conductance decay rate.
        - gamma_s: Synaptic decay rate.
        - gamma: Membrane potential decay rate.
        - beta: Inverse synaptic timescale.
        - V_th: Neuronal firing threshold.
        - E_s: Synaptic reversal potential.
        - E_c: Leakege potential equilibrium potential.
        - a_r: Synaptic rise time constant.
        - a_d: Synaptic decay time constant.
        - C: Membrane capacitance.
        """
        self.dt = dt
        self.num_neurons = num_neurons

        if self.num_neurons is None:
            raise ValueError("num_neurons must be specified.")

        # Expand scalar parameters to arrays
        self.Veq = self._expand_to_array(Veq, num_neurons)  # Equilibrium potential
        self.Seq = self._expand_to_array(Seq, (num_neurons, num_neurons))  # Synaptic state equilibrium

        # Initialize membrane potential and synaptic state arrays
        self.Vs = Vs if Vs is not None else np.full((10, num_neurons), self.Veq)  # Default resolution = 10
        self.resolution = self.Vs.shape[0]  # Resolution determined by Vs shape

        self.Ss = Ss if Ss is not None else np.full((self.resolution, num_neurons, num_neurons), self.Seq)

        # Compute deviations from equilibrium states
        self.delta_Vs = self.Vs - self.Veq[None, :]
        self.delta_Ss = self.Ss - self.Seq[None, :, :]

        # Expand parameters for all neurons
        # Gap junction conductance
        self.gamma_g = self._expand_to_array(gamma_g, (num_neurons, num_neurons))

        # Chemmical synapse parameters
        self.gamma_s = self._expand_to_array(gamma_s, (num_neurons, num_neurons))
        self.beta = self._expand_to_array(beta, (num_neurons, num_neurons))
        self.V_th = self._expand_to_array(V_th, (num_neurons, num_neurons))
        self.E_s = self._expand_to_array(E_s, (num_neurons, num_neurons))
        self.a_r = self._expand_to_array(a_r, (num_neurons, num_neurons))
        self.a_d = self._expand_to_array(a_d, (num_neurons, num_neurons))

        # One compartiment model cell parameters
        self.gamma = self._expand_to_array(gamma, num_neurons)  # Membrane potential decay rate
        self.E_c = self._expand_to_array(E_c, num_neurons) 
        self.C = self._expand_to_array(C, num_neurons)  # Capacitance

        # Initialize Green's function arrays
        self.sigma_0 = np.zeros((self.resolution, self.resolution, num_neurons, num_neurons))
        self.gg_0 = np.zeros_like(self.sigma_0)
        self.gs_0 = np.zeros_like(self.sigma_0)
        self.g_0 = np.zeros_like(self.sigma_0)
        self.sigma = np.zeros_like(self.sigma_0)
        self.pi = np.zeros_like(self.sigma_0)
        self.g = np.zeros_like(self.sigma_0)

    def _expand_to_array(self, value: Union[float, np.ndarray], shape: Tuple[int, ...]) -> np.ndarray:
        """
        Expand a scalar value to an array of the given shape, or validate an existing array.
        """
        if np.isscalar(value):
            return np.full(shape, value)
        elif isinstance(value, np.ndarray):
            if value.shape == shape:
                return value
            else:
                raise ValueError(f"Expected shape {shape}, but got {value.shape}")
        else:
            raise TypeError(f"Expected scalar or numpy array, but got {type(value)}")

    def heaviside(self, t: np.ndarray) -> np.ndarray:
        """
        Heaviside step function.

        Parameters:
            t (np.ndarray): Input array.

        Returns:
            np.ndarray: Heaviside step function applied to the input.
        """
        return np.where(t >= 0, 1.0, 0.0)

    def update_V(self,  V: np.ndarray):
        """
        Update the membrane potential and synaptic state dynamics.

        Parameters:
            V (np.ndarray): Membrane potential.
        """
        # Compute synaptic activation
        self.V = V
    
    def synaptic_activation(self, V: np.ndarray, beta: np.ndarray, V_th: np.ndarray) -> np.ndarray:
        """
        Compute the synaptic activation function.

        Parameters:
            V (np.ndarray): Membrane potential.
            beta (np.ndarray): Synaptic activation steepness.
            V_th (np.ndarray): Threshold potential.

        Returns:
            np.ndarray: Synaptic activation values.
        """
        return 1 / (1 + np.exp(-beta * (V - V_th)))

    def d_synaptic_activation(self, V: np.ndarray, beta: np.ndarray, V_th: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the synaptic activation function.

        Parameters:
            V (np.ndarray): Membrane potential.
            beta (np.ndarray): Synaptic activation steepness.
            V_th (np.ndarray): Threshold potential.

        Returns:
            np.ndarray: Derivative of the synaptic activation function.
        """
        exp_term = np.exp(-beta * (V - V_th))
        return (beta * exp_term) / (1 + exp_term) ** 2

    def compute_direct_negf_eq(self, dt, resolution,):
        """
        Compute the equilibrium Green's functions for the LIF network.

        Parameter:
            degree_max (int): Maximum number of nodes that compromise a path for signal propagation.
        """
        self.resolution = resolution
        self.dt = dt

        time_diff = np.arange(self.resolution)[:, None] - np.arange(self.resolution)
        heaviside_diff = self.heaviside(time_diff)

        # Vectorized computation of exp_term
        exp_term = np.exp(-time_diff[..., None, None] * 
                          (self.a_d[None, None, ...] - 
                           self.a_r[None, None, ...] / 
                           (1 + np.exp(-self.beta[None, None, ...] * 
                            (self.Veq[None, None, ...] - self.V_th[None, None, ...])))))

        # Compute sigma_0
        self.sigma_0 = (heaviside_diff[..., None, None] * 
                        self.a_r[None, None, ...] * 
                        (1 - self.Seq[None, None, ...]) * 
                        self.d_synaptic_activation(self.Veq[None, None, ...], 
                                                  self.beta[None, None, ...], 
                                                  self.V_th[None, None, ...]) * 
                        exp_term)

        # Compute gg_0 and gs_0
        exp_term_gg = np.exp(-time_diff[..., None, None] * 
                       (self.gamma[None, None, ...] + 
                        np.sum(self.gamma_g[None, None, ...], axis=-1) + 
                        np.sum(self.gamma_s[None, None, ...] * self.Seq[None, None, ...], axis=-1)))

        self.gg_0 = heaviside_diff[..., None, None] * self.gamma_g[None, None, ...] * exp_term_gg
        self.gs_0 = heaviside_diff[..., None, None] * self.gamma_s[None, None, ...] * (self.Es[None, None, ...] - self.Veq[None, None, ...]) * exp_term_gg

        # Compute g_0 using convolution
        for t in range(self.resolution):
            for t_prime in range(t):
                self.g_0[t, t_prime] = self.gg_0[t, t_prime] + convolution(self.gs_0[t, t_prime:t], self.sigma_0[t_prime:t, t_prime], self.dt, 8)
        return self.g_0

    def compute_direct_negf(self, dt, resolution,
        Vs: np.ndarray = None,
        iteration_index_MAX = 4
        ):
        """
            Compute the nonequilibrium Green's functions for the LIF network.

            Parameter:
                n_neigh_max (int): Maximum number of neighbors for fitting the effective NEGF for nodes not direct connected.
        """
        self.resolution = resolution
        self.dt = dt
        self.Vs = Vs
        self.delta_Vs = self.Vs - self.Veq[None, :]

        self.Ss = np.full((self.resolution, self.num_neurons, self.num_neurons), self.Seq) # Initialize synaptic state dynamics for iterative approximation
        self.delta_Ss = self.Ss - self.Seq[None, :, :]

        for _ in range(iteration_index_MAX):  # Iterative approximation for Neumann series approximation
            for i in range(self.num_neurons):
                for j in range(self.num_neurons):
                    for t in range(self.resolution):
                        if self.Vs[t, j] == self.Veq[j]:
                            self.sigma[t, :, i, j] = 0.0
                        else:
                            synaptic_diff = (self.synaptic_activation(self.Vs[t, j], self.beta[i, j], self.V_th[i, j]) - 
                                             self.synaptic_activation(self.Veq[j], self.beta[i, j], self.V_th[i, j])) / self.delta_Vs[t, j]
                            self.sigma[t, :, i, j] = (self.sigma_0[t, :, i, j] / 
                                                      self.d_synaptic_activation(self.Veq[j], self.beta[i, j], self.V_th[i, j]) * 
                                                      synaptic_diff * (1 - (self.Ss[t, i, j] / (1 - self.Seq[i, j]))))

                        self.Ss[t, i, j] = convolution(self.sigma[t, :t, i, j], self.delta_Vs[:t, j], self.dt, 8)

                        for t_prime in range(t):
                            self.pi[t, t_prime, i, j] = convolution(self.gs_0[t, t_prime:t, i, j], 
                                                                   (1 - (self.delta_Vs[t_prime:t, i] / (self.Es[i, j] - self.Veq[i]))) * 
                                                                   self.sigma[t_prime:t, t_prime, i, j], self.dt, 8)
                            self.g[t, t_prime, i, j] = self.gg_0[t, t_prime, i, j] + self.pi[t, t_prime, i, j]
        return self.g

def non_translational_conv(self, K1, K2, dt=None):
    """
    Computes a non-translational convolution between two kernel matrices.

    Parameters:
    - K1: np.ndarray, first kernel matrix
    - K2: np.ndarray, second kernel matrix
    - dt: float, optional time step (defaults to self.dt)

    Returns:
    - np.ndarray: Output convolution matrix
    """
    if dt is None:
        dt = self.dt

    out = np.zeros_like(K1)  # Fix: Should be based on K1, not self.K1

    for t in range(self.resolution):
        for t_prime in range(t):  # Fix: Prevents accessing out-of-bounds indices
            out[t, t_prime] = convolution(K1[t, t_prime:t], K2[t_prime:t, t_prime], dt, 8)  # Fix slicing
    return out


def compute_effective_negf(self, gf_order_max: int = 2):
    """
    Computes the effective non-equilibrium Green's function (NEGF) up to a specified order.

    Parameters:
    - gf_order_max: int, maximum order of Green's function iterations

    Returns:
    - np.ndarray: Effective Green's function matrix
    """
    self.resolution = self.g.shape[0]  # Ensure resolution is set properly

    G = np.copy(self.g)  # First-order Green's function

    for path_len in range(1, gf_order_max + 1):  # Fix: Starts from 1 for path contributions
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i == j:
                    continue  # Skip self-connections
                if self.gamma_g[i, j] == 0 and self.gamma_s[i, j] == 0:
                    continue  # Skip if no interaction
                
                for k in range(self.num_neurons):
                    if k == j or k == i:
                        continue  # Avoid self-loops
                    if (self.gamma_g[i, k] == 0 and self.gamma_s[i, k] == 0) or \
                       (self.gamma_g[k, j] == 0 and self.gamma_s[k, j] == 0):
                        continue  # Ensure path is valid

                    # Update Green's function iteratively
                    G[:, :, i, j] += self.non_translational_conv(self.g[:, :, i, k], G[:, :, k, j])

    return G


    def eval(self,x,dtype=np.float64,drop_branches=None):
        '''Evaluates the NEGFs in the time domain.
        
        Parameters
        ----------
        t: array_like
            Time axis. All times should be positive.
        dtype: type (optional)
            Type of the output array. Default: np.float64        
        drop_branches: int or array_like of int
            Branches to be ignored in the evaluation. Default: None.
            
        Returns
        -------
        out: numpy.ndarray
            ExponentialConvolution evaluated on x.
        '''
        assert np.all(x>=0)
        
        if drop_branches is not None:
            try: len(drop_branches)
            except: drop_branches = [drop_branches]
        
        out = np.zeros_like(x,dtype=dtype)
            
        for exp in self.exp[-1]:
            # Skip terms that are in excluded branches
            branch = exp["branch"]
            if drop_branches is not None:
                if branch in drop_branches: continue
                
            g = exp["g"]
            factor = exp["factor"]
            power_t = exp["power_t"]
            
            if power_t==0: mult=1.
            else: mult=np.power(x,power_t)
            out += factor*mult*np.exp(-g*x)
            
        return out
    





    @classmethod
    def fit(
        cls,
        signal: np.ndarray,
        dt: float,
        n_neigh_max: int = 2,
        rms_limits: Optional[Tuple[int, int]] = None,
        auto_stop: bool = False,
        rms_tol: float = 1e-2,
        method: Optional[str] = None,
        routine: str = "least_squares",
        p0: Optional[np.ndarray] = None
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit the LIF model to the given signal.

        Parameters:
            signal (np.ndarray): Signal to fit (shape: [resolution, num_neurons]).
            dt (float): Time step.
            n_neigh_max (int): Maximum number of neighbors for fitting.
            rms_limits (Optional[Tuple[int, int]]): Time limits for RMS calculation.
            auto_stop (bool): Whether to stop fitting early if RMS improvement is below tolerance.
            rms_tol (float): Tolerance for early stopping.
            method (Optional[str]): Optimization method for `scipy.optimize`.
            routine (str): Optimization routine ("minimize" or "least_squares").

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Fitted parameters, branch parameters, and residuals.
        """
                
        num_neurons = signal.shape[1]
        resolution = signal.shape[0]
    
        Veq = np.mean(signal, axis=0)
        Seq = np.zeros((num_neurons, num_neurons))
        Vs = np.zeros((resolution, num_neurons))
        delta_Vs = np.zeros_like(Vs)
        delta_Ss = np.zeros((resolution, num_neurons, num_neurons))
        gamma_g = np.zeros((num_neurons, num_neurons))
        gamma_s = np.zeros_like(gamma_g)
        gamma = np.zeros(num_neurons)
        beta = np.zeros_like(gamma_g)
        V_th = np.zeros_like(gamma_g)
        Es = np.zeros_like(gamma_g)
        a_r = np.zeros_like(gamma_g)
        a_d = np.zeros_like(gamma_g)

        lif = cls(num_neurons, dt, Veq, Seq, Vs, delta_Vs, delta_Ss, gamma_g, gamma_s, gamma, beta, V_th, Es, a_r, a_d)
        lif.compute_equilibrium_green_functions()
        lif.compute_nonequilibrium_green_functions()

        if p0 is None:
            p0 = np.random.rand(num_neurons)

        if routine == "minimize":
            error = lambda p, x, y: np.sum(np.power(convolution(x, cls.eci(x, p), dt, 8) - y, 2))
            res = minimize(error, p0, args=(signal, signal), method=method)
        elif routine == "least_squares":
            residuals = lambda p, x, y: convolution(x, cls.eci(x, p), dt, 8) - y
            res = least_squares(residuals, p0, args=(signal, signal), method=method)

        return res.x, None, None
