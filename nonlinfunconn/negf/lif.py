import numpy as np
from scipy.optimize import minimize, least_squares
from typing import Optional, Tuple, Union
from nonlinfunconn import convolution
from ..utils.nontt_conv import  nontt_conv

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
        ts: np.ndarray = None,
        dt: float = 1.0,
        num_neurons: int = None,
        Veq: Union[float, np.ndarray] = None,  # Equilibrium membrane potential
        Seq: Union[float, np.ndarray] = None,  # Equilibrium synaptic state
        Vs: np.ndarray = None,  # Membrane potential dynamics (time series)
        Ss: np.ndarray = None,  # Synaptic state dynamics (time series)
        gamma_g: Union[float, np.ndarray] = 10.0,  # Conductance decay rate
        gamma_s: Union[float, np.ndarray] = 10.0,  # Synaptic decay rate
        gamma: Union[float, np.ndarray] = 10.0,  # Membrane potential decay rate
        beta: Union[float, np.ndarray] = 125,  # Inverse synaptic timescale
        Vth: Union[float, np.ndarray] = None,  # Threshold potential for spiking
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
        - Vth: Neuronal firing threshold.
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

        # Initialize membrane potential and synaptic state arrays
        self.Vs = Vs if Vs is not None else np.full((10, num_neurons), self.Veq)  # Default resolution = 10
        self.resolution = self.Vs.shape[0]  # Resolution determined by Vs shape

        if ts is None:
            self.ts = np.arange(0, self.resolution * self.dt, self.dt)
        else:
            self.ts = ts

        # Expand parameters for all neurons
        # Gap junction conductance
        self.gamma_g = self._expand_to_array(gamma_g, (num_neurons, num_neurons))

        # Chemmical synapse parameters
        self.gamma_s = self._expand_to_array(gamma_s, (num_neurons, num_neurons))
        self.E_s =    self._expand_to_array(E_s, (num_neurons, num_neurons))

        self.beta = self._expand_to_array(beta, (num_neurons, num_neurons))

        self.a_r = self._expand_to_array(a_r, (num_neurons, num_neurons))
        self.a_d = self._expand_to_array(a_d, (num_neurons, num_neurons))

        # One compartiment model cell parameters
        self.gamma = self._expand_to_array(gamma, num_neurons)  # Membrane potential decay rate
        self.E_c = self._expand_to_array(E_c, num_neurons) 
        self.C = self._expand_to_array(C, num_neurons)  # Capacitance

        # find Vth as the equilibrium value, so the chemical synapse as term phi = 0.5, half oppened channels
        _Veq, _Seq = self.find_eq_self_consistent()#self.find_equilibrium(np.zeros((num_neurons)))
        if Vth is None:
            Vth = _Veq
        self.Vth = self._expand_to_array(Vth, (num_neurons, num_neurons))

        if np.all(Veq == None):
            self.Veq = _Veq
        else:
            self.Veq = Veq
        
        if np.all(Seq == None):
            self.Seq = _Seq
        else:
            self.Seq = self._expand_to_array(Seq, (num_neurons, num_neurons))

        self.Ss = Ss if Ss is not None else np.full((self.resolution, num_neurons, num_neurons), self.Seq)

        # Compute deviations from equilibrium states
        self.delta_Vs = self.Vs - self.Veq[None, :]
        self.delta_Ss = self.Ss - self.Seq[None, :, :]

        # Initialize Green's function arrays
        self.sigma0 = np.zeros((self.resolution, self.resolution, num_neurons, num_neurons))
        self.gg0 = np.zeros_like(self.sigma0)
        self.gs0 = np.zeros_like(self.sigma0)
        self.g0 = np.zeros_like(self.sigma0)
        self.sigma = np.zeros_like(self.sigma0)
        self.pi = np.zeros_like(self.sigma0)
        self.g = np.zeros_like(self.sigma0)

        self.g_computed_flag = False
        self.g_eq_computed_flag = False

        self.G = np.zeros_like(self.g)      # Effective Green's function

    def _expand_to_array(self, value: Union[float, np.ndarray], shape: Tuple[int, ...]) -> np.ndarray:   #### This must be an .util method
        """
        Expand a scalar value to an array of the given shape, expand 1D arrays to 2D if necessary,
        or validate an existing array.

        Args:
            value: A scalar, 1D array, or 2D array.
            shape: The target shape as a tuple (e.g., (rows, cols)).

        Returns:
            A NumPy array of the specified shape.

        Raises:
            ValueError: If the input array cannot be expanded to the target shape.
            TypeError: If the input is neither a scalar nor a NumPy array.
        """
        if np.isscalar(value):
            return np.full(shape, value)

        if not isinstance(value, np.ndarray):
            raise TypeError(f"Expected scalar or numpy array, but got {type(value)}")

        if value.shape == shape:
            return value

        if value.ndim == 1:
            out = np.zeros(shape)
            for i in range(shape[0]):
                out[i, :] = value
            return out
        else:
            raise ValueError(
                f"1D array length {value.shape[0]} does not match either dimension of the target shape {shape}"
            )
        
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
        Y = self.E_c \
            - np.sum((self.gamma_s * S / self.gamma) * (V_exp - self.E_s), axis=1) \
            - np.sum((self.gamma_g / self.gamma) * (V_exp - V[:, None]), axis=1)
        
        return Y

    def find_eq_self_consistent(self, maxit=100000, damp=1e-3, tol=5e-4):
        """
        Find equilibrium membrane potentials using an iterative self-consistent method.

        Parameters:
        - maxit: Maximum number of iterations (default: 100000).
        - damp: Initial damping factor for stability (default: 1e-3).
        - tol: Convergence tolerance (default: 5e-5).

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
            V = V_old + damp * (V_new - V_old)
            
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

    def update_V(self,  V: np.ndarray):
        """
        Update the membrane potential and synaptic state dynamics.

        Parameters:
            V (np.ndarray): Membrane potential.
        """
        # Compute synaptic activation
        self.V = V
    
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

    def compute_direct_equilibrium_green_functions(self):
        self.g_eq_computed_flag = True 

        ts_diff = self.ts[:, np.newaxis] - self.ts
        heaviside_func = self.heaviside(ts_diff)

        gamma_sum = self.gamma[:, np.newaxis] + np.sum(self.gamma_g, axis=1)[:, np.newaxis] + np.sum(self.gamma_s * self.Seq, axis=1)[:, np.newaxis]

        for i in range(self.num_neurons):
            if gamma_sum[i] == self.gamma[i]: 
                continue  # Skip unconnected nodes

            exp_factor_gs_gg = np.exp(-ts_diff * gamma_sum[i])

            for j in range(self.num_neurons):
                if i == j or (self.gamma_g[i, j] == 0 and self.gamma_s[i, j] == 0):
                    continue  # Skip self-interaction & null kernels

                a_r, a_d, beta, Veq, Vth = self.a_r[i, j], self.a_d[i, j], self.beta[i, j], self.Veq[j], self.Vth[i, j]
                
                
                exp_factor_synaptic = np.exp(-ts_diff * (a_d - a_r / (1 + np.exp(-beta * (Veq - Vth)))))
                synaptic_factor = a_r * (1 - self.Seq[i, j]) * self.d_synaptic_activation(Veq, beta, Vth)

                self.sigma0[:, :, i, j] = heaviside_func * synaptic_factor * exp_factor_synaptic
                self.gg0[:, :, i, j] = heaviside_func * self.gamma_g[i, j] * exp_factor_gs_gg
                self.gs0[:, :, i, j] = heaviside_func * self.gamma_s[i, j] * (self.E_s[i, j] - self.Veq[i]) * exp_factor_gs_gg

                conv_s = nontt_conv(self.gs0[:, :, i, j], self.sigma0[:, :, i, j], self.dt)
                self.g0[:, :, i, j] = self.gg0[:, :, i, j] + conv_s
                """
                import matplotlib.pyplot as plt
                plt.figure(figsize=(6, 4), dpi=200)
            
                plt.plot(self.ts[-1] - self.ts, self.g0[-1, :, i, j], linewidth=1.4, linestyle='solid')
                
                plt.xlabel("t - t′ (s)")
                plt.ylabel(f"g0 ({i},{j})")
                plt.legend([f"g0 ({i},{j})"], bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)

                plt.grid(False)
                plt.box(True)
                
                plt.show()
                """
            return self.g0

    def compute_direct_negf(self, Vs: np.ndarray = None,  dt : float = 1.0, iteration_index_MAX=5):
        """
        Compute the nonequilibrium Green's functions for the LIF network.

        Parameters:
            dt (float): Time step for simulation.
            resolution (int): Resolution of the simulation.
            Vs (np.ndarray): Membrane potential dynamics (time series).
            iteration_index_MAX (int): Maximum number of iterations for the Neumann series approximation.
        """


        self.g_computed_flag = True
        if Vs != None : 
            self.resolution = Vs.shape[0]
            self.Vs = Vs
        self.dt = dt
        self.delta_Vs = self.Vs - self.Veq[None, :]

        if self.g_eq_computed_flag==False: _ = self.compute_direct_equilibrium_green_functions()

        self.Ss = np.full((self.resolution, self.num_neurons, self.num_neurons), self.Seq)  # Initialize synaptic state dynamics for iterative approximation
        self.delta_Ss = self.Ss - self.Seq[None, :, :]

        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if self.gamma_g[i, j] == 0 and self.gamma_s[i, j]==0: continue # avoid computing null kernell (no connection)
                synaptic_diff = np.zeros_like(self.delta_Vs[:, j])
                non_zero_indices = np.abs(self.delta_Vs[:, j]) >= 0.0001
                synaptic_diff[non_zero_indices] = (
                    self.synaptic_activation(self.Vs[:, j], self.beta[i, j], self.Vth[i, j]) - 
                    self.synaptic_activation(self.Veq[None, j], self.beta[i, j], self.Vth[i, j])
                )[non_zero_indices] / self.delta_Vs[non_zero_indices, j]

                for _ in range(iteration_index_MAX):  # Iterative approximation for self consistent series approximation
                    self.sigma[:, :, i, j] = (
                        self.sigma0[:, :, i, j] / 
                        self.d_synaptic_activation(self.Veq[j]*np.ones((self.resolution)), self.beta[i, j], self.Vth[i, j])[None, :] * 
                        synaptic_diff[None, :] * (1 - (self.delta_Ss[None, :, i, j] / (1 - self.Seq[i, j])))
                    )

                    self.delta_Ss[:, i, j] = nontt_conv(self.sigma[:, :, i, j], self.delta_Vs[:, j], self.dt)

                self.pi[:, :, i, j] = nontt_conv(
                    self.gs0[:, :, i, j], 
                    (1 - (self.delta_Vs[None, :, i] / (self.E_s[i, j] - self.Veq[i]))) *  self.sigma[:, :, i, j], self.dt
                )
                self.g[:, :, i, j] = self.gg0[:, :, i, j] + self.pi[:, :, i, j]

                """ # for debbuging/testing
                import matplotlib.pyplot as plt
                
                plt.plot(self.ts[-1] - self.ts, self.g[-1, :, i, j])
                plt.xlabel('Time')
                plt.ylabel('Green\'s Function')
                plt.title(f'Green\'s Function for Neurons {i} and {j}')
                plt.show()
                """



        self.G = np.copy(self.g)  # First approximation for effective Green's function
        return self.g

    def compute_effective_negf(self, g, max_paths_len: int = 2):
        """
        Computes the effective non-equilibrium Green's function (NEGF) up to a specified order.

        Parameters:
        - max_paths_len: int, maximum path length for effective green function computation. 1 corresponds to direct paths. 2 corresponds to direct and one indirect path (reaches second neighbors).

        Returns:
        - np.ndarray: Effective Green's function matrix
        """

        G = np.copy(g)  # First-order Green's function

        for path_len in range(2, max_paths_len + 1):  # Ensure this aligns with expected logic
            for i in range(self.num_neurons):
                for j in range(self.num_neurons):
                    if i == j or np.all(g[:, :, i, j] == 0):
                        continue  # Skip self-connections and no-interaction pairs
                    
                    for k in range(self.num_neurons):
                        if k in (i, j) or np.all(g[:, :, i, k] == 0) or np.all(g[:, :, k, j] == 0):
                            continue  # Skip invalid paths
                        
                        # Iterative update of Green's function
                        G[:, :, i, j] += nontt_conv(g[:, :, i, k], G[:, :, k, j])


        return G


    def eval(self, x, dtype=np.float64, drop_branches=None):
        """
        Evaluates the NEGFs in the time domain.

        Parameters:
        - x: array_like, Time axis. All times should be positive.
        - dtype: type (optional), Type of the output array. Default: np.float64        
        - drop_branches: int or array_like of int, Branches to be ignored in the evaluation. Default: None.
                
        Returns:
        - np.ndarray: ExponentialConvolution evaluated on x.
        """
        assert np.all(x >= 0)
            
        if drop_branches is not None:
            try:
                len(drop_branches)
            except:
                drop_branches = [drop_branches]
            
        out = np.zeros_like(x, dtype=dtype)
                
        for exp in self.exp[-1]:
            # Skip terms that are in excluded branches
            branch = exp["branch"]
            if drop_branches is not None and branch in drop_branches:
                continue
                    
            g = exp["g"]
            factor = exp["factor"]
            power_t = exp["power_t"]
                
            if power_t == 0:
                mult = 1.0
            else:
                mult = np.power(x, power_t)
            out += factor * mult * np.exp(-g * x)
                
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
        - signal (np.ndarray): Signal to fit (shape: [resolution, num_neurons]).
        - dt (float): Time step.
        - n_neigh_max (int): Maximum number of neighbors for fitting.
        - rms_limits (Optional[Tuple[int, int]]): Time limits for RMS calculation.
        - auto_stop (bool): Whether to stop fitting early if RMS improvement is below tolerance.
        - rms_tol (float): Tolerance for early stopping.
        - method (Optional[str]): Optimization method for `scipy.optimize`.
        - routine (str): Optimization routine ("minimize" or "least_squares").

        Returns:
        - Tuple[np.ndarray, np.ndarray, np.ndarray]: Fitted parameters, branch parameters, and residuals.
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
        Vth = np.zeros_like(gamma_g)
        E_s = np.zeros_like(gamma_g)
        a_r = np.zeros_like(gamma_g)
        a_d = np.zeros_like(gamma_g)

        lif = cls(num_neurons, dt, Veq, Seq, Vs, delta_Vs, delta_Ss, gamma_g, gamma_s, gamma, beta, Vth, E_s, a_r, a_d)
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
