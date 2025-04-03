import numpy as np
from typing import Optional, Tuple, Union
from scipy.optimize import minimize, least_squares

from nonlinfunconn import convolution
from ..utils.nontt_conv import  nontt_conv



class LIF():
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
        """

        self.attribute_list = [
            "gamma_g", "gamma_s", "gamma", "beta", "Vth",
            "E_c", "E_s", "a_r", "a_d", "C"
        ]
        
        if num_neurons is None:
            raise ValueError("num_neurons must be specified.")
        
        self.dt = dt
        self.num_neurons = num_neurons
        
        # Ensure resolution can be determined
        self.Vs = Vs if Vs is not None else np.zeros((self.resolution, num_neurons))
        self.resolution = self.Vs.shape[0]
        
        self.ts = ts if ts is not None else np.arange(0, self.resolution * self.dt, self.dt)
        

        # Expand parameters to appropriate shapes
        self.gamma_g = self._expand_to_array(gamma_g, (num_neurons, num_neurons))
        self.gamma_s = self._expand_to_array(gamma_s, (num_neurons, num_neurons))
        self.E_s = self._expand_to_array(E_s, (num_neurons, num_neurons))
        self.beta = self._expand_to_array(beta, (num_neurons, num_neurons))
        self.a_r = self._expand_to_array(a_r, (num_neurons, num_neurons))
        self.a_d = self._expand_to_array(a_d, (num_neurons, num_neurons))
        self.gamma = self._expand_to_array(gamma, num_neurons)
        self.E_c = self._expand_to_array(E_c, num_neurons)
        self.C = self._expand_to_array(C, num_neurons)
        
        # find Vth as the equilibrium value, so the chemical synapse as term phi = 0.5, half oppened channels
        _Veq, _Seq = self.find_eq_self_consistent()#self.find_equilibrium(np.zeros((num_neurons)))
        if Vth is None:
            Vth = _Veq
        self.Vth = self._expand_to_array(Vth, (num_neurons, num_neurons))

        if np.all(Veq) == None:
            self.Veq = _Veq
        else:
            self.Veq = Veq
        
        if np.all(Seq) == None:
            self.Seq = _Seq
        else:
            self.Seq = self._expand_to_array(Seq, (num_neurons, num_neurons))

        # Initialize synaptic state
        self.Ss = Ss if Ss is not None else np.full((self.resolution, num_neurons, num_neurons), self.Seq)
        # Compute deviations from equilibrium states

        self.delta_Vs = self.Vs - self.Veq[None, :]
        self.delta_Ss = self.Ss - self.Seq[None, :, :]
        
        # Initialize Green's function arrays
        shape = (self.resolution, self.resolution, num_neurons, num_neurons)
        self.sigma0 = np.zeros(shape)
        self.gg0 = np.zeros_like(self.sigma0)
        self.gs0 = np.zeros_like(self.sigma0)
        self.g0 = np.zeros_like(self.sigma0)
        self.sigma = np.zeros_like(self.sigma0)
        self.pi = np.zeros_like(self.sigma0)
        self.g = np.zeros_like(self.sigma0)
        self.G = np.zeros_like(self.g)
        
        # Flags for computation tracking
        self.g_computed_flag = False
        self.g_eq_computed_flag = False

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
                
        return self.g0

    def compute_direct_negf(self, Vs, p: np.ndarray = None, dt : float = 1.0, iteration_index_MAX=5, return_estimated_V=False):
        """
        Compute the nonequilibrium Green's functions for the LIF network.

        Parameters:
            dt (float): Time step for simulation.
            resolution (int): Resolution of the simulation.
            Vs (np.ndarray): Membrane potential dynamics (time series).
            iteration_index_MAX (int): Maximum number of iterations for the Neumann series approximation.
        """

        # Update class attributes with optimized parameters
        if p is not None:
            offset = 0
            for attr in self.attribute_list:
                size = getattr(self, attr).size
                setattr(self, attr, p[offset:offset + size].reshape(getattr(self, attr).shape))
                offset += size


        self.g_computed_flag = True
        self.Veq = Vs[0,:]
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

        self.G = np.copy(self.g)  # First approximation for effective Green's function

        if return_estimated_V:
            est_V = self.Veq # in the future change for V0
            for i in range(self.num_neurons):
                for j in range(self.num_neurons):
                    est_V[i] += nontt_conv(self.g[:, :, i, j], self.delta_Vs)
            
            return self.g , est_V
        else:
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
        num_eff_neurons = G.shape[-1]

        for _ in range(2, max_paths_len+1):
            for i in range(num_eff_neurons):
                for j in range(num_eff_neurons):
                    if i==j: continue # Avoids self loops
                    for k in range(num_eff_neurons):

                        if np.all(g[:, :, i, k] == 0) or np.all(g[:, :, k, j] == 0):
                            continue  # Ensure connectivity exists before doing the computation
                        
                        # Iterative update of Green's function while preventing revisits
                        G[:, :, i, j] += nontt_conv(g[:, :, i, k], G[:, :, k, j])
    
        return G

    def fit(
        self,
        Y: np.ndarray,
        n_neigh_max: int = 2,
        rms_limits: Optional[Tuple[int, int]] = None,
        auto_stop: bool = False,
        rms_tol: float = 1e-2,
        method: Optional[str] = 'trf',
        routine: str = "least_squares",
        p0: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Fit the LIF model to the given signal.

        Parameters:
        - x (np.ndarray): Input data (e.g., stimulus signal).
        - y (np.ndarray): Target neural activation data (shape: [resolution, num_neurons]).
        - dt (float): Time step.
        - n_neigh_max (int): Maximum number of neighbors for fitting (unused).
        - rms_limits (Optional[Tuple[int, int]]): Time limits for RMS calculation (unused).
        - auto_stop (bool): Whether to stop fitting early if RMS improvement is below tolerance.
        - rms_tol (float): Tolerance for early stopping.
        - method (Optional[str]): Optimization method for `scipy.optimize`.
        - routine (str): Optimization routine ("minimize" or "least_squares").
        - p0 (Optional[np.ndarray]): Initial parameter values.

        Returns:
        - Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]: 
        Fitted parameters, branch parameters (if applicable), and residuals (if applicable).
        """
        
        n_trials, time_resolution, n_neurons = Y.shape

        # Ensure necessary attributes exist
        for attr in self.attribute_list:
            if not hasattr(self, attr):
                raise AttributeError(f"Missing required class attribute: {attr}")

        # Flatten model parameters for optimization
        param_list = [getattr(self, attr).flatten() for attr in self.attribute_list]
        p0 = np.concatenate(param_list) if p0 is None else p0
        original_params = p0.copy()

        # Define error function
        def error(p, X, Y):
            Y_predicted = np.zeros_like(Y)
            err = 0.0
            for trial_idx in range(X.shape[0]):  # Iterate over trials
            
                # Ensure compute_direct_negf is implemented and correct
                _, Y_predicted[trial_idx, :, :] = self.compute_direct_negf(Vs=X[trial_idx, :, :], p=p, return_estimated_V=True)
                err += np.sum((Y_predicted[trial_idx, :, :] - Y[trial_idx, :, :]) ** 2)
            err /= n_trials
            # Compute squared error
            return err

        # Choose optimization method
        if routine == "minimize":
            res = minimize(error, p0, args=(Y, Y), method='trf')
        elif routine == "least_squares":
            res = least_squares(error, p0, args=(Y, Y), method='trf')
        else:
            raise ValueError(f"Invalid routine '{routine}'. Choose 'minimize' or 'least_squares'.")

        # Update class attributes with optimized parameters
        offset = 0
        for attr in self.attribute_list:
            size = getattr(self, attr).size
            setattr(self, attr, res.x[offset:offset + size].reshape(getattr(self, attr).shape))
            offset += size

        return res.x, None, None
