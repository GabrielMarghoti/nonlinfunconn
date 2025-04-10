import numpy as np
from typing import Optional, Tuple, Union
import copy
from joblib import Parallel, delayed

from nonlinfunconn import convolution
from ..utils.nontt_conv import  nontt_conv



class LIF():
    """
    Leaky Integrate-and-Fire (LIF) neural network model with Green's function computation.
    """
    
    def __init__(
        self,
        num_neurons: int = None,
        Veq: Union[float, np.ndarray] = None,  # Equilibrium membrane potential
        Seq: Union[float, np.ndarray] = None,  # Equilibrium synaptic state
        Vs: np.ndarray = None,  # Membrane potential dynamics (time series)
        Ss: np.ndarray = None,  # Synaptic state dynamics (time series)
        C: Union[float, np.ndarray] = 1e-09,  # Membrane capacitance
        gamma_g: Union[float, np.ndarray] = 1,  # Conductance decay rate
        gamma_s: Union[float, np.ndarray] = 1,  # Synaptic decay rate
        gap_conductance: Union[float, np.ndarray] = 1e-10,  # Gap junction conductance
        syn_conductance: Union[float, np.ndarray] = 1e-10,  # Gap junction conductance
        gamma: Union[float, np.ndarray] = 1e-11,  # Membrane potential decay rate
        beta: Union[float, np.ndarray] = 0.125,  # Inverse synaptic timescale
        Vth: Union[float, np.ndarray] = None,  # Threshold potential for spiking
        E_c: Union[float, np.ndarray] = -60.0,  # Equilibrium membrane potential
        E_s: Union[float, np.ndarray] = 0.0,  # Synaptic reversal potential
        a_r: Union[float, np.ndarray] = 1.0,  # Synaptic rise time constant
        a_d: Union[float, np.ndarray] = 5.0,  # Synaptic decay time constant
    ):
        """
        Initialize the LIF model.
        """

        self.attribute_list = [
            "gamma_g", "gamma_s", "gap_cond", "syn_cond", "C", "gamma", "beta", "Vth",
            "E_c", "E_s", "a_r", "a_d"
        ]
        
        if num_neurons is None:
            raise ValueError("num_neurons must be specified.")
        

        self.num_neurons = num_neurons

        # Expand parameters to appropriate shapes
        self.gamma_g = self._expand_to_array(gamma_g, (num_neurons, num_neurons))
        self.gamma_s = self._expand_to_array(gamma_s, (num_neurons, num_neurons))
        self.gap_cond = self._expand_to_array(gap_conductance, (num_neurons, num_neurons))
        self.syn_cond = self._expand_to_array(syn_conductance, (num_neurons, num_neurons))
        self.E_s = self._expand_to_array(E_s, (num_neurons, num_neurons))
        self.beta = self._expand_to_array(beta, (num_neurons, num_neurons))
        self.a_r = self._expand_to_array(a_r, (num_neurons, num_neurons))
        self.a_d = self._expand_to_array(a_d, (num_neurons, num_neurons))
        self.C = self._expand_to_array(C, num_neurons)
        self.gamma = self._expand_to_array(gamma, num_neurons)
        self.E_c = self._expand_to_array(E_c, num_neurons)
        
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

        # Flags for computation tracking
        #self.g_computed_flag = False
        #elf.g_eq_computed_flag = False

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
    def get_parameters_array(self, include_adj_matrix: bool = True) -> np.ndarray:
            """Flatten and concatenate model parameters into a single vector."""
            if not include_adj_matrix:
                self.attribute_list.remove("gamma_g")
                self.attribute_list.remove("gamma_s")
            param_list = [getattr(self, attr).flatten() for attr in self.attribute_list]
            return np.concatenate(param_list)

    def set_parameters(self, p):
        """Update model attributes from a flattened parameter array."""
        offset = 0
        for attr in self.attribute_list:
            arr = getattr(self, attr)
            size = arr.size
            new_vals = p[offset:offset + size].reshape(arr.shape)
            setattr(self, attr, new_vals)
            offset += size

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

    def compute_direct_equilibrium_green_functions(self, num_neurons=None ,time_len=None, dt=None, p: np.ndarray = None):

        self.time_len = time_len if time_len is not None else self.time_len
        self.num_neurons = num_neurons if num_neurons is not None else self.num_neurons
        self.dt = dt if dt is not None else self.dt
        
        green_functions_shape = (self.time_len, self.time_len, self.num_neurons, self.num_neurons)
        # Initialize Green's function arrays
        self.sigma0 = np.zeros(green_functions_shape)
        self.gg0 = np.zeros(green_functions_shape)
        self.gs0 = np.zeros(green_functions_shape)
        self.g0 = np.zeros(green_functions_shape)

        self.ts = np.arange(0, self.time_len * self.dt, self.dt)

        # Update class attributes with optimized parameters
        if p is not None:
            offset = 0
            for attr in self.attribute_list:
                size = getattr(self, attr).size
                setattr(self, attr, p[offset:offset + size].reshape(getattr(self, attr).shape))
                offset += size
        ts_diff = self.ts[:, np.newaxis] - self.ts

        heaviside_func = self.heaviside(ts_diff)

        gamma_sum = (self.gamma/self.C)[:, np.newaxis] + np.sum((self.gamma_g*self.gap_cond/self.C), axis=1)[:, np.newaxis] + np.sum((self.gamma_s*self.syn_cond/self.C) * self.Seq, axis=1)[:, np.newaxis]
       
        for i in range(self.num_neurons):
            exp_factor_gs_gg = np.exp(-ts_diff * gamma_sum[i])

            for j in range(self.num_neurons):
                if i == j or (self.gamma_g[i, j] == 0 and self.gamma_s[i, j] == 0):
                    continue  # Skip self-interaction & null kernels
                
                a_r, a_d, beta, Veq, Vth = self.a_r[i, j], self.a_d[i, j], self.beta[i, j], self.Veq[j], self.Vth[i, j]
                
                exp_factor_synaptic = np.exp(-ts_diff * (a_d - a_r / (1 + np.exp(-beta * (Veq - Vth)))))
                synaptic_factor = a_r * (1 - self.Seq[i, j]) * self.d_synaptic_activation(Veq, beta, Vth)

                self.sigma0[:, :, i, j] = heaviside_func * synaptic_factor * exp_factor_synaptic
                self.gg0[:, :, i, j] = heaviside_func * (self.gamma_g[i, j]*self.gap_cond[i,j]/self.C[i]) * exp_factor_gs_gg
                self.gs0[:, :, i, j] = heaviside_func * (self.gamma_s[i, j]*self.syn_cond[i,j]/self.C[i]) * (self.E_s[i, j] - self.Veq[i]) * exp_factor_gs_gg

                conv_s = nontt_conv(self.gs0[:, :, i, j], self.sigma0[:, :, i, j], self.dt)
                self.g0[:, :, i, j] = self.gg0[:, :, i, j] + conv_s
        

        return self.g0

    def compute_direct_negf(self, Vs, dt, p: np.ndarray = None, iteration_index_MAX=10, return_estimated_V=False):
        """
        Compute the nonequilibrium Green's functions for the LIF network.

        Parameters:
            dt (float): Time step for simulation.
            resolution (int): Resolution of the simulation.
            Vs (np.ndarray): Membrane potential dynamics (time series).
            iteration_index_MAX (int): Maximum number of iterations for the Neumann series approximation.
        """

        self.time_len = len(Vs)
        self.num_neurons = Vs.shape[1]
        self.dt = dt if dt is not None else self.dt
        
        self.ts = np.arange(0, self.time_len * self.dt, self.dt)
        
        # Initialize Green's function arrays
        green_functions_shape = (self.time_len, self.time_len, self.num_neurons, self.num_neurons)
        V_shape = (self.time_len, self.num_neurons)
        S_shape = (self.time_len, self.num_neurons, self.num_neurons)

        self.sigma = np.zeros(green_functions_shape)
        self.pi = np.zeros(green_functions_shape)
        self.g = np.zeros(green_functions_shape)
        self.G = np.zeros(green_functions_shape)

        self.V = np.zeros(V_shape)
        
        self.delta_Vs = np.zeros(V_shape)

        self.Ss = np.zeros(S_shape)
        self.delta_Ss = np.zeros(S_shape)

        # Synaptic state dynamics is hidden, so the model estimates it. the initial guess is the equilibrium value
        self.Ss = np.repeat(self.Seq[np.newaxis, :, :], self.time_len, axis=0)  # Initialize synaptic state dynamics for iterative approximation
        

        # Update class attributes with optimized parameters
        if p is not None:
            offset = 0
            for attr in self.attribute_list:
                size = getattr(self, attr).size
                setattr(self, attr, p[offset:offset + size].reshape(getattr(self, attr).shape))
                offset += size

        self.Veq = Vs[0,:]
        
        self.Vs = Vs
        self.delta_Vs = self.Vs - self.Veq[None, :]

        self.compute_direct_equilibrium_green_functions()
        
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if self.gamma_g[i, j] == 0 and self.gamma_s[i, j]==0: continue # avoid computing null kernell (no connection)
                synaptic_diff = np.zeros_like(self.delta_Vs[:, j])
                non_zero_indices = np.abs(self.delta_Vs[:, j]) >= 0.000001
                synaptic_diff[non_zero_indices] = (
                    self.synaptic_activation(self.Vs[:, j], self.beta[i, j], self.Vth[i, j]) - 
                    self.synaptic_activation(self.Veq[None, j], self.beta[i, j], self.Vth[i, j])
                )[non_zero_indices] / self.delta_Vs[non_zero_indices, j]

                prev_delta_S = np.copy(self.delta_Ss[:, i, j])
                for _ in range(iteration_index_MAX):  # Iterative approximation for self consistent series approximation
                                                      # in the future, this should be a while loop with a convergence criterion tol.
                    
                    self.sigma[:, :, i, j] = (
                        self.sigma0[:, :, i, j] / 
                        self.d_synaptic_activation(self.Veq[j] * np.ones((self.time_len)), self.beta[i, j], self.Vth[i, j])[None, :] * 
                        synaptic_diff[None, :] * (1 - (self.delta_Ss[:, i, j] / (1 - self.Seq[i, j])))
                    )

                    self.delta_Ss[:, i, j] = nontt_conv(self.sigma[:, :, i, j], self.delta_Vs[:, j], self.dt)
                    if np.all(np.abs(self.delta_Ss[:, i, j] - prev_delta_S) < 1e-3):   # stop self consistent iteration if change is smaller than a tolerance
                        break
                    prev_delta_S = np.copy(self.delta_Ss[:, i, j])
                    

                self.pi[:, :, i, j] = nontt_conv(
                    self.gs0[:, :, i, j], 
                    (1 - (self.delta_Vs[:, i, None] / (self.E_s[i, j] - self.Veq[i]))) * self.sigma[:, :, i, j], self.dt
                )
                self.g[:, :, i, j] = self.gg0[:, :, i, j] + self.pi[:, :, i, j]

        self.G = np.copy(self.g)  # First approximation for effective Green's function

        if return_estimated_V:
            est_V = np.zeros_like(self.Vs) 
            for i in range(self.num_neurons):
                est_V[:,i] += self.Veq[i] * np.ones((self.time_len))
                for j in range(self.num_neurons):
                    est_V[:,i] += nontt_conv(self.g[:, :, i, j], self.delta_Vs[:, j], self.dt)
            
            return self.g , est_V
        else:
            return self.g

    def compute_effective_negf(self, g, max_paths_len: int = 2, neuron_pair = "all"):
        """
        Computes the effective non-equilibrium Green's function (NEGF) up to a specified order.

        Parameters:
        - max_paths_len: int, maximum path length for effective green function computation. 1 corresponds to direct paths. 2 corresponds to direct and one indirect path (reaches second neighbors).

        - neuron_pair: str, "all" for all neuron pairs or a specific pair (i, j) to compute the effective Green's function from j to i.

        Returns:
        - np.ndarray: Effective Green's function matrix
        """
        if neuron_pair == "all":
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
        else:
            # for now it only consider second neighbors, review
            i, j = neuron_pair
            G = np.copy(g[:, :, i, j])
            for k in range(self.num_neurons):
                if np.all(g[:, :, i, k] == 0) or np.all(g[:, :, k, j] == 0):
                    continue
                G += nontt_conv(g[:, :, i, k], g[:, :, k, j])

        
        return G


    def fit(
        self,
        Y: np.ndarray,
        dt = None,
        fit_linear_model: bool = False,
        include_adj_matrix: bool = True,
        parameter_to_fit_list: Optional[list] = None,
        n_neigh_max: int = 2,
        rms_limits: Optional[Tuple[int, int]] = None,
        auto_stop: bool = True,
        rms_tol: float = 1e-3,
        max_iters: int = 1000,
        learning_rate: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        p0: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Fit the model using Adam gradient descent.
        """

        if dt is not None:
            self.dt = dt
        else:
            self.dt = self.dt # use the previous
            raise ValueError("Time step (dt) must be provided.")

        n_trials, time_len, n_neurons = Y.shape

        # Check for required attributes
        parameter_to_fit_list = self.attribute_list if parameter_to_fit_list is None else parameter_to_fit_list

        for attr in parameter_to_fit_list:
            if not hasattr(self, attr):
                raise AttributeError(f"Missing required class attribute: {attr}")

        # Initialize parameters
        p0 = self.get_parameters_array(include_adj_matrix) if p0 is None else p0

        p = p0.copy()
        m = np.zeros_like(p)
        v = np.zeros_like(p)

        def loss(variant_self, p, X, Y):
            Y_pred = np.zeros_like(Y)
            err = 0.0
            for trial_idx in range(X.shape[0]):
                variant_self.Vs = Y[trial_idx]
                variant_self.delta_Vs = self.Vs - self.Veq[None, :]
                if fit_linear_model:
                    g0 = variant_self.compute_direct_equilibrium_green_functions(num_neurons=n_neurons, time_len=time_len, dt = self.dt, p=p)
                    est_V = np.zeros((time_len, variant_self.num_neurons)) 
                    for i in range(variant_self.num_neurons):
                        est_V[:,i] += self.Veq[i] * np.ones((self.time_len))
                        for j in range(variant_self.num_neurons):
                            est_V[:,i] += nontt_conv(g0[:, :, i, j], self.delta_Vs[:, j], self.dt)
            
                    Y_pred[trial_idx, :, :] = est_V
                else:
                    _, Y_pred[trial_idx, :, :] = variant_self.compute_direct_negf(Vs=X[trial_idx], dt = self.dt, p=p, return_estimated_V=True)

                err += np.sqrt(np.sum((Y_pred[trial_idx] - Y[trial_idx]) ** 2))
            return err / (n_trials* time_len * n_neurons)

        def compute_grad(p, X, Y, epsilon=1e-2):
            grad = np.zeros_like(p)
            loss_0 = loss(self, p, X, Y)

            # Adjust epsilon based on the magnitude of each parameter
            adjusted_epsilon = epsilon * np.maximum(np.abs(p), 1e-4)
            def compute_single_grad(i):
                p_eps = p.copy()
                p_eps[i] += adjusted_epsilon[i]
                loss_eps = loss(copy.deepcopy(self), p_eps, X, Y)
                return (loss_eps - loss_0) / adjusted_epsilon[i]

            grad = np.array(Parallel(n_jobs=-1)(delayed(compute_single_grad)(i) for i in range(len(p))))

            return grad

        prev_loss = float('inf')
        for t in range(1, max_iters + 1):
            grad = compute_grad(p, Y, Y)

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)

            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            p -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

            current_loss = loss(self, p, Y, Y)
            if t % 10 == 0 or t == 1:
                print(f"Iteration {t}, Loss: {current_loss:.6f}")

            if auto_stop and abs(prev_loss - current_loss) < rms_tol and current_loss < 0.1*prev_loss:
                print(f"Early stopping at iteration {t}. Loss improvement < {rms_tol}")
                break
            prev_loss = current_loss

        # Update model parameters
        self.set_parameters(p)

        return p, None, None
