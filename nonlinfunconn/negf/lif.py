import numpy as np
from scipy.optimize import minimize, least_squares
from typing import Optional, Tuple, Union
from nonlinfunconn import convolution
from ..utils.irrarray import irrarray

class LIF:
    """
    Leaky Integrate-and-Fire (LIF) neural network model with Green's function computation.
    This class implements equilibrium and non-equilibrium Green's functions for a network of LIF neurons.
    """

    def __init__(
        self,
        num_neurons: int,
        dt: float,
        Veq: Union[float, np.ndarray],
        Seq: Union[float, np.ndarray],
        Vs: np.ndarray,
        delta_Vs: np.ndarray,
        delta_Ss: np.ndarray,
        gamma_g: Union[float, np.ndarray],
        gamma_s: Union[float, np.ndarray],
        gamma: Union[float, np.ndarray],
        beta: Union[float, np.ndarray],
        V_th: Union[float, np.ndarray],
        Es: Union[float, np.ndarray],
        a_r: Union[float, np.ndarray],
        a_d: Union[float, np.ndarray],
    ):
        """
        Initialize the LIF model.

        Parameters:
            num_neurons (int): Number of neurons in the network.
            dt (float): Time step for simulations.
            Veq (Union[float, np.ndarray]): Equilibrium membrane potentials.
            Seq (Union[float, np.ndarray]): Equilibrium synaptic states.
            Vs (np.ndarray): Membrane potentials over time (shape: [resolution, num_neurons]).
            delta_Vs (np.ndarray): Deviation from equilibrium membrane potentials (shape: [resolution, num_neurons]).
            delta_Ss (np.ndarray): Deviation from equilibrium synaptic states (shape: [resolution, num_neurons, num_neurons]).
            gamma_g (Union[float, np.ndarray]): Gap junction coupling strengths.
            gamma_s (Union[float, np.ndarray]): Chemical synapse coupling strengths.
            gamma (Union[float, np.ndarray]): Leakage rates.
            beta (Union[float, np.ndarray]): Synaptic activation steepness.
            V_th (Union[float, np.ndarray]): Threshold potentials.
            Es (Union[float, np.ndarray]): Synaptic reversal potentials.
            a_r (Union[float, np.ndarray]): Synaptic rise rates.
            a_d (Union[float, np.ndarray]): Synaptic decay rates.
        """
        self.num_neurons = num_neurons
        self.dt = dt
        self.resolution = Vs.shape[0]

        # Expand scalar parameters to arrays if necessary
        self.Veq = self._expand_to_array(Veq, num_neurons)
        self.Seq = self._expand_to_array(Seq, (num_neurons, num_neurons))
        self.Vs = Vs
        self.delta_Vs = delta_Vs
        self.delta_Ss = delta_Ss
        self.gamma_g = self._expand_to_array(gamma_g, (num_neurons, num_neurons))
        self.gamma_s = self._expand_to_array(gamma_s, (num_neurons, num_neurons))
        self.gamma = self._expand_to_array(gamma, num_neurons)
        self.beta = self._expand_to_array(beta, (num_neurons, num_neurons))
        self.V_th = self._expand_to_array(V_th, (num_neurons, num_neurons))
        self.Es = self._expand_to_array(Es, (num_neurons, num_neurons))
        self.a_r = self._expand_to_array(a_r, (num_neurons, num_neurons))
        self.a_d = self._expand_to_array(a_d, (num_neurons, num_neurons))

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

        Parameters:
            value (Union[float, np.ndarray]): Input value or array.
            shape (Tuple[int, ...]): Desired shape of the output array.

        Returns:
            np.ndarray: Array of the specified shape.
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

    def compute_equilibrium_green_functions(self,
        n_neigh_max: int = 2):
        """
        Compute the equilibrium Green's functions for the LIF network.

        Parameter:
            n_neigh_max (int): Maximum number of neighbors for fitting the effective NEGF for nodes not direct connected.
        """
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

    def compute_nonequilibrium_green_functions(self,
        n_neigh_max: int = 2):
        """
            Compute the nonequilibrium Green's functions for the LIF network.

            Parameter:
                n_neigh_max (int): Maximum number of neighbors for fitting the effective NEGF for nodes not direct connected.
        """
        conv_sigma_V = np.zeros((self.resolution, self.num_neurons, self.num_neurons))

        for itr in range(3):  # Iterative approximation
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
                                                      synaptic_diff * (1 - (conv_sigma_V[t, i, j] / (1 - self.Seq[i, j]))))

                        conv_sigma_V[t, i, j] = convolution(self.sigma[t, :t, i, j], self.delta_Vs[:t, j], self.dt, 8)

                        for t_prime in range(t):
                            self.pi[t, t_prime, i, j] = convolution(self.gs_0[t, t_prime:t, i, j], 
                                                                   (1 - (self.delta_Vs[t_prime:t, i] / (self.Es[i, j] - self.Veq[i]))) * 
                                                                   self.sigma[t_prime:t, t_prime, i, j], self.dt, 8)
                            self.g[t, t_prime, i, j] = self.gg_0[t, t_prime, i, j] + self.pi[t, t_prime, i, j]


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
