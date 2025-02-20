
import numpy as np

from nonlinfunconn import convolution

class LIF:
    def __init__(self, num_neurons, dt, Veq, Seq, Vs, delta_Vs, delta_Ss, gamma_g, gamma_s, gamma, beta, V_th, Es, a_r, a_d):
        """
        Initialize the LIF (Leaky Integrate-and-Fire) model.
        The NEGF of LIF network requires as entry: the model parameters, nodes states and initial conditions (typically assumed as the equilibrium) variable values.

        Parameters:
            num_neurons (int): Number of neurons.
            dt: Time step.
            Veq (float or np.ndarray): Equilibrium membrane potentials.
            Seq (float or np.ndarray): Equilibrium synaptic states.
            Vs (np.ndarray): Membrane potentials over time.
            delta_Vs (np.ndarray): Deviation from equilibrium membrane potentials.
            delta_Ss (np.ndarray): Deviation from equilibrium synaptic states.
            gamma_g (float or np.ndarray): Gap junction coupling strengths.
            gamma_s (float or np.ndarray): Chemical synapse coupling strengths.
            gamma (float or np.ndarray): Leakage rates.
            beta (float or np.ndarray): Synaptic activation steepness.
            V_th (float or np.ndarray): Threshold potentials.
            Es (float or np.ndarray): Synaptic reversal potentials.
            a_r (float or np.ndarray): Synaptic rise rates.
            a_d (float or np.ndarray): Synaptic decay rates.
        """

        self.num_neurons = num_neurons
        self.dt = dt
        self.resolution = Vs.shape[1]

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

        # Initialize Green function arrays
        self.sigma_0 = np.zeros((self.resolution, self.resolution, num_neurons, num_neurons))
        self.gg_0 = np.zeros((self.resolution, self.resolution, num_neurons, num_neurons))
        self.gs_0 = np.zeros((self.resolution, self.resolution, num_neurons, num_neurons))
        self.g_0 = np.zeros((self.resolution, self.resolution, num_neurons, num_neurons))
        self.sigma = np.zeros((self.resolution, self.resolution, num_neurons, num_neurons))
        self.pi = np.zeros((self.resolution, self.resolution, num_neurons, num_neurons))
        self.g = np.zeros((self.resolution, self.resolution, num_neurons, num_neurons))

    def _expand_to_array(self, value, shape):
        if np.isscalar(value):
            return np.full(shape, value)
        elif isinstance(value, np.ndarray):
            if value.shape == shape:
                return value
            else:
                raise ValueError(f"Expected shape {shape}, but got {value.shape}")
        else:
            raise TypeError(f"Expected scalar or numpy array, but got {type(value)}")

    def heaviside(self, t):
        return np.where(t >= 0, 1.0, 0.0)

    def synaptic_activation(self, V, beta, V_th):
        return 1 / (1 + np.exp(-beta * (V - V_th)))

    def d_synaptic_activation(self, V, beta, V_th):
        exp_term = np.exp(-beta * (V - V_th))
        return (beta * exp_term) / (1 + exp_term) ** 2

    def compute_equilibrium_green_functions(self):
        time_diff = self.time_array[:, None] - self.time_array
        heaviside_diff = self.heaviside(time_diff)

        for i in range(self.num_neurons):
            for j in range(i, self.num_neurons):
                exp_term = np.exp(-time_diff * (self.a_d[i, j] - self.a_r[i, j] / (1 + np.exp(-self.beta[i, j] * (self.Veq[j] - self.V_th[i, j])))))
                self.sigma_0[:, :, i, j] = heaviside_diff * self.a_r[i, j] * (1 - self.Seq[i, j]) * self.d_synaptic_activation(self.Veq[j], self.beta[i, j], self.V_th[i, j]) * exp_term

                exp_term_gg = np.exp(-time_diff * (self.gamma[i] + np.sum(self.gamma_g[i, :]) + np.sum(self.gamma_s[i, :] * self.Seq[i, :])))
                self.gg_0[:, :, i, j] = heaviside_diff * self.gamma_g[i, j] * exp_term_gg

                self.gs_0[:, :, i, j] = heaviside_diff * self.gamma_s[i, j] * (self.Es[i, j] - self.Veq[i]) * exp_term_gg

                for t in range(self.resolution):
                    for t_prime in range(t):
                        self.g_0[t, t_prime, i, j] = self.gg_0[t, t_prime, i, j] + convolution(self.gs_0[t, t_prime:t, i, j], self.sigma_0[t_prime:t, t_prime, i, j], self.dt, 8)

    def compute_nonequilibrium_green_functions(self):
        conv_sigma_V = np.zeros((self.resolution, self.num_neurons, self.num_neurons))

        for itr in range(3):  # Iterative method to approximate sigma
            for i in range(self.num_neurons):
                for j in range(i, self.num_neurons):
                    for t in range(self.resolution):
                        for t_prime in range(t):
                            if self.Vs[t_prime, j] == self.Veq[j]:
                                self.sigma[t, t_prime, i, j] = 0.0
                            else:
                                synaptic_diff = (self.synaptic_activation(self.Vs[t_prime, j], self.beta[i, j], self.V_th[i, j]) - self.synaptic_activation(self.Veq[j], self.beta[i, j], self.V_th[i, j])) / self.delta_Vs[t_prime, j]
                                self.sigma[t, t_prime, i, j] = self.sigma_0[t, t_prime, i, j] / self.d_synaptic_activation(self.Veq[j], self.beta[i, j], self.V_th[i, j]) * synaptic_diff * (1 - (conv_sigma_V[t_prime, i, j] / (1 - self.Seq[i, j])))

                        conv_sigma_V[t, i, j] = convolution(self.sigma[t, :t, i, j], self.delta_Vs[:t, j], self.dt, 8)

                        for t_prime in range(t):
                            self.pi[t, t_prime, i, j] = convolution(self.gs_0[t, t_prime:t, i, j], (1 - (self.delta_Vs[t_prime:t, i] / (self.Es[i, j] - self.Veq[i]))) * self.sigma[t_prime:t, t_prime, i, j], self.dt, 8)
                            self.g[t, t_prime, i, j] = self.gg_0[t, t_prime, i, j] + self.pi[t, t_prime, i, j]