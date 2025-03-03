
import numpy as np

from nonlinfunconn import convolution


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
        self.resolution = Vs.shape[0]
        
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
        
        self._initialize_green_functions()

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

    def _initialize_green_functions(self):
        """Initialize all Green function matrices with zeros."""
        shape = (self.resolution, self.resolution, self.num_neurons, self.num_neurons)
        self.sigma_0 = np.zeros(shape)
        self.gg_0 = np.zeros(shape)
        self.gs_0 = np.zeros(shape)
        self.g_0 = np.zeros(shape)
        self.sigma = np.zeros(shape)
        self.pi = np.zeros(shape)
        self.g = np.zeros(shape)

    @staticmethod
    def heaviside(t):
        return np.where(t >= 0, 1.0, 0.0)
    
    def synaptic_activation(self, V: np.ndarray, beta: np.ndarray, V_th: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-beta * (V - V_th)))

    def d_synaptic_activation(self, V, beta, V_th):
        exp_term = np.exp(-beta * (V - V_th))
        return (beta * exp_term) / (1 + exp_term) ** 2

    def compute_equilibrium_green_functions(self):
        """Vectorized computation of equilibrium Green's functions."""
        time_diff = np.subtract.outer(self.time_array, self.time_array)
        heaviside_diff = self.heaviside(time_diff)

        exp_term = np.exp(-time_diff[:, :, None, None] * 
                           (self.a_d - self.a_r / (1 + np.exp(-self.beta * (self.Veq[:, None] - self.V_th)))))
        self.sigma_0 = heaviside_diff[:, :, None, None] * self.a_r * (1 - self.Seq) * \
                       self.d_synaptic_activation(self.Veq[:, None], self.beta, self.V_th) * exp_term

        exp_term_gg = np.exp(-time_diff[:, :, None, None] * (self.gamma + np.sum(self.gamma_g, axis=1)[:, None] +
                        np.sum(self.gamma_s * self.Seq, axis=1)[:, None]))
        self.gg_0 = heaviside_diff[:, :, None, None] * self.gamma_g * exp_term_gg
        self.gs_0 = heaviside_diff[:, :, None, None] * self.gamma_s * (self.Es - self.Veq[:, None]) * exp_term_gg

        self.g_0 = self.gg_0 + convolution(self.gs_0, self.sigma_0, self.dt, 8)

    def compute_nonequilibrium_green_functions(self):
        """Iteratively compute non-equilibrium Green's functions."""
        conv_sigma_V = np.zeros((self.resolution, self.num_neurons, self.num_neurons))
        for _ in range(3):  # Iterative refinement
            synaptic_diff = (self.synaptic_activation(self.Vs[:, None, :], self.beta, self.V_th) -
                             self.synaptic_activation(self.Veq[:, None], self.beta, self.V_th)) / self.delta_Vs[:, None, :]
            
            self.sigma = self.sigma_0 / self.d_synaptic_activation(self.Veq[:, None], self.beta, self.V_th) * synaptic_diff
            conv_sigma_V = convolution(self.sigma, self.delta_Vs, self.dt, 8)
            
            self.pi = convolution(self.gs_0, (1 - (self.delta_Vs[:, None] / (self.Es - self.Veq[:, None]))) * self.sigma, self.dt, 8)
            self.g = self.gg_0 + self.pi

    def fit(self, x, dt, method='least_squares', auto_stop=False, rms_tol=1e-2):
        """
        Fit the LIF model to input data using least-squares optimization.

        Parameters:
            x (np.ndarray): Signal to fit and use as input. Can be real data or synthetic data but it needs to encompass all the nodes which nonlinear signal propagation pass through.
            dt: Time step.

        Returns:
            params: Fitted parameters.
            residuals: Residuals.
        """
        
        from scipy.optimize import least_squares
        
        """
        num_neurons = x.shape[1]
        Veq = np.mean(x, axis=0)
        params = np.random.rand(num_neurons)
        
        def residuals(p):
            return convolution(x, self.synaptic_activation(x, p, Veq), dt, 8) - x
        
        res = least_squares(residuals, params, method=method)
        return res.x, res.cost
        """
        rms = []
        
        y_norm = np.sum(y)
        if np.isnan(y_norm) or np.isinf(y_norm) or y_norm==0:
            return None, None, None
        else:
            yb = y/y_norm
        #yb = y/y_norm
        stim_norm = np.sum(stim)
        stimb = stim/stim_norm
        
        p0_tot_prev_ = np.array([])
        p_prev = np.array([])
        n_in_prev = np.array([],dtype=int)
                
        for n_branches in np.arange(0,n_branches_max):
            
            rms_cur_b = []
            for i in np.arange(n_hops_min,n_hops_max+1):
                p0_cur_b_ = 0.2+np.arange(i)*0.03+n_branches*0.02
                if n_branches == 0:
                    A0 = 1.
                else:
                    A0 = (-1)**n_branches
                p0_cur_b_ = np.append(A0,p0_cur_b_)
                
                p0_tot_ = np.append(p0_tot_prev_,p0_cur_b_)
                p0_tot = mf.struct.irrarray(p0_tot_,[np.append(n_in_prev,i+1)],["branch"])
                
                lower_bounds = -np.inf*np.ones_like(p0_tot)
                upper_bounds = np.inf*np.ones_like(lower_bounds)
                special_idx = np.append(0,np.cumsum(n_in_prev))
                for q in np.arange(len(lower_bounds)):
                    if q not in special_idx:
                        lower_bounds[q] = 0.0
                    else:
                        branch_idx = np.where(special_idx==q)[0][0]
                        #if branch_idx%2==1: upper_bounds[q]=0.0
                        #elif branch_idx%2==0 and n_branches>0: lower_bounds[q]=0.0
                        
                if routine == "minimize":
                    error = lambda p,x,y: np.sum(np.power(pp.convolution(stimb,cls.eci(x,p),dt,8)-y,2))
                    res = minimize(error,p0_tot,args=(x,yb),method=method)
                    p_cur_b = res.x
                elif routine == "least_squares":
                    residuals = lambda p,x,y: pp.convolution(stimb,cls.eci(x,p),dt,8)-y
                    #residuals = cls.residuals_branching_least_squares
                    res = least_squares(residuals,p0_tot,args=(x,yb),method=method,bounds=(lower_bounds,upper_bounds))
                    p_cur_b = res.x
                rms_cur_b.append(np.sqrt(np.sum(np.power((cls.eci(x,p_cur_b)-yb)[rms_limits[0]:rms_limits[1]],2))))
                
                if auto_stop and i>n_hops_min:
                    delta_rms_rel = np.abs(rms_cur_b[-1]-rms_cur_b[-2])/rms_cur_b[-2]
                    if delta_rms_rel<rms_tol:break

            p0_tot_prev_= p0_tot_
            rms.append(rms_cur_b[-1])
            
            if auto_stop and n_branches>0:
                delta_rms_rel = (rms[-1]-rms[-2])/rms[-2]
                if np.abs(delta_rms_rel)<rms_tol and delta_rms_rel<0:
                    n_in_prev = np.append(n_in_prev,i+1)
                    p_prev = p_cur_b
                    break
                elif np.abs(delta_rms_rel)<rms_tol and delta_rms_rel>=0: 
                    rms.pop(-1)
                    break
                else:
                    n_in_prev = np.append(n_in_prev,i+1)
                    p_prev = p_cur_b
            else:
                n_in_prev = np.append(n_in_prev,i+1)
                p_prev = p_cur_b
        
        special_idx = np.append(0,np.cumsum(n_in_prev))
        for j in special_idx[:-1]:
            #FIXME BE CAREFUL WITH THIS. IF YOU HAVE REJOINING OF BRANCHES, IT
            # IS NOT GOING TO BEHAVE WELL!
            p_prev[j] *= y_norm/stim_norm
            
        gc.collect()
            
        return p_prev,n_in_prev,rms

# Example Usage:
# lif = LIF(num_neurons=10, dt=0.01, Veq=..., Seq=..., Vs=..., delta_Vs=..., delta_Ss=..., gamma_g=..., gamma_s=..., gamma=..., beta=..., V_th=..., Es=..., a_r=..., a_d=...)
# lif.compute_equilibrium_green_functions()
# lif.compute_nonequilibrium_green_functions()
# fitted_params, error = lif.fit(x, dt=0.01)
