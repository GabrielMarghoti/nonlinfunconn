import numpy as np
from typing import Optional, Tuple, Union

from nonlinfunconn import convolution
from ..utils.nontt_conv import  nontt_conv
from ..utils.expandtoarray import expandtoarray


class NM:
    """
    Neural Mass model class.
    """
    
    def __init__(self, num_nodes: int, params: dict):
        """
        Initialize the NM model with parameters passed as a dictionary.
        The first argument is the number of regions (nodes) in the network.
        The second argument is a dictionary of parameters. The keys in the dictionary should match the expected parameter names.
        The parameters are:
        - tau: Poputation relaxation time constant
        - w: Synaptic weight
        - beta: Steepness of the sigmoid function
        - x0: Threshold potential for synapse activation

        The parameters are used to set up the model. If a parameter is not provided, a default value is used.
        The parameters are expanded to the appropriate shape based on the number of neurons in the network.
        """
        # Default parameters
        default_params = {
            "tau": 10.0,       
            "w": 1.0,               # adjacency matrix
            "beta": 1.0,                # Steepness of the sigmoid function
            "xth": 0.0,                 # Threshold potential for synapse activation
        }


        self.parameters = default_params.copy()  # Copy default parameters to instance variable
        # Update default parameters with provided ones
        self.parameters.update(params)

        # Assign parameters to instance variables
        self.N = num_nodes

        # Expand parameters to appropriate shapes
        self.tau      = expandtoarray(self.parameters["tau"] , (num_nodes))
        self.w        = expandtoarray(self.parameters["w"]   , (num_nodes, num_nodes))
        self.beta     = expandtoarray(self.parameters["beta"], (num_nodes, num_nodes))
        self.xth      = expandtoarray(self.parameters["xth"] , (num_nodes, num_nodes))
        
        self.parameters.update({
            "tau": self.tau,      
            "w": self.w,  
            "beta": self.beta, 
            "xth": self.xth,   
        })


    def heaviside(self, t: np.ndarray) -> np.ndarray:
        """
        Heaviside step function.

        Parameters:
            t (np.ndarray): Input array.

        Returns:
            np.ndarray: Heaviside step function applied to the input.
        """
        return np.where(t >= 0, 1.0, 0.0)

    
    def phi(self, x: np.ndarray, beta: np.ndarray, xth: np.ndarray) -> np.ndarray:
        """
        Compute the synaptic activation function.

        Parameters:
            V (np.ndarray): Membrane potential.
            beta (np.ndarray): Synaptic activation steepness.
            Vth (np.ndarray): Threshold potential.

        Returns:
            np.ndarray: Synaptic activation values.
        """
        return 1 / (1 + np.exp(-beta * (x - xth)))


    def compute_direct_green_functions(self, xs,  dt = None, pop_i=None, pop_j=None, iteration_index_MAX=10, p=None, return_estimated_variables=False):
        """
        Compute the nonequilibrium Green's functions for the LIF network.

        Parameters:
            dt (float): Time step for simulation.
            xs (np.ndarray):poputation activity.
            iteration_index_MAX (int): Max iterations for Neumann series.
            return_estimated_variables (bool): If True, also return estimated activity.
        """
        
        num_nodes, time_len = xs.shape
        
        self.time_len = time_len if time_len is not None else self.time_len
        self.dt = dt if dt is not None else self.dt
        
        # Update class attributes 
        if p is not None:
            for key, value in p.items():
                setattr(self, key, expandtoarray(value, getattr(self, key).shape))

        x0 = xs[:, 0]  # first time point for each neuron
        delta_xs = xs - x0[:, None]  # (pop_idx, time)

        self.ts = np.arange(0, self.time_len * self.dt, self.dt)
        ts_diff =  self.ts[:, np.newaxis] - self.ts # (time_len, time_len)
        ts_diff = np.minimum(ts_diff, 0)
        heaviside_func = self.heaviside(ts_diff)

        # SHAPE: (num_nodes, num_nodes, time_len, time_len)
        green_shape = (num_nodes, num_nodes, time_len, time_len)

        g = np.zeros(green_shape)

        for i in range(num_nodes):
            for j in range(num_nodes):
                if self.w[i, j] == 0:
                    continue  # Skip non-connected neurons
                
                small_delta_mask = np.abs(delta_xs[j]) >= 0.0001
                
                # Compute Green's function as before
                g[i, j][:, small_delta_mask] = heaviside_func[:, small_delta_mask] * np.exp(-ts_diff[:, small_delta_mask] / self.tau[i]) * self.w[i, j] * (
                    (self.phi(xs[j, small_delta_mask], self.beta[i, j], self.xth[i, j]) - self.phi(xs[j, 0], self.beta[i, j], self.xth[i, j]))
                    / (delta_xs[j, small_delta_mask])
                )[None, :]

                

            
        if return_estimated_variables:
            est_x = np.zeros_like(xs)
            for i in range(num_nodes):
                est_x[i, :] += x0[i]
                for j in range(num_nodes):
                    est_x[i] += nontt_conv(g[i, j], delta_xs[j], self.dt)

            return g, est_x

        return g
