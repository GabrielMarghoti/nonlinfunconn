import numpy as np
from .integration import integral

class nontt_conv(np.ndarray):
    '''
    Non-translational convolution class. 
    Inherits from np.ndarray and adds a method to compute a non-translational convolution between two kernel matrices or a kernel and input signal.
    '''
    
    columnNames = ["z","y","x"]
    upToIndex = {}

    def __new__(cls, K1 : np.ndarray, K2 : np.ndarray, dt : float = 1.0):
        """
        Computes a non-translational convolution between two kernel matrices.

        Parameters:
        - K1: np.ndarray, first kernel matrix
        - K2: np.ndarray, second kernel matrix or input signal
        - dt: float, optional time step (defaults to self.dt)

        Returns:
        - np.ndarray: Output convolution matrix
        """
        resolution = K1.shape[0]

        out = np.zeros_like(K2) 

        # case K2 is another kernel
        if K1.shape == K2.shape:
            for t in range(resolution):
                for t_prime in range(t+1):  
                    out[t, t_prime] = integral(K1[t, t_prime:t] * K2[t_prime:t, t_prime], dt, 8)  
        # case K2 is the signal
        else:  
            for t in range(resolution):
                out[t] = integral(K1[t, :t] * K2[:t], dt, 8)
            
        return out
