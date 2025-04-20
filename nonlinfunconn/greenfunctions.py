import numpy as np
from typing import Optional, Tuple, Union
import copy
from joblib import Parallel, delayed

from nonlinfunconn import convolution
from .utils.nontt_conv import  nontt_conv


class GreenFunctions:
    """
    Class to store Green's functions for a given nonlinear dinamical model.

    This class focuses on two primary Green's functions: 
    - **Interaction Green's functions**: Describe signal propagation between different nodes.
    - **Self-interaction Green's functions**: Capture the intrinsic, self-dynamics of individual nodes.

    Model-specific internal Green's functions are defined within the model class and are not directly accessed by this class.

    The GreenFunctions class is initialized with a number of nodes and a model (either as a string identifier or a class). 
    It provides methods to:
    - Compute different types of Green's functions (Equilibrium, Direct, Effective, Self-interaction, Paths),
    - Fit model parameters using observed data,
    - Predict node activity,
    - Perform convolutions between Green's functions and input signals.

    This abstraction allows flexible use of different networks models under a unified interface.
    """
    
    def __init__(
        self,
        model: object,  # Accept an instance of the model
        dt: float,
        x: Optional[np.ndarray],
    ):
        """
        Initialize the GreenFunctions class.
        This class sets up the model and the Green functions for different trials (with same time length).

        Parameters
        ----------
        model : object
            An instance of a model class from the models subpackage.
        dt : float
            Time step for the simulation. Shall be provided with the nodes states for proper time scaling.
        x : np.ndarray
            Input signal data for each trial and each node. In the shape (num_trials, num_nodes, time_len). 
        """

        self.n_trials, self.n_nodes, self.time_len = x.shape

        if not hasattr(model, "parameters"):
            raise TypeError("Model must be an instance of a class from the models subpackage.")

        self.model_instance = model

        # Setup time and system states
        self.dt = dt
        self.t = np.arange(self.time_len) * self.dt

        self.x = x

        # Initialize Green's function arrays
        self.g0 = np.zeros((self.n_nodes, self.n_nodes, self.time_len, self.time_len))
        self.g  = np.zeros((self.n_trials, self.n_nodes, self.n_nodes, self.time_len, self.time_len))

        # Compute Green's functions
        self.g0 = self.model_instance.compute_direct_equilibrium_green_functions(time_len=self.time_len, dt=self.dt)

        self.g = np.array(Parallel(n_jobs=-1)(
            delayed(self.model_instance.compute_direct_green_functions)(self.x[trial_idx], dt=self.dt) for trial_idx in range(self.n_trials)
        ))

    def get_parameters_array(self, parameters = None, constrain=None):
        """
        Flatten and concatenate model parameters into a single vector.

        Parameters:
            include_adj_matrix (bool): Whether to include adjacency matrix-related parameters (gamma_g, gamma_s).
            constrain (tuple): A tuple containing min and max constraints for parameters.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Flattened parameter array, min constraints, and max constraints.
        """
        param_list = []
        _min_constrain_list = []
        _max_constrain_list = []

        if parameters == None:
            parameters = self.model_instance.parameters.keys()

        for attr in parameters:
            param_list.append(getattr(self.model_instance, attr).flatten())
            if constrain is not None:
                try: 
                    _min_constrain_list.append(np.full_like(getattr(self.model_instance, attr), constrain[0][attr]).flatten())
                    _max_constrain_list.append(np.full_like(getattr(self.model_instance, attr), constrain[1][attr]).flatten())
                except:
                    _min_constrain_list.append(np.full_like(getattr(self.model_instance, attr), -np.inf).flatten())
                    _max_constrain_list.append(np.full_like(getattr(self.model_instance, attr), np.inf).flatten())


        param_array = np.concatenate(param_list)
        min_constrain_array = np.concatenate(_min_constrain_list) if constrain is not None else None
        max_constrain_array = np.concatenate(_max_constrain_list) if constrain is not None else None
        
        return param_array, min_constrain_array, max_constrain_array


    def set_parameters(self, p):
        """Update model attributes from a flattened parameter array."""
        offset = 0

        for attr in self.model_instance.parameters.keys():
            arr = getattr(self.model_instance, attr)
            size = arr.size
            new_vals = p[offset:offset + size].reshape(arr.shape)
            setattr(self.model_instance, attr, new_vals)
            offset += size

    def path_G(self, path, trial_idx = None, linear_model = False):
        """
        Compute the Green's function contribution for a specific path and trial.

        Parameters:
        - path: list of int, sequence of nodes representing the path.
        - trial_idx: int, index of the trial.

        Returns:
        - np.ndarray: Convolution result for the given path and trial.
        """
        if linear_model:
            _g = np.copy(self.g0) if linear_model else np.copy(self.g)
        elif trial_idx is not None:
            _g = np.copy(self.g[trial_idx])
            _G_k_j = _g[path[1], path[0], :, :]
            for idx in range(1, len(path) - 1):
                _G_k_j = nontt_conv(_g[path[idx + 1], path[idx], :, :], _G_k_j, self.dt)

            return _G_k_j
        else:
            _G_k_j = _g[:, path[1], path[0], :, :]
            for trial_idx in range(self.n_trials):
                _g = np.copy(self.g[trial_idx])
                _G_k_j[trial_idx] = _g[path[1], path[0], :, :]
                for idx in range(1, len(path) - 1):
                    _G_k_j[trial_idx] = nontt_conv(_g[path[idx + 1], path[idx], :, :], _G_k_j[trial_idx], self.dt)

            return _G_k_j
            
    def total_G(self, max_paths_len, linear_model=False, node_pair = "all"):
        """
        Computes the effective non-equilibrium Green's function (NEGF) up to a specified order.

        Parameters:
        - max_paths_len: int, maximum path length for effective green function computation. 1 corresponds to direct paths. 2 corresponds to direct and one indirect path (reaches second neighbors).

        - node_pair: str, "all" for all node pairs or a specific pair (i, j) to compute the effective Green's function from j to i.

        Returns:
        - np.ndarray: Effective Green's function matrix up to a maximum path length 'max_path_len'
        """
        if node_pair == "all":
            G = np.copy(self.g0) if linear_model else np.copy(self.g) # First-order Green's function is the direct one
            _g = np.copy(self.g0) if linear_model else np.copy(self.g)
            num_eff_nodes = G.shape[-1]

            def compute_trial_update(trial_idx):
                G_trial = np.copy(G[trial_idx])
                for _ in range(2, max_paths_len + 1):
                    for i in range(num_eff_nodes):
                        for j in range(num_eff_nodes):
                            if i == j:
                                continue  # Avoids self loops
                            for k in range(num_eff_nodes):
                                if np.all(_g[i, k, :, :] == 0) or np.all(G_trial[k, j, :, :] == 0):
                                    continue  # Ensure connectivity exists before doing the computation
                                # Iterative update of Green's function while preventing revisits
                                G_trial[i, j] += nontt_conv(_g[i, k], G_trial[k, j], self.dt)
                return G_trial

            G = np.array(Parallel(n_jobs=-1)(delayed(compute_trial_update)(trial_idx) for trial_idx in range(self.n_trials)))
        else:

            i, j = node_pair

            paths = self._find_paths(start_node=j, end_node=i, max_paths_len=max_paths_len)

            def compute_trial_update(trial_idx):
                G_trial = np.copy(G[trial_idx])
                for path in paths:
                    G_trial[i, j, :, :] += self.path_G(path, trial_idx)
                return G_trial

            G = np.array(Parallel(n_jobs=-1)(
                delayed(compute_trial_update)(trial_idx) for trial_idx in range(self.n_trials)
            ))

        return G


    def _find_paths(self, start_node, end_node=None, max_paths_len=1):
        """
        Find all paths starting from a given node up to a specified maximum path length.

        Parameters:
        - start_node: int, the starting node for the paths.
        - end_node: int, optional, the ending node for the paths. If specified, only paths ending at this node are returned.
        - max_paths_len: int, the maximum length of the paths.

        Returns:
        - list of lists: Each inner list represents a path as a sequence of nodes.
        """

        paths = [[start_node]]  # Initialize paths starting from the start_node
        for _ in range(max_paths_len - 1):
            new_paths = []
            for path in paths:
                last_node = path[-1]
                for k in range(self.n_nodes):
                    if k not in path and not np.all(self.g[k, last_node, :, :] == 0):
                        # Ensure the path does not return to any previous node
                        new_paths.append(path + [k])
            paths = new_paths

        if end_node is not None:
            # Filter paths that end at the specified end_node
            paths = [path for path in paths if path[-1] == end_node]

        return paths

    def ADAM_fit(
        self,
        dt: float,
        x: Optional[np.ndarray],
        fit_linear_model: bool = False,
        include_adj_matrix: bool = True,
        parameter_to_fit_list: Optional[list] = None,
        n_neigh_max: int = 2,
        rms_limits: Optional[Tuple[int, int]] = None,
        constrain=None, 
        auto_stop: bool = True,
        rms_tol: float = 1e-4,
        max_iters: int = 1000,
        learning_rate: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-6,
        p0: Optional[np.ndarray] = None,
        loss_method = 'correlation',
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Fit the model using Adam gradient descent with parameter dict support.
        """
        import copy
        from joblib import Parallel, delayed

        if dt is None:
            dt = self.dt 

        target = x

        parameter_to_fit_list = self.model_instance.parameters.keys() if parameter_to_fit_list is None else parameter_to_fit_list

        for attr in parameter_to_fit_list:
            if not hasattr(self.model_instance, attr):
                raise AttributeError(f"Missing required class attribute: {attr}")

        p = self.model_instance.parameters if p0 is None else p0

        m_dict = {k: np.zeros_like(v) for k, v in p.items()}
        v_dict = {k: np.zeros_like(v) for k, v in p.items()}

        def correlation_loss(y_pred, y_true):
            vx = y_pred - np.mean(y_pred, axis=1, keepdims=True)
            vy = y_true - np.mean(y_true, axis=1, keepdims=True)

            numerator = np.sum(vx * vy, axis=1)
            denominator = np.sqrt(np.sum(vx ** 2, axis=1)) * np.sqrt(np.sum(vy ** 2, axis=1))
            
            corr = numerator / (denominator + 1e-8)  # Add epsilon to avoid div by 0
            return 1 - np.mean(corr)  # Average over nodes
        
        def loss(variant_self, p, X, Y):

            err = 0.0
            for trial_idx in range(X.shape[0]):
                X_trial = X[trial_idx]
                Y_trial = Y[trial_idx]
                delta_X = X_trial - X_trial[:, [0]]

                if fit_linear_model:
                    g0 = variant_self.compute_direct_equilibrium_green_functions(num_nodes=self.n_nodes, time_len=Y_trial.shape[1], dt=dt, p=p)
                    est_V = np.zeros((variant_self.num_nodes, self.time_len)) 
                    for i in range(variant_self.num_nodes):
                        est_V[i] += self.Veq[i]
                        for j in range(variant_self.num_nodes):
                            est_V[i] += nontt_conv(g0[i, j], delta_X[j], dt)
                    Y_pred = est_V
                else:
                    _, Y_pred = self.model_instance.compute_direct_green_functions(X_trial, dt, p=p, return_estimated_V=True)

                if loss_method == 'correlation':
                    err += correlation_loss(Y_pred, Y_trial)
                else:
                    err += np.sqrt(np.sum((Y_pred - Y_trial) ** 2)) / (self.n_trials * self.time_len * self.n_nodes)
            return err 

        def compute_grad(param_dict, X, Y, epsilon=1e-6):
            loss_0 = loss(self, param_dict, X, Y)
            grad_dict = {}

            for key in param_dict:
                grad_dict[key] = np.zeros_like(param_dict[key])

                def compute_single_grad(idx):
                    perturbed = copy.deepcopy(param_dict)
                    perturbed[key] = param_dict[key].copy()
                    perturbed[key][idx] += epsilon
                    loss_eps = loss(copy.deepcopy(self), perturbed, X, Y)
                    return (loss_eps - loss_0) / epsilon

                if np.isscalar(param_dict[key]):
                    grads = [compute_single_grad(0)]
                else:
                    grads = Parallel(n_jobs=-1)(delayed(compute_single_grad)(i) for i in np.ndindex(param_dict[key].shape))
                grad_dict[key] = np.array(grads).reshape(param_dict[key].shape)

            return grad_dict

        prev_loss = float('inf')

        for t in range(1, max_iters + 1):
            grad_dict = compute_grad(p, x, target)

            for key in p:
                m_dict[key] = beta1 * m_dict[key] + (1 - beta1) * grad_dict[key]
                v_dict[key] = beta2 * v_dict[key] + (1 - beta2) * (grad_dict[key] ** 2)

                m_hat = m_dict[key] / (1 - beta1 ** t)
                v_hat = v_dict[key] / (1 - beta2 ** t)

                p[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)
                # Apply constraints if provided
                if constrain is not None and key in constrain[0]:
                    p[key] = np.clip(p[key], constrain[0][key], constrain[1][key])

            current_loss = loss(self, p, x, target)

            if t % 10 == 0 or t == 1:
                print(f"Iteration {t}, Loss: {current_loss:.6f}")

            if auto_stop and abs(prev_loss - current_loss) < rms_tol:
                print(f"Early stopping at iteration {t}. Loss improvement < {rms_tol}")
                break
            prev_loss = current_loss

        # Final update to model
        self.model_instance.parameters = p

        # Update Green's functions
        self.g0 = self.model_instance.compute_direct_equilibrium_green_functions(time_len=self.time_len, dt=self.dt, p=p)

        self.g = np.array(Parallel(n_jobs=-1)(
            delayed(self.model_instance.compute_direct_green_functions)(self.x[trial_idx], dt=self.dt, p=p) for trial_idx in range(self.n_trials)
        ))

        return p
