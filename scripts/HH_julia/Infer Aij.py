#####2.用Gabriel的代码，但是不把非线性项放后面，看结果如何---
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm

def conv(X, Y, interval):
    mask = ~np.isnan(X) & ~np.isnan(Y)
    product = X[mask] * Y[mask]
    scale = (interval[-1] - interval[0]) / len(interval)
    result = scale * np.sum(product)
    return result


# 修改 sum_conv_g_ij_delta_x_j 函数中的访问方式
def sum_conv_g_ij_delta_x_j(DeltaX, *params):
    resolution = 1000
    t_eval = np.linspace(0, 20, resolution)
    N = 4
    A = np.reshape(params, (N, N))
    estimated_x_i_from_convolutions = np.zeros((N, resolution))
    for t in range(1, resolution):
        for i in range(N):
            A[i, i] = 0.0
            for j in range(N):
                # 确保 DeltaX 是二维的
                estimated_x_i_from_convolutions[i, t] += conv(A[i, j] * np.exp(-(t_eval[t] - t_eval[:t])) * (1 - DeltaX[i, :t] / (1 - 0)),DeltaX[j, :t], t_eval[:t])
    return estimated_x_i_from_convolutions.flatten()



if __name__ == "__main__":
    N = 4
    A = np.array([[0, 1, 0, 0],
                  [1, 0, 1, 1],
                  [0, 0, 0, 1],
                  [1, 0, 1, 0]])
    X0 = np.random.rand(N)
    resolution = 1000
    tolerance = 1e-8
    max_iterations = 1000
    t_span = (0, 20)
    t_eval = np.linspace(t_span[0], t_span[1], resolution)

    def nonlinear_equation_original(t, X, N, A):
        dX = np.zeros_like(X)
        for i in range(N):
            sum_term = 0
            for j in range(N):
                sum_term += A[i, j] * (1 - X[i]) * X[j]
            dX[i] = sum_term - X[i]
        return dX

    for iteration in tqdm(range(max_iterations), desc="Iterating"):
        sol = solve_ivp(lambda t, X: nonlinear_equation_original(t, X, N, A), t_span, X0, t_eval=t_eval)
        change = np.linalg.norm(sol.y[:, -1] - sol.y[:, -2])
        if change < tolerance:
            break
        else:
            X0 = sol.y[:, -1]
    Xeq = X0

    def nonlinear_equation_stimuli(t, X, N, A):
        dX = np.zeros_like(X)
        for i in range(N):
            sum_term = 0
            for j in range(N):
                sum_term += A[i, j] * (1 - X[i]) * X[j]
            stim = 1 if i == 3 and 2 < t < 8 else 0
            dX[i] = sum_term - X[i] + stim
        return dX

    sol = solve_ivp(lambda t, X: nonlinear_equation_stimuli(t, X, N, A), t_span, Xeq, t_eval=t_eval)

    DeltaX = sol.y - np.tile(Xeq.reshape(-1, 1), (1, len(sol.t)))

    plt.figure(figsize=(10, 6))
    for i in range(N):
        plt.plot(sol.t, DeltaX[i], label=f'Node {i + 1}')
    plt.title('Time Evolution of the System for Each Node')
    plt.xlabel('Time')
    plt.ylabel('Node State')
    plt.legend()
    plt.grid(True)

    nonlinear_term = np.zeros((N * N, resolution))
    for i in range(N):
        for j in range(N):
            for t_idx in range(resolution):
                nonlinear_term[(i * N + j), t_idx] = (1 - Xeq[i]) * (1 - (DeltaX[i, t_idx] / (1 - Xeq[i]))) * DeltaX[j, t_idx]

    A_guess = np.array([0.5, 0.5, 0.5, 0.5,
                        0.5, 0.5, 0.5, 0.5,
                        0.5, 0.5, 0.5, 0.5,
                        0.5, 0.5, 0.5, 0.5])  # Total 16 parameters

    fitted_params, pcov = curve_fit(sum_conv_g_ij_delta_x_j, DeltaX, DeltaX.flatten(), p0=A_guess,bounds=(np.zeros((N * N)), np.ones((N * N))))

    fitted_curve = sum_conv_g_ij_delta_x_j(DeltaX, fitted_params)

    A_fitted = np.reshape(fitted_params[0:N * N], (N, N))
    print("Fitted parameters for the network:")
    print(A_fitted)

    line_colors = ['b', 'g', 'r', 'c']

    plt.figure(figsize=(10, 24))

    for i in range(N):
        plt.subplot(N, 1, i + 1)
        plt.plot(t_eval, fitted_curve[i * resolution:(i + 1) * resolution], label=f'Fit', color=line_colors[i])
        plt.scatter(t_eval, DeltaX[i, :], label=f'Observed', color='gray', alpha=0.6)
        plt.ylabel(f'Node {i + 1} State')

        plt.legend()
        plt.grid(True)

    plt.xlabel('Time')
    plt.show()



###1.用Gabriel的方法尝试dxi/dt=-xi+aij*sin(xj)模型，看结果如何------结果很好
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm


def conv(X, Y, interval):
    """
    Perform a convolution-like operation between two arrays X and Y over a given interval.

    Args:
    - X (numpy array): First array.
    - Y (numpy array): Second array.
    - interval (numpy array): Array defining the time interval.

    Returns:
    - result (float): The result of the convolution-like operation.
    """
    # Handle missing values by masking NaNs
    mask = ~np.isnan(X) & ~np.isnan(Y)

    # Element-wise multiplication of X and Y, skipping missing values
    product = X[mask] * Y[mask]

    # Calculate the scaling factor based on the interval
    scale = (interval[-1] - interval[0]) / len(interval)

    # Perform the summation
    result = scale * np.sum(product)

    return result


def sum_conv_g_ij_delta_x_j(nonlinear_term, *params):
    """
    Estimate x_i based on the convolution of g_ij and delta_x_j.

    Args:
    - nonlinear_term (numpy array): Array containing the nonlinear terms, combination of DeltaX_j and DeltaX_i states.
    - params (list): List of parameters to be optimized.

    Returns:
    - estimated_x_i_from_convolutions (numpy array): Array of estimated x_i values.
    """
    resolution = 1000
    t_eval = np.linspace(0, 20, resolution)
    N = 4
    A = np.reshape(params, (N, N))
    estimated_x_i_from_convolutions = np.zeros((N, resolution))
    for t in range(1, resolution):
        for i in range(N):
            A[i, i] = 0.0
            for j in range(N):
                estimated_x_i_from_convolutions[i, t] += conv(A[i, j] * np.exp(-(t_eval[t] - t_eval[:t])),nonlinear_term[(i * N + j), :t], t_eval[:t])
    return estimated_x_i_from_convolutions.flatten()


# Example usage
if __name__ == "__main__":
    N = 4  # Number of nodes
    A = np.array([[0, 1, 0, 0],  # Network structure 4 → 3 → 2 → 1
                  [1, 0, 1, 1],
                  [0, 0, 0, 1],
                  [1, 0, 1, 0]])
    X0 = np.random.rand(N)  # Initial random state for nodes
    resolution = 1000
    tolerance = 1e-8
    max_iterations = 1000
    t_span = (0, 20)  # Time span for integration
    t_eval = np.linspace(t_span[0], t_span[1], resolution)  # Evaluation points for the solution


    def nonlinear_equation(t, X, N, A):
        dX = np.zeros_like(X)
        for i in range(N):
            dX[i] = -X[i]
            for j in range(N):
                dX[i] += A[i, j] * np.sin(X[j])
            dX[i] = dX[i]
        return dX


    for iteration in tqdm(range(max_iterations), desc="Iterating"):
        sol = solve_ivp(lambda t, X: nonlinear_equation(t, X, N, A), t_span, X0, t_eval=t_eval)
        change = np.linalg.norm(sol.y[:, -1] - sol.y[:, -2])
        if change < tolerance:
            break
        else:
            X0 = sol.y[:, -1]  # Update initial condition
    Xeq = X0  # Final equilibrium state


    def nonlinear_equation_stimuli(t, X, N, A):
        dX = np.zeros_like(X)
        for i in range(N):
            dX[i] = -X[i]
            for j in range(N):
                dX[i] += A[i, j] * np.sin(X[j])
            stim = 0
            if i == 3 and 2<t < 8:
                stim = 0.3*np.sin(t)
            else:
                stim = 0
            dX[i] = dX[i] + stim
        return dX


    sol = solve_ivp(lambda t, X: nonlinear_equation_stimuli(t, X, N, A), t_span, Xeq, t_eval=t_eval)

    DeltaX = sol.y - np.tile(Xeq.reshape(-1, 1),
                             (1, len(sol.t)))  # Add random fluctuation np.random.normal(0, 0.1, (N, resolution))

    plt.figure(figsize=(10, 6))
    for i in range(N):
        plt.plot(sol.t, DeltaX[i], label=f'Node {i + 1}')
    plt.title('Time Evolution of the System for Each Node')
    plt.xlabel('Time')
    plt.ylabel('Node State')
    plt.legend()
    plt.grid(True)
    nonlinear_term = np.zeros((N * N, resolution))
    for i in range(N):
        for j in range(N):
            for t_idx in range(resolution):
                nonlinear_term[(i * N + j), t_idx] = np.sin(Xeq[j]+DeltaX[j, t_idx])-np.sin(Xeq[j])

    # Initial guess for matrix A
    A_guess = np.array([[0.0, 0.5, 0.5, 0.5],
                        [0.5, 0.0, 0.5, 0.5],
                        [0.5, 0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.5, 0.0]])

    # Fit parameters using observed values
    fitted_params, pcov = curve_fit(sum_conv_g_ij_delta_x_j, nonlinear_term, DeltaX.flatten(), p0=A_guess.flatten(),bounds=(np.zeros((N * N)), np.ones((N * N))))

    fitted_curve = sum_conv_g_ij_delta_x_j(nonlinear_term, fitted_params)

    A_fitted = np.reshape(fitted_params[0:N * N], (N, N))
    print("Fitted parameters for the network:")
    print(A_fitted)

line_colors = ['b', 'g', 'r', 'c']  # Blue, Green, Red, Cyan for line fits

plt.figure(figsize=(10, 24))  # Adjusted height for better visibility in 4x1 layout

# Creating subplots
for i in range(N):
    plt.subplot(N, 1, i + 1)  # Create a 4x1 grid of subplots, selecting the (i+1)th one
    # Plot fitted curves for the current node
    plt.plot(t_eval, fitted_curve[i * resolution:(i + 1) * resolution], label=f'Fit', color=line_colors[i])
    # Scatter plot for the observed data for the current node
    plt.scatter(t_eval, DeltaX[i, :], label=f'Observed', color='gray', alpha=0.6)  # Adjusting index for observed data
    plt.ylabel(f'Node {i + 1} State')

    # Setting titles and labels
    plt.legend()
    plt.grid(True)

plt.xlabel('Time')
plt.show()





