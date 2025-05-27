import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define symbolic variables
t = sp.Symbol('t')
t_prime = sp.Symbol('t_prime')
t_vals = sp.Symbol('t_vals')  # Discrete time points for fitting
x_signal = sp.Function('x_signal')(t_vals)
y_signal = sp.Function('y_signal')(t_vals)  # Observed output
params = sp.symbols('a0:2 b0:2')  # Kernel parameters (amplitude and decay constants for 2 exponentials)

# Define symbolic kernel
def exp_kernel_symbolic(t, params):
    """Symbolic exponential kernel as a function of time t and parameters."""
    kernel_expr = sum(params[2 * i] * sp.exp(-params[2 * i + 1] * t) for i in range(len(params) // 2))
    return kernel_expr

# Define symbolic convolution
def conv_symbolic(X, kernel, t_eval):
    """Symbolically compute the convolution of X with kernel over time."""
    return sp.integrate(X.subs(t, t_prime) * kernel.subs(t, t_eval - t_prime), (t_prime, 0, t_eval))

# Define symbolic kernel fitting function
kernel = exp_kernel_symbolic(t - t_prime, params)
conv_expr = conv_symbolic(x_signal, kernel, t_vals)

# Symbolic residual for fitting
residual = y_signal - conv_expr

# Define the least-squares objective
objective = sp.Sum(residual**2, (t_vals, 0, t_vals))  # Sum over time points

# Derivatives for fitting (parameter gradients)
gradients = [sp.diff(objective, p) for p in params]

# Display symbolic results
print("Convolution Expression:")
sp.pprint(conv_expr)
print("\nObjective Function:")
sp.pprint(objective)
print("\nParameter Gradients:")
for grad in gradients:
    sp.pprint(grad)

# Convert to numerical functions for fitting
objective_func = sp.lambdify((t_vals, x_signal, y_signal, *params), objective, 'numpy')
grad_funcs = [sp.lambdify((t_vals, x_signal, y_signal, *params), grad, 'numpy') for grad in gradients]

# Example: Fitting the kernel to data
from scipy.optimize import minimize

# Generate example data
time_points = np.linspace(0, 20, 500)
x_signal_numeric = np.sin(time_points)  # Example input signal
true_params = [1.0, 0.1, 0.5, 0.05]  # True kernel parameters
kernel_numeric = sum(true_params[2 * i] * np.exp(-true_params[2 * i + 1] * time_points) for i in range(len(true_params) // 2))
y_signal_numeric = np.convolve(x_signal_numeric, kernel_numeric, mode='full')[:len(time_points)] + np.random.normal(0, 0.05, len(time_points))

# Define the numerical objective for optimization
def numerical_objective(params):
    return objective_func(time_points, x_signal_numeric, y_signal_numeric, *params)

# Fit the kernel
initial_guess = [0.8, 0.2, 0.3, 0.1]  # Initial parameter guess
result = minimize(numerical_objective, initial_guess, method='L-BFGS-B', bounds=[(0, None)] * len(initial_guess))

# Extract fitted parameters
fitted_params = result.x

# Compare true and fitted kernels
fitted_kernel = sum(fitted_params[2 * i] * np.exp(-fitted_params[2 * i + 1] * time_points) for i in range(len(fitted_params) // 2))

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time_points, kernel_numeric, label="True Kernel", linestyle='--', color='blue')
plt.plot(time_points, fitted_kernel, label="Fitted Kernel", linestyle='-', color='red')
plt.legend()
plt.xlabel("Time")
plt.ylabel("Kernel Amplitude")
plt.title("True vs. Fitted Kernel")
plt.grid()
plt.show()
