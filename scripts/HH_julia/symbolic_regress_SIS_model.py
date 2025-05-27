import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm
from pysr import PySRRegressor

# Function to solve the ODE system without stimulus
def nonlinear_equation_original(t, X, N, A):
    dX = np.zeros_like(X)
    for i in range(N):
        sum_term = np.sum(A[i] * (1 - X[i]) * X) - A[i, i] * (1 - X[i]) * X[i]
        dX[i] = sum_term - X[i]
    return dX

# Function to solve the ODE system with stimulus on node 3 between time 2 and 8
def nonlinear_equation_stimuli(t, X, N, A):
    dX = np.zeros_like(X)
    for i in range(N):
        sum_term = np.sum(A[i] * (1 - X[i]) * X) - A[i, i] * (1 - X[i]) * X[i]
        stim = 1 if i == 3 and 2 < t < 8 else 0
        dX[i] = sum_term - X[i] + stim
    return dX

if __name__ == "__main__":
    N = 4  # Number of nodes
    A = np.array([[0, 1, 0, 0],  # Network structure 4 → 3 → 2 → 1
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]])

    X0 = np.random.rand(N)  # Initial random state for nodes
    resolution = 500
    t_span = (0, 20)  # Time span for integration
    t_eval = np.linspace(t_span[0], t_span[1], resolution)  # Evaluation points

    # Solve the system without stimulus
    tolerance = 1e-8
    max_iterations = 100
    for iteration in tqdm(range(max_iterations), desc="Iterating"):
        sol = solve_ivp(lambda t, X: nonlinear_equation_original(t, X, N, A), t_span, X0, t_eval=t_eval)
        change = np.linalg.norm(sol.y[:, -1] - sol.y[:, -2])
        if change < tolerance:
            break
        X0 = sol.y[:, -1]  # Update initial condition
    Xeq = X0  # Final equilibrium state

    # Solve the system with stimulus
    sol = solve_ivp(lambda t, X: nonlinear_equation_stimuli(t, X, N, A), t_span, Xeq, t_eval=t_eval)

    # Add random noise to the solution
    DeltaX = sol.y + np.random.normal(0, 0.05, (N, resolution)) - np.tile(Xeq.reshape(-1, 1), (1, len(sol.t)))

    # Data preparation for symbolic regression
    input_data = DeltaX.T  # Transpose for symbolic regression input
    target_data = DeltaX[3, :]  # Target data for node 3

    # Symbolic regression using PySR
    model = PySRRegressor(
        niterations=40,           # Number of iterations for searching
        binary_operators=["+", "*", "-", "/"],  # Binary operations
        unary_operators=["exp", "log", "sin", "cos"],  # Unary operations
        populations=1000,          # Genetic programming population
        loss="loss",               # Default loss
        verbosity=1,               # Verbosity level
    )

    # Train symbolic regression model to fit the relationship
    model.fit(input_data, target_data)

    # Display the discovered equation
    print("Best equation discovered for DeltaX[3]:")
    print(model.get_best())

    # Predict the curve based on symbolic regression
    predicted_curve = model.predict(input_data)

    # Plot the original vs symbolic regression fitted curves
    plt.figure(figsize=(10, 6))
    plt.plot(t_eval, DeltaX[3, :], label="Observed Node 3", color="gray", alpha=0.6)
    plt.plot(t_eval, predicted_curve, label="Symbolic Regression Fit", color="red")
    plt.title("Observed vs Symbolic Regression Fit for Node 3")
    plt.xlabel("Time")
    plt.ylabel("Node 3 State")
    plt.legend()
    plt.grid(True)
    plt.show()
