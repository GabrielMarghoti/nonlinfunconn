import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from tqdm import tqdm
# 定义向量化版本的 g_ij_model
def g_ij_model_vectorized(t, tprime, A, x_eq_i, x_eq_j, delta_x_i):
    return A * (1 - x_eq_i) * np.exp(-(1 + A * x_eq_j) * (t - tprime)) * (1 -  delta_x_i) / (1 - x_eq_i)
# 定义参数
N = 4
X0 = np.random.rand(N)
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 1000)
resolution = 1000
tolerance = 1e-10
max_iterations = 100
A_true = np.array([[0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1],
                   [0, 0, 0, 0]])
# 定义非线性方程
def nonlinear_equation_original(t, X, N, A):
    dX = np.zeros_like(X)
    for i in range(N):
        sum_term = 0
        for j in range(N):
            sum_term += A[i, j] * (1 - X[i]) * X[j]
        dX[i] = sum_term - X[i]
    return dX

# 迭代求解平衡点
for iteration in tqdm(range(max_iterations), desc="Iterating"):
    sol = solve_ivp(lambda t, X: nonlinear_equation_original(t, X, N, A_true), t_span, X0, t_eval=t_eval)
    change = np.linalg.norm(sol.y[:, -1] - sol.y[:, -2])
    if change < tolerance:
        break
    else:
        X0 = sol.y[:, -1]
Xeq = X0
print("Xeq:", Xeq)

# 求解带刺激的方程
def nonlinear_equation_stimuli(t, X, N, A):
    dX = np.zeros_like(X)
    for i in range(N):
        if (t>1 or t<2) and i==1:
            sum_term = 0.5
        else:
            sum_term = 0.0
        for j in range(N):
            sum_term += A[i, j] * (1 - X[i]) * X[j]
        dX[i] = sum_term - X[i]
    return dX

# Simulation
sol = solve_ivp(lambda t, X: nonlinear_equation_stimuli(t, X, N, A_true), t_span, Xeq, t_eval=t_eval)
DeltaX = sol.y - np.tile(Xeq.reshape(-1, 1), (1, len(sol.t)))
t = sol.t
V1, V2, V3, V4 = sol.y[0, :], sol.y[1, :], sol.y[2, :], sol.y[3, :]

# Plot the simulated variables
plt.figure(figsize=(10, 6))

plt.subplot(4, 1, 1)
plt.plot(t, V1, label='Node 1 (Origin)', color='black')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(t, V2, label='Node 1', color='orange')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(t, V3, label='Node 1', color='green')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(t, V4, label='Node 1', color='blue')
plt.legend()

plt.xlabel('Time')
plt.show()

# 计算真实的 g_ij
g_true = np.zeros((resolution, resolution, N, N))
g_data = np.zeros((resolution, resolution, N, N))
t_values = np.linspace(0, 10, resolution)
g_0 = np.zeros((resolution, resolution, N, N))

# 计算 g_0
for i, j in [(i, j) for i in range(N) for j in range(N)]:
    for t_prime in range(resolution):
        condition_array = np.where(t_values >= t_values[t_prime], 1, 0)
        g_0[:, t_prime, i, j] = condition_array * A_true[i, j] * (1 - Xeq[i]) * np.exp(-(t_values - t_values[t_prime]))

# 定义非线性格林函数
pi = np.zeros((resolution, resolution, N, N))
for i in range(N):
    for j in range(N):
        for t in range(1, resolution):
            for tprime in range(t):
                if A_true[i, j] == 0 or DeltaX[j, tprime] == 0:
                    g_true[t, tprime, i, j] = 0
                else:
                    pi[t, tprime, i, j] = g_0[t, tprime, i, j] * DeltaX[i, tprime] / (1 - Xeq[i])
                    g_true[t, tprime, i, j] = g_0[t, tprime, i, j] - pi[t, tprime, i, j]

# 添加扰动到 g_data
noise = 0.0 # 0.1 * np.random.normal(size=g_true.shape)
g_data = g_true + noise

# 优化损失函数
def loss_function(A_flat):
    A = A_flat.reshape((N, N))
    t_values_matrix = t_values[:, np.newaxis]
    loss = 0
    for i in tqdm(range(N), desc="Rows Progress"):
        for j in range(N):
            if A[i, j] != 0:
                g_model = g_ij_model_vectorized(t_values_matrix, t_values_matrix.T, A[i, j], Xeq[i], Xeq[j], DeltaX[i])
                loss += np.sum((g_model - g_data[:, :, i, j]) ** 2)
    return loss

# 初始猜测
initial_guess = np.random.rand(N, N)
# 使用 minimize 进行优化
result = minimize(loss_function, initial_guess.flatten(), method='BFGS', options={'maxiter': 100, 'disp': True})
# 重塑为 N x N 矩阵
A_estimated = result.x.reshape((N, N))
# 输出估计的 A 矩阵
print("Estimated A matrix:")
print(A_estimated)
