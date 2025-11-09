from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from functions import get_plot

N = 50
a = 1
b = 0.05
x0 = 1
c = 0.2
d = -0.5
y0 = 1
c_ = np.array([x0 * a ** i for i in range(N + 1)] + [0] * (N + 1)).T
d_ = np.array([0] * (N + 1) + [y0 * c ** i for i in range(N + 1)]).T
A = np.array([np.array([0.0] * (2 * N)) for i in range(2 * N + 2)])
B = np.array([np.array([0.0] * (2 * N)) for i in range(2 * N + 2)])
D = np.array([np.array([0.0] * (2 * N)) for i in range(N)])
A = A.astype(np.float64)
for i in range(N):
    D[i][i] = 1
    D[i][i + N] = -1
    for j in range(N):
        if i >= j:
            A[i + 1][j] = a ** (i - j) * b
            B[N + 1 + i + 1, N + j] = c ** (i - j) * d

gamma = 1
M = (A.T @ A + B.T @ B + gamma * D.T @ D)
optimal_uv = -(np.linalg.inv(M)) @ (c_.T @ A + d_.T @ A).T
x = A @ optimal_uv + c_
y = B @ optimal_uv + d_
x = x[:N + 1]
y = y[N + 1:]
u = optimal_uv[:N]
v = optimal_uv[N:]

print(gamma * np.linalg.norm(optimal_uv) ** 2 + np.linalg.norm(x) ** 2)
get_plot([i for i in range(N + 1)], x, "Optimal Trajectories x_u*","Time step i", "x",)
get_plot([i for i in range(1, N + 1)], u, "Optimal Control Signals u*","Time step i", "u*")
get_plot([i for i in range(N + 1)], y, "Optimal Trajectories y_v*","Time step i", "y")
get_plot([i for i in range(1, N + 1)], v, "Optimal Control Signals v*", "Time step i", "v*")
