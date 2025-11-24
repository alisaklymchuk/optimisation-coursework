from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from functions import get_plot, J_1, J_2

N = 50
a = 1
b = 0.05
x0 = 1
c = 0.2
d = -0.5
y0 = 1
c_ = np.array([x0 * a ** i for i in range(N + 1)] + [y0 * c ** i for i in range(N + 1)]).T
A = np.array([np.array([0.0] * (2 * N)) for i in range(2 * N + 2)])
D = np.array([np.array([0.0] * (2 * N)) for i in range(2 * N)])
A = A.astype(np.float64)
for i in range(N):
    D[i][i] = 1
    D[i][i + N] = -1
    for j in range(N):
        if i >= j:
            A[i + 1][j] = a ** (i - j) * b
            A[N + 1 + i + 1][N + j] = c ** (i - j) * d
J1 = []
J2 = []
gammas = []

for power in np.arange(-5.0, 5.1, 0.1):
    gamma = 10 ** power
    gammas.append(power)
    M = (A.T @ A + gamma * D.T @ D)
    optimal_uv = -(np.linalg.inv(M)) @ A.T @ c_
    x = A @ optimal_uv + c_
    y = A @ optimal_uv + c_
    x = x[:N + 1]
    y = y[N + 1:]
    u = optimal_uv[:N]
    v = optimal_uv[N:]
    print(np.linalg.norm(x) ** 2, np.linalg.norm(y) ** 2)
    J1.append(J_1(x, y))
    J2.append(J_2(u, v))

get_plot(J1, J2, "2c - Pareto", "J₁(γ)", "J₂(γ)")

fig = go.Figure()
fig.add_trace(go.Scatter(x=gammas, y=J1, name="J₁(log(γ))"))
fig.add_trace(go.Scatter(x=gammas, y=J2, name="J₂(log(γ))"))
fig.update_layout(
    xaxis_title="log(γ)",
    yaxis_title="J(γ)"
)
fig.write_image("images/2c - Cost functions.png")