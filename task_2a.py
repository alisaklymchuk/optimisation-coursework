from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

N = 50
a = 1
b = -0.01
x0 = 1
c = np.array([x0 * a ** i for i in range(N + 1)]).T
A = np.array([np.array([0.0] * N) for i in range(N + 1)])
A = A.astype(np.float64)
for i in range(N):
    for j in range(N):
        if i >= j:
            A[i + 1][j] = a ** (i - j) * b

fig_u = go.Figure()
fig_x = go.Figure()

for gamma in [0.001, 0.01, 0.1, 1]:
    M = (A.T @ A + gamma * np.eye(N))
    optimal_u = -(np.linalg.inv(M)) @ A.T @ c
    x = A @ optimal_u + c
    print(gamma * np.linalg.norm(optimal_u) ** 2 + np.linalg.norm(x) ** 2)
    fig_u.add_trace(go.Scatter(y=optimal_u, mode='lines+markers', name=f'gamma={gamma}'))
    fig_x.add_trace(go.Scatter(y=x, mode='lines+markers', name=f'gamma={gamma}'))

fig_u.update_layout(
    title="Optimal Control Signals u*",
    xaxis_title="Time step i",
    yaxis_title="u*",
)
fig_u.show()

fig_x.update_layout(
    title="Optimal Trajectories x_u*",
    xaxis_title="Time step i",
    yaxis_title="x",
)
fig_x.show()