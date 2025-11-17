from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from functions import get_plot, L_e, f, grad_f
from numpy.linalg import norm

params = [[1, 1, 0], [1, 0, 1], [0.1, 0, 1]]

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
D = np.array([np.array([0.0] * (2 * N)) for i in range(2 * N)])
A = A.astype(np.float64)
for i in range(N):
    D[i][i] = 1
    D[i][i + N] = -1
    for j in range(N):
        if i >= j:
            A[i + 1][j] = a ** (i - j) * b
            B[N + 1 + i + 1][N + j] = c ** (i - j) * d

# w_t A w + b_t w + c
b = 2 * c_.T @ A + 2 * d_.T @ B
c = c_.T @ c_ + d_.T @ d_
tolerance = 10 ** (-5)

x_plot = go.Figure()
y_plot = go.Figure()
u_plot = go.Figure()
v_plot = go.Figure()
for i in range(len(params)):
    e = params[i][0]
    gamma2 = params[i][1]
    gamma1 = params[i][2]

    M = (A.T @ A + B.T @ B + gamma2 * D.T @ D)
    L = 2 * norm(M + gamma1 * D.T @ D, ord=2)
    t = 1 / L

    w = np.zeros(2 * N)
    cnt = 0

    while (norm(grad_f(M, b, w, D, e)) > tolerance):
        cnt += 1
        w = w - t * grad_f(M, b, w, D, e)

    x = A @ w + c_
    y = B @ w + d_
    x = x[:N + 1]
    y = y[N + 1:]
    u = w[:N]
    v = w[N:]
    x_plot.add_trace(go.Scatter(x=[i for i in range(N + 1)], y=x, name = "Part "f"{i + 1}"))
    y_plot.add_trace(go.Scatter(x=[i for i in range(N + 1)], y=y, name = "Part "f"{i + 1}"))
    u_plot.add_trace(go.Scatter(x=[i + 1 for i in range(N)], y=u, name = "Part "f"{i + 1}"))
    v_plot.add_trace(go.Scatter(x=[i + 1 for i in range(N)], y=v, name = "Part "f"{i + 1}"))
x_plot.update_layout(
    xaxis_title="Time step i",
    yaxis_title="x_i"
)
y_plot.update_layout(
    xaxis_title="Time step i",
    yaxis_title="y_i"
)
u_plot.update_layout(
    xaxis_title="Time step i",
    yaxis_title="u_i"
)
v_plot.update_layout(
    xaxis_title="Time step i",
    yaxis_title="v_i"
)
x_plot.write_image("images/2d - x_plot.png")
y_plot.write_image("images/2d - y_plot.png")
u_plot.write_image("images/2d - u_plot.png")
v_plot.write_image("images/2d - v_plot.png")