import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from numpy.linalg import norm

def phi(x, n):
    return np.array([x ** i for i in range(n)])

def grad_phi(x, n):
    return np.array([0] + [(i + 1) * x ** i for i in range(n - 1)])

def J_1(x, y):
    return np.linalg.norm(x) ** 2 + np.linalg.norm(y) ** 2

def J_2(u, v):
    return np.linalg.norm(u - v) ** 2

def L_e(u, e):
    if abs(u) <= e:
        return u ** 2 / 2
    return e * (abs(u) - e / 2)

def L_prime(u, e):
    if abs(u) <= e:
        return u
    if u > 0:
        return  e
    return -e

def grad_L_e(u, e):
    return np.array([L_prime(u_i, e) for u_i in u])

def norm1(u, e):
    result = 0
    for i in u:
        result += L_e(i, e)
    return result

def f(x, y, u, v, gamma1, gamma2, e):
    return norm(x) ** 2 + norm(y) ** 2 + gamma2 * norm(u - v) ** 2 + gamma1 * norm1(u - v)

def grad_f(A, b, w, D, e):
    return 2 * A @ w + b + grad_L_e(D @ w, e)

def get_plot(x, y, title, xaxis, yaxis):
    fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines+markers'))
    fig.update_layout(
        xaxis_title=xaxis,
        yaxis_title=yaxis
    )
    fig.write_image("images/"f"{title}.png")

def get_theta(X, Y):
    model = LinearRegression(fit_intercept=False)
    model.fit(X, Y)
    return np.array(model.coef_)