import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def phi(x, n):
    return np.array([x ** i for i in range(n)])

def grad_phi(x, n):
    return np.array([0] + [(i + 1) * x ** i for i in range(n - 1)])

def J_1(x, y):
    return np.linalg.norm(x) ** 2 + np.linalg.norm(y) ** 2

def J_2(u, v):
    return np.linalg.norm(u - v) ** 2

def get_plot(x, y, title, xaxis, yaxis):
    fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines+markers'))
    fig.update_layout(
        title={
            'text':title,
            'x': 0.5,  # centers the title
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title=xaxis,
        yaxis_title=yaxis
    )
    fig.show()

def get_theta(X, Y):
    model = LinearRegression(fit_intercept=False)
    model.fit(X, Y)
    return np.array(model.coef_)