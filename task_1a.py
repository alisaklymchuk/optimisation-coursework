import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from functions import phi, get_theta, get_plot

train_df = pd.read_csv("data/trainingIa.dat", sep=r"\s+", header=None)
train_df.columns = ["x", "y"]
x = train_df["x"]
y = train_df["y"]
validation_df = pd.read_csv("data/validationIa.dat", sep=r"\s+", header=None)
validation_df.columns = ["x", "y"]
validation_x = validation_df["x"]
validation_y = validation_df["y"]
m = len(validation_x)

N = 20
MSE = [0] * N
log_MSE = [0] * N
for n in range(1, N + 1):
    X = np.array([phi(xi, n) for xi in x])
    Y = y
    theta = get_theta(X, Y)
    predictions = np.array([theta.T @ phi(validation_x[i], n) for i in range(m)])
    MSE[n - 1] = np.linalg.norm(predictions - validation_y) ** 2 / m
    log_MSE[n - 1] = np.log(MSE[n - 1]) / np.log(10)

# MSE vs degree plot
get_plot([n for n in range(1, N + 1)], MSE, "1a - MSE vs Degree", "Degree",
         "Mean Squared Error")


# log MSE vs degree plot
get_plot([n for n in range(1, N + 1)], log_MSE, "1a - Log MSE vs Degree",
         "Degree", "Log(Mean Squared Error)")
"Mean Squared Error vs Degree"

# n = 10 is when MSE <= 10^-3
n = 10
MSE = [0] * len(x)
log_MSE = [0] * len(x)
for training_data_size in range(1, len(x) + 1):
    X = np.array([phi(x[i], n) for i in range(training_data_size)])
    Y = y[:training_data_size]
    theta = get_theta(X, Y)
    predictions = np.array([theta.T @ phi(validation_x[i], n) for i in range(m)])
    MSE[training_data_size - 1] = np.linalg.norm(predictions - validation_y) ** 2 / m
    log_MSE[training_data_size - 1] = np.log(MSE[training_data_size - 1]) / np.log(10)

get_plot([n for n in range(1, len(x) + 1)], MSE, "1a - MSE vs Number of training points",
         "Number of training points", "Mean Squared Error")

x = np.linspace(-2, 2, 1000)
y = np.array([theta.T @ phi(x[i], n) for i in range(1000)])
fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines'))
fig.update_layout(
    xaxis_title="x",
    yaxis_title="V(x)"
)
fig.write_image("images/1a - Learned function V(x).png")