import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from functions import *

train_df = pd.read_csv("data/trainingIb.dat", sep=r"\s+", header=None)
train_df.columns = ["x", "y", "dy"]
x = np.array(train_df["x"])
y = np.array(train_df["y"])
dy = np.array(train_df["dy"])
validation_df = pd.read_csv("data/validationIa.dat", sep=r"\s+", header=None)
validation_df.columns = ["x", "y"]
validation_x = validation_df["x"]
validation_y = validation_df["y"]
m = len(validation_x)

N = 20
MSE = [0] * N
log_MSE = [0] * N
# create a vector of values that we want to get (vector b)
Y = np.concatenate((y, dy), axis=0)
for n in range(1, N + 1):
    A0 = np.array([phi(xi, n) for xi in x])
    A1 = np.array([grad_phi(xi, n) for xi in x])
    X = np.concatenate((A0, A1), axis=0)
    theta = get_theta(X, Y)
    predictions = np.array([theta.T @ phi(validation_x[i], n) for i in range(m)])
    MSE[n - 1] = np.linalg.norm(predictions - validation_y) ** 2 / m
    log_MSE[n - 1] = np.log(MSE[n - 1]) / np.log(10)

# MSE vs degree plot
get_plot([n for n in range(1, N + 1)], MSE, "Mean Squared Error vs Degree", "Degree",
         "Mean Squared Error")

# log MSE vs degree plot
get_plot([n for n in range(1, N + 1)], log_MSE, "Log(Mean Squared Error) vs Degree",
         "Degree", "Log(Mean Squared Error)")
"Mean Squared Error vs Degree"

# n = 10 is when MSE <= 10^-3
n = 10
log_MSE = [0] * len(x)
for training_data_size in range(1, len(x) + 1):
    A0 = np.array([phi(x[i], n) for i in range(training_data_size)])
    A1 = np.array([grad_phi(x[i], n) for i in range(training_data_size)])
    X = np.concatenate((A0, A1), axis=0)
    Y = np.concatenate((y[:training_data_size], dy[:training_data_size]), axis = 0)
    theta = get_theta(X, Y)
    predictions = np.array([theta.T @ phi(validation_x[i], n) for i in range(m)])
    mse = np.linalg.norm(predictions - validation_y) ** 2 / m
    log_MSE[training_data_size - 1] = np.log(mse) / np.log(10)

get_plot([n for n in range(1, len(x) + 1)], log_MSE, "Log(Mean Squared Error) vs Number of training points",
         "Number of training points", "Log(Mean Squared Error)")