from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from functions import get_plot, L_e, f, grad_f
from numpy.linalg import norm

params = [[1, 1, 0], [1, 0, 1], [0.1, 0, 1]]

e = 1
gamma2 = 1
gamma1 = 0

