from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from functions import get_plot, L_e, f


x = np.arange(-2.0, 2.1, 0.001)
e = 0.01
y = [L_e(x, e) for x in x]

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=[L_e(x, 0.1) for x in x], name="ϵ = 0.1"))
# shows that L_ϵ(u) is approximation of f(u) = abs(u) * ϵ - ϵ^2/2
# fig.add_trace(go.Scatter(x=x, y=[abs(x) * 0.1 - 0.005 for x in x], name="ϵ = 0.1"))
# fig.add_trace(go.Scatter(x=x, y=[L_e(x, 0.05) for x in x], name="ϵ = 0.05"))
# fig.add_trace(go.Scatter(x=x, y=[L_e(x, 0.01) for x in x], name="ϵ = 0.01"))
fig.update_layout(
    title="L_ϵ(u) for different ϵ",
    xaxis_title="L_ϵ(u)",
    yaxis_title="u"
)
fig.show()

params = [[1, 1, 0], [1, 0, 1], [0.1, 0, 1]]

e = 1
gamma2 = 1
gamma1 = 0

