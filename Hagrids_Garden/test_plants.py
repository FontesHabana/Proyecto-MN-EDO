"""
test_plants.py

Example to test plant logistic growth model.
"""
import numpy as np
import plotly.graph_objects as go

from core.plants_models import logistic_growth
from core.solvers import solve_ivp_model

#==============================================================================================
# 1. Define initial conditions and parameters
#==============================================================================================
    
P0=[0.1]          #initial plant population
t_span =[0,20]  #time range
dt=0.1          #time step for RK4

params={
    "r":0.5,     #growth rate
    "K": 10.0    #carrying capacity
}

#==============================================================================================
# 2. Solve using solve_ivp
#==============================================================================================
    
sol= solve_ivp_model(logistic_growth,P0,t_span,params)
t_ivp=sol.t
P_ivp=sol.y[0]


#==============================================================================================
# 3. Plot the results using Plotly
#==============================================================================================

fig=go.Figure()

fig.add_trace(go.Scatter(x=t_ivp,y=P_ivp,mode='lines', name='solve_ivp'))

fig.update_layout(
    title="Logistic Growth of Magical Plant",
    xaxis_title="time",
    yaxis_title="Plant Population",
    template="plotly_dark"
)

fig.show()