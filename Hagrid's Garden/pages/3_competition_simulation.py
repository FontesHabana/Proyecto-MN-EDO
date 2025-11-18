"""
3_competition simulation.py

Interactive Streamlit page to simulate two populations copeting for shared resources.
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from core.competition_models import two_species_competition
from core.solvers import solve_ivp_model


#==============================================================================================
# 1. Page setup
#============================================================================================== 
   
st.set_page_config(page_title="Two Species Competition Simulation",layout='wide')
st.title("🌱🌟Two Species Competition Simulator")

st.markdown(
    "Adjust the parameters below to see how two magical populatiions compete for limited resources!"
)

#==============================================================================================
# 2. Interactive sliders for parameters
#============================================================================================== 

P0 = st.slider("Initial Plant Population (P0)", min_value=0.0,max_value=5.0,value=0.1,step=0.1) 
C0 = st.slider("Initial Creature Population (C0)", min_value=0.0,max_value=5.0,value=0.1,step=0.1) 

r_p = st.slider("Plant Growth Rate(r_p)", min_value=0.1,max_value=2.0,value=0.5,step=0.1) 
r_c = st.slider("Creature Growth Rate(r_c)", min_value=0.1,max_value=2.0,value=0.5,step=0.1) 

K = st.slider("Carrying Capacity (K)", min_value=1.0,max_value=20.0,value=10.0,step=1.0)

alpha=st.slider("Effect of Plants on Creatures (alpha)", min_value=0.0,max_value=2.0,value=1.0,step=0.01)
beta=st.slider("Effect of Creatures on Plants (beta)", min_value=0.0,max_value=2.0,value=1.0,step=0.01)


t_end=st.slider("Simulation Time", min_value=1,max_value=50,value=20,step=1)

params={
    "r_p":r_p,
    "r_c":r_c,
    "K":K,
    "alpha":alpha,
    "beta":beta}

t_span=[0,t_end]
Y0=np.array([P0,C0]).flatten()

#==============================================================================================
# 3. Solving using solve_ivp
#============================================================================================== 
sol=solve_ivp_model(two_species_competition,Y0,t_span,params)
t_ivp=sol.t
P_ivp=sol.y[0]
C_ivp=sol.y[1]
#==============================================================================================
# TODO 4. Solving using RK4
#============================================================================================== 

#==============================================================================================
# 5. Plot the results
#============================================================================================== 

fig=go.Figure()

fig.add_trace(go.Scatter(x=t_ivp,y=P_ivp,mode='lines', name='Plants 🌱'))
fig.add_trace(go.Scatter(x=t_ivp,y=C_ivp,mode='lines', name='Creatures 🦄'))


fig.update_layout(
    title="Two Species Competition Over TIme",
    xaxis_title="Time",
    yaxis_title="Population",
    template="plotly_dark"
)


#==============================================================================================
# 6. Display plot in Streamlit
#============================================================================================== 

st.plotly_chart(fig,use_container_width=True) 
