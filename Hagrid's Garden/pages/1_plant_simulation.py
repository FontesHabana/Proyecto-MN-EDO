"""
1_plant_simulation.py


Interactive Streamlit page to simulate logistic growth of magical plants.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from core.plants_models import logistic_growth
from core.solvers import solve_ivp_model






#==============================================================================================
# 1. Page setup
#============================================================================================== 
 

   
st.set_page_config(page_title="Magical Plants Simulator",layout='wide',initial_sidebar_state="collapsed")
st.title("🌱 Magical Plants Growth Simulator")

st.divider()
sidebar_col,center,main_col=st.columns([1.2,0.3,4],gap="small", vertical_alignment="top")

with sidebar_col:
        
    st.markdown(
        "**Adjust the parameters below to see how a magical plant grows over time!**"
    )

    #==============================================================================================
    # 2. Interactive sliders for parameters
    #============================================================================================== 

    P0 = st.slider("Initial Plant Population (P0)", min_value=0.0,max_value=5.0,value=0.1,step=0.1) 
    r = st.slider("Growth rate(r)", min_value=0.1,max_value=2.0,value=0.5,step=0.1) 
    K = st.slider("Carrying Capacity (K)", min_value=1.0,max_value=20.0,value=10.0,step=1.0)
    t_end=st.slider("Simulation Time", min_value=1,max_value=50,value=20,step=1)

    params={"r":r,"K":K}
    t_span=[0,t_end]

#==============================================================================================
# 3. Solving using solve_ivp
#============================================================================================== 
sol=solve_ivp_model(logistic_growth,np.array([P0]),t_span,params)
t_ivp=sol.t
P_ivp=sol.y[0]
#==============================================================================================
# TODO 4. Solving using Improveds Euler
#============================================================================================== 
#==============================================================================================
# TODO 5. Solving using RK4
#============================================================================================== 

#==============================================================================================
# 6. Plot the results
#============================================================================================== 
with main_col:
    fig=go.Figure()

    fig.add_trace(go.Scatter(x=t_ivp,y=P_ivp,mode='lines', name='solve_ivp'))

    fig.update_layout(
        title="Logisitc Growth of Magical Plant",
        xaxis_title="Time",
        yaxis_title="Plant Population",
        template="plotly_dark"
    )


    #==============================================================================================
    # 7. Display plot in Streamlit
    #============================================================================================== 

    st.plotly_chart(fig,use_container_width=True) 
