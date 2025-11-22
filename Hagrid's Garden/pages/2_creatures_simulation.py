"""
2_creatures_simulation.py


Interactive Streamlit page to simulate predator-prey interactions using the classic Lotka-Volterra model.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from core.creatures_models import lotka_volterra
from core.solvers import solve_ivp_model

#==============================================================================================
# 1. Page setup
#============================================================================================== 
 

   
st.set_page_config(page_title="Magical Plants Simulator",layout='wide',initial_sidebar_state="collapsed")

st.title("🦁🦓 Predator-Prey Ecosystem Simulator")

st.divider()
sidebar_col,center,main_col=st.columns([1.2,0.3,4],gap="small", vertical_alignment="top")



with sidebar_col:
        
    st.markdown(
        "**Adjust the parameters below to see how magical creatures interact through predator-prey dynamics!**"
    )

    #==============================================================================================
    # 2. Interactive sliders for parameters
    #============================================================================================== 

    X0 = st.slider("Initial Prey Population (X0)", min_value=0.0,max_value=50.0,value=10.0,step=1.0) 
    Y0 = st.slider("Initial Predator Population (Y0)", min_value=0.0,max_value=50.0,value=10.0,step=1.0) 
   
   
    alpha=st.slider("Prey Growth Rate (ɑ)", min_value=0.0,max_value=3.0,value=1.0,step=0.1)
    beta=st.slider("Predation Growth Rate (β)", min_value=0.0,max_value=3.0,value=1.0,step=0.1)

    delta=st.slider("Growth per Prey Eaten (δ)", min_value=0.0,max_value=1.0,value=0.075,step=0.01)
    gamma=st.slider("Predator Death Rate (γ)", min_value=0.0,max_value=3.0,value=1.5,step=0.1)

    t_end=st.slider("Simulation Time", min_value=1,max_value=50,value=20,step=1)

    params={
        "alpha":alpha,
        "beta":beta,
        "delta":delta,
        "gamma":gamma
    }
    t_span=[0,t_end]
    Y0=np.array([X0,Y0]).flatten()

#==============================================================================================
# 3. Solving using solve_ivp
#============================================================================================== 
sol=solve_ivp_model(lotka_volterra,Y0,t_span,params)
t_ivp=sol.t
prey_ivp=sol.y[0]
predator_ivp=sol.y[1]

#==============================================================================================
# 4. Plot the results
#============================================================================================== 
with main_col:
    fig=go.Figure()

    fig.add_trace(go.Scatter(x=t_ivp,y=prey_ivp,mode='lines', name='🦓 Prey'))
    fig.add_trace(go.Scatter(x=t_ivp,y=predator_ivp,mode='lines', name='🦁 Predator'))

    fig.update_layout(
        title="Predator-Prey Dynamics Over Time",
        xaxis_title="Time",
        yaxis_title="Population",
        template="plotly_dark"
    )


    #==============================================================================================
    # 7. Display plot in Streamlit
    #============================================================================================== 

    st.plotly_chart(fig,use_container_width=True) 
