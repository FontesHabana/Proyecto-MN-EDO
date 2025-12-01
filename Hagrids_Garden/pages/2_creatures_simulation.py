"""
2_creatures_simulation.py

Interactive Streamlit page to simulate predator-prey interactions using the classic Lotka-Volterra model.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.creatures_models import lotka_volterra
from core.solvers import solve_ivp_model

# ==============================================================================================
# 1. Page setup & Custom Styles
# ============================================================================================== 

st.set_page_config(
    page_title="Santuario de Criaturas | Arcane Lab",
    layout='wide',
    initial_sidebar_state="expanded"
)

st.title("🦁⚡ Santuario: Dinámica de Depredadores")
# 2. Lore & Mathematical Model
# ============================================================================================== 
with st.container(border=True):
    col_lore, col_img = st.columns([3, 1])
    with col_lore:
        st.markdown("""
        **Bitácora del Archimagister:**
        
    Observamos la eterna danza entre los **Conejos de Cristal (Presas)** y 
    los **Zorros Espirituales (Depredadores)**. 
    A diferencia de la agricultura estática, aquí el equilibrio es cíclico.
    
    Si los conejos proliferan, los zorros tienen más alimento y se multiplican. 
    Cuando hay demasiados zorros, los conejos disminuyen, causando que los zorros mueran de hambre, 
    reiniciando el ciclo. Usa el modelo **Lotka-Volterra** para armonizar el ecosistema.
        """)
  
# Lore text with the specific blue border style


# LaTeX Equations
col_eq1, col_eq2 = st.columns(2)

with col_eq1:
    st.info("📜 **Ecuaciones de Equilibrio (Lotka-Volterra)**")
    st.latex(r"\frac{dx}{dt} = \alpha x - \beta x y")
    st.latex(r"\frac{dy}{dt} = \delta x y - \gamma y")

with col_eq2:
    st.success("🔮 **Glosario de Variables**")
    st.markdown(r"""
    - $x$: Población de **Presas** (Conejos).
    - $y$: Población de **Depredadores** (Zorros).
    - $\alpha$: Tasa de crecimiento natural de presas.
    - $\beta$: Tasa de depredación (caza).
    - $\delta$: Eficiencia (cuánto crece el depredador al comer).
    - $\gamma$: Tasa de mortalidad natural de depredadores.
    """)

st.divider()

# ==============================================================================================
# 3. Interactive Layout
# ============================================================================================== 

sidebar_col, center, main_col = st.columns([1.2, 0.1, 4], gap="small", vertical_alignment="top")

with sidebar_col:
    st.subheader("🎛️ Panel de Control")
    st.write("Ajusta las variables del ecosistema:")

    # ----------------------------------------------------------------------
    # Sliders for parameters (Renamed to Spanish)
    # ---------------------------------------------------------------------- 

    st.markdown("---")
    st.markdown("#### 🌱 Población Inicial")
    X0 = st.slider("Presas Iniciales (x)", min_value=0.0, max_value=100.0, value=40.0, step=1.0) 
    Y0_slider = st.slider("Depredadores Iniciales (y)", min_value=0.0, max_value=50.0, value=9.0, step=1.0) 
   
    st.markdown("#### ⚡ Dinámica")
    alpha = st.slider("Reproducción Presas (α)", min_value=0.0, max_value=2.0, value=0.1, step=0.05, help="Qué tan rápido nacen los conejos si no hay zorros.")
    beta = st.slider("Tasa de Caza (β)", min_value=0.0, max_value=0.1, value=0.02, step=0.001, format="%.3f", help="Probabilidad de que un conejo sea cazado.")

    delta = st.slider("Eficiencia de Conversión (δ)", min_value=0.0, max_value=0.1, value=0.01, step=0.001, format="%.3f", help="Cuántos zorros nuevos nacen por cada conejo comido.")
    gamma = st.slider("Mortalidad Depredadores (γ)", min_value=0.0, max_value=1.0, value=0.1, step=0.05, help="Qué tan rápido mueren los zorros sin comida.")

    t_end = st.slider("Tiempo de Simulación", min_value=10, max_value=200, value=100, step=10)

    # Dictionary for solver
    params = {
        "alpha": alpha,
        "beta": beta,
        "delta": delta,
        "gamma": gamma
    }
    t_span = [0, t_end]
    
    # Prepare initial state array
    initial_state = np.array([X0, Y0_slider])

# ==============================================================================================
# 4. Solving using solve_ivp
# ============================================================================================== 

sol = solve_ivp_model(lotka_volterra, initial_state, t_span, params)
t_ivp = sol.t
prey_ivp = sol.y[0]
predator_ivp = sol.y[1]

# ==============================================================================================
# 5. Plot the results
# ============================================================================================== 

with main_col:
    # Use tabs to show different perspectives of the data
    tab1, tab2 = st.tabs(["📈 Evolución Temporal", "🌀 Ciclo de Fase"])

    # --- Tab 1: Time Series (Time vs Population) ---
    with tab1:
        fig_time = go.Figure()

        # Prey Trace (Cyan/Blue)
        fig_time.add_trace(go.Scatter(
            x=t_ivp, y=prey_ivp, 
            mode='lines', 
            name='🦓 Presas (Cristal)',
            line=dict(color='#22d3ee', width=2),
            fill='tozeroy',
            fillcolor='rgba(34, 211, 238, 0.1)'
        ))

        # Predator Trace (Red/Orange)
        fig_time.add_trace(go.Scatter(
            x=t_ivp, y=predator_ivp, 
            mode='lines', 
            name='🦁 Depredadores (Espíritus)',
            line=dict(color='#f472b6', width=2),
            fill='tozeroy',
            fillcolor='rgba(244, 114, 182, 0.1)'
        ))

        fig_time.update_layout(
            title="Dinámica Poblacional en el Tiempo",
            xaxis_title="Tiempo",
            yaxis_title="Cantidad de Criaturas",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_time, use_container_width=True)

    # --- Tab 2: Phase Portrait (Prey vs Predator) ---
    with tab2:
        fig_phase = go.Figure()

        fig_phase.add_trace(go.Scatter(
            x=prey_ivp, y=predator_ivp,
            mode='lines',
            name='Ciclo de Fase',
            line=dict(color='#a78bfa', width=3), # Purple line
        ))

        # Mark the start point
        fig_phase.add_trace(go.Scatter(
            x=[prey_ivp[0]], y=[predator_ivp[0]],
            mode='markers',
            name='Inicio',
            marker=dict(color='white', size=10, symbol='diamond')
        ))

        fig_phase.update_layout(
            title="Retrato de Fase (Ciclicidad)",
            xaxis_title="Población de Presas (x)",
            yaxis_title="Población de Depredadores (y)",
            hovermode="closest"
        )
        
        # Add annotation explaining the cycle
        st.plotly_chart(fig_phase, use_container_width=True)
        st.caption("ℹ️ *Si la gráfica forma un círculo cerrado, el sistema es estable y periódico.*")
        
        
        