"""
2_creatures_simulation.py

Interactive Streamlit page to simulate predator-prey interactions using the classic Lotka-Volterra model.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Importamos también tus métodos manuales
from core.creatures_models import lotka_volterra
from core.solvers import solve_ivp_model, improved_euler, runge_kutta_4

# ==============================================================================================
# 1. Page setup & Custom Styles
# ==============================================================================================

st.set_page_config(
    page_title="Santuario de Criaturas | Arcane Lab",
    layout='wide',
    initial_sidebar_state="expanded"
)

st.title("🦁⚡ Santuario: Dinámica de Depredadores")

# ==============================================================================================
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
        reiniciando el ciclo.
        """)

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
    - $\alpha, \beta$: Crecimiento presas / Tasa de caza.
    - $\delta, \gamma$: Eficiencia depredador / Mortalidad natural.
    """)

st.divider()

# ==============================================================================================
# 3. Interactive Layout
# ==============================================================================================

sidebar_col, center, main_col = st.columns([1.3, 0.1, 4], gap="small", vertical_alignment="top")

with sidebar_col:
    st.subheader("🎛️ Panel de Control")
    st.markdown("##### ⚙️ Motor Numérico")
    solver_method = st.selectbox(
        "Método de Integración",
        ["Scipy (Adaptativo)", "Euler Mejorado", "Runge-Kutta 4"]
    )

    h = 0.1
    if solver_method != "Scipy (Adaptativo)":
        h = st.slider("Paso de tiempo (h)", 0.01, 5.0, 0.1, 0.01)
    st.markdown("---")

    # --- Parameters ---
    st.markdown("##### 🌱 Población Inicial")
    X0 = st.slider("Presas Iniciales (x)", 0.0, 100.0, 40.0, 1.0)
    Y0_slider = st.slider("Depredadores Iniciales (y)", 0.0, 50.0, 9.0, 1.0)

    st.markdown("##### ⚡ Dinámica")
    alpha = st.slider("Reproducción Presas (α)", 0.0, 2.0, 0.1, 0.05)
    beta = st.slider("Tasa de Caza (β)", 0.0, 0.1, 0.02, 0.001, format="%.3f")
    delta = st.slider("Eficiencia (δ)", 0.0, 0.1, 0.01, 0.001, format="%.3f")
    gamma = st.slider("Mortalidad Zorros (γ)", 0.0, 1.0, 0.1, 0.05)

    t_end = st.slider("Tiempo de Simulación", 10, 200, 100, 10)

    params = {
        "alpha": alpha, "beta": beta,
        "delta": delta, "gamma": gamma
    }
    t_span = [0, t_end]
    initial_state = np.array([X0, Y0_slider])

# ==============================================================================================
# 4. Solving execution logic
# ==============================================================================================

t_sim = []
prey_sim = []
predator_sim = []

if solver_method == "Scipy (Adaptativo)":
    sol = solve_ivp_model(lotka_volterra, initial_state, t_span, params)
    t_sim = sol.t
    prey_sim = sol.y[0]
    predator_sim = sol.y[1]

elif solver_method == "Euler Mejorado":
    t_sim, y_sim = improved_euler(lotka_volterra, 0, initial_state, h, t_end, params)
    prey_sim = y_sim[:, 0]
    predator_sim = y_sim[:, 1]

elif solver_method == "Runge-Kutta 4":
    t_sim, y_sim = runge_kutta_4(lotka_volterra, 0, initial_state, h, t_end, params)
    prey_sim = y_sim[:, 0]
    predator_sim = y_sim[:, 1]

# ==============================================================================================
# 5. Plot the results
# ==============================================================================================

with main_col:
    tab1, tab2 = st.tabs(["📈 Evolución Temporal", "🌀 Ciclo de Fase"])

    # --- Tab 1: Time Series ---
    with tab1:
        fig_time = go.Figure()

        # Prey Trace
        fig_time.add_trace(go.Scatter(
            x=t_sim, y=prey_sim,
            mode='lines',
            name='🦓 Presas (Cristal)',
            line=dict(color='#22d3ee', width=2),
            fill='tozeroy',
            fillcolor='rgba(34, 211, 238, 0.1)'
        ))

        # Predator Trace
        fig_time.add_trace(go.Scatter(
            x=t_sim, y=predator_sim,
            mode='lines',
            name='🦁 Depredadores (Espíritus)',
            line=dict(color='#f472b6', width=2),
            fill='tozeroy',
            fillcolor='rgba(244, 114, 182, 0.1)'
        ))

        fig_time.update_layout(
            title=f"Dinámica Poblacional ({solver_method})",
            xaxis_title="Tiempo",
            yaxis_title="Cantidad de Criaturas",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_time, use_container_width=True)

    # --- Tab 2: Phase Portrait ---
    with tab2:
        fig_phase = go.Figure()

        fig_phase.add_trace(go.Scatter(
            x=prey_sim, y=predator_sim,
            mode='lines',
            name='Ciclo de Fase',
            line=dict(color='#a78bfa', width=3),
        ))

        # Start point
        fig_phase.add_trace(go.Scatter(
            x=[prey_sim[0]], y=[predator_sim[0]],
            mode='markers',
            name='Inicio',
            marker=dict(color='white', size=10, symbol='diamond')
        ))

        fig_phase.update_layout(
            title=f"Retrato de Fase - {solver_method}",
            xaxis_title="Población de Presas (x)",
            yaxis_title="Población de Depredadores (y)",
            hovermode="closest"
        )

        st.plotly_chart(fig_phase, use_container_width=True)
        st.caption("ℹ️ *Si usas un paso (h) muy grande en Métodos numéricos, verás cómo el círculo se rompe y espiraliza hacia afuera (error numérico).*")