"""
1_plant_simulation.py

Interactive Streamlit page to simulate logistic growth of magical plants.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Asegúrate de que estas rutas existan en tu proyecto
from core.plants_models import logistic_growth
from core.solvers import solve_ivp_model

# ============================================================
# 1. Page setup & Lore
# ============================================================

st.set_page_config(
    page_title="Magical Plants Simulator",
    layout='wide',
    initial_sidebar_state="collapsed"
)

st.title("🌱 Cámara de Cultivo: Flora Arcana")


with st.container(border=True):
    col_lore, col_img = st.columns([3, 1])
    with col_lore:
        st.markdown("""
        **Bitácora del Archimagister:**
        
        Estás observando la **Vitis Aetherea** (Vid Etérea). Esta planta mágica no crece infinitamente; 
        su expansión está limitada por la saturación de maná del entorno.
        
        Utilizamos el **Modelo Logístico** para predecir su comportamiento: al principio crece exponencialmente, 
        pero se ralentiza al acercarse al límite de energía del ambiente.
        """)
  
st.divider()

# LaTeX Section
col_eq1, col_eq2 = st.columns(2)
with col_eq1:
    st.info("📜 **Modelo Matemático (Logístico)**")
    st.latex(r"\frac{dP}{dt} = r \cdot P \left(1 - \frac{P}{K}\right)")

with col_eq2:
    st.warning("🔮 **Variables Arcanas**")
    st.markdown(r"""
    - $P(t)$: Población de la planta (Biomasa mágica).
    - $r$: **Tasa de Vitalidad** (Velocidad de crecimiento).
    - $K$: **Saturación de Maná** (Capacidad de carga máxima).
    """)

st.divider()

sidebar_col, center, main_col = st.columns([1.2, 0.1, 4], gap="small", vertical_alignment="top")

with sidebar_col:
    st.subheader("🎛️ Panel de Control")
    st.markdown("**Calibra el entorno mágico:**")

    # ============================================================
    # 2. Interactive sliders for parameters
    # ============================================================

    P0 = st.slider("Semilla Inicial (P0)", min_value=0.1, max_value=5.0, value=0.1, step=0.1, help="Biomasa inicial al tiempo 0.")
    r = st.slider("Tasa de Vitalidad (r)", min_value=0.1, max_value=2.0, value=0.5, step=0.1, help="Velocidad intrínseca de crecimiento.")
    K = st.slider("Saturación de Maná (K)", min_value=1.0, max_value=20.0, value=10.0, step=1.0, help="Límite máximo que el entorno soporta.")
    t_end = st.slider("Tiempo de Simulación", min_value=5, max_value=100, value=30, step=5)

    # Method selector to test future implementations
    method_selector = st.selectbox(
        "Método de Resolución",
        ["Scipy (Exacto)", "Euler Mejorado", "Runge-Kutta 4"],
        index=0
    )

    params = {"r": r, "K": K}
    t_span = [0, t_end]

# Variables to store plotting results
# By default, we use Scipy. If other methods are selected, they will overwrite these.
t_plot = []
P_plot = []
method_name = "Sin Datos"

# ============================================================
# 3. Solving using solve_ivp (Reference)
# ============================================================
sol = solve_ivp_model(logistic_growth, np.array([P0]), t_span, params)
t_scipy = sol.t
P_scipy = sol.y[0]

# Default plotting values
t_plot = t_scipy
P_plot = P_scipy
method_name = "Scipy (Reference)"

# ============================================================
# TODO 4. Solving using Improved Euler
# ============================================================
if method_selector == "Euler Mejorado":
    # HERE YOU MUST IMPLEMENT YOUR CODE
    # Example structure (uncomment when implemented):
    # t_euler, P_euler = my_improved_euler_function(logistic_growth, P0, t_span, params)
    
    # Placeholder actual:
    st.toast("⚠️ Euler Mejorado aún no implementado. Mostrando Scipy.", icon="🧪")
    # t_plot = t_euler
    # P_plot = P_euler
    # method_name = "Euler Mejorado"
    pass 

# ============================================================
# TODO 5. Solving using RK4
# ============================================================
if method_selector == "Runge-Kutta 4":
     # HERE YOU MUST IMPLEMENT YOUR CODE
    # Example structure (uncomment when implemented):
    # t_rk4, P_rk4 = my_rk4_function(logistic_growth, P0, t_span, params)
    
    # Current Placeholder:
    st.toast("⚠️ RK4 aún no implementado. Mostrando Scipy.", icon="🧪")
    # t_plot = t_rk4
    # P_plot = P_rk4
    # method_name = "RK4"
    pass

# ============================================================
# 6. Plot the results
# ============================================================
with main_col:
    fig = go.Figure()

    # 1. Simulation Line (Magic Purple - Visible in both Black/White modes)
    fig.add_trace(go.Scatter(
        x=t_plot, 
        y=P_plot, 
        mode='lines', 
        name=f'Crecimiento ({method_name})',
        line=dict(color='#8b5cf6', width=4)
    ))

     # 2. Carrying Capacity Line (Reference)
    fig.add_trace(go.Scatter(
        x=[0, t_end],
        y=[K, K],
        mode='lines',
        name='Saturación (K)',
        line=dict(color='#ef4444', width=2, dash='dash') 
    ))

    ## Layout configuration adaptable to theme
    fig.update_layout(
        title="📈 Dinámica de Crecimiento Arcano",
        xaxis_title="Tiempo (Ciclos)",
        yaxis_title="Biomasa (P)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # ============================================================
    # 7. Display plot in Streamlit
    # ============================================================
    
    st.plotly_chart(fig, use_container_width=True)

    # Quick metrics below the graph
    m1, m2, m3 = st.columns(3)
    m1.metric("Población Final", f"{P_plot[-1]:.2f}")
    m2.metric("Saturación (K)", f"{K}")
    
    # Calculate occupation percentage
    pct_ocupacion = (P_plot[-1] / K) * 100
    m3.metric("% Ocupación de Maná", f"{pct_ocupacion:.1f}%")