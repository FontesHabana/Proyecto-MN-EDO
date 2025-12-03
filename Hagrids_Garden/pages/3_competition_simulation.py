"""
3_competition_simulation.py

Interactive Streamlit page to simulate two populations competing for shared resources.
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Imports actualizados
from core.competition_models import two_species_competition
from core.solvers import solve_ivp_model, improved_euler, runge_kutta_4

# ==============================================================================================
# 1. Page setup & Custom Styles
# ==============================================================================================

st.set_page_config(
    page_title="Arena de Competición | Arcane Lab",
    layout='wide',
    initial_sidebar_state="expanded"
)

st.title("⚔️ Arena Elemental: Competición de Especies")

# ==============================================================================================
# 2. Lore & Mathematical Model
# ==============================================================================================

with st.container(border=True):
    col_lore, col_img = st.columns([3, 1])
    with col_lore:
        st.markdown("""
        **Bitácora del Archimagister:**
        
        Dos especies mágicas intentan colonizar el mismo nodo de energía: 
        **Hongos Ígneos (P)** vs **Cristales Gélidos (C)**.
        
        Ambos consumen el mismo **Maná (K)**. Su supervivencia depende no solo de qué tan rápido crecen, 
        sino de cuánto se "estorban" mutuamente.
        """)

# LaTeX Equations
col_math1, col_math2 = st.columns(2)
with col_math1:
    st.info("📜 **Modelo de Competencia (Gause-Witt)**")
    st.latex(r"\frac{dP}{dt} = r_p P \left(1 - \frac{P + \alpha C}{K}\right)")
    st.latex(r"\frac{dC}{dt} = r_c C \left(1 - \frac{C + \beta P}{K}\right)")

with col_math2:
    st.success("🔮 **Variables de Conflicto**")
    st.markdown(r"""
    - $P, C$: Población de Hongos y Cristales.
    - $r_p, r_c$: Velocidad de crecimiento.
    - $K$: **Maná Total** (Recursos compartidos).
    - $\alpha, \beta$: Coeficientes de interferencia mutua.
    """)

st.divider()

# ==============================================================================================
# 3. Interactive Layout
# ==============================================================================================

sidebar_col, center, main_col = st.columns([1.3, 0.1, 4], gap="small", vertical_alignment="top")

with sidebar_col:
    st.subheader("🎛️ Configuración de la Arena")
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
    st.markdown("##### 🚩 Inicio")
    P0 = st.slider("Población Hongos (P0)", 0.0, 10.0, 2.0, 0.1)
    C0 = st.slider("Población Cristales (C0)", 0.0, 10.0, 2.0, 0.1)

    st.markdown("##### ⚡ Capacidades")
    r_p = st.slider("Tasa Hongos (r_p)", 0.1, 2.0, 1.0, 0.1)
    r_c = st.slider("Tasa Cristales (r_c)", 0.1, 2.0, 0.8, 0.1)
    K = st.slider("Maná Disponible (K)", 1.0, 30.0, 15.0, 1.0)

    st.markdown("##### ⚔️ Interacción")
    alpha = st.slider("Daño Cristales -> Hongos (α)", 0.0, 2.0, 0.5, 0.1)
    beta = st.slider("Daño Hongos -> Cristales (β)", 0.0, 2.0, 0.5, 0.1)

    t_end = st.slider("Duración", 5, 100, 40, 5)

    params = {
        "r_p": r_p, "r_c": r_c,
        "K": K, "alpha": alpha, "beta": beta
    }
    t_span = [0, t_end]
    Y0 = np.array([P0, C0]).flatten()

# ==============================================================================================
# 4. Solving execution logic
# ==============================================================================================

t_sim = []
P_sim = []
C_sim = []

if solver_method == "Scipy (Adaptativo)":
    sol = solve_ivp_model(two_species_competition, Y0, t_span, params)
    t_sim = sol.t
    P_sim = sol.y[0]
    C_sim = sol.y[1]

elif solver_method == "Euler Mejorado":
    # Custom solver devuelve matriz (steps, 2)
    t_sim, y_sim = improved_euler(two_species_competition, 0, Y0, h, t_end, params)
    P_sim = y_sim[:, 0]
    C_sim = y_sim[:, 1]

elif solver_method == "Runge-Kutta 4":
    t_sim, y_sim = runge_kutta_4(two_species_competition, 0, Y0, h, t_end, params)
    P_sim = y_sim[:, 0]
    C_sim = y_sim[:, 1]

# ==============================================================================================
# 5. Plot the results
# ==============================================================================================
with main_col:

    tab1, tab2 = st.tabs(["📊 Evolución Temporal", "🌀 Retrato de Fase"])

    # --- Tab 1: Time Series ---
    with tab1:
        fig = go.Figure()

        # Hongos (Red)
        fig.add_trace(go.Scatter(
            x=t_sim, y=P_sim,
            mode='lines', name='🔥 Hongos Ígneos',
            line=dict(color='#ef4444', width=3)
        ))

        # Cristales (Blue)
        fig.add_trace(go.Scatter(
            x=t_sim, y=C_sim,
            mode='lines', name='❄️ Cristales Gélidos',
            line=dict(color='#3b82f6', width=3)
        ))

        # Limit line (K)
        fig.add_trace(go.Scatter(
            x=[0, t_end], y=[K, K],
            mode='lines', name='Límite de Maná (K)',
            line=dict(color='gray', dash='dash', width=1)
        ))

        fig.update_layout(
            title=f"Batalla por el Maná ({solver_method})",
            xaxis_title="Tiempo",
            yaxis_title="Población",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Tab 2: Phase Plane ---
    with tab2:
        fig_phase = go.Figure()

        fig_phase.add_trace(go.Scatter(
            x=P_sim, y=C_sim,
            mode='lines',
            name='Trayectoria',
            line=dict(color='#a855f7', width=3)
        ))

        # Start marker
        fig_phase.add_trace(go.Scatter(
            x=[P0], y=[C0],
            mode='markers', name='Inicio',
            marker=dict(size=10, color='white')
        ))

        # End marker
        fig_phase.add_trace(go.Scatter(
            x=[P_sim[-1]], y=[C_sim[-1]],
            mode='markers', name='Estado Final',
            marker=dict(size=12, color='yellow', symbol='star')
        ))

        fig_phase.update_layout(
            title=f"Mapa de Dominio - {solver_method}",
            xaxis_title="Población Hongos (P)",
            yaxis_title="Población Cristales (C)",
            hovermode="closest"
        )

        st.plotly_chart(fig_phase, use_container_width=True)

        # Analysis
        if P_sim[-1] < 0.1 and C_sim[-1] > 0.1:
            st.info("❄️ **Resultado:** Los Cristales han dominado la arena.")
        elif C_sim[-1] < 0.1 and P_sim[-1] > 0.1:
            st.warning("🔥 **Resultado:** Los Hongos han conquistado el territorio.")
        elif P_sim[-1] > 0.1 and C_sim[-1] > 0.1:
            st.success("🤝 **Resultado:** Coexistencia estable alcanzada.")