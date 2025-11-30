"""
3_competition_simulation.py

Interactive Streamlit page to simulate two populations competing for shared resources.
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.competition_models import two_species_competition
from core.solvers import solve_ivp_model


# ==============================================================================================
# 1. Page setup & Custom Styles
# ============================================================================================== 
   
st.set_page_config(
    page_title="Arena de Competición | Arcane Lab",
    layout='wide',
    initial_sidebar_state="expanded"
)

# Custom CSS (Lore box without left border as requested)
st.markdown("""
<style>
    .lore-container {
        background-color: #111827; /* Dark gray/blue background */
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 25px;
        color: #e5e7eb;
        border: 1px solid #374151; /* Subtle full border instead of left accent */
    }
    .species-1 { color: #f87171; font-weight: bold; } /* Red */
    .species-2 { color: #60a5fa; font-weight: bold; } /* Blue */
</style>
""", unsafe_allow_html=True)

st.title("⚔️ Arena Elemental: Competición de Especies")

# ==============================================================================================
# 2. Lore & Mathematical Model
# ============================================================================================== 

# Lore text without the left border style

with st.container(border=True):
    col_lore, col_img = st.columns([3, 1])
    with col_lore:
        st.markdown("""
        **Bitácora del Archimagister:**
        
    Dos especies mágicas intentan colonizar el mismo nodo de energía. 
    Por un lado, los **Hongos Ígneos (P)**, agresivos y rápidos. 
    Por otro, los **Cristales Gélidos (C)**, defensivos y resistentes.
    
      
    Ambos consumen el mismo **Maná (K)**. Su supervivencia depende no solo de qué tan rápido crecen, 
    sino de cuánto se "estorban" mutuamente (Coeficientes de competencia).
    ¿Podrán coexistir, o una extinguirá a la otra?
        """)
  

# LaTeX Equations
col_math1, col_math2 = st.columns(2)

with col_math1:
    st.info("📜 **Modelo de Competencia (Gause-Witt)**")
    # Equation showing shared Carrying Capacity (K)
    st.latex(r"\frac{dP}{dt} = r_p P \left(1 - \frac{P + \alpha C}{K}\right)")
    st.latex(r"\frac{dC}{dt} = r_c C \left(1 - \frac{C + \beta P}{K}\right)")

with col_math2:
    st.success("🔮 **Variables de Conflicto**")
    st.markdown(r"""
    - $P, C$: Población de Hongos y Cristales.
    - $r_p, r_c$: Velocidad de crecimiento de cada especie.
    - $K$: **Maná Total** (Recursos compartidos).
    - $\alpha$: Cuánto afecta la presencia de **Cristales** a los Hongos.
    - $\beta$: Cuánto afecta la presencia de **Hongos** a los Cristales.
    """)

st.divider()

# ==============================================================================================
# 3. Interactive Layout
# ============================================================================================== 

sidebar_col, center, main_col = st.columns([1.2, 0.1, 4], gap="small", vertical_alignment="top")

with sidebar_col:
    st.subheader("🎛️ Configuración de la Arena")
    st.write("Define las reglas del combate:")

    # ----------------------------------------------------------------------
    # Interactive sliders for parameters
    # ---------------------------------------------------------------------- 

    st.markdown("---")
    st.markdown("#### 🚩 Inicio")
    P0 = st.slider("Población Hongos (P0)", min_value=0.0, max_value=10.0, value=2.0, step=0.1) 
    C0 = st.slider("Población Cristales (C0)", min_value=0.0, max_value=10.0, value=2.0, step=0.1) 

    st.markdown("#### ⚡ Capacidades")
    r_p = st.slider("Tasa Crecimiento Hongos (r_p)", min_value=0.1, max_value=2.0, value=1.0, step=0.1) 
    r_c = st.slider("Tasa Crecimiento Cristales (r_c)", min_value=0.1, max_value=2.0, value=0.8, step=0.1) 
    K = st.slider("Maná Disponible (K)", min_value=1.0, max_value=30.0, value=15.0, step=1.0)

    st.markdown("#### ⚔️ Interacción")
    # Alpha: Effect of C on P
    alpha = st.slider("Daño de Cristales a Hongos (α)", min_value=0.0, max_value=2.0, value=0.5, step=0.1, 
                      help="Si es alto, los Cristales son tóxicos para los Hongos.")
    # Beta: Effect of P on C
    beta = st.slider("Daño de Hongos a Cristales (β)", min_value=0.0, max_value=2.0, value=0.5, step=0.1,
                     help="Si es alto, los Hongos sofocan a los Cristales.")

    t_end = st.slider("Duración del Experimento", min_value=5, max_value=100, value=40, step=5)

    params = {
        "r_p": r_p,
        "r_c": r_c,
        "K": K,
        "alpha": alpha,
        "beta": beta
    }

    t_span = [0, t_end]
    Y0 = np.array([P0, C0]).flatten()

# ==============================================================================================
# 4. Solving using solve_ivp
# ============================================================================================== 
sol = solve_ivp_model(two_species_competition, Y0, t_span, params)
t_ivp = sol.t
P_ivp = sol.y[0]
C_ivp = sol.y[1]

# ==============================================================================================
# TODO 4. Solving using RK4
# ============================================================================================== 
# (Placeholder for future implementation)

# ==============================================================================================
# 5. Plot the results
# ============================================================================================== 
with main_col:
    
    # Tabs for different visualizations (Time Series vs Phase Plane)
    tab1, tab2 = st.tabs(["📊 Evolución Temporal", "🌀 Retrato de Fase"])

    # --- Tab 1: Time Series ---
    with tab1:
        fig = go.Figure()
        
        # Trace for Species 1 (Fire/Red)
        fig.add_trace(go.Scatter(
            x=t_ivp, y=P_ivp, 
            mode='lines', 
            name='🔥 Hongos Ígneos',
            line=dict(color='#ef4444', width=3)
        ))
        
        # Trace for Species 2 (Ice/Blue)
        fig.add_trace(go.Scatter(
            x=t_ivp, y=C_ivp, 
            mode='lines', 
            name='❄️ Cristales Gélidos',
            line=dict(color='#3b82f6', width=3)
        ))
        
        # Limit line (K)
        fig.add_trace(go.Scatter(
            x=[0, t_end], y=[K, K],
            mode='lines',
            name='Límite de Maná (K)',
            line=dict(color='gray', dash='dash', width=1)
        ))

        fig.update_layout(
            title="Batalla por el Maná",
            xaxis_title="Tiempo",
            yaxis_title="Población",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Tab 2: Phase Plane (Visualizing the equilibrium point) ---
    with tab2:
        fig_phase = go.Figure()

        fig_phase.add_trace(go.Scatter(
            x=P_ivp, y=C_ivp,
            mode='lines',
            name='Trayectoria',
            line=dict(color='#a855f7', width=3) # Purple trajectory
        ))
        
        # Start point marker
        fig_phase.add_trace(go.Scatter(
            x=[P0], y=[C0],
            mode='markers',
            name='Inicio',
            marker=dict(size=10, color='white')
        ))

        # End point marker (Equilibrium?)
        fig_phase.add_trace(go.Scatter(
            x=[P_ivp[-1]], y=[C_ivp[-1]],
            mode='markers',
            name='Estado Final',
            marker=dict(size=12, color='yellow', symbol='star')
        ))

        fig_phase.update_layout(
            title="Mapa de Dominio (Fase)",
            xaxis_title="Población Hongos (P)",
            yaxis_title="Población Cristales (C)",
            hovermode="closest"
        )
        
        st.plotly_chart(fig_phase, use_container_width=True)
        
        # Brief analysis text
        if P_ivp[-1] < 0.1 and C_ivp[-1] > 0.1:
            st.info("❄️ **Resultado:** Los Cristales han dominado la arena.")
        elif C_ivp[-1] < 0.1 and P_ivp[-1] > 0.1:
            st.warning("🔥 **Resultado:** Los Hongos han conquistado el territorio.")
        elif P_ivp[-1] > 0.1 and C_ivp[-1] > 0.1:
            st.success("🤝 **Resultado:** Coexistencia estable alcanzada.")