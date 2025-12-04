"""
Interactive Streamlit page to simulate logistic growth of magical plants.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Importaciones de tus módulos
from core.plants_models import logistic_growth, logistic_growth_inverse
from core.solvers import (
    solve_ivp_model,
    improved_euler,
    runge_kutta_4,
    compare_numerical_methods
)

# ============================================================
# 0. Helpers & Configuration
# ============================================================

# Adaptador para que Scipy se comporte igual que tus funciones euler/rk4
# (recibiendo h, aunque Scipy es adaptativo, usaremos h como max_step)
def scipy_adapter(f, t0, y0, h, tf, params):
    # h actúa como max_step para forzar resolución si el usuario quiere
    step = h if h > 0 else 0.1
    sol = solve_ivp_model(f, np.array([y0]), [t0, tf], params, max_step=step)
    return sol.t, sol.y[0]

METHOD_MAP = {
    "Euler Mejorado": improved_euler,
    "Runge-Kutta 4": runge_kutta_4,
    "Scipy (Exacto)": scipy_adapter
}

# Inicialización del Estado (Con un elemento por defecto)
if 'comparison_stack' not in st.session_state:
    st.session_state['comparison_stack'] = [
        {
            "name": "Euler Mejorado (h=0.50)",
            "method_name": "Euler Mejorado",
            "h": 0.5
        }
    ]

# ============================================================
# 1. Page setup & Lore
# ============================================================

st.set_page_config(
    page_title="Magical Plants Simulator",
    layout='wide',
    initial_sidebar_state="expanded"
)

st.title("🌱 Cámara de Cultivo: Flora Arcana")

with st.container(border=True):
    col_lore, col_img = st.columns([3, 1])
    with col_lore:
        st.markdown("""
        **Bitácora del Archimagister:**
        
        Utiliza el **Stack de Comparación** en el panel lateral para añadir diferentes métodos numéricos
        o variaciones de precisión (paso $h$). 
        
        Observa cómo compiten entre sí en la gráfica interactiva y analiza sus errores.
        """)

st.divider()

sidebar_col, center, main_col = st.columns([1.3, 0.1, 4], gap="small", vertical_alignment="top")

with sidebar_col:
    st.subheader("🎛️ Configuración")

    # --- Parámetros Físicos ---
    st.markdown("##### 1. Entorno Mágico")
    P0 = st.slider("Semilla (P0)", 0.1, 5.0, 0.1, 0.1)
    r = st.slider("Vitalidad (r)", 0.1, 2.0, 1.2, 0.1)
    K = st.slider("Maná (K)", 1.0, 20.0, 15.0, 1.0)
    t_end = st.slider("Tiempo Total", 5, 100, 60, 5)

    params = {"r": r, "K": K}

    st.markdown("---")

    # --- Constructor de Comparaciones ---
    st.markdown("##### 2. Stack de Comparación")

    with st.container(border=True):
        c1, c2 = st.columns([1.5, 1])
        with c1:
            comp_method = st.selectbox(
                "Método",
                ["Euler Mejorado", "Runge-Kutta 4", "Scipy (Exacto)"],
                key="comp_select"
            )
        with c2:
            comp_h = st.number_input("Paso (h)", 0.01, 5.0, 0.5, 0.1, key="comp_h")

        if st.button("➕ Añadir Gráfica", use_container_width=True):
            # Añadimos a la sesión
            entry = {
                "name": f"{comp_method} (h={comp_h})",
                "method_name": comp_method,
                "h": comp_h
            }
            st.session_state['comparison_stack'].append(entry)

    # --- Lista Activa ---
    st.markdown("###### Elementos Activos:")

    if not st.session_state['comparison_stack']:
        st.caption("⚠️ La lista está vacía.")
    else:
        # Iteramos con índice invertido para mostrar el más reciente arriba (opcional)
        # o normal. Usaremos enumerate normal.
        for i, item in enumerate(st.session_state['comparison_stack']):
            # Contenedor pequeño para cada item
            cols = st.columns([0.1, 3, 1])
            cols[0].write(f"**{i+1}.**")
            cols[1].caption(f"{item['name']}")
            if cols[2].button("❌", key=f"del_{i}"):
                st.session_state['comparison_stack'].pop(i)
                st.rerun()

        if st.button("Limpiar Todo", type="secondary", use_container_width=True):
            st.session_state['comparison_stack'] = []
            st.rerun()


# ============================================================
# 4. Visualization Logic
# ============================================================
with main_col:
    tab1, tab2 = st.tabs(["📈 Trayectorias", "🧪 Análisis de Error"])

    # --- TAB 1: Visualización ---
    with tab1:
        fig = go.Figure()

        # 1. Background Reference (Siempre útil tener la verdad absoluta de fondo tenue)
        # Calculamos una referencia exacta oculta para contexto visual
        sol_ref = solve_ivp_model(logistic_growth, np.array([P0]), [0, t_end], params, max_step=0.1)
        fig.add_trace(go.Scatter(
            x=sol_ref.t, y=sol_ref.y[0],
            mode='lines',
            name='Referencia (Oculta)',
            line=dict(color='lightgray', width=4),
            opacity=0.3,
            showlegend=False
        ))

        # 2. Plot Stack Items
        if not st.session_state['comparison_stack']:
            st.info("👈 Añade elementos desde el panel lateral para comenzar la simulación.")

        else:
            for i, item in enumerate(st.session_state['comparison_stack']):
                solver_func = METHOD_MAP[item['method_name']]

                # Ejecutar solver
                try:
                    t_res, P_res = solver_func(logistic_growth, 0, P0, item['h'], t_end, params)

                    fig.add_trace(go.Scatter(
                        x=t_res, y=P_res,
                        mode='lines+markers' if len(t_res) < 30 else 'lines',
                        name=item['name'],
                        line=dict(width=2),
                        marker=dict(size=5)
                    ))
                except Exception as e:
                    st.error(f"Error calculando {item['name']}: {e}")

        # 3. Línea de Saturación K
        fig.add_trace(go.Scatter(
            x=[0, t_end], y=[K, K],
            mode='lines', name='Saturación (K)',
            line=dict(color='#ef4444', width=1, dash='dash')
        ))

        fig.update_layout(
            title="Comparativa de Trayectorias",
            xaxis_title="Tiempo",
            yaxis_title="Biomasa (P)",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)


    # --- TAB 2: Errores ---
    with tab2:
        if not st.session_state['comparison_stack']:
            st.warning("No hay datos para comparar.")
        else:
            st.subheader("Comparativa de Precisión")

            # Convertimos stack al formato para solvers.py
            methods_config = []
            for item in st.session_state['comparison_stack']:
                methods_config.append({
                    'name': item['name'],
                    'method': METHOD_MAP[item['method_name']],
                    'h': item['h']
                })

            col_scale, _ = st.columns([1, 4])
            with col_scale:
                scale_opt = st.radio("Escala Error:", ["log", "linear"], horizontal=True)

            try:
                fig_comp = compare_numerical_methods(
                    f=logistic_growth,
                    f_inv=logistic_growth_inverse,
                    t0=0,
                    y0=P0,
                    tf=t_end,
                    methods_config=methods_config,
                    scale=scale_opt,
                    f_params=params
                )
                st.pyplot(fig_comp)
            except Exception as e:
                st.error(f"Error en el análisis de convergencia: {e}")
                st.info("Si usas pasos muy grandes (h > 1.5), es posible que los métodos exploten.")