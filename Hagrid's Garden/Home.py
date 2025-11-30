import streamlit as st

# ============================================================
# Page config
# ============================================================

st.set_page_config(
    page_title="Arcane Dynamics Lab",
    layout="wide",
)

# ============================================================
# Custom CSS styling
# ============================================================

st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}

.hero {
    padding: 40px;
    text-align: center;
    background: linear-gradient(135deg, #1c2333, #2f3b52);
    color: white;
    border-radius: 15px;
    margin-bottom: 30px;
}

.hero-title {
    font-size: 48px;
    font-weight: 700;
    margin-bottom: 10px;
}

.hero-subtitle {
    font-size: 22px;
    opacity: 0.85;
}

.section-title {
    font-size: 30px;
    font-weight: 600;
    border-left: 6px solid #7aa2f7;
    padding-left: 10px;
    margin-top: 40px;
}

.feature-card {
    background: #111827;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #374151;
    transition: 0.2s;
}

.feature-card:hover {
    border-color: #60a5fa;
    box-shadow: 0px 0px 12px rgba(96,165,250,0.35);
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# Hero Section (banner)
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">Arcane Dynamics Lab</div>
    <div class="hero-subtitle">
        Un laboratorio mágico para explorar ecuaciones diferenciales.<br>
        Observa, simula y experimenta con plantas encantadas, criaturas fantásticas y sistemas creados por ti.
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Introduction Section
# ============================================================

st.markdown('<div class="section-title">¿Qué es este proyecto?</div>', unsafe_allow_html=True)

st.write("""
Arcane Dynamics Lab es un entorno interactivo diseñado para aprender y explorar ecuaciones diferenciales ordinarias (EDOs)  
de una manera visual, divertida y científicamente rigurosa.

Aquí podrás:

- Simular modelos clásicos y mágicos.  
- Visualizar soluciones con gráficos dinámicos.  
- Manipular parámetros para ver cómo cambia la dinámica del sistema.  
- Crear tus propios experimentos matemáticos.

""")


# ============================================================
# Feature cards with navigation buttons
# ============================================================

st.markdown('<div class="section-title">Explora el laboratorio</div>', unsafe_allow_html=True)

col1, col2, col3,col4 = st.columns(4)

with col1:
    st.image("assets\images\Captura de pantalla 2025-09-28 185748.png", use_container_width=True)
    st.subheader("🌱 Simulación de plantas")
    st.write("Estudia el crecimiento de plantas mágicas con modelos logísticos y recursos limitados.")
    st.page_link("pages/1_plant_simulation.py", label="Ir a Plant Simulation →")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.image("assets\images\Captura de pantalla 2025-09-28 185748.png", use_container_width=True)
    st.subheader("🐾 SSimulación de criaturas")
    st.write("Explora modelos depredador-presa y otras dinámicas mágicas.")
    st.page_link("pages/2_creatures_simulation.py", label="Ir a Creatures Simulation →")
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.image("assets\images\Captura de pantalla 2025-09-28 185748.png", use_container_width=True)
    st.subheader("🦁Competición de especies")
    st.write("Dos especies compitiendo por recursos liimitados.")
    st.page_link("pages/3_competition_simulation.py", label="Ir a Competition Simulation →")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.image("assets\images\Captura de pantalla 2025-09-28 185748.png", use_container_width=True)
    st.subheader("🔮 Panel de Experimentos")
    st.write("Combina EDOs, diseña sistemas personalizados y ejecuta experimentos complejos.")
    st.page_link("pages/4_experiment_panel.py", label="Ir a Experiment Panel →")
    st.markdown('</div>', unsafe_allow_html=True)
# Footer (optional)

st.markdown("---")
st.caption(" Proyecto de simulación de EDOs y Matemática Numérica · 2025")