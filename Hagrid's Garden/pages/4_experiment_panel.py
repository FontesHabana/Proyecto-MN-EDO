import streamlit as st
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Importamos nuestros módulos core
# Nota: Asegúrate de correr streamlit desde la carpeta raíz
from core.models_registry import MODELS_REGISTRY
from core.experiment_manager import ExperimentManager

st.set_page_config(page_title="Panel de Experimentos", layout="wide", page_icon="🧪")



# --- 1. GESTIÓN DE ESTADO (Session State) ---
# Necesitamos guardar los experimentos en memoria para que no se borren al recargar
if "experiments" not in st.session_state:
    st.session_state.experiments = [] # Lista de diccionarios
# Aquí guardaremos los modelos creados por el usuario en esta sesión
if "custom_models" not in st.session_state:
    st.session_state.custom_models = {}
# Función auxiliar para fusionar registros (Estándar + Usuario)
def get_full_registry():
    full_reg = MODELS_REGISTRY.copy()
    full_reg.update(st.session_state.custom_models)
    return full_reg

st.title("🧪 Experiment Panel: Gestor Dinámico de EDOs")
st.divider()
sidebar_col,center,main_col=st.columns([1.2,0.3,4],gap="small", vertical_alignment="top")

# --- 2. BARRA LATERAL: CREAR EXPERIMENTOS ---
with sidebar_col:
    sb_tab1, sb_tab2, sb_tab3 = st.tabs(["⚙️ Configurar", "💾 Guardar/Cargar", "🛠 Nuevo Modelo"])
   
    with sb_tab1:
        st.header("🛠 Configurar Modelo")
        
        # A. Selección del Modelo Base (incluye modelos personalizados)
        full_reg = get_full_registry()
        model_key = st.selectbox("Selecciona un Modelo", list(full_reg.keys()), format_func=lambda x: full_reg[x]["display_name"])
        model_info = full_reg[model_key]
        
        experiment_name=st.text_input("Nombre del experimento",max_chars=50)
        
        st.divider()
        
        # B. Generación Dinámica de Parámetros (Sliders)
        st.subheader("Parámetros")
        current_params = {}
        
        # Iteramos sobre el diccionario de parámetros del modelo seleccionado
        for p_name, p_cfg in model_info["params"].items():
            current_params[p_name] = st.slider(
                f"{p_name} ({p_cfg['desc']})",
                min_value=float(p_cfg["min"]),
                max_value=float(p_cfg["max"]),
                value=float(p_cfg["default"]),
                step=float(p_cfg["step"]),
                key=f"slider_{model_key}_{p_name}" # Key única
            )
        
        # C. Generación Dinámica de Condiciones Iniciales
        st.subheader("Condiciones Iniciales (t=0)")
        current_y0 = []
        for idx, y_cfg in enumerate(model_info["y0_config"]):
            val = st.number_input(
                f"{y_cfg['label']}",
                value=float(y_cfg['default']),
                key=f"input_{model_key}_y0_{idx}"
            )
            current_y0.append(val)
            
        # D. Configuración de Tiempo
        t_max = st.number_input("Tiempo Final", value=50.0, step=5.0)
        
        # E. Botón para Añadir al "Stage"
        st.divider()
        btn_add = st.button("➕ Añadir al Panel de Experimentos", use_container_width=True)

        if btn_add:
            new_exp = {
                "id": len(st.session_state.experiments) + 1,
                "name": experiment_name,
                "model_key": model_key,
                "model_name": model_info["display_name"],
                "params": current_params,
                "y0": current_y0,
                "t_span": [0, t_max]
            }
            st.session_state.experiments.append(new_exp)
            st.success(f"Añadido: {model_info['display_name']}")

     # >>> PESTAÑA 2: GUARDAR Y CARGAR (JSON) <<<
    with sb_tab2:
        st.subheader("Exportar")
        json_str = ExperimentManager.to_json(st.session_state.experiments)
        st.download_button(
            label="📥 Descargar Experimentos (.json)",
            data=json_str,
            file_name="mis_experimentos.json",
            mime="application/json"
        )
        
        st.subheader("Importar")
        uploaded_file = st.file_uploader("Subir JSON", type=["json"])
        if uploaded_file is not None:
            if st.button("Cargar archivo"):
                content = uploaded_file.read().decode("utf-8")
                loaded_exps = ExperimentManager.from_json(content)
                if loaded_exps:
                    st.session_state.experiments.extend(loaded_exps)
                    st.success(f"Se cargaron {len(loaded_exps)} experimentos.")
                    st.rerun()

    # >>> PESTAÑA 3: CREAR NUEVO MODELO (Usuario) <<<
    with sb_tab3:
        st.markdown("### 🛠 Constructor de Sistemas EDO")
        st.info("Define tus variables y el sistema detectará los parámetros automáticamente.")

        cust_name = st.text_input("Nombre del Modelo", "Modelo Personalizado")
        cust_id = "custom_" + str(int(time.time()))

        # 1. Definir Variables de Estado
        vars_input = st.text_input("Variables de estado (separadas por coma)", "x, y", help="Ej: S, I, R  o  x, y")
        
        # Limpieza de variables
        state_vars = [v.strip() for v in vars_input.split(",") if v.strip()]
        
        if not state_vars:
            st.warning("Escribe al menos una variable (ej: x).")
        else:
            st.markdown("---")
            st.write("Define las ecuaciones ($d/dt$):")
            
            # 2. Generar inputs para cada ecuación
            equations = {}
            for var in state_vars:
                equations[var] = st.text_input(
                    f"d{var}/dt =", 
                    value="", 
                    placeholder=f"Ej: alpha * {var} - beta * {var} * ...",
                    key=f"eq_{var}"
                )

            # Botón de Procesamiento
            if st.button("🚀 Compilar y Crear Modelo"):
                try:
                    # Diccionario de traducciones español -> Python/NumPy
                    translations = {
                        'seno': 'np.sin',
                        'sen': 'np.sin',
                        'coseno': 'np.cos',
                        'cos': 'np.cos',
                        'tangente': 'np.tan',
                        'tan': 'np.tan',
                        'arcoseno': 'np.arcsin',
                        'arcocoseno': 'np.arccos',
                        'arcotangente': 'np.arctan',
                        'exponencial': 'np.exp',
                        'logaritmo': 'np.log',
                        'raiz': 'np.sqrt',
                        'abs': 'np.abs',
                        'absoluto': 'np.abs',
                        'sinh': 'np.sinh',
                        'cosh': 'np.cosh',
                        'tanh': 'np.tanh'
                    }
                    
                    # --- PASO A: ANÁLISIS DE PARÁMETROS (AST) ---
                    found_params = set()
                    # Palabras reservadas que NO son parámetros
                    reserved = set(state_vars + ['t', 'np', 'math', 'e', 'pi', 'sin', 'cos', 'exp', 'log', 'sqrt', 'abs', 
                                                  'tan', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh'])
                    
                    # Aplicar traducciones a cada ecuación
                    translated_equations = {}
                    for var, eq_str in equations.items():
                        eq_translated = eq_str
                        for spanish, python in translations.items():
                            # Reemplazar palabras completas (con límites de palabra)
                            import re
                            eq_translated = re.sub(r'\b' + spanish + r'\b', python, eq_translated)
                        # Reemplazar constantes
                        eq_translated = re.sub(r'\bpi\b', 'np.pi', eq_translated)
                        eq_translated = re.sub(r'\be\b', 'np.e', eq_translated)
                        translated_equations[var] = eq_translated
                    
                    # Envolver cada ecuación entre paréntesis para evitar errores de sintaxis
                    all_eq_str = " ".join([f"({eq})" if eq.strip() else "0" for eq in translated_equations.values()])
                    
                    # Parseamos el string para buscar identificadores (nombres)
                    tree = ast.parse(all_eq_str)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Name):
                            if node.id not in reserved:
                                found_params.add(node.id)
                    
                    param_list = sorted(list(found_params))
                    
                    # --- PASO B: CONSTRUCCIÓN DE LA FUNCIÓN ---
                    # Creamos una función wrapper que desempaqueta 'y' y los 'params'
                    
                    # Mapa de índices para el vector y: {'x': 0, 'y': 1}
                    var_map = {v: i for i, v in enumerate(state_vars)}
                    
                    # Construimos el código de la función como string dinámico
                    # Si no hay parámetros, la función no tendrá argumentos adicionales
                    if param_list:
                        func_code = f"def dynamic_ode(t, y, {', '.join(param_list)}):\n"
                    else:
                        func_code = "def dynamic_ode(t, y):\n"
                    
                    # Desempaquetar variables de estado del array 'y'
                    for var, idx in var_map.items():
                        func_code += f"    {var} = y[{idx}]\n"
                    
                    # Evaluar ecuaciones (usando las traducidas)
                    results = []
                    for var in state_vars:
                        eq_clean = translated_equations[var] if translated_equations[var].strip() else "0"
                        results.append(eq_clean)
                    
                    func_code += f"    return np.array([{', '.join(results)}])"
                    
                    # --- PASO C: COMPILAR Y GUARDAR ---
                    # Ejecutamos el string para crear la función en el espacio local
                    # Incluimos funciones matemáticas comunes para que estén disponibles
                    local_scope = {
                        "np": np,
                        "sin": np.sin,
                        "cos": np.cos,
                        "tan": np.tan,
                        "exp": np.exp,
                        "log": np.log,
                        "sqrt": np.sqrt,
                        "abs": np.abs,
                        "pi": np.pi,
                        "e": np.e,
                        "arcsin": np.arcsin,
                        "arccos": np.arccos,
                        "arctan": np.arctan,
                        "sinh": np.sinh,
                        "cosh": np.cosh,
                        "tanh": np.tanh
                    }
                    exec(func_code, local_scope)
                    dynamic_ode_func = local_scope["dynamic_ode"]
                    
                    # Crear un wrapper que convierta el diccionario de params a argumentos individuales
                    # solve_ivp_model llama: model_func(t, y, params_dict)
                    # pero dynamic_ode espera: dynamic_ode(t, y, param1, param2, ...)
                    if param_list:
                        def wrapper_func(t, y, params_dict):
                            # Extraer valores de los parámetros en el orden correcto
                            param_values = [params_dict[p] for p in param_list]
                            return dynamic_ode_func(t, y, *param_values)
                    else:
                        # Si no hay parámetros, no necesitamos desempaquetar nada
                        def wrapper_func(t, y, params_dict):
                            return dynamic_ode_func(t, y)
                    
                    # Construir diccionario de parámetros para el registro
                    params_dict_reg = {}
                    for p in param_list:
                        params_dict_reg[p] = {
                            "min": 0.0, "max": 5.0, "default": 1.0, "step": 0.05, "desc": "Auto-detected"
                        }
                    
                    # Construir diccionario de y0
                    y0_reg = []
                    for v in state_vars:
                        y0_reg.append({"label": f"{v} inicial", "default": 1.0})

                    new_model_entry = {
                        "display_name": f" {cust_name}",
                        "type": "system" if len(state_vars) > 1 else "ode",
                        "function": wrapper_func,
                        "params": params_dict_reg,
                        "y0_config": y0_reg
                    }
                    
                    # Guardar en Session State
                    st.session_state.custom_models[cust_id] = new_model_entry
                    
                    st.success(f"¡Modelo compilado con éxito! Detectamos los parámetros: {param_list}")
                    time.sleep(1) # Pequeña pausa para ver el mensaje
                    st.rerun() # Recargar para que aparezca en la lista
                    
                except SyntaxError:
                    st.error("Error de sintaxis en tus ecuaciones. Revisa paréntesis y operadores (*).")
                except Exception as e:
                    st.error(f"Error interno al compilar: {e}")



    
# --- 3. ÁREA PRINCIPAL: VISUALIZACIÓN Y EJECUCIÓN ---
with main_col:
    # Contenedor de Métricas rápidas
    total_exp=len(st.session_state.experiments)
    if total_exp > 0:
        st.markdown(f"### 📂 Experimentos Activos: {total_exp}")
    else:
        st.info("👈 Configura y añade un modelo desde el panel izquierdo para comenzar.")

    # Creamos pestañas para vista individual o comparativa
    tab1, tab2 = st.tabs(["🔬 Vista Individual", "📊 Comparación Global"])

    with tab1:
        # Iteramos sobre cada experimento guardado
        for i, exp in enumerate(st.session_state.experiments):
            with st.expander(f"Experimento #{exp['id']}: {exp['name']} ({exp['model_name']})", expanded=True):
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.write("**Parámetros:**")
                    st.json(exp["params"])
                    st.write(f"**Inicial:** {exp['y0']}")
                    
                    # Selector de variables para graficar
                    full_reg = get_full_registry()
                    if exp["model_key"] in full_reg:
                        var_labels = [cfg['label'] for cfg in full_reg[exp['model_key']]['y0_config']]
                        selected_vars = st.multiselect(
                            "Variables a mostrar:",
                            options=list(range(len(var_labels))),
                            default=list(range(len(var_labels))),
                            format_func=lambda x: var_labels[x],
                            key=f"vars_exp_{i}"
                        )
                    else:
                        selected_vars = [0]
                    
                    if st.button(f"🗑 Eliminar #{exp['id']}", key=f"del_{i}"):
                        st.session_state.experiments.pop(i)
                        st.rerun()

                with col2:
                    # EJECUCIÓN EN TIEMPO REAL
                    # Aquí llamamos al Manager
                    try:
                        t, y = ExperimentManager.run_simulation(
                            exp["model_key"], 
                            exp["params"], 
                            exp["y0"], 
                            exp["t_span"],
                            st.session_state.custom_models
                        )
                        
                        # Graficar
                        fig, ax = plt.subplots(figsize=(8, 3))
                        
                        # Manejo inteligente de graficas (Si es sistema o escalar)
                        if y.ndim > 1 and y.shape[0] > 1: # Es un sistema (ej. Lotka Volterra)
                            full_reg = get_full_registry()
                            if exp['model_key'] in full_reg:
                                labels = [cfg['label'] for cfg in full_reg[exp['model_key']]['y0_config']]
                            else:
                                labels = [f"Var {i}" for i in range(y.shape[0])]
                            
                            # Graficar solo las variables seleccionadas
                            for line_idx in selected_vars:
                                if line_idx < y.shape[0]:
                                    ax.plot(t, y[line_idx], label=labels[line_idx])
                        else: # Es escalar
                            ax.plot(t, y.flatten(), label="y(t)")
                            
                        ax.set_title(f"Dinámica: {exp['model_name']}")
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Error al simular: {e}")

    with tab2:
        if total_exp < 2:
            st.warning("Necesitas al menos 2 experimentos para comparar.")
        else:
            st.subheader("Superposición de Resultados")
            
            # Selector de experimentos
            exp_options = {f"#{i+1} {e['model_name']} (ID:{e['id']})": i for i, e in enumerate(st.session_state.experiments)}
            selected_indices = st.multiselect("Selecciona experimentos:", list(exp_options.keys()), default=list(exp_options.keys()))
            
            # Configuración individual para cada experimento
            exp_var_selections = {}
            if selected_indices:
                st.markdown("**Selecciona las variables para cada experimento:**")
                for name in selected_indices:
                    idx = exp_options[name]
                    exp = st.session_state.experiments[idx]
                    full_reg = get_full_registry()
                    
                    if exp["model_key"] in full_reg:
                        var_labels = [cfg['label'] for cfg in full_reg[exp["model_key"]]['y0_config']]
                        
                        # Selector de variables específico para este experimento
                        selected_vars = st.multiselect(
                            f"📊 {exp['model_name']} (Exp #{idx+1}):",
                            options=list(range(len(var_labels))),
                            default=[0],
                            format_func=lambda x: var_labels[x] if x < len(var_labels) else f"Var {x}",
                            key=f"comp_vars_exp_{idx}"
                        )
                        
                        exp_var_selections[name] = {
                            'exp': exp,
                            'idx': idx,
                            'vars': selected_vars,
                            'labels': var_labels
                        }

            # Graficar
            if exp_var_selections:
                fig_comp, ax_comp = plt.subplots(figsize=(10, 5))
                
                for name, config in exp_var_selections.items():
                    exp = config['exp']
                    idx = config['idx']
                    selected_vars = config['vars']
                    var_labels = config['labels']
                    
                    if selected_vars:  # Solo graficar si hay variables seleccionadas
                        t, y = ExperimentManager.run_simulation(
                            exp["model_key"], exp["params"], exp["y0"], exp["t_span"], st.session_state.custom_models
                        )
                        
                        # Graficar cada variable seleccionada para este experimento
                        for var_idx in selected_vars:
                            var_label = var_labels[var_idx] if var_idx < len(var_labels) else f"Var {var_idx}"
                            
                            if y.ndim > 1 and y.shape[0] > var_idx:
                                data_to_plot = y[var_idx]
                            elif y.ndim == 1 and var_idx == 0:
                                data_to_plot = y.flatten()
                            else:
                                continue  # Saltar si la variable no existe
                            
                            ax_comp.plot(t, data_to_plot, label=f"Exp #{idx+1}: {exp['model_name']} [{var_label}]")
                
                ax_comp.set_title("Comparativa Directa")
                ax_comp.set_xlabel("Tiempo")
                ax_comp.set_ylabel("Valor de la Variable")
                ax_comp.grid(True)
                ax_comp.legend()
                st.pyplot(fig_comp)
            elif selected_indices:
                st.info("Selecciona al menos una variable para cada experimento.")
            # (Aquí iría la lógica para graficar múltiples experimentos en un solo ax)