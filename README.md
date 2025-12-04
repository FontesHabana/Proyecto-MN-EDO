# 🌱✨ Arcane Dynamics Lab
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hagrids-garden.streamlit.app/)
## Descripción del Proyecto

[**Arcane Dynamics Lab**](https://hagrids-garden.streamlit.app/) es un laboratorio interactivo diseñado para explorar y simular ecuaciones diferenciales ordinarias (EDOs) de manera visual e intuitiva. Este proyecto combina conceptos matemáticos rigurosos con una narrativa mágica, permitiendo a los usuarios experimentar con modelos poblacionales, sistemas dinámicos y métodos numéricos a través de una interfaz web desarrollada con Streamlit.

El proyecto está ambientado en un universo de fantasía donde plantas encantadas, criaturas mágicas y ecosistemas fantásticos obedecen las leyes de las ecuaciones diferenciales. Es ideal para estudiantes, profesores e investigadores que deseen aprender o enseñar matemática numérica de forma práctica y entretenida.

---

## 🎯 Características Principales

### 1. **Simulación de Plantas Mágicas** 🌱
- Modelos de crecimiento logístico con recursos limitados
- Visualización interactiva del crecimiento poblacional
- Parámetros ajustables: población inicial, tasa de crecimiento, capacidad de carga
- Comparación entre métodos numéricos: `scipy.odeint`, Euler mejorado, RK4

### 2. **Simulación de Criaturas** 🐾
- Modelos depredador-presa (Lotka-Volterra)
- Dinámicas de poblaciones con interacciones mágicas
- Control de parámetros de reproducción, mortalidad e interacción
- Gráficos dinámicos con Plotly

### 3. **Competición entre Especies** 🦁🌿
- Sistema de dos especies compitiendo por recursos compartidos
- Modelo de competencia con coeficientes de interacción ajustables
- Análisis de equilibrios y comportamiento dinámico
- Visualización de trayectorias en el plano de fase

### 4. **Panel de Experimentos** 🔮
- Constructor personalizado de EDOs
- Diseño de sistemas dinámicos complejos
- Experimentación con múltiples condiciones iniciales
- Exportación de datos y gráficos

---

## 🛠️ Tecnologías Utilizadas

- **Python 3.10+**
- **Streamlit**: Framework para aplicaciones web interactivas
- **NumPy**: Cálculos numéricos y álgebra lineal
- **SciPy**: Solución de ecuaciones diferenciales (`odeint`, `solve_ivp`)
- **Matplotlib**: Visualización de datos estática
- **Plotly**: Gráficos interactivos y dinámicos
- **Pandas**: Manipulación de datos tabulares

---

## 📦 Instalación

### Requisitos Previos
- Python 3.10 o superior
- pip (gestor de paquetes de Python)
- Git (opcional, para clonar el repositorio)

### Pasos de Instalación

1. **Clonar el repositorio** (o descargar el código):
```bash
git clone https://github.com/FontesHabana/Proyecto-MN.git
cd Proyecto-MN/Hagrids_Garden
```

2. **Crear un entorno virtual** (recomendado):
```bash
python -m venv venv
```

3. **Activar el entorno virtual**:
   - En Windows:
     ```bash
     venv\Scripts\activate
     ```
   - En macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Instalar las dependencias**:
```bash
pip install -r requirements.txt
```

---

## 🚀 Uso

### Iniciar la Aplicación

Para ejecutar el laboratorio interactivo, usa el siguiente comando desde la carpeta `Hagrids_Garden`:

```bash
streamlit run Home.py
```

Esto abrirá automáticamente la aplicación en tu navegador predeterminado (normalmente en `http://localhost:8501`).

### Navegación

La aplicación consta de varias páginas accesibles desde el menú lateral:

1. **🏠 Home**: Página principal con descripción del proyecto
2. **🌱 Plant Simulation**: Simulación de crecimiento de plantas
3. **🐾 Creatures Simulation**: Modelos depredador-presa
4. **🦁 Competition Simulation**: Competencia entre dos especies
5. **🔮 Experiment Panel**: Panel de experimentos personalizados

---

## 📂 Estructura del Proyecto

```
Hagrids_Garden/
│
├── Home.py                      # Página principal de la aplicación
├── requirements.txt             # Dependencias del proyecto
├── README.md                    # Este archivo
│
├── assets/                      # Recursos multimedia
│   ├── styles.css              # Estilos CSS personalizados
│   └── images/                 # Imágenes para la interfaz
│
├── core/                        # Módulos principales
│   ├── __init__.py
│   ├── plants_models.py        # Modelos de crecimiento de plantas
│   ├── creatures_models.py     # Modelos de criaturas (depredador-presa)
│   ├── competition_models.py   # Modelos de competencia entre especies
│   ├── ecosystem_models.py     # Modelos de ecosistemas complejos
│   ├── solvers.py              # Métodos numéricos (Euler, RK4, etc.)
│   ├── models_registry.py      # Registro de modelos disponibles
│   ├── experiment_manager.py   # Gestión de experimentos
│   ├── custom_model_builder.py # Constructor de modelos personalizados
│   └── data_loader.py          # Carga de datos y parámetros
│
├── data/                        # Datos de configuración
│   ├── plant_profiles.json     # Perfiles de plantas mágicas
│   ├── creature_profiles.json  # Perfiles de criaturas
│   └── default_parameters.json # Parámetros por defecto
│
├── pages/                       # Páginas de Streamlit
│   ├── 1_plant_simulation.py   # Página de simulación de plantas
│   ├── 2_creatures_simulation.py # Página de simulación de criaturas
│   ├── 3_competition_simulation.py # Página de competencia
│   └── 4_experiment_panel.py   # Panel de experimentos
│
├── utils/                       # Utilidades
│   ├── __init__.py
│   ├── plot_utils.py           # Funciones auxiliares para gráficos
│   ├── random_events.py        # Generador de eventos aleatorios
│   └── style.py                # Estilos y temas personalizados
│
└── test_plants.py              # Tests unitarios
```

---

## 🧪 Modelos Matemáticos Implementados

### 1. Crecimiento Logístico
```
dP/dt = rP(1 - P/K)
```
Donde:
- `P`: Población
- `r`: Tasa de crecimiento
- `K`: Capacidad de carga

### 2. Modelo Lotka-Volterra (Depredador-Presa)
```
dP/dt = αP - βPC
dC/dt = δPC - γC
```
Donde:
- `P`: Población de presas
- `C`: Población de depredadores
- `α, β, γ, δ`: Parámetros de interacción

### 3. Competencia entre Especies
```
dP/dt = r_p·P(1 - (P + α·C)/K)
dC/dt = r_c·C(1 - (C + β·P)/K)
```
Donde:
- `P, C`: Poblaciones de dos especies
- `r_p, r_c`: Tasas de crecimiento
- `α, β`: Coeficientes de competencia
- `K`: Capacidad de carga compartida

---

## 🎓 Métodos Numéricos

El proyecto implementa varios métodos de solución de EDOs:

1. **Método de Euler Mejorado** (orden 2)
2. **Método Runge-Kutta de 4º orden (RK4)** (orden 4)
3. **Métodos de pasos múltiples Adams-Bashforth** (orden 4)
4. **Métodos de pasos múltiples Adams-Bashforth-Moulton** (orden 5)
5. **scipy.integrate.odeint** (adaptativo, orden variable)
6. **scipy.integrate.solve_ivp** (adaptativo, múltiples métodos)

---

## 🧑‍💻 Contribuciones

Las contribuciones son bienvenidas. Si deseas colaborar:

1. Haz un fork del repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Haz commit de tus cambios (`git commit -m 'Añadir nueva funcionalidad'`)
4. Haz push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

---

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

---

## 👥 Autores

- **Adrian Estévez Álvarez**
- **Javier Fontes Basabe**
- **Karla Yisel Ramírez Garcell**
- Universidad de La Habana
- Curso: Matemática Numérica y Ecuaciones Diferenciales Ordinarias
- Año: 2025

---

## 📧 Contacto

Para preguntas, sugerencias o reportar problemas:
- GitHub Issues: [https://github.com/FontesHabana/Proyecto-MN/issues](https://github.com/FontesHabana/Proyecto-MN/issues)
- Email: [adrian.estevez@estudiantes.matcom.uh.cu](adrian.estevez@estudiantes.matcom.uh.cu)
- Email: [javierfontbas@gmail.com](javierfontbas@gmail.com)
- Email: [karla.yramirez@estudiantes.matcom.uh.cu](karla.yramirez@estudiantes.matcom.uh.cu)
---

## 🙏 Agradecimientos

- A los profesores del curso de Matemática Numérica
- A la comunidad de Streamlit por su excelente framework
- A todos los colaboradores del proyecto

---



**¡Disfruta explorando el mundo mágico de las ecuaciones diferenciales!** ✨🧙‍♂️
