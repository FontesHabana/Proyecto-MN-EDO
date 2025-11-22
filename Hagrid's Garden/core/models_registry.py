import numpy as np

from core.creatures_models import lotka_volterra
from core.plants_models import logistic_growth
from core.competition_models import two_species_competition

# 2. El Registro Maestro (Diccionario de configuración)
MODELS_REGISTRY = {
    "logistic": {
        "display_name": "Crecimiento Logístico",
        "type": "ode", # Ecuación simple
        "function": logistic_growth,
        "params": {
            "r": {"min": 0.0, "max": 2.0, "default": 0.5, "step": 0.1, "desc": "Tasa de crecimiento"},
            "K": {"min": 1.0, "max": 100.0, "default": 40.0, "step": 1.0, "desc": "Capacidad de carga"},
        },
        "y0_config": [
            {"label": "Población Inicial", "default": 10.0}
        ]
    },
    
    "lotka_volterra": {
        "display_name": "Lotka–Volterra (Predador-Presa)",
        "type": "system", # Sistema de ecuaciones
        "function": lotka_volterra,
        "params": {
            "alpha": {"min": 0.0, "max": 3.0, "default": 1.1, "step": 0.1, "desc": "Tasa crec. presas"},
            "beta":  {"min": 0.0, "max": 1.0, "default": 0.4, "step": 0.05, "desc": "Tasa depredación"},
            "delta": {"min": 0.0, "max": 1.0, "default": 0.1, "step": 0.05, "desc": "Eficiencia conversión"},
            "gamma": {"min": 0.0, "max": 3.0, "default": 0.4, "step": 0.1, "desc": "Mortalidad depredador"},
        },
        "y0_config": [
            {"label": "Presas (x)", "default": 10.0},
            {"label": "Depredadores (y)", "default": 2.0}
        ]
    },
    
    "two_species_competition": {
        "display_name": "Recursos limitados",
        "type": "system", # Sistema de ecuaciones
        "function": two_species_competition,
        "params": {
            "r_p": {"min": 0.0, "max": 2.0, "default": 0.5, "step": 0.1, "desc": "Tasa crec. especie A"},
            "r_c": {"min": 0.0, "max": 2.0, "default": 0.5, "step": 0.1, "desc": "Tasa crec. especie B"},
            "K": {"min": 1.0, "max": 100.0, "default": 40.0, "step": 1.0, "desc": "Capacidad de carga"},
            "alpha": {"min": 0.0, "max": 2.0, "default": 1., "step": 0.01, "desc": "Efecto A en B"},
            "beta":  {"min": 0.0, "max": 1.0, "default": 0.1, "step": 0.01, "desc": "Efecto B en A"},
           
        },
        "y0_config": [
            {"label": "Especie A (x)", "default": 10.0},
            {"label": "Especie B (y)", "default": 2.0}
        ]
    }
}