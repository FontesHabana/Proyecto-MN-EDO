import json
import numpy as np
from scipy.integrate import solve_ivp
from core.models_registry import MODELS_REGISTRY
from core.solvers import solve_ivp_model
class ExperimentManager:
    def __init__(self):
        pass # En el futuro aquí podemos manejar el historial global

    @staticmethod
    def run_simulation(model_key, params, y0, t_span, custom_models=None):
        """
        Ejecuta una simulación basada en la clave del modelo.
        """
        # Fusionar registros si hay modelos personalizados
        full_registry = MODELS_REGISTRY.copy()
        if custom_models:
            full_registry.update(custom_models)
        
        if model_key not in full_registry:
            raise ValueError(f"Modelo {model_key} no encontrado.")
        
        model_data = full_registry[model_key]
        func = model_data["function"]
        
        # Preparar argumentos para solve_ivp
        # solve_ivp espera f(t, y, *args), así que pasamos args
        args = tuple(params.values())
        
        
        
        sol=solve_ivp_model(func,y0,t_span,params)
            
      
        
        return sol.t, sol.y
    
    
    @staticmethod
    def to_json(experiments_list):
        """Convierte la lista de experimentos a un string JSON válido"""
        # Convertimos tipos de numpy a nativos de python para evitar errores de JSON
        def convert(o):
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            raise TypeError
            
        return json.dumps(experiments_list, default=convert, indent=2)

    @staticmethod
    def from_json(json_str):
        """Carga experimentos desde un string JSON"""
        try:
            return json.loads(json_str)
        except Exception as e:
            print(f"Error cargando JSON: {e}")
            return []

# Instancia global (opcional, útil si guardamos estado aquí)
manager = ExperimentManager()