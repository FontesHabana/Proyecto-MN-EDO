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
    def to_json(experiments_list, custom_models=None):
        """
        Convierte la lista de experimentos y modelos personalizados a JSON.
        Se guardan solo los metadatos de recreación de los modelos custom.
        """
        data = {
            "experiments": experiments_list,
            "custom_definitions": {}
        }
        
        if custom_models:
            for key, model_data in custom_models.items():
                # Solo guardamos si tiene la metadata de origen (_source)
                if "_source" in model_data:
                    data["custom_definitions"][key] = model_data["_source"]

        def convert(o):
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            raise TypeError
            
        return json.dumps(data, default=convert, indent=2)

    @staticmethod
    def from_json(json_str):
        """
        Carga experimentos y reconstruye modelos personalizados.
        Retorna: (lista_experimentos, dict_custom_models)
        """
        try:
            data = json.loads(json_str)
            
            # Soporte retrocompatibilidad (si es una lista plana)
            if isinstance(data, list):
                return data, {}
                
            experiments = data.get("experiments", [])
            custom_definitions = data.get("custom_definitions", {})
            
            rebuilt_models = {}
            if custom_definitions:
                from core.custom_model_builder import CustomModelBuilder
                
                for key, source in custom_definitions.items():
                    try:
                        # Reconstruir el modelo
                        model_entry, _ = CustomModelBuilder.build_model(
                            model_name=source["name"],
                            state_vars=source["state_vars"],
                            equations=source["equations"]
                        )
                        # Restaurar la metadata _source para futuros guardados
                        model_entry["_source"] = source
                        rebuilt_models[key] = model_entry
                    except Exception as e:
                        print(f"Error reconstruyendo modelo {key}: {e}")
            
            return experiments, rebuilt_models
            
        except Exception as e:
            print(f"Error cargando JSON: {e}")
            return [], {}

# Instancia global (opcional, útil si guardamos estado aquí)
manager = ExperimentManager()