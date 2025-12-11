"""
custom_model_builder.py

Clase para construir modelos EDO personalizados dinámicamente.
Maneja el parsing, traducción de funciones matemáticas y compilación de ecuaciones.
"""

import ast
import re
import numpy as np
from typing import Dict, List, Tuple, Callable


class CustomModelBuilder:
    """Constructor de modelos EDO personalizados."""
    
    # Diccionario de traducciones español -> Python/NumPy
    TRANSLATIONS = {
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
    
    # Palabras reservadas que NO son parámetros
    RESERVED_WORDS = [
        't', 'np', 'math', 'e', 'pi', 'sin', 'cos', 'exp', 'log', 'sqrt', 'abs',
        'tan', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh'
    ]
    
    # Scope con funciones matemáticas
    MATH_SCOPE = {
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
    
    @staticmethod
    def translate_equation(equation: str) -> str:
        """
        Traduce una ecuación de español/notación simple a Python/NumPy.
        
        Args:
            equation: Ecuación en texto (puede contener seno, coseno, etc.)
            
        Returns:
            Ecuación traducida a sintaxis Python/NumPy
        """
        eq_translated = equation
        
        # Aplicar traducciones
        for spanish, python in CustomModelBuilder.TRANSLATIONS.items():
            eq_translated = re.sub(r'\b' + spanish + r'\b', python, eq_translated)
        
        # Reemplazar constantes
        eq_translated = re.sub(r'\bpi\b', 'np.pi', eq_translated)
        eq_translated = re.sub(r'\be\b', 'np.e', eq_translated)
        
        return eq_translated
    
    @staticmethod
    def detect_parameters(equations: Dict[str, str], state_vars: List[str]) -> List[str]:
        """
        Detecta automáticamente los parámetros en las ecuaciones usando AST.
        
        Args:
            equations: Diccionario {variable: ecuación}
            state_vars: Lista de variables de estado
            
        Returns:
            Lista ordenada de parámetros detectados
        """
        found_params = set()
        reserved = set(state_vars + CustomModelBuilder.RESERVED_WORDS)
        
        # Traducir ecuaciones primero
        translated_eqs = [CustomModelBuilder.translate_equation(eq) for eq in equations.values()]
        
        # Envolver cada ecuación entre paréntesis para evitar errores de sintaxis
        all_eq_str = " ".join([f"({eq})" if eq.strip() else "0" for eq in translated_eqs])
        
        # Parsear usando AST
        tree = ast.parse(all_eq_str)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id not in reserved:
                    found_params.add(node.id)
        
        return sorted(list(found_params))
    
    @staticmethod
    def build_function(state_vars: List[str], 
                      equations: Dict[str, str], 
                      param_list: List[str]) -> Callable:
        """
        Construye dinámicamente la función EDO.
        
        Args:
            state_vars: Lista de variables de estado
            equations: Diccionario {variable: ecuación}
            param_list: Lista de parámetros
            
        Returns:
            Función compilada
        """
        # Traducir ecuaciones
        translated_equations = {var: CustomModelBuilder.translate_equation(eq) 
                              for var, eq in equations.items()}
        
        # Mapa de índices para el vector y: {'x': 0, 'y': 1}
        var_map = {v: i for i, v in enumerate(state_vars)}
        
        # Construir código de la función
        if param_list:
            func_code = f"def dynamic_ode(t, _y_internal, {', '.join(param_list)}):\n"
        else:
            func_code = "def dynamic_ode(t, _y_internal):\n"
        
        # Desempaquetar variables de estado del array 'y'
        for var, idx in var_map.items():
            func_code += f"    {var} = _y_internal[{idx}]\n"
        
        # Evaluar ecuaciones (usando las traducidas)
        results = []
        for var in state_vars:
            eq_clean = translated_equations[var] if translated_equations[var].strip() else "0"
            results.append(eq_clean)
        
        func_code += f"    return np.array([{', '.join(results)}])"
        
        # Ejecutar el string para crear la función
        local_scope = CustomModelBuilder.MATH_SCOPE.copy()
        exec(func_code, local_scope)
        
        return local_scope["dynamic_ode"]
    
    @staticmethod
    def create_wrapper(dynamic_func: Callable, param_list: List[str]) -> Callable:
        """
        Crea un wrapper que convierte diccionario de params a argumentos individuales.
        
        Args:
            dynamic_func: Función dinámica generada
            param_list: Lista de parámetros
            
        Returns:
            Función wrapper compatible con solve_ivp_model
        """
        if param_list:
            def wrapper_func(y, t, params_dict):
                # Ensure y is an array (scipy might pass scalar for single ODE)
                y = np.atleast_1d(y)
                param_values = [params_dict[p] for p in param_list]
                return dynamic_func(t, y, *param_values)
        else:
            def wrapper_func(y, t, params_dict):
                # Ensure y is an array
                y = np.atleast_1d(y)
                return dynamic_func(t, y)
        
        return wrapper_func
    
    @classmethod
    def build_model(cls, 
                   model_name: str,
                   state_vars: List[str], 
                   equations: Dict[str, str],
                   param_ranges: Dict[str, Dict] = None) -> Tuple[Dict, List[str]]:
        """
        Método principal para construir un modelo completo.
        
        Args:
            model_name: Nombre del modelo
            state_vars: Lista de variables de estado
            equations: Diccionario {variable: ecuación}
            param_ranges: Opcional, rangos personalizados para parámetros
            
        Returns:
            Tupla (model_entry, param_list)
        """
        # Detectar parámetros
        param_list = cls.detect_parameters(equations, state_vars)
        
        # Construir función dinámica
        dynamic_func = cls.build_function(state_vars, equations, param_list)
        
        # Crear wrapper
        wrapper_func = cls.create_wrapper(dynamic_func, param_list)
        
        # Construir diccionario de parámetros
        params_dict_reg = {}
        for p in param_list:
            if param_ranges and p in param_ranges:
                params_dict_reg[p] = param_ranges[p]
            else:
                params_dict_reg[p] = {
                    "min": 0.0, 
                    "max": 5.0, 
                    "default": 1.0, 
                    "step": 0.05, 
                    "desc": "Auto-detected"
                }
        
        # Construir diccionario de y0
        y0_reg = []
        for v in state_vars:
            y0_reg.append({"label": f"{v} inicial", "default": 1.0})
        
        # Crear entrada del modelo
        model_entry = {
            "display_name": f" {model_name}",
            "type": "system" if len(state_vars) > 1 else "ode",
            "function": wrapper_func,
            "params": params_dict_reg,
            "y0_config": y0_reg
        }
        
        return model_entry, param_list
