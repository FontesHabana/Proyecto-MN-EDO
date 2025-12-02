"""
plants_model.py

Mathematical models for simulating the growth of magical plants.
Contains functions that define dP/dt for different plant types.
"""

from typing import Dict,Callable
import numpy as np

#==============================================================================================
# 1. Logistic growth model (limited population with carrying capacity)
#==============================================================================================

def logistic_growth(P:float, t:float, params: Dict[str,float])->float:
    """
    Logistic growth model:
    dP/dt=r*P*(1-P/k)
    
    Expected parameters:
    - r: intrinsic growth rate
    - K: carrying capacity
    """
    r=params["r"]
    K=params["K"]
    
    return r*P*(1-P/K)

#==============================================================================================
# 2. Model registry
#==============================================================================================
    
PLANT_MODELS: Dict[str,Callable]={
    "logistic":logistic_growth,
}



#==============================================================================================
# 3. Helper function to retrieve plant models
#==============================================================================================
 
def get_plant_model(name: str)->Callable:
     """
     Returns the function associated with the plant model name.
     """
     if name not in PLANT_MODELS:
         raise ValueError(f"Unknown plant model: {name}")
     return PLANT_MODELS[name]