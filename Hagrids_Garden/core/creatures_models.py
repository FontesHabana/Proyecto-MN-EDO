"""
creatures_model.py

Mathematical models for simulating the predator-prey interactions.
Contains functions that define predator-prey interactions.
"""

from typing import Dict,Callable
import numpy as np

#==============================================================================================
# 1. Lotka-Volterra model 
#==============================================================================================

def lotka_volterra(Y:np.ndarray, t:float, params: Dict[str,float])->np.ndarray:
    """
    Computes derivatives for the Lotka-Volterra predator-prey system.
    Y[0]=prey population
    Y[1]=predator population
    """
    alpha=params["alpha"]
    beta=params["beta"]
    gamma=params["gamma"]
    delta=params["delta"]
    x,y=Y
    dX=alpha*x-beta*x*y
    dY=delta*x*y-gamma*y
    return np.array([dX,dY])

#==============================================================================================
# 2. Model registry
#==============================================================================================
    
PREDATOR_PREY_MODELS: Dict[str,Callable]={
    "lotka-volterra":lotka_volterra,
}



#==============================================================================================
# 3. Helper function to retrieve predator-prey models
#==============================================================================================
 
def get_predator_prey_model(name: str)->Callable:
     """
     Returns the function associated with the predator-prey model name.
     """
     if name not in PREDATOR_PREY_MODELS:
         raise ValueError(f"Unknown predator prey model: {name}")
     return PREDATOR_PREY_MODELS[name]