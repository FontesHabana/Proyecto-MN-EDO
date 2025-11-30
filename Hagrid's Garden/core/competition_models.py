"""
competition_models.py

Models for simulating two populations competiting for limited resources
"""

from typing import Dict, Callable
import numpy as np

#==============================================================================================
# 1. Two sprecies competition model
#============================================================================================== 
def two_species_competition(t: float, Y:np.ndarray,params:Dict[str,float])->np.ndarray:
    """
    Two populations competing for shared resources.
    
    Y: array with with [P , C]
    params: dictionary with keys:
        r_p: plant growth  rate
        r_c:creature growth rate
        K: carrying capacity
        alpha; effect of plants on creatures
        beta: effect of creatures on plants
    """
    P,C=Y
    r_p=params["r_p"]
    r_c=params["r_c"]
    K=params["K"]
    alpha=params.get("alpha",1.0)
    beta=params.get("beta",1.0)
    dP=r_p*P*(1-(P+alpha*C)/K)
    dC=r_c*C*(1-(C+beta*P)/K)
    
    return np.array([dP,dC])

#==============================================================================================
# 2. Model registry
#==============================================================================================
    
COMPETITION_MODELS: Dict[str,Callable]={
    "logistic":two_species_competition,
}



#==============================================================================================
# 3. Helper function to retrieve plant models
#==============================================================================================
 
def get_competition_model(name: str)->Callable:
     """
     Returns the function associated with the competition model name.
     """
     if name not in COMPETITION_MODELS:
         raise ValueError(f"Unknown competition model: {name}")
     return COMPETITION_MODELS[name]
 
 