"""
solvers.py

Module to integrate ODE models (plants,creatures, ecosystem).
Supports multiple integration methods:
solve_ivp, Euler,RK4.
"""
from typing import Callable,Dict,List
import numpy as np
from scipy.integrate import solve_ivp


#==============================================================================================
# 1. Solver using SciPy solve_ivp
#============================================================================================== 
   
def solve_ivp_model(model_func: Callable,
                    P0:float,t_span:List[float],
                    params:Dict[str,float],
                    method:str="RK45",
                    max_step:float=0.1):
    """
    Solve an ODE using SciPy's solve_ivp.
    
    Args:
        model_func: callable(t,P,params)->dP/dt
        P0: initial condition
        t_span: [t_start, t_end]
        params: dictionary of parameters for the model
        method: integration method supported by solve_ivp (default RK45)
        max_step: maximum step size 
    Returns:
        solution: object with t and y attributes    
        """
    sol=solve_ivp(fun=lambda t, P:model_func(t,P,params),
                  t_span=t_span,
                  y0=P0,
                  method=method,
                  max_step=max_step)
    return sol

#==============================================================================================
# TODO 2. Solver using Improved Eulers
#==============================================================================================
    
#==============================================================================================
# TODO 3. Solver using RK4
#==============================================================================================
    