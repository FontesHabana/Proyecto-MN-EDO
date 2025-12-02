"""
solvers.py

Module to integrate ODE models (plants, creatures, ecosystem).
Supports multiple integration methods:
solve_ivp, Improved Euler, Runge Kutta 4.
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
    sol=solve_ivp(fun=lambda t, P:model_func(P,t,params),
                  t_span=t_span,
                  y0=P0,
                  method=method,
                  max_step=max_step)
    return sol


def improved_euler(f, x0, y0, h, xf, f_params):
    """
    Solve an ODE using the Improved Euler (Heun's) method.
    Supports both scalar and vector (system of) ODEs.

    Parameters
    ----------
    f : callable
        The ODE function f(y, x, params) representing dy/dx = f(y, x, params).
    x0 : float
        Initial x value.
    y0 : float or array-like
        Initial y value(s). Can be scalar or array for systems of ODEs.
    h : float
        Step size.
    xf : float
        Final x value.
    f_params: dict
        Parameters for f

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        Arrays of x values and corresponding approximated y values.
        For systems, y_points has shape (n_steps+1, n_equations).
    """
    n_steps = int(np.ceil((xf - x0) / h))

    # Convert y0 to numpy array to handle both scalar and vector cases
    y0_array = np.atleast_1d(y0)
    n_eqs = len(y0_array)

    x_points = np.zeros(n_steps + 1)
    y_points = np.zeros((n_steps + 1, n_eqs))

    x_points[0] = x0
    y_points[0] = y0_array

    for i in range(n_steps):
        x_i = x_points[i]
        y_i = y_points[i]

        # Predictor step
        slope1 = f(y_i, x_i, params=f_params)
        y_predict = y_i + h * slope1

        # Corrector step
        slope2 = f(y_predict, x_i + h, params=f_params)
        y_correct = y_i + (h / 2.0) * (slope1 + slope2)

        x_points[i + 1] = x_i + h
        y_points[i + 1] = y_correct

    # If original input was scalar, return scalar output
    if n_eqs == 1:
        return x_points, y_points[:, 0]
    else:
        return x_points, y_points

def runge_kutta_4(f, x0, y0, h, xf, f_params):
    """
    Solve an ODE using the 4th-order Runge-Kutta (RK4) method.
    Supports both scalar and vector (system of) ODEs.

    Parameters
    ----------
    f : callable
        The ODE function f(y, x, params) representing dy/dx = f(y, x, params).
    x0 : float
        Initial x value.
    y0 : float or array-like
        Initial y value(s). Can be scalar or array for systems of ODEs.
    h : float
        Step size.
    xf : float
        Final x value (target).
    f_params: dict
        Parameters for f

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        Arrays of x values and corresponding approximated y values.
        For systems, y_points has shape (n_steps+1, n_equations).
    """
    n_steps = int(np.ceil((xf - x0) / h))

    # Convert y0 to numpy array to handle both scalar and vector cases
    y0_array = np.atleast_1d(y0)
    n_eqs = len(y0_array)

    x_points = np.zeros(n_steps + 1)
    y_points = np.zeros((n_steps + 1, n_eqs))

    x_points[0] = x0
    y_points[0] = y0_array

    for i in range(n_steps):
        x = x_points[i]
        y = y_points[i]

        k1 = h * f(y, x, params=f_params)
        k2 = h * f(y + k1 / 2, x + h / 2, params=f_params)
        k3 = h * f(y + k2 / 2, x + h / 2, params=f_params)
        k4 = h * f(y + k3, x + h, params=f_params)

        y_new = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x_new = x + h

        x_points[i + 1] = x_new
        y_points[i + 1] = y_new

    # If original input was scalar, return scalar output
    if n_eqs == 1:
        return x_points, y_points[:, 0]
    else:
        return x_points, y_points