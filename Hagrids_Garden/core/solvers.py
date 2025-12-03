"""
solvers.py

Module to integrate ODE models (plants, creatures, ecosystem).
Supports multiple integration methods:
solve_ivp, Improved Euler, Runge Kutta 4.
"""
from typing import Callable,Dict,List
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.gridspec import GridSpec

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


def compare_numerical_methods(f, f_inv, t0, y0, tf, methods_config, scale='log', f_params={}):
    """
    Compares multiple numerical methods and plots their errors.
    """
    if scale not in ('linear', 'log'):
        raise ValueError(f"scale must be 'linear' or 'log', got '{scale}'")

    h_ref = 0.001
    t_ref = np.arange(t0, tf + h_ref, h_ref)

    y_ref = odeint(f, y0, t_ref, args=(f_params,)).flatten()

    fig = plt.figure(figsize=(16, 7), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)

    num_methods = len(methods_config)
    colors = plt.cm.tab10(np.arange(num_methods) % 10)

    results = {}
    for config in methods_config:
        name = config['name']
        method = config['method']
        h = config['h']
        t_num, y_num = method(f, t0, y0, h, tf, f_params)
        results[name] = {'t': t_num, 'y': y_num, 'h': h}

    # ========== PLOT 1: Forward Error at Each Step ==========
    ax1 = fig.add_subplot(gs[0, 0])

    for idx, config in enumerate(methods_config):
        name = config['name']
        t_num = results[name]['t']
        y_num = results[name]['y']

        y_ref_interp = np.interp(t_num, t_ref, y_ref)
        forward_error = np.abs(y_num - y_ref_interp)

        if scale == 'log':
            plot_data = np.where(forward_error == 0, np.nan, forward_error)
            ax1.semilogy(t_num, plot_data, 'o-', label=name, color=colors[idx], markersize=4, alpha=0.8)
        else:
            ax1.plot(t_num, forward_error, 'o-', label=name, color=colors[idx], markersize=4, alpha=0.8)

    ax1.set_xlabel('t', fontsize=11)
    ax1.set_ylabel('|Forward Error| (in y)', fontsize=11)
    ax1.set_title(f'Forward Error (Scale: {scale})', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3, which='both')

    # ========== PLOT 2: Backward Error at Each Step ==========
    ax2 = fig.add_subplot(gs[0, 1])

    for idx, config in enumerate(methods_config):
        name = config['name']
        t_num = results[name]['t']
        y_num = results[name]['y']

        t_inv = np.array([f_inv(val, f_params, y0) for val in y_num])
        # --------------------------------

        backward_error = np.abs(t_num - t_inv)

        if scale == 'log':
            plot_data = np.where(backward_error == 0, np.nan, backward_error)
            ax2.semilogy(t_num, plot_data, 'o-', label=name, color=colors[idx], markersize=4, alpha=0.8)
        else:
            ax2.plot(t_num, backward_error, 'o-', label=name, color=colors[idx], markersize=4, alpha=0.8)

    ax2.set_xlabel('t', fontsize=11)
    ax2.set_ylabel('|Backward Error| (in t)', fontsize=11)
    ax2.set_title(f'Backward Error (Scale: {scale})', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')

    fig.suptitle('Comparative Analysis of Errors in Numerical Methods', fontsize=14, fontweight='bold')

    return fig