# homework1/problem7/optimizer.py
# This script implements vanilla gradient descent for a 1D non-convex function.


# Importing necessary libraries
import numpy as np # for numerical computations


def f(x):
    """Objective function: f(x) = x^4 - 3x^3 + 2"""
    return x**4 - 3*x**3 + 2


def grad_f(x):
    """Gradient: f'(x) = 4x^3 - 9x^2"""
    return 4*x**3 - 9*x**2


def gradient_descent(x0, step_size, max_iter):
    """
    Vanilla gradient descent for 1D function.
    
    Returns:
        trajectory: Array of x values at each iteration
        values: Array of f(x) values at each iteration
    """
    x = x0
    trajectory = [x]
    values = [f(x)]
    
    for _ in range(max_iter):
        gradient = grad_f(x)
        x = x - step_size * gradient
        trajectory.append(x)
        values.append(f(x))
        
    return np.array(trajectory), np.array(values)