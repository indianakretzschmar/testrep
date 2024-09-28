import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

# Constants
Ti = 20   # Initial temperature in degrees Celsius
Ts = -15  # Surface temperature in degrees Celsius
alpha = 0.138 * 1e-6  # Thermal diffusivity in m^2/s
t = 60 * 24 * 3600  # Time in seconds (60 days)

# Function f(x)
def f(x):
    return erf(x / (2 * np.sqrt(alpha * t))) - 3/7

# Derivative f'(x)
def f_prime(x):
    return (2 / np.sqrt(np.pi)) * np.exp(-(x / (2 * np.sqrt(alpha * t)))**2) * (1 / (2 * np.sqrt(alpha * t)))

# Bisection Method
def bisection_method(f, a, b, tol):
    if f(a) * f(b) >= 0:
        print("Bisection method assumes a change of sign between a and b.")
        return None
    c = a
    while (b - a) / 2 > tol:
        c = (a + b) / 2
        if f(c) == 0:
            break
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return c
