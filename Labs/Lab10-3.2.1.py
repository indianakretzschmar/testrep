import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.integrate import quad

def eval_legendre(n, x):
    # Initialize a vector to store Legendre polynomials up to order n
    p = np.zeros(n + 1)
    p[0] = 1  # ϕ0(x) = 1
    if n > 0:
        p[1] = x  # ϕ1(x) = x
    # Use the recursion formula for higher orders
    for k in range(1, n):
        p[k + 1] = ((2 * k + 1) * x * p[k] - k * p[k - 1]) / (k + 1)
    return p

# Define the polynomial function φ_j(x)
def phi_j(x, j):
    # Use the eval_legendre function or define the Legendre polynomials manually
    return eval_legendre(j, x)[j]

# Define the integrand for the numerator: f(x) * φ_j(x) * w(x)
def f_phi_w(x, f, phi_j, w, j):
    return f(x) * phi_j(x, j) * w(x)

# Define the integrand for the denominator (normalization): φ_j(x)^2 * w(x)
def phi_sq_w(x, phi_j, w, j):
    return phi_j(x, j)**2 * w(x)

# Evaluate the coefficient a_j using a single line
def eval_aj(f, phi_j, w, j, a, b):
    return quad(lambda x: f_phi_w(x, f, phi_j, w, j), a, b)[0] / quad(lambda x: phi_sq_w(x, phi_j, w, j), a, b)[0]

# Define the function f, weight function w, and interval [a, b]
f = lambda x: np.exp(x)  # Example function
w = lambda x: 1          # Example weight function (uniform weight)
a, b = -1, 1             # Example interval

# Calculate the coefficient a_j for j = 2
j = 2
aj = eval_aj(f, phi_j, w, j, a, b)
print(f"The coefficient a_{j} is: {aj}")

