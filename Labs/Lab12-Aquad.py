import numpy as np

# Composite Trapezoidal Rule
def composite_trapezoidal(a, b, f, N):
    h = (b - a) / (N - 1)
    result = 0.5 * (f(a) + f(b))
    for i in range(1, N - 1):
        result += f(a + i * h)
    result *= h
    return result

# Composite Simpson's Rule
def composite_simpsons(a, b, f, N):
    if (N - 1) % 2 != 0:
        raise ValueError("N must be an odd number for Simpson's rule.")
    
    h = (b - a) / (N - 1)
    result = f(a) + f(b)
    
    for i in range(1, N - 1, 2):
        result += 4 * f(a + i * h)
    for i in range(2, N - 2, 2):
        result += 2 * f(a + i * h)
    
    result *= h / 3
    return result

interval_count_trap = 0
interval_count_simp = 0
interval_count_gauss = 0

# Adaptive Trapezoidal Rule with interval counter
def adaptive_trapezoidal(f, a, b, tol, N=5):
    global interval_count_trap
    interval_count_trap += 1
    mid = (a + b) / 2
    coarse = composite_trapezoidal(a, b, f, N)
    fine = composite_trapezoidal(a, mid, f, N) + composite_trapezoidal(mid, b, f, N)
    if abs(coarse - fine) < tol:
        return fine
    else:
        return adaptive_trapezoidal(f, a, mid, tol / 2, N) + adaptive_trapezoidal(f, mid, b, tol / 2, N)

# Adaptive Simpson's Rule with interval counter
def adaptive_simpsons(f, a, b, tol, N=5):
    global interval_count_simp
    interval_count_simp += 1
    mid = (a + b) / 2
    coarse = composite_simpsons(a, b, f, N)
    fine = composite_simpsons(a, mid, f, N) + composite_simpsons(mid, b, f, N)
    if abs(coarse - fine) < tol:
        return fine
    else:
        return adaptive_simpsons(f, a, mid, tol / 2, N) + adaptive_simpsons(f, mid, b, tol / 2, N)

# Adaptive Gaussian Quadrature with interval counter
def adaptive_gaussian(f, a, b, tol):
    global interval_count_gauss
    interval_count_gauss += 1
    # Use scipy's quad as a substitute for Gaussian quadrature with tolerance
    result, error = spi.quad(f, a, b, epsabs=tol)
    return result

import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt


# Define the integrand
def integrand(x):
    return np.sin(1 / x)


# Parameters
a = 0.1
b = 2.0
tolerance = 1e-3
n_values = range(5, 51, 5)  # Number of nodes for non-adaptive methods

# True result using scipy's quad
true_result, _ = spi.quad(integrand, a, b, epsabs=tolerance)

# Store results for plotting
trap_results = []
simp_results = []
gauss_results = []
trap_errors = []
simp_errors = []
gauss_errors = []

# Non-adaptive calculations for each n
for n in n_values:
    trap_approx = composite_trapezoidal(a, b, integrand, n)
    simp_approx = composite_simpsons(a, b, integrand, n) if (n - 1) % 2 == 0 else None
    
    trap_results.append(trap_approx)
    trap_errors.append(abs(true_result - trap_approx))
    
    if simp_approx is not None:
        simp_results.append(simp_approx)
        simp_errors.append(abs(true_result - simp_approx))

# Adaptive results
adaptive_trap_result = adaptive_trapezoidal(integrand, a, b, tolerance)
adaptive_simp_result = adaptive_simpsons(integrand, a, b, tolerance)
adaptive_gauss_result, _ = adaptive_gaussian(integrand, a, b, tolerance)

adaptive_trap_error = abs(true_result - adaptive_trap_result)
adaptive_simp_error = abs(true_result - adaptive_simp_result)
adaptive_gauss_error = abs(true_result - adaptive_gauss_result)

# Plotting results
plt.figure(figsize=(12, 6))

# Plot errors for non-adaptive methods
plt.plot(n_values, trap_errors, label="Non-Adaptive Trapezoidal Error", marker='o')
plt.plot(n_values[:len(simp_errors)], simp_errors, label="Non-Adaptive Simpson's Error", marker='o')

# Plot errors for adaptive methods as horizontal lines
plt.axhline(y=adaptive_trap_error, color='blue', linestyle='--', label="Adaptive Trapezoidal Error")
plt.axhline(y=adaptive_simp_error, color='orange', linestyle='--', label="Adaptive Simpson's Error")
plt.axhline(y=adaptive_gauss_error, color='green', linestyle='--', label="Adaptive Gaussian Error")

# Add labels and title
plt.xlabel("Number of Intervals (N)")
plt.ylabel("Absolute Error")
plt.yscale("log")  # Log scale for better comparison
plt.title("Error Comparison: Adaptive vs Non-Adaptive Quadrature Methods")
plt.legend()
plt.grid(True)
plt.show()