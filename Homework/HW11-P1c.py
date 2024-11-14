from scipy.integrate import quad
import numpy as np

def f(s):
    return 1 / (1 + s**2)


def integrand(s):
    return 1 / (1 + s**2)

def composite_trapezoidal(a, b, n):
    h = (b - a) / n
    result = 0.5 * (integrand(a) + integrand(b))
    for i in range(1, n):
        result += integrand(a + i * h)
    result *= h
    return result

def composite_simpsons(a, b, n):
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's rule.")
    h = (b - a) / n
    result = integrand(a) + integrand(b)
    for i in range(1, n, 2):
        result += 4 * integrand(a + i * h)
    for i in range(2, n-1, 2):
        result += 2 * integrand(a + i * h)
    result *= h / 3
    return result

def approximate_integral(a, b, n, method="trapezoidal"):
    if method == "trapezoidal":
        return composite_trapezoidal(a, b, n)
    elif method == "simpsons":
        return composite_simpsons(a, b, n)
    else:
        raise ValueError("Method must be 'trapezoidal' or 'simpsons'.")

# Define parameters
a = -5  # lower limit
b = 5   # upper limit
n = 10  # number of subintervals (for Simpson's rule, n must be even)

# Calculate exact result
exact_result = 2 * np.arctan(5)

# Choose method
method = "trapezoidal"  # Choose either "trapezoidal" or "simpsons"

# Compute approximation
approx_result = approximate_integral(a, b, n, method)

# Calculate absolute error
absolute_error = abs(exact_result - approx_result)


n_trap = 1291
n_simp = 108
# Reuse the composite_trapezoidal and composite_simpsons functions from the previous code
approx_trap = composite_trapezoidal(a, b, n_trap)
approx_simp = composite_simpsons(a, b, n_simp)

# Compute using scipy's quad with different tolerances
scipy_result_default, scipy_error_default = quad(f, a, b)
scipy_result_1e4, scipy_error_1e4 = quad(f, a, b, epsabs=1e-4)

# Number of function evaluations
trap_evals = n_trap + 1  # n+1 evaluations for Trapezoidal
simp_evals = n_simp + 1  # n+1 evaluations for Simpson's

# Output results
print(f"Trapezoidal (n={n_trap}): Approx = {approx_trap}, Func Evals = {trap_evals}")
print(f"Simpson's (n={n_simp}): Approx = {approx_simp}, Func Evals = {simp_evals}")
print("\nscipy quad results:")
print(f"Default tolerance (1e-6): Approx = {scipy_result_default}, Error = {scipy_error_default}")
print(f"Tolerance 1e-4: Approx = {scipy_result_1e4}, Error = {scipy_error_1e4}")
