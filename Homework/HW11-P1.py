import numpy as np

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

# Display results
print(f"The approximation of the integral using {method} rule is: {approx_result}")
print(f"The exact value of the integral is: {exact_result}")
print(f"The absolute error is: {absolute_error}")
