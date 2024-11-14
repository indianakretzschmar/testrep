import numpy as np

# Define the transformed integrand with the t factor
def transformed_integrand(t):
    return np.cos(1 / t) * t

# Composite Simpson's Rule function
def composite_simpsons_for_transformed_integral(a, b, n):
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's rule.")
    
    h = (b - a) / n
    result = transformed_integrand(a) + transformed_integrand(b)
    
    for i in range(1, n, 2):
        result += 4 * transformed_integrand(a + i * h)
    for i in range(2, n-1, 2):
        result += 2 * transformed_integrand(a + i * h)
    
    result *= h / 3
    return result  # Include the negative sign from the transformation

# Parameters for the integral
a = 1e-20  # slightly above zero to avoid division by zero
b = 1      # upper limit after transformation
n = 4      # number of intervals (5 nodes)

# Calculate the approximation
approximation = composite_simpsons_for_transformed_integral(a, b, n)
print(f"The approximation of the integral is: {approximation}")
