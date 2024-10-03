import numpy as np

# Define the function and true derivative
f = lambda x: np.cos(x)
f_prime_exact = lambda x: -np.sin(x)

# The point where we approximate the derivative
s = np.pi / 2

# Generate values of h
h_values = 0.01 * 2.0 ** (-np.arange(0, 10))

# Forward difference approximation
forward_diff = (f(s + h_values) - f(s)) / h_values

# Centered difference approximation
centered_diff = (f(s + h_values) - f(s - h_values)) / (2 * h_values)

# Exact derivative
exact_derivative = f_prime_exact(s)

# Print results
print("h values:", h_values)
print("Forward Difference Approximations:", forward_diff)
print("Centered Difference Approximations:", centered_diff)
print("Exact Derivative:", exact_derivative)


import matplotlib.pyplot as plt

# Calculate errors
error_forward = np.abs(forward_diff - exact_derivative)
error_centered = np.abs(centered_diff - exact_derivative)

# Plot errors on a log-log plot
plt.plot(h_values, error_forward, label="Forward Difference Error", marker='o')
plt.plot(h_values, error_centered, label="Centered Difference Error", marker='x')

# Add labels and legend
plt.xlabel("h")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.show()