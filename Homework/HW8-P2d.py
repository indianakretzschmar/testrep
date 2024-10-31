import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Define the function and its derivative for clamped spline
def f(x):
    return 1 / (1 + x**2)

def f_prime(x):
    return -2 * x / (1 + x**2)**2

# True function values
x_values = np.linspace(-5, 5, 1000)
y_true = f(x_values)

# Node counts for interpolation
node_counts = [5, 10, 15, 20]

# Plot the Clamped Cubic Spline interpolation with Chebyshev nodes for each n on a single plot
plt.figure(figsize=(12, 5))
plt.plot(x_values, y_true, 'k--', label="True Function")
for n in node_counts:
    # Generate Chebyshev nodes
    nodes = -5 * np.cos((2 * np.arange(n) + 1) * np.pi / (2 * n))
    fnodes = f(nodes)

    # Clamped Cubic Spline Interpolation
    clamped_spline = CubicSpline(nodes, fnodes, bc_type=((1, f_prime(nodes[0])), (1, f_prime(nodes[-1]))))
    y_clamped_spline = clamped_spline(x_values)

    # Plot interpolation
    plt.plot(x_values, y_clamped_spline, label=f"Clamped Cubic Spline (n = {n})")

plt.scatter(nodes, fnodes, color="red", label="Chebyshev Nodes")
plt.title("Clamped Cubic Spline Interpolation with Chebyshev Nodes")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()

# Plot the error for each n on a single plot
plt.figure(figsize=(12, 5))
for n in node_counts:
    # Generate Chebyshev nodes
    nodes = -5 * np.cos((2 * np.arange(n) + 1) * np.pi / (2 * n))
    fnodes = f(nodes)
    clamped_spline = CubicSpline(nodes, fnodes, bc_type=((1, f_prime(nodes[0])), (1, f_prime(nodes[-1]))))
    y_clamped_spline = clamped_spline(x_values)

    # Calculate and plot error
    error = np.abs(y_true - y_clamped_spline)
    plt.plot(x_values, error, label=f"Error (n = {n})")

plt.title("Clamped Cubic Spline Interpolation Error with Chebyshev Nodes")
plt.xlabel("x")
plt.ylabel("Error")
plt.yscale("log")
plt.legend()
plt.show()
