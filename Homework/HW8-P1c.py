import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Define the function
def f(x):
    return 1 / (1 + x**2)

# True function values
x_values = np.linspace(-5, 5, 1000)
y_true = f(x_values)

# Node counts for interpolation
node_counts = [5, 10, 15, 20]

# Plot the Natural Cubic Spline interpolation for each n on a single plot
plt.figure(figsize=(12, 5))
plt.plot(x_values, y_true, 'k--', label="True Function")
for n in node_counts:
    nodes = np.linspace(-5, 5, n)
    fnodes = f(nodes)

    # Natural Cubic Spline Interpolation
    natural_spline = CubicSpline(nodes, fnodes, bc_type='natural')
    y_natural_spline = natural_spline(x_values)

    # Plot interpolation
    plt.plot(x_values, y_natural_spline, label=f"Natural Cubic Spline (n = {n})")

plt.scatter(nodes, fnodes, color="red", label="Interpolation Nodes")
plt.title("Natural Cubic Spline Interpolation")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()

# Plot the error for each n on a single plot
plt.figure(figsize=(12, 5))
for n in node_counts:
    nodes = np.linspace(-5, 5, n)
    fnodes = f(nodes)
    natural_spline = CubicSpline(nodes, fnodes, bc_type='natural')
    y_natural_spline = natural_spline(x_values)

    # Calculate and plot error
    error = np.abs(y_true - y_natural_spline)
    plt.plot(x_values, error, label=f"Error (n = {n})")

plt.title("Natural Cubic Spline Interpolation Error")
plt.xlabel("x")
plt.ylabel("Error")
plt.yscale("log")
plt.legend()
plt.show()
