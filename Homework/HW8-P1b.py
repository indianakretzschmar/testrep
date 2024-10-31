import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# Define the function
def f(x):
    return 1 / (1 + x**2)

# True function values
x_values = np.linspace(-5, 5, 1000)
y_true = f(x_values)

# Node counts for interpolation
node_counts = [5, 10, 15, 20]

# Plot the Hermite (PCHIP) interpolation for each n on a single plot
plt.figure(figsize=(12, 5))
plt.plot(x_values, y_true, 'k--', label="True Function")
for n in node_counts:
    nodes = np.linspace(-5, 5, n)
    fnodes = f(nodes)

    # Hermite (PCHIP) Interpolation
    hermite = PchipInterpolator(nodes, fnodes)
    y_hermite = hermite(x_values)

    # Plot interpolation
    plt.plot(x_values, y_hermite, label=f"Hermite (PCHIP) (n = {n})")

plt.scatter(nodes, fnodes, color="red", label="Interpolation Nodes")
plt.title("Hermite (PCHIP) Interpolation")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()

# Plot the error for each n on a single plot
plt.figure(figsize=(12, 5))
for n in node_counts:
    nodes = np.linspace(-5, 5, n)
    fnodes = f(nodes)
    hermite = PchipInterpolator(nodes, fnodes)
    y_hermite = hermite(x_values)

    # Calculate and plot error
    error = np.abs(y_true - y_hermite)
    plt.plot(x_values, error, label=f"Error (n = {n})")

plt.title("Hermite (PCHIP) Interpolation Error")
plt.xlabel("x")
plt.ylabel("Error")
plt.yscale("log")
plt.legend()
plt.show()
