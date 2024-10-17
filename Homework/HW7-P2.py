import numpy as np
import matplotlib.pyplot as plt

# Function f(x)
def f(x):
    return 1 / (1 + (10 * x)**2)

# Barycentric weights calculation
def barycentric_weights(x):
    n = len(x)
    w = np.ones(n)
    for j in range(n):
        for i in range(n):
            if i != j:
                w[j] /= (x[j] - x[i])
    return w

# Barycentric interpolation formula
def barycentric_interpolation(xi, yi, w, x_eval):
    numerator = np.zeros_like(x_eval)
    denominator = np.zeros_like(x_eval)
    
    # Evaluate the interpolation at each point in x_eval
    for j in range(len(xi)):
        temp = w[j] / (x_eval - xi[j])
        numerator += temp * yi[j]
        denominator += temp
    
    return numerator / denominator

# Interpolation function using barycentric interpolation
def interpolate_barycentric(N):
    # Create the points x_i = -1 + (i - 1)h, h = 2 / (N - 1)
    h = 2 / (N - 1)
    x_i = np.linspace(-1, 1, N)
    y_i = f(x_i)

    # Calculate the weights w_j
    w = barycentric_weights(x_i)

    return x_i, y_i, w

# Plotting function
def plot_barycentric_interpolation(N):
    x_i, y_i, w = interpolate_barycentric(N)
    
    # Plot the original points
    plt.plot(x_i, y_i, 'o', label="Data points")

    # Create a finer grid for plotting the polynomial and the original function
    x_fine = np.linspace(-1, 1, 1001)
    f_fine = f(x_fine)
    
    # Evaluate the polynomial on the finer grid using barycentric interpolation
    p_fine = barycentric_interpolation(x_i, y_i, w, x_fine)

    # Plot the polynomial and the original function
    plt.plot(x_fine, p_fine, label=f"Barycentric polynomial (N={N})")
    plt.plot(x_fine, f_fine, '--', label="Original function f(x)")

    # Plot settings
    plt.title(f"Barycentric Interpolation with N={N}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

# Test different values of N
for N in range(2, 21):  # Trying values from N=2 to N=20
    plot_barycentric_interpolation(N)
