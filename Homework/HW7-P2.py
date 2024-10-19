import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 1 / (1 + (10 * x)**2)

def barycentric_weights(x):
    n = len(x)
    w = np.ones(n)
    for j in range(n):
        for i in range(n):
            if i != j:
                w[j] /= (x[j] - x[i])
    return w

def barycentric_interpolation(xi, yi, w, x_eval):
    numerator = np.zeros_like(x_eval)
    denominator = np.zeros_like(x_eval)
    
    for j in range(len(xi)):
        temp = w[j] / (x_eval - xi[j])
        numerator += temp * yi[j]
        denominator += temp
    
    return numerator / denominator

def interpolate_barycentric(N):
    h = 2 / (N - 1)
    x_i = np.linspace(-1, 1, N)
    y_i = f(x_i)

    w = barycentric_weights(x_i)

    return x_i, y_i, w

# Plotting function
def plot_barycentric_interpolation(N):
    x_i, y_i, w = interpolate_barycentric(N)
   
    x_fine = np.linspace(-1, 1, 1001)
    f_fine = f(x_fine)
    
    p_fine = barycentric_interpolation(x_i, y_i, w, x_fine)

    plt.plot(x_fine, p_fine,'o', label=f"Barycentric polynomial (N={N})")
    plt.plot(x_fine, f_fine, 'o', label="Original function f(x)")

    plt.title(f"Barycentric Interpolation with N={N}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

    err = abs(f_fine - p_fine)
    
    plt.semilogy(x_fine, err,'o', label='Absolute error')
    plt.legend()
    plt.show()

plot_barycentric_interpolation(19)
