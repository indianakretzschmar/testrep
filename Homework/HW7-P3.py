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

def chebyshev_points(N):
    return np.cos((2 * np.arange(1, N + 1) - 1) * np.pi / (2 * N))

def interpolate_barycentric_chebyshev(N):
    x_i = chebyshev_points(N)
    y_i = f(x_i)

    w = barycentric_weights(x_i)

    return x_i, y_i, w

def plot_barycentric_interpolation_chebyshev(N):
    x_i, y_i, w = interpolate_barycentric_chebyshev(N)
    
    plt.plot(x_i, y_i, 'o', label="Chebyshev points")

    x_fine = np.linspace(-1, 1, 1001)
    f_fine = f(x_fine)
    
    p_fine = barycentric_interpolation(x_i, y_i, w, x_fine)

    plt.plot(x_fine, p_fine, label=f"Barycentric polynomial (N={N})")
    plt.plot(x_fine, f_fine, '--', label="Original function f(x)")

    plt.title(f"Barycentric Interpolation with Chebyshev Points (N={N})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

    err = abs(f_fine - p_fine)
    
    plt.semilogy(x_fine, err, label='Absolute error')
    plt.legend()
    plt.show()

plot_barycentric_interpolation_chebyshev(2)


