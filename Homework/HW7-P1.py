import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 1 / (1 + (10 * x)**2)

def vandermonde(x):
    n = len(x)
    V = np.vander(x, N=n, increasing=True)
    return V

def evaluate_polynomial(c, x):
    n = len(c)
    p_x = np.polyval(np.flip(c), x)  # Using polyval with flipped coefficients
    return p_x

def interpolate_polynomial(N):
    h = 2 / (N - 1)
    x_i = np.linspace(-1, 1, N)
    y_i = f(x_i)

    V = vandermonde(x_i)

    c = np.linalg.solve(V, y_i)

    return c, x_i, y_i

# Plotting function
def plot_interpolation(N):
    c, x_i, y_i = interpolate_polynomial(N)

    x_fine = np.linspace(-1, 1, 1001)
    f_fine = f(x_fine)
    
    p_fine = evaluate_polynomial(c, x_fine)

    plt.plot(x_fine, p_fine, 'o',  label=f"Interpolating polynomial (N={N})")
    plt.plot(x_fine, p_fine, '--')

    plt.plot(x_fine, f_fine, 'o', label="Original function f(x)")

    plt.title(f"Polynomial Interpolation with N={N}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_interpolation(19)
