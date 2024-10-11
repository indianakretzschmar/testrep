import numpy as np
import matplotlib.pyplot as plt

# Function to construct the Vandermonde matrix
def vandermonde_matrix(x):
    N = len(x)
    V = np.vander(x, increasing=True)
    return V

# Function to solve for the coefficients of the monomial expansion
def monomial_coefficients(x, y):
    V = vandermonde_matrix(x)
    a = np.linalg.solve(V, y)
    return a

# Function to evaluate the polynomial at new points
def evaluate_monomial(a, x):
    # Polynomial evaluation using Horner's method
    p = np.polyval(a[::-1], x)
    return p

# Example function f(x) = 1 / (1 + (10x)^2)
def f(x):
    return 1 / (1 + (10*x)**2)

def absolute_error(p_eval, x_eval):
    return np.abs(f(x_eval) - p_eval)

def interpolate_monomial(N):
    xj = np.linspace(-1, 1, N)  # interpolation nodes
    yj = f(xj)  # function values at interpolation points
    a = monomial_coefficients(xj, yj)  # get the coefficients

    # Evaluate the polynomial at 1000 points
    x_eval = np.linspace(-1, 1, 1000)
    p_eval = evaluate_monomial(a, x_eval)
    return x_eval, p_eval, xj, yj

# Updated plot for Monomial Expansion
def plot_monomial(N):
    x_eval, p_eval, xj, yj = interpolate_monomial(N)
    error = absolute_error(p_eval, x_eval)

    plt.figure(figsize=(10,5))

    # Plot the approximation
    plt.subplot(1, 2, 1)
    plt.plot(x_eval, p_eval, label="Monomial Expansion")
    plt.scatter(xj, yj, color='red', zorder=5, label="Interpolation Points")
    plt.plot(x_eval, f(x_eval), label="Original Function", linestyle='dashed')
    plt.title(f"Monomial Expansion Interpolation for N={N}")
    plt.legend()

    # Plot the absolute error
    plt.subplot(1, 2, 2)
    plt.plot(x_eval, error, label="Absolute Error")
    plt.title(f"Absolute Error for N={N}")
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_monomial(20)