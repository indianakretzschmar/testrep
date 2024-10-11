import numpy as np
import matplotlib.pyplot as plt

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

# Function to compute Lagrange basis polynomials
def lagrange_basis(x, xj, j):
    Lj = np.ones_like(x)
    for i, xi in enumerate(xj):
        if i != j:
            Lj *= (x - xi) / (xj[j] - xi)
    return Lj

# Function to evaluate the Lagrange polynomial
def evaluate_lagrange(x, xj, yj):
    N = len(xj)
    p = np.zeros_like(x)
    for j in range(N):
        Lj = lagrange_basis(x, xj, j)
        p += yj[j] * Lj
    return p

# Function to calculate the absolute error
def absolute_error(p_eval, x_eval):
    return np.abs(f(x_eval) - p_eval)

# Generate interpolation points and evaluate the polynomial
def interpolate_lagrange(N):
    xj = np.linspace(-1, 1, N)  # interpolation nodes
    yj = f(xj)  # function values at interpolation points

    # Evaluate the polynomial at 1000 points
    x_eval = np.linspace(-1, 1, 1000)
    p_eval = evaluate_lagrange(x_eval, xj, yj)
    return x_eval, p_eval, xj, yj

def plot_lagrange(N):
    x_eval, p_eval, xj, yj = interpolate_lagrange(N)
    error = absolute_error(p_eval, x_eval)

    plt.figure(figsize=(10,5))

    # Plot the approximation
    plt.subplot(1, 2, 1)
    plt.plot(x_eval, p_eval, label="Lagrange Polynomials")
    plt.scatter(xj, yj, color='red', zorder=5, label="Interpolation Points")
    plt.plot(x_eval, f(x_eval), label="Original Function", linestyle='dashed')
    plt.title(f"Lagrange Polynomial Interpolation for N={N}")
    plt.legend()

    # Plot the absolute error
    plt.subplot(1, 2, 2)
    plt.plot(x_eval, error, label="Absolute Error")
    plt.title(f"Absolute Error for N={N}")
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_lagrange(5)