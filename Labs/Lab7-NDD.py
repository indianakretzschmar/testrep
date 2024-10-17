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


# Function to compute Newton's divided differences
def newton_divided_diff(xj, yj):
    N = len(xj)
    coeff = np.copy(yj)
    for j in range(1, N):
        coeff[j:N] = (coeff[j:N] - coeff[j-1:N-1]) / (xj[j:N] - xj[0:N-j])
    return coeff

# Function to evaluate the Newton polynomial using the divided differences
def evaluate_newton(x, xj, coeff):
    N = len(xj)
    p = coeff[-1] * np.ones_like(x)
    for j in range(N-2, -1, -1):
        p = p * (x - xj[j]) + coeff[j]
    return p

# Function to calculate the absolute error
def absolute_error(p_eval, x_eval):
    return np.abs(f(x_eval) - p_eval)

# Generate interpolation points and evaluate the polynomial
def interpolate_newton(N):
    xj = np.linspace(-1, 1, N)  # interpolation nodes
    yj = f(xj)  # function values at interpolation points
    coeff = newton_divided_diff(xj, yj)  # divided difference coefficients

    # Evaluate the polynomial at 1000 points
    x_eval = np.linspace(-1, 1, 1000)
    p_eval = evaluate_newton(x_eval, xj, coeff)
    return x_eval, p_eval, xj, yj

def plot_newton(N):
    x_eval, p_eval, xj, yj = interpolate_newton(N)
    error = absolute_error(p_eval, x_eval)

    plt.figure(figsize=(10,5))

    # Plot the approximation
    plt.subplot(1, 2, 1)
    plt.plot(x_eval, p_eval, label="Newton-Divided Differences")
    plt.scatter(xj, yj, color='red', zorder=5, label="Interpolation Points")
    plt.plot(x_eval, f(x_eval), label="Original Function", linestyle='dashed')
    plt.title(f"Newton-Divided Differences Interpolation for N={N}")
    plt.legend()

    # Plot the absolute error
    plt.subplot(1, 2, 2)
    plt.plot(x_eval, error, label="Absolute Error")
    plt.title(f"Absolute Error for N={N}")
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_newton(5)




