import numpy as np
import matplotlib.pyplot as plt

# Function f(x)
def f(x):
    return 1 / (1 + (10 * x)**2)

# Vandermonde matrix constructor
def vandermonde(x):
    n = len(x)
    V = np.vander(x, N=n, increasing=True)
    return V

# Polynomial evaluation function
def evaluate_polynomial(c, x):
    n = len(c)
    p_x = np.polyval(np.flip(c), x)  # Using polyval with flipped coefficients
    return p_x

# Interpolation function
def interpolate_polynomial(N):
    # Create the points x_i = -1 + (i - 1)h, h = 2 / (N - 1)
    h = 2 / (N - 1)
    x_i = np.linspace(-1, 1, N)
    y_i = f(x_i)

    # Construct Vandermonde matrix
    V = vandermonde(x_i)

    # Solve for the coefficients c
    c = np.linalg.solve(V, y_i)

    return c, x_i, y_i

# Plotting function
def plot_interpolation(N):
    c, x_i, y_i = interpolate_polynomial(N)
    
    # Plot the original points
    #plt.plot(x_i, y_i, 'o', label="Data points")

    # Create a finer grid for plotting the polynomial and the original function
    x_fine = np.linspace(-1, 1, 1001)
    f_fine = f(x_fine)
    
    # Evaluate the polynomial on the finer grid
    p_fine = evaluate_polynomial(c, x_fine)

    # Plot the polynomial and the original function
    plt.plot(x_fine, p_fine, 'o',  label=f"Interpolating polynomial (N={N})")
    plt.plot(x_fine, p_fine, '--')

    plt.plot(x_fine, f_fine, 'o', label="Original function f(x)")

    # Plot settings
    plt.title(f"Polynomial Interpolation with N={N}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

# Test different values of N
#for N in range(2, 21):  # Trying values from N=2 to N=20
plot_interpolation(19)
