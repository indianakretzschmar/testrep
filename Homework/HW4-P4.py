import numpy as np

# Define the function f and its derivative f'
def f(x):
    return np.exp(3*x) - 27*x**6 + 27*x**4*np.exp(x) - 9*x**2*np.exp(2*x)

def df(x):
    return 3*np.exp(3*x) - 162*x**5 + 27*x**4*np.exp(x) + 108*x**3*np.exp(x) - 18*x**2*np.exp(2*x) - 18*x*np.exp(2*x)

# Initial guess
x0 = 4
# Number of iterations
n_iter = 10

# Standard Newton's Method
def newton_method(f, df, x0, n_iter):
    x = x0
    for _ in range(n_iter):
        x = x - f(x) / df(x)
    return x

def ddf(x):
    return (27*x**4 + 324*x**3 + 81*x**4)*np.exp(x) + (54*x**2 + 54*x)*np.exp(2*x) + 9*np.exp(3*x) - 810*x**4

# Corrected modified Newton's Method using g(x) = f(x) / f'(x)
def modified_newton(f, df, ddf, x0, n_iter):
    x = x0
    for _ in range(n_iter):
        x = x - f(x) / (df(x) - 0.5 * f(x) * ddf(x) / df(x))  # Applying the quotient rule correction
    return x

# Corrected implementation of the fixed-point iteration from problem 2c with m = 3
def fixed_point_iteration_2c(f, df, x0, n_iter, m=3):
    x = x0
    for _ in range(n_iter):
        x = x - m * f(x) / df(x)
    return x

# Re-run the methods with the refined details



# Compute roots using the different methods
root_newton = newton_method(f, df, x0, n_iter)
root_modified_2c = modified_newton(f, df, ddf, x0, n_iter)
root_fixed_point_2c = fixed_point_iteration_2c(f, df, x0, n_iter)

print(root_newton)
print(root_modified_2c) 
print(root_fixed_point_2c)