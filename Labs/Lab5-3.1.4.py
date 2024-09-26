import numpy as np

def driver():
    # Define the function and its derivatives
    f = lambda x: np.sin(x)           # Example function
    df = lambda x: np.cos(x)          # First derivative of f
    ddf = lambda x: -np.sin(x)        # Second derivative of f

    # Initial interval and tolerance
    a = 0.5
    b = (3 * np.pi) / 4
    tol = 1e-5

    # Call the hybrid method
    [astar, ier] = bisection_newton(f, df, ddf, a, b, tol)
    print('The approximate root is:', astar)
    print('The error message reads:', 'Success' if ier == 0 else 'Failed')
    print('f(astar) =', f(astar))

def bisection_newton(f, df, ddf, a, b, tol):
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        return [a, 1]  # No root in the interval

    if fa == 0:
        return [a, 0]

    if fb == 0:
        return [b, 0]

    d = 0.5 * (a + b)
    while abs(d - a) > tol:
        fd = f(d)
        if fd == 0:
            return [d, 0]

        # Check if d is within the basin of convergence for Newton's method
        if abs(f(d) * ddf(d)) < abs(df(d)**2):
            # Newton's method starts here
            return newton_method(f, df, d, tol)

        # Usual bisection steps
        if fa * fd < 0:
            b = d
        else:
            a = d
            fa = fd

        d = 0.5 * (a + b)

    return [d, 0]  # If the bisection method converges without switching to Newton's

def newton_method(f, df, x0, tol):
    # Newton's method to refine the root approximation
    max_iter = 1000  # Avoid infinite loops
    for _ in range(max_iter):
        fx0 = f(x0)
        dfx0 = df(x0)
        if abs(dfx0) < np.finfo(float).eps:  # Avoid division by zero
            return [x0, 1]  # Derivative too small, possible failure
        x1 = x0 - fx0 / dfx0
        if abs(x1 - x0) < tol:  # Check if the result is within the desired tolerance
            return [x1, 0]
        x0 = x1
    return [x0, 1]  # Return after max iterations if no convergence

driver()
