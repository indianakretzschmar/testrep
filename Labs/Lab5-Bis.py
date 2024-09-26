import numpy as np

def driver():
    f = lambda x: np.sin(x)           # Function whose root is to be found
    df = lambda x: np.cos(x)          # Derivative of f
    ddf = lambda x: -np.sin(x)        # Second derivative of f

    a = 0.5
    b = (3 * np.pi) / 4
    tol = 1e-5

    [astar, ier] = bisection(f, df, ddf, a, b, tol)
    print('The approximate root is:', astar)
    print('The error message reads:', 'Success' if ier == 0 else 'Failed')
    print('f(astar) =', f(astar))

def bisection(f, df, ddf, a, b, tol):
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        return [a, 1]  # No change in sign

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
            return [d, 0]  # Midpoint is within the basin

        # Usual bisection steps
        if fa * fd < 0:
            b = d
        else:
            a = d
            fa = fd

        d = 0.5 * (a + b)

    return [d, 0]  # Convergence if reached here without finding the exact root

driver()
