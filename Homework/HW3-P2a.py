import numpy as np

def driver():

    # use routine for x^3 + x - 4
    f = lambda x:  x**9 - 45*x**8 + 1260*x**7 - 17640*x**6 + 137214*x**5 - 615084*x**4 + 1640250*x**3 - 2460375*x**2 + 1953125*x - 781250

    a = 1
    b = 4
    tol = 1e-3

    [astar, ier, count] = bisection(f, a, b, tol)
    print('The approximate root is:', astar)
    print('The error message reads:', ier)
    print('f(astar) =', f(astar))
    print('Number of iterations:', count)

def bisection(f, a, b, tol):
    # Inputs: f,a,b - function and endpoints of initial interval
    # tol - bisection stops when interval length < tol
    # Returns: astar - approximation of root, ier - error message

    # First verify there is a root in the interval
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        ier = 1
        astar = a
        return [astar, ier]

    # Verify end points are not a root
    if fa == 0:
        astar = a
        ier = 0
        return [astar, ier]

    if fb == 0:
        astar = b
        ier = 0
        return [astar, ier]

    count = 0
    d = 0.5 * (a + b)
    while abs(d - a) > tol:
        fd = f(d)
        if fd == 0:
            astar = d
            ier = 0
            return [astar, ier]
        if fa * fd < 0:
            b = d
        else:
            a = d
            fa = fd
        d = 0.5 * (a + b)
        count += 1

    astar = d
    ier = 0
    return [astar, ier, count]

driver()
