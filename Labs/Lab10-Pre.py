import numpy as np


def eval_legendre(n, x):
    # Initialize a vector to store Legendre polynomials up to order n
    p = np.zeros(n + 1)
    p[0] = 1  # ϕ0(x) = 1
    if n > 0:
        p[1] = x  # ϕ1(x) = x
    # Use the recursion formula for higher orders
    for k in range(1, n):
        p[k + 1] = ((2 * k + 1) * x * p[k] - k * p[k - 1]) / (k + 1)
    return p

