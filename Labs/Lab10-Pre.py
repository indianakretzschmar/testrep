import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.integrate import quad

def driver():
    # function you want to approximate
    f = lambda x: math.exp(x)
    # Interval of interest
    a = -1
    b = 1
    # weight function
    w = lambda x: 1.
    # order of approximation
    n = 2
    # Number of points you want to sample in [a,b]
    N = 1000
    xeval = np.linspace(a, b, N + 1)
    pval = np.zeros(N + 1)
    for kk in range(N + 1):
        pval[kk] = eval_legendre_expansion(f, a, b, w, n, xeval[kk])
    ''' create vector with exact values'''
    fex = np.zeros(N + 1)
    for kk in range(N + 1):
        fex[kk] = f(xeval[kk])
    plt.figure()
    plt.plot(xeval, fex, 'ro-', label='f(x)')
    plt.plot(xeval, pval, 'bs--', label='Expansion')
    plt.legend()
    plt.show()
    err = abs(pval - fex)
    plt.semilogy(xeval, err, 'ro--', label='error')
    plt.legend()
    plt.show()

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

def eval_legendre_expansion(f, a, b, w, n, x):
    # Generate Legendre polynomials at x using eval_legendre
    p = eval_legendre(n, x)
    
    # Initialize the sum to 0
    pval = 0.0
    
    for j in range(n + 1):
        # Define phi_j(x) as a lambda function for the current Legendre polynomial
        phi_j = lambda x: eval_legendre(j, x)[j]
        
        # Define phi_j^2(x) * w(x)
        phi_j_sq = lambda x: phi_j(x)**2 * w(x)
        
        # Calculate the normalization factor using quad
        norm_fac, err = quad(phi_j_sq, a, b)
        
        # Define the integrand for the coefficient a_j
        func_j = lambda x: phi_j(x) * f(x) * w(x) / norm_fac
        
        # Calculate the coefficient a_j using quad
        aj, err = quad(func_j, a, b)
        
        # Accumulate the value of the expansion at x
        pval += aj * p[j]
    
    return pval

if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    driver()
