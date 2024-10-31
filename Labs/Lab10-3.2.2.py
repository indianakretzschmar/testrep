import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.integrate import quad

# Function to evaluate Legendre polynomials at x up to order n
def eval_legendre(n, x):
    p = np.zeros(n + 1)
    p[0] = 1  # ϕ0(x) = 1
    if n > 0:
        p[1] = x  # ϕ1(x) = x
    for k in range(1, n):
        p[k + 1] = ((2 * k + 1) * x * p[k] - k * p[k - 1]) / (k + 1)
    return p

# Function to evaluate the coefficient a_j
def eval_aj(f, phi_j, w, j, a, b):
    return quad(lambda x: f(x) * phi_j(x, j) * w(x), a, b)[0] / quad(lambda x: phi_j(x, j)**2 * w(x), a, b)[0]

# Function for Legendre expansion approximation
def eval_legendre_expansion(f, a, b, w, n, x):
    # Evaluate Legendre polynomials up to order n at x
    p = eval_legendre(n, x)
    
    # Initialize sum
    pval = 0.0
    
    for j in range(n + 1):
        # Define phi_j(x) for the j-th Legendre polynomial
        phi_j = lambda x, j=j: eval_legendre(j, x)[j]
        
        # Calculate the coefficient a_j
        aj = eval_aj(f, phi_j, w, j, a, b)
        
        # Accumulate the contribution to the polynomial approximation at x
        pval += aj * p[j]
    
    return pval

# Main driver function
def driver():
    # Function to approximate
    #f = lambda x: math.exp(x)
    f = lambda x: 1 / (1 + x**2)
    # Interval of interest
    a, b = -1, 1
    # Weight function
    w = lambda x: 1.0
    # Order of approximation
    n = 2
    # Number of points to sample in [a, b]
    N = 1000
    xeval = np.linspace(a, b, N + 1)
    pval = np.zeros(N + 1)
    
    # Evaluate the Legendre expansion at each x in xeval
    for kk in range(N + 1):
        pval[kk] = eval_legendre_expansion(f, a, b, w, n, xeval[kk])
    
    # Create vector with exact values of f(x)
    fex = np.array([f(x) for x in xeval])
    
    # Plot f(x) and the Legendre expansion approximation
    plt.figure()
    plt.plot(xeval, fex, 'ro-', label='f(x)')
    plt.plot(xeval, pval, 'bs--', label='Expansion')
    plt.legend()
    plt.show()
    
    # Plot the error between the approximation and exact values
    err = np.abs(pval - fex)
    plt.semilogy(xeval, err, 'ro--', label='Error')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    driver()
