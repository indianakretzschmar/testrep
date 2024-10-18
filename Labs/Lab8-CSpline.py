import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm

def driver():
    f = lambda x: 1 / (1 + (10 * x)**2)
    a = -1
    b = 1
    # Number of intervals
    Nint = 10
    xint = np.linspace(a, b, Nint + 1)
    yint = f(xint)

    # Create points to evaluate
    Neval = 100
    xeval = np.linspace(xint[0], xint[Nint], Neval + 1)

    # Create the coefficients for the natural spline
    (M, C, D) = create_natural_spline(yint, xint, Nint)

    # Evaluate the cubic spline
    yeval = eval_cubic_spline(xeval, Neval, xint, Nint, M, C, D)

    # Evaluate the function at the evaluation points
    fex = f(xeval)
    nerr = norm(fex - yeval)
    print('nerr = ', nerr)

    # Plot the exact function and natural spline
    plt.figure()
    plt.plot(xeval, fex, 'ro-', label='exact function')
    plt.plot(xeval, yeval, 'bs--', label='natural spline')
    plt.legend()
    plt.show()

    # Plot the absolute error
    err = abs(yeval - fex)
    plt.figure()
    plt.semilogy(xeval, err, 'ro--', label='absolute error')
    plt.legend()
    plt.show()

def create_natural_spline(yint, xint, N):
    # Create the right-hand side for the linear system
    b = np.zeros(N + 1)
    h = np.zeros(N)

    for i in range(1, N):
        h[i-1] = xint[i] - xint[i-1]
        b[i] = (yint[i+1] - yint[i]) / (xint[i+1] - xint[i]) - (yint[i] - yint[i-1]) / (xint[i] - xint[i-1])

    h[N-1] = xint[N] - xint[N-1]

    # Create matrix A for the system
    A = np.zeros((N + 1, N + 1))
    A[0, 0] = 1  # Natural spline condition M_0 = 0
    A[N, N] = 1  # Natural spline condition M_N = 0

    for i in range(1, N):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]

    # Invert matrix A to solve for M
    Ainv = inv(A)
    M = Ainv @ b  # Solve for M

    # Calculate the coefficients C and D
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
        C[j] = (yint[j+1] - yint[j]) / h[j] - h[j] * (M[j+1] + 2 * M[j]) / 6
        D[j] = (M[j+1] - M[j]) / h[j]

    return (M, C, D)

def eval_local_spline(xeval, xi, xip, Mi, Mip, C, D):
    # Evaluate the local cubic spline on the interval [xi, xip]
    hi = xip - xi
    term1 = Mi * ((xip - xeval)**3) / (6 * hi)
    term2 = Mip * ((xeval - xi)**3) / (6 * hi)
    term3 = C * (xeval - xi)
    term4 = D
    yeval = term1 + term2 + term3 + term4
    return yeval

def eval_cubic_spline(xeval, Neval, xint, Nint, M, C, D):
    yeval = np.zeros(Neval + 1)
    for j in range(Nint):
        # Find indices of xeval in interval (xint[j], xint[j+1])
        atmp = xint[j]
        btmp = xint[j+1]

        # Get indices where xeval lies within the interval
        ind = np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]

        # Evaluate the spline locally
        yloc = eval_local_spline(xloc, atmp, btmp, M[j], M[j+1], C[j], D[j])

        # Copy into yeval
        yeval[ind] = yloc

    return yeval

driver()
