import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as scila
import time

def driver():
    # Define the sizes to test
    sizes = [100, 500, 1000, 2000, 4000, 5000]
    
    for N in sizes:
        print(f"Testing for N = {N}:")

        ''' Right hand side'''
        b = np.random.rand(N, 1)
        A = np.random.rand(N, N)

        # Time the SCIPY solve method
        start_time = time.time()
        x = scila.solve(A, b)
        scipy_solve_time = time.time() - start_time
        test = np.matmul(A, x)
        r = la.norm(test - b)
        print(f"Scipy solve residual: {r}, Time: {scipy_solve_time:.5f} seconds")

        # Time the LU factorization method
        # Step 1: LU factorization
        start_time = time.time()
        P, L, U = scila.lu(A)  # LU factorization
        lu_factorization_time = time.time() - start_time

        # Step 2: Solve L * y = b
        start_time = time.time()
        y = scila.solve(L, b)  # Solve L * y = b
        solve_L_time = time.time() - start_time

        # Step 3: Solve U * x = y
        start_time = time.time()
        x_lu = scila.solve(U, y)  # Solve U * x = y
        solve_U_time = time.time() - start_time

        # Total LU solve time
        lu_solve_time = solve_L_time + solve_U_time
        test_lu = np.matmul(A, x_lu)
        r_lu = la.norm(test_lu - b)

        print(f"LU factorization residual: {r_lu}, LU factorization time: {lu_factorization_time:.5f} seconds, Solve time: {lu_solve_time:.5f} seconds")

def create_rect(N, M):
    ''' this subroutine creates an ill-conditioned rectangular matrix'''
    a = np.linspace(1, 10, M)
    d = 10**(-a)

    D2 = np.zeros((N, M))
    for j in range(0, M):
        D2[j, j] = d[j]

    '''' create matrices needed to manufacture the low rank matrix'''
    A = np.random.rand(N, N)
    Q1, R = la.qr(A)
    test = np.matmul(Q1, R)
    A = np.random.rand(M, M)
    Q2, R = la.qr(A)
    test = np.matmul(Q2, R)

    B = np.matmul(Q1, D2)
    B = np.matmul(B, Q2)
    return B

if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    driver()
