import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as scila
import time

def driver():
    # Define the sizes to test
    sizes = [100, 500, 1000, 2000, 4000, 5000]
    right_hand_sides = [1, 5, 10, 50, 100]  # Test with different numbers of right-hand sides
    
    for N in sizes:
        print(f"\nTesting for N = {N}:")
        
        for M in right_hand_sides:  # Loop through the number of right-hand sides
            print(f"  Right-hand sides = {M}:")
            
            ''' Right hand side (M columns for M right-hand sides)'''
            B = np.random.rand(N, M)
            A = np.random.rand(N, N)

            # Time the SCIPY solve method
            start_time = time.time()
            X_scipy = scila.solve(A, B)
            scipy_solve_time = time.time() - start_time
            residual_scipy = la.norm(np.matmul(A, X_scipy) - B)

            # Time the LU factorization method
            # Step 1: LU factorization
            start_time = time.time()
            P, L, U = scila.lu(A)  # LU factorization
            lu_factorization_time = time.time() - start_time

            # Step 2: Solve L * Y = B
            start_time = time.time()
            Y = scila.solve(L, B)  # Solve L * Y = B
            solve_L_time = time.time() - start_time

            # Step 3: Solve U * X = Y
            start_time = time.time()
            X_lu = scila.solve(U, Y)  # Solve U * X = Y
            solve_U_time = time.time() - start_time

            # Total LU solve time
            lu_solve_time = solve_L_time + solve_U_time
            residual_lu = la.norm(np.matmul(A, X_lu) - B)

            # Print results for comparison
            print(f"    Scipy solve residual: {residual_scipy:.5e}, Time: {scipy_solve_time:.5f} seconds")
            print(f"    LU factorization residual: {residual_lu:.5e}, LU factorization time: {lu_factorization_time:.5f} seconds, Solve time: {lu_solve_time:.5f} seconds")

            # Check if LU is faster than SciPy solve
            if lu_solve_time < scipy_solve_time:
                print(f"    LU is faster for {M} right-hand sides.")
            else:
                print(f"    SciPy solve is faster for {M} right-hand sides.")

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
