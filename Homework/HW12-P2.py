import numpy as np
import pandas as pd

# Define the matrix A
A = np.array([
    [12, 10,  4],
    [10,  8, -5],
    [ 4, -5,  3]
], dtype=float)

def householder_tridiagonalize(A):
    """
    Perform a similarity transformation using Householder matrices
    to bring the symmetric matrix A to tridiagonal form.
    """
    n = A.shape[0]
    T = A.copy()  # Copy of the matrix to be transformed

    for k in range(n-2):  # Iterate over each column except the last two
        # Extract the subvector starting at the diagonal element
        x = T[k+1:, k]

        # Compute the Householder vector u
        norm_x = np.linalg.norm(x)
        u = x.copy()
        u[0] += np.sign(x[0]) * norm_x
        u /= np.linalg.norm(u)

        # Form the Householder matrix Hk
        Hk = np.eye(n)
        Hk[k+1:, k+1:] -= 2.0 * np.outer(u, u)

        # Apply the similarity transformation: T = Hk * T * Hk.T
        T = Hk @ T @ Hk.T

    return T

# Compute the tridiagonal form
T_tridiagonal = householder_tridiagonalize(A)

# Display the resulting tridiagonal matrix as a DataFrame
df_tridiagonal = pd.DataFrame(T_tridiagonal, columns=["Column 1", "Column 2", "Column 3"])
print("Tridiagonal Matrix of A:")
print(df_tridiagonal)
