import numpy as np
import pandas as pd

def hilbert_matrix(n):
    """Generate the Hilbert matrix of size n x n."""
    return np.array([[1 / (i + j - 1) for j in range(1, n + 1)] for i in range(1, n + 1)])

def power_method(A, tol=1e-8, max_iter=1000):
    """
    Perform the power method to find the dominant eigenvalue and eigenvector.
    
    Parameters:
    A (ndarray): The matrix for which to find the dominant eigenvalue.
    tol (float): The tolerance for convergence.
    max_iter (int): The maximum number of iterations.
    
    Returns:
    dominant_eigenvalue, dominant_eigenvector, iterations
    """
    n = A.shape[0]
    x = np.random.rand(n)  # Random initial vector
    x = x / np.linalg.norm(x)  # Normalize
    
    for k in range(max_iter):
        x_next = A @ x  # Matrix-vector product
        x_next = x_next / np.linalg.norm(x_next)  # Normalize
        eigenvalue = np.dot(x_next.T, A @ x_next)  # Rayleigh quotient
        
        # Check for convergence
        if np.linalg.norm(A @ x_next - eigenvalue * x_next) < tol:
            return eigenvalue, x_next, k + 1
        
        x = x_next
    
    raise ValueError("Power method did not converge within the maximum number of iterations")

# Test the power method on Hilbert matrices for n = 4, 8, ..., 20
results = []
for n in range(4, 21, 4):
    H = hilbert_matrix(n)
    dominant_eigenvalue, dominant_eigenvector, iterations = power_method(H)
    results.append((n, dominant_eigenvalue, iterations))

# Create a DataFrame and print the results
results_df = pd.DataFrame(results, columns=["Matrix Size (n)", "Dominant Eigenvalue", "Iterations"])
print("Power Method Results for Hilbert Matrices:")
print(results_df)
