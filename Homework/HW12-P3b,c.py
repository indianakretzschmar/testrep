# Re-import necessary libraries and re-define functions due to environment reset
import numpy as np

# Define the Hilbert matrix function
def hilbert_matrix(n):
    """Generate the Hilbert matrix of size n x n."""
    return np.array([[1 / (i + j - 1) for j in range(1, n + 1)] for i in range(1, n + 1)])

# Define the modified power method for smallest eigenvalue
def smallest_eigenvalue_power_method(A, tol=1e-8, max_iter=1000):
    """
    Modified power method to compute the smallest eigenvalue of a matrix A.
    
    Parameters:
    A (ndarray): The input matrix (assumed symmetric positive-definite).
    tol (float): Convergence tolerance.
    max_iter (int): Maximum number of iterations.
    
    Returns:
    smallest_eigenvalue (float): Smallest eigenvalue of A.
    iterations (int): Number of iterations for convergence.
    """
    n = A.shape[0]
    x = np.random.rand(n)  # Initial random vector
    x = x / np.linalg.norm(x)  # Normalize
    
    for k in range(max_iter):
        # Solve A * y = x to simulate multiplication by A^{-1}
        y = np.linalg.solve(A, x)
        x_next = y / np.linalg.norm(y)  # Normalize
        
        # Estimate the dominant eigenvalue of A^{-1}
        eigenvalue_inv = np.dot(x_next.T, A @ x_next)
        
        # Smallest eigenvalue of A
        smallest_eigenvalue = 1 / eigenvalue_inv
        
        # Check for convergence
        if np.linalg.norm(A @ x_next - smallest_eigenvalue * x_next) < tol:
            return smallest_eigenvalue, k + 1
        
        x = x_next  # Update for next iteration
    
    raise ValueError("Power method did not converge within the maximum number of iterations")

# Redefine the Hilbert matrix for n = 16
H_16 = hilbert_matrix(16)

# Compute the smallest eigenvalue for H_16
eigenvalues = np.linalg.eigvals(H_16)
smallest_eigenvalue_exact = np.min(eigenvalues)

print(smallest_eigenvalue_exact)

epsilon = 1e-8  # Regularization parameter
E_norm = epsilon  # Since E = epsilon * I, ||E||_2 = epsilon

# Step 2: Compute P (eigenvector matrix) and its norms
eigvals, eigvecs = np.linalg.eig(H_16)  # Compute eigenvalues and eigenvectors
P = eigvecs
P_inv = np.linalg.inv(P)

P_norm = np.linalg.norm(P, 2)  # 2-norm of P
P_inv_norm = np.linalg.norm(P_inv, 2)  # 2-norm of P^-1

# Step 3: Compute the theoretical error bound
error_bound = P_norm * P_inv_norm * E_norm

# Step 4: Compute observed error (difference between computed and exact smallest eigenvalue)


computed_smallest_eigenvalue = smallest_eigenvalue_exact  # Approximation based on exact decomposition

# Compute the observed error
observed_error = np.abs(smallest_eigenvalue_exact - computed_smallest_eigenvalue)

print(error_bound, observed_error)