import numpy as np

def fixed_point_iteration(g, x0, tol, max_iter):
    """
    Fixed-point iteration subroutine that returns a column vector of approximations
    at all iterations.
    
    Parameters:
    g: The function for fixed-point iteration (g(x))
    x0: Initial guess for the fixed point
    tol: Tolerance for stopping criterion
    max_iter: Maximum number of iterations
    
    Returns:
    x: Column vector of size (iterations, 1) with the approximations at each iteration
    """
    # Initialize the column vector to store approximations
    x = np.zeros((max_iter, 1))
    x[0] = x0  # First entry is the initial guess
    
    # Start iterating
    for i in range(1, max_iter):
        x[i] = g(x[i-1])  # Apply the fixed-point iteration formula
        # Check if the change is below the tolerance
        if abs(x[i] - x[i-1]) < tol:
            return x[:i+1]  # Return the approximations up to the current iteration
    
    # If max_iter reached, return all iterations
    return x

# Example usage:
# Define the function g(x) for the iteration
def g(x):
    return (x + 1)**(1/3)

# Call the subroutine with initial guess x0, tolerance, and maximum iterations
approximations = fixed_point_iteration(g, 0.5, 1e-5, 100)
print(approximations)
