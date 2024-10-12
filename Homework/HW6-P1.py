import numpy as np

# Define the system of equations
def F(x):
    return np.array([
        x[0]**2 + x[1]**2 - 4,  # f(x, y) = x^2 + y^2 - 4
        np.exp(x[0]) + x[1] - 1  # g(x, y) = e^x + y - 1
    ])

# Define the Jacobian matrix
def J(x):
    return np.array([
        [2 * x[0], 2 * x[1]],  # Partial derivatives of f
        [np.exp(x[0]), 1]      # Partial derivatives of g
    ])

# Helper function to regularize the Jacobian if singular
def regularize_jacobian(J, epsilon=1e-8):
    return J + epsilon * np.eye(len(J))

# Broyden's Method with Jacobian regularization
def broyden_method(x0, tol=1e-10, max_iter=100, epsilon=1e-8):
    x = np.array(x0)
    B = J(x)  # Initial Jacobian approximation
    for i in range(max_iter):
        try:
            # Regularize the Jacobian if needed to avoid singularity
            delta_x = np.linalg.solve(regularize_jacobian(B, epsilon), -F(x))
        except np.linalg.LinAlgError:
            print("Jacobian is singular, adding perturbation.")
            B = regularize_jacobian(B, epsilon)
            delta_x = np.linalg.solve(B, -F(x))
        
        x_new = x + delta_x

        # Check for convergence
        if np.linalg.norm(delta_x) < tol:
            print(f'Broyden method converged in {i} iterations.')
            return x_new

        # Update Jacobian approximation using Broyden's update
        y = F(x_new) - F(x)
        B += np.outer((y - B @ delta_x), delta_x) / np.dot(delta_x, delta_x)
        
        x = x_new
    
    print('Broyden method did not converge within the maximum number of iterations.')
    return x

# Lazy Newton's Method with damping
def lazy_newton_method(x0, tol=1e-10, max_iter=100, jac_update_tol=1e-3, epsilon=1e-8, damping=0.8):
    x = np.array(x0)
    J_current = J(x)  # Initial Jacobian
    for i in range(max_iter):
        try:
            # Regularize the Jacobian if needed to avoid singularity
            delta_x = np.linalg.solve(regularize_jacobian(J_current, epsilon), -F(x))
        except np.linalg.LinAlgError:
            print("Jacobian is singular, adding perturbation.")
            J_current = regularize_jacobian(J_current, epsilon)
            delta_x = np.linalg.solve(J_current, -F(x))
        
        # Apply damping to the step size
        delta_x = damping * delta_x
        
        x_new = x + delta_x

        # Check for convergence
        if np.linalg.norm(delta_x) < tol:
            print(f'Lazy Newton method converged in {i} iterations.')
            return x_new

        # Only update the Jacobian if the change is large enough
        if np.linalg.norm(delta_x) > jac_update_tol:
            J_current = J(x_new)  # Update the Jacobian

        x = x_new
    
    print('Lazy Newton method did not converge within the maximum number of iterations.')
    return x

# Testing both methods with the regularization and damping strategies
initial_guesses = [
    [1, 1],
    [1, -1],
    [0, 0]
]

print("Broyden's Method Results:")
for guess in initial_guesses:
    result_broyden = broyden_method(guess)
    print(f'Initial guess {guess} -> Solution: {result_broyden}\n')

print("Lazy Newton's Method Results:")
for guess in initial_guesses:
    result_lazy_newton = lazy_newton_method(guess)
    print(f'Initial guess {guess} -> Solution: {result_lazy_newton}\n')
