import numpy as np

def F(x):
    return np.array([4 * x[0]**2 + x[1]**2 - 4, x[0] + x[1] - np.sin(x[0] - x[1])])

def J(x):
    return np.array([
        [8 * x[0], 2 * x[1]],
        [1 - np.cos(x[0] - x[1]), 1 + np.cos(x[0] - x[1])]
    ])

def slacker_newton(x0, tol=1e-10, max_iter=100):
    x = x0
    J_inv = np.linalg.inv(J(x))  # Initial Jacobian inverse
    for i in range(max_iter):
        Fx = F(x)
        delta_x = -J_inv @ Fx
        x_new = x + delta_x

        if np.linalg.norm(x_new - x + J_inv @ Fx) > tol:
            # Re-evaluate the Jacobian if condition is met
            J_inv = np.linalg.inv(J(x_new))
        
        x = x_new

        if np.linalg.norm(Fx) < tol:
            print(f"Converged after {i+1} iterations.")
            return x
        
    print("Did not converge within the maximum number of iterations.")
    return x

x0 = np.array([1, 0])

solution = slacker_newton(x0)
print("Solution:", solution)
