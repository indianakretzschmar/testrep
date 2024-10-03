import numpy as np

def F(x):
    return np.array([4 * x[0]**2 + x[1]**2 - 4, x[0] + x[1] - np.sin(x[0] - x[1])])

def approximate_jacobian(x, h_factor):
    n = len(x)
    J_approx = np.zeros((n, n))
    h = h_factor * np.abs(x)  

    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1 

        J_approx[:, j] = (F(x + h[j] * e_j) - F(x)) / h[j]

    return J_approx

def newton_with_approx_jacobian(x0, tol=1e-10, max_iter=100, h_factor=1e-7):
    x = x0
    for i in range(max_iter):
        Fx = F(x)
        J_approx = approximate_jacobian(x, h_factor)
        
        delta_x = np.linalg.solve(J_approx, -Fx)
        
        x_new = x + delta_x
        
        if np.linalg.norm(Fx) < tol:
            print(f"Converged after {i+1} iterations with h_factor = {h_factor}.")
            return x
        
        x = x_new

    print("Did not converge within the maximum number of iterations.")
    return x

x0 = np.array([1, 0])

solution_1 = newton_with_approx_jacobian(x0, h_factor=1e-7)
print("Solution with h_factor = 1e-7:", solution_1)

solution_2 = newton_with_approx_jacobian(x0, h_factor=1e-3)
print("Solution with h_factor = 1e-3:", solution_2)
