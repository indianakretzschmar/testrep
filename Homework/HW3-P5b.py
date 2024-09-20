import numpy as np

def iteration_function(x):
    return -np.sin(2 * x) + 5 * x / 4 - 3 / 4

def fixed_point_iteration(x0, tol=1e-10, max_iter=1000):
    x = x0
    for i in range(max_iter):
        x_next = iteration_function(x)
        if abs(x_next - x) < tol:
            return x_next, i  
        x = x_next
    return None, max_iter  

initial_guesses = np.array([-.89,-.544,1.7,3.16,4.4])
roots = []

for x0 in initial_guesses:
    root, iterations = fixed_point_iteration(x0)
    if root is not None:
        roots.append(root)
        print(f"Root found: {root} after {iterations} iterations with initial guess {x0}")

unique_roots = np.unique(np.round(roots, decimals=10))
print("Unique roots:", unique_roots)
