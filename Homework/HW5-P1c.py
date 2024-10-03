import numpy as np

def J(x, y):
    return np.array([
        [6 * x, -2 * y],
        [3 * y**2 - 3 * x**2, 6 * x * y]
    ])

def f(x,y):
    return 3*x**2 - y**2

def g(x,y):
    return 3*x*y**2 - x**3 - 1

# Starting guess
x, y = 1, 1

for i in range(10):  # Let's also do 10 iterations here
    inv_J = np.linalg.inv(J(x, y))
    values = np.array([f(x, y), g(x, y)])
    corrections = inv_J @ values
    x -= corrections[0]
    y -= corrections[1]
    print(f"Newton Iteration {i+1}: x = {x:.4f}, y = {y:.4f}")
