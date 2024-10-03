import numpy as np

def f(x,y):
    return 3*x**2 - y**2

def g(x,y):
    return 3*x*y**2 - x**3 - 1

A = np.array([[1/6, 1/18],
          [0, 1/6]])

x, y = 1, 1

for i in range(20):
    values = np.array([f(x,y), g(x,y)])
    corrections = A @ values
    x -= corrections[0]
    y -= corrections[1]
    print(f"Iteration {i+1}: x = {x:.4f}, y = {y:.4f}")

