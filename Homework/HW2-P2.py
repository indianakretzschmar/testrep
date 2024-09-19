import numpy as np

A = 0.5 * np.array([[1, 1], [1 + 10e-10, 1 - 10e-10]])

norm_A = np.linalg.norm(A)

A_inv = np.linalg.inv(A)

norm_A_inv = np.linalg.norm(A_inv)

condition_number_formula = norm_A * norm_A_inv

print(norm_A)
print(norm_A_inv)
print(condition_number_formula)
