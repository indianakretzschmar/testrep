import numpy as np

A = 0.5 * np.array([[1, 1], [1 + 10e-10, 1 - 10e-10]])
B = np.array([[1],[1]])
norm_b = np.linalg.norm(B)
# Calculate the norm of A
norm_A = np.linalg.norm(A)

# Calculate the inverse of A
A_inv = np.linalg.inv(A)

# Calculate the norm of the inverse of A
norm_A_inv = np.linalg.norm(A_inv)

# Calculate the condition number using the formula
condition_number_formula = norm_A * norm_A_inv
print(norm_b)

# print(norm_A)
# print(norm_A_inv)
# print(condition_number_formula)
