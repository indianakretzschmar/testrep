import numpy as np

# Define A' and b'
A_prime = np.array([[1, 3],
                    [12, -2],
                    [20, 0],
                    [6, 21]])

b_prime = np.array([1, 4, 15, 12])

# Calculate A'^T * b'
A_prime_T_b_prime = A_prime.T @ b_prime
print(A_prime_T_b_prime)

# Calculate A'^T * A'
A_prime_T_A_prime = A_prime.T @ A_prime
print(A_prime_T_A_prime)
# Solve for x using the normal equations A'^T * A' * x = A'^T * b'
x = np.linalg.solve(A_prime_T_A_prime, A_prime_T_b_prime)
print(x)
