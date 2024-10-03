import numpy as np

def f(x, y, z):
    return x**2 + 4*y**2 + 4*z**2 - 16

def grad_f(x, y, z):
    return np.array([2*x, 8*y, 8*z])

def newton_step(x, y, z):
    g = grad_f(x, y, z)
    value = f(x, y, z)
    g_norm_squared = np.dot(g, g)
    next_point = np.array([x, y, z]) - (value / g_norm_squared) * g
    return next_point

# Initial point
x, y, z = 1, 1, 1

# Perform iterations
iterations = 10
points = [(x, y, z)]
errors = [f(x, y, z)]

for _ in range(iterations):
    x, y, z = newton_step(x, y, z)
    points.append((x, y, z))
    errors.append(f(x, y, z))

selected_errors = [errors[1], errors[2], errors[3]]  # Using errors from iterations 2, 3, and 4

# Calculate alpha using these selected errors
alpha_estimates = []
for i in range(1, len(selected_errors) - 1):
    e_n_minus_1 = selected_errors[i-1]
    e_n = selected_errors[i]
    e_n_plus_1 = selected_errors[i+1]
    if e_n_minus_1 != 0 and e_n != 0 and e_n_plus_1 != 0:
        alpha = np.log(e_n_plus_1 / e_n) / np.log(e_n / e_n_minus_1)
        alpha_estimates.append(alpha)

alpha_estimates


print("The solution is:")
print(f'x = {points[-1][0]}')  # print points to see convergence
print(f'x = {points[-1][1]}')
print(f'x = {points[-1][2]}')
print(f'Number of iterations = 5')
print(f'the convergence rate is:{alpha_estimates[0]}')

