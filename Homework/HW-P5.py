import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
# Define the function and its derivative
def f(x):
    return x**6 - x - 1

def df(x):
    return 6*x**5 - 1

# Newton's Method
def newtons_method(f, df, x0, max_iter=10):
    errors = [abs(x0 - alpha)]  # Initial error
    xn = x0
    for n in range(max_iter):
        fxn = f(xn)
        dfxn = df(xn)
        if dfxn == 0:
            return xn, errors, "Derivative zero. No solution found."
        xn1 = xn - fxn / dfxn
        errors.append(abs(xn1 - alpha))
        xn = xn1
    return xn, errors, None

# Secant Method
def secant_method(f, x0, x1, max_iter=10):
    errors = [abs(x0 - alpha), abs(x1 - alpha)]  # Initial errors
    xn_minus1 = x0
    xn = x1
    for n in range(max_iter):
        fxn_minus1 = f(xn_minus1)
        fxn = f(xn)
        if fxn - fxn_minus1 == 0:
            return xn, errors, "Zero difference. No solution found."
        xn1 = xn - fxn * (xn - xn_minus1) / (fxn - fxn_minus1)
        errors.append(abs(xn1 - alpha))
        xn_minus1 = xn
        xn = xn1
    return xn, errors, None

# Exact root approximation
alpha = fsolve(f, 2)[0]

# Run methods
x0_newton = 2
x0_secant, x1_secant = 2, 1
root_newton, errors_newton, error_msg_newton = newtons_method(f, df, x0_newton)
root_secant, errors_secant, error_msg_secant = secant_method(f, x0_secant, x1_secant)

# Print formatted results
print(f"Approximation of root with Newton's Method")
print(f"the approximate root is {root_newton:.12f}")
print(f"the error message reads: {error_msg_newton if error_msg_newton else '0'}")
print(f"Number of iterations: {len(errors_newton)}")
print("Error for Newton")
print("----------------")
for error in errors_newton:
    print(f"{error:.12f}")

print("\nApproximation of root with Secant Method")
print(f"the approximate root is {root_secant:.12f}")
print(f"the error message reads: {error_msg_secant if error_msg_secant else '0'}")
print(f"Number of iterations: {len(errors_secant)}")
print("Error for Secant")
print("----------------")
for error in errors_secant:
    print(f"{error:.12f}")


plt.figure(figsize=(10, 5))
plt.loglog(errors_newton[:-1], errors_newton[1:], 'o-', label='Newton\'s Method')
plt.loglog(errors_secant[:-1], errors_secant[1:], 's-', label='Secant Method')
plt.xlabel('|xk - α|')
plt.ylabel('|xk+1 - α|')
plt.title('Error Convergence Plot')
plt.legend()
plt.grid(True)
plt.show()