import numpy as np
import matplotlib.pyplot as plt

# Define the function and the interval
x_vals = np.linspace(0, 5, 500)
f_exact = np.sin(x_vals)

# Define the Maclaurin polynomial of degree 6 for sin(x)
def maclaurin_sin6(x):
    return x - (x**3) / 6 + (x**5) / 120

# Calculate the Maclaurin polynomial
f_maclaurin = maclaurin_sin6(x_vals)

# Define the Padé approximations based on the formulas given
# Padé approximation P^2_4(x) from the first image
def pade_2_4_specific(x):
    return x / (1 + (x**2) / 6 + (7 * x**4) / 360)

# Padé approximation P^4_2(x) from the second image
def pade_4_2_specific(x):
    return (x - (7 * x**3) / 60) / (1 + x**2 / 20)

# Compute values for each approximation
f_pade_2_4_specific = pade_2_4_specific(x_vals)
f_pade_4_2_specific = pade_4_2_specific(x_vals)

# Calculate errors for each approximation and Maclaurin
error_maclaurin = np.abs(f_exact - f_maclaurin)
error_pade_2_4_specific = np.abs(f_exact - f_pade_2_4_specific)
error_pade_4_2_specific = np.abs(f_exact - f_pade_4_2_specific)

# Plot the errors with distinct colors for better visibility
plt.figure(figsize=(12, 8))
plt.plot(x_vals, error_maclaurin, color="blue", label="Maclaurin (Degree 6) Error")
plt.plot(x_vals, error_pade_2_4_specific, color="green", label="Padé $ P^2_4(x) $ Error")
plt.plot(x_vals, error_pade_4_2_specific, color="red", label="Padé $ P^4_2(x) $ Error")

plt.xlabel("x")
plt.ylabel("Error")
plt.title("Error Comparison of Specific Padé Approximations and Maclaurin Polynomial for sin(x)")
plt.legend()
plt.yscale("log")
plt.grid(True)
plt.show()