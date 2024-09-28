import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# Constants
Ti = 20   # Initial temperature in degrees Celsius
Ts = -15  # Surface temperature in degrees Celsius
alpha = 0.138 * 1e-6  # Thermal diffusivity in m^2/s
t = 60 * 24 * 3600  # Time in seconds (60 days)

# Function f(x)
def f(x):
    return erf(x / (2 * np.sqrt(alpha * t))) - 3/7

# Derivative f'(x)
def f_prime(x):
    return (2 / np.sqrt(np.pi)) * np.exp(-(x / (2 * np.sqrt(alpha * t)))**2) * (1 / (2 * np.sqrt(alpha * t)))

# Plotting range for x
x_values = np.linspace(0, 10, 400)
f_values = f(x_values)

# Plotting f(x)
plt.figure(figsize=(10, 5))
plt.plot(x_values, f_values, label=r"$f(x) = \text{erf}\left(\frac{x}{2\sqrt{\alpha t}}\right) - \frac{3}{7}$")
plt.axhline(0, color='gray', lw=0.5)
plt.title("Plot of the function $f(x)$")
plt.xlabel("Depth x (meters)")
plt.ylabel("$f(x)$")
plt.legend()
plt.grid(True)
plt.show()
