import numpy as np
import matplotlib.pyplot as plt

# Define two functions for the expressions
def func1(x, delta):
    return np.cos(x + delta) - np.cos(x)

def func2(x, delta):
    return -delta*np.sin(x)

# Define delta and x values
delta = np.logspace(-16, 0, 100)
x1 = np.pi
x2 = 1e6

# Compute the differences for x1 = pi
y11 = func1(x1, delta)
y12 = func2(x1, delta)
y13 = np.abs(y11 - y12)

# Compute the differences for x2 = 10^6
y21 = func1(x2, delta)
y22 = func2(x2, delta)
y23 = np.abs(y21 - y22)

# Plot for x = pi
plt.plot(delta, y13)
plt.xlabel("delta")
plt.ylabel("Difference between two expressions")
plt.title("Difference between the two expressions for x = pi")
plt.show()

# Plot for x = 10^6
plt.plot(delta, y23)
plt.xlabel("delta")
plt.ylabel("Difference between two expressions")
plt.title("Difference between the two expressions for x = 10^6")
plt.show()
