import numpy as np
import matplotlib.pyplot as plt
import random

#Part (a)
theta = np.linspace(0, 2 * np.pi, 500)
R = 1.2
delta_r = 0.1
f = 15
p = 0
x = R * (1 + delta_r * np.sin(f * theta + p)) * np.cos(theta)
y = R * (1 + delta_r * np.sin(f * theta + p)) * np.sin(theta)
plt.figure()
plt.plot(x, y)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Wavy Circle for R=1.2, δr=0.1, f=15, p=0")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#Part (b)
plt.figure()

for i in range(1, 11): 
    R = i
    delta_r = 0.05
    f = 2 + i
    p = random.uniform(0, 2)
    x = R * (1 + delta_r * np.sin(f * theta + p)) * np.cos(theta)
    y = R * (1 + delta_r * np.sin(f * theta + p)) * np.sin(theta)
    plt.plot(x, y)
    
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Wavy Circles with varying R, δr, f, and random p")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
